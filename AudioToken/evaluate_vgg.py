import numpy as np
import torch
from typing import Dict, List, Optional, Union
from PIL import Image
import clip
import wav2clip
from torcheval.metrics import FrechetInceptionDistance as FID
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import to_pil_image
from transformers import BlipProcessor, BlipForConditionalGeneration
import PIL

import random
random_seed=45
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

class_names = [
    "airplane flyby", "ambulance siren", "baby crying", "baby laughter", "car engine idling",
    "cat purring", "chainsawing trees", "church bell ringing", "train whistling", "train horning",
    "cow lowing", "dog barking", "elk bugling", "fire truck siren", "gibbon howling",
    "hail", "lawn mowing", "volcano explosion", "stream burbling", "waterfall burbling",
    "orchestra", "owl hooting", "people cheering", "people crowd", "people marching",
    "playing bassoon", "playing drum kit", "playing harp", "playing harpsichord", "playing snare drum",
    "playing timbales", "printer printing", "scuba diving", "sea waves", "sheep bleating",
    "singing choir", "skiing", "slot machine", "snake hissing", "tractor digging",
    "underwater bubbling", "railroad car, train wagon", "train wheels squealing", "baltimore oriole calling",
    "car engine knocking", "driving snowmobile", "wood thrush calling", "hedge trimmer running",
    "ice cream truck, ice cream van", "woodpecker pecking tree"
]

TensorLike = Union[torch.Tensor, np.ndarray, List[float]]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_DEVICE_STR = "cuda" if device.type == "cuda" else "cpu"
_CLIP_B32 = None
_PRE_B32 = None
_CLIP_L14 = None
_PRE_L14 = None
_WAV2C = None
_BLIP_MODEL = None
_BLIP_PROC = None

MAX_NEG = 256   # cap negatives to avoid unbounded growth

def _load_models_once():
    global _CLIP_B32, _PRE_B32, _CLIP_L14, _PRE_L14, _WAV2C, _BLIP_MODEL, _BLIP_PROC
    if _WAV2C is None:
        # wav2clip MUST stay on CPU
        _WAV2C = wav2clip.get_model().cpu().eval()

    if _CLIP_B32 is None or _CLIP_L14 is None:
        _CLIP_B32, _PRE_B32 = clip.load("ViT-B/32", device=_DEVICE_STR)
        _CLIP_L14, _PRE_L14 = clip.load("ViT-L/14", device=_DEVICE_STR)
        _CLIP_B32.eval().to(device)
        _CLIP_L14.eval().to(device)
    if _BLIP_MODEL is None:
        _BLIP_PROC = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        _BLIP_MODEL = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(device).eval()

def ensure_tensor(x, device_arg=None, dtype=None):
    if isinstance(x, torch.Tensor):
        t = x
    else:
        t = torch.tensor(np.array(x))
    if dtype is not None:
        t = t.to(dtype)
    dev = device_arg if device_arg is not None else device
    if dev is not None:
        t = t.to(dev)
    return t

def normalize_embeddings(emb, eps=1e-10):
    norm = emb.norm(dim=-1, keepdim=True).clamp_min(eps)
    return emb / norm

def pairwise_cos_sim(A, B):
    return torch.matmul(A.float(), B.float().t())



def _to_pil(img):
    # 1. Convert to numpy if it's a PyTorch tensor
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
        
    # 2. Squeeze out unnecessary dimensions of size 1 
    # This turns (1, 1, 512, 512) into (512, 512)
    # or (1, 3, 512, 512) into (3, 512, 512)
    img = np.squeeze(img)
    
    # 3. Handle PyTorch Channel-First format (C, H, W) -> PIL Channel-Last format (H, W, C)
    if img.ndim == 3 and img.shape[0] in [1, 3, 4]:
        img = np.transpose(img, (1, 2, 0))
        
    # 4. If we ended up with (H, W, 1), squeeze it down to (H, W) for PIL grayscale
    if img.ndim == 3 and img.shape[2] == 1:
        img = np.squeeze(img, axis=2)
        
    # 5. Ensure the data type is uint8 (0-255) before passing to PIL
    if img.dtype != np.uint8:
        if np.issubdtype(img.dtype, np.floating):
            # FIX: Force float arrays into [0, 1] bounds before multiplying by 255
            # This perfectly mimics V1's behavior and prevents integer underflow
            img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

    return Image.fromarray(img).convert("RGB")



def get_text_embeddings(model, template="a photo of a {}"):
    prompts = [template.format(c) for c in class_names]
    tokens = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        text_emb = model.encode_text(tokens)  # (N, D)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    return text_emb 

def caption_label_similarity(image, labels: List[str], num_captions: int = 2) -> Dict:
    image = _to_pil(image)

    # ---- Caption generation ----
    inputs = _BLIP_PROC(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = _BLIP_MODEL.generate(
            **inputs,
            max_length=30,
            num_return_sequences=num_captions
        )
    captions = _BLIP_PROC.batch_decode(out, skip_special_tokens=True)

    # ---- Encode captions + labels ----
    cap_tokens = clip.tokenize(captions).to(device)
    lbl_tokens = clip.tokenize(labels).to(device)

    with torch.no_grad():
        cap_emb = normalize_embeddings(
            _CLIP_L14.encode_text(cap_tokens).float()
        )
        lbl_emb = normalize_embeddings(
            _CLIP_L14.encode_text(lbl_tokens).float()
        )

    # (C, L)
    sims = cap_emb @ lbl_emb.T

    # Best caption per label
    best_per_label = sims.max(dim=0).values  # (L,)
    scores = torch.clamp(best_per_label, min=1e-6)
    gm_cls = torch.exp(torch.mean(torch.log(scores)))

    return {
        "captions": captions,
        "per_label_similarity": best_per_label.cpu().tolist(),
        "gm": gm_cls.item()
    }

def caption_caption_similarity(
    gen_img,
    real_imgs: List,
    num_captions: int = 2
) -> Dict:
    gen_img = _to_pil(gen_img)
    real_imgs = [_to_pil(img) for img in real_imgs]

    # ---- Generate captions for generated image ----
    gen_inputs = _BLIP_PROC(images=gen_img, return_tensors="pt").to(device)
    with torch.no_grad():
        gen_out = _BLIP_MODEL.generate(
            **gen_inputs,
            max_length=30,
            num_return_sequences=num_captions
        )
    gen_captions = _BLIP_PROC.batch_decode(gen_out, skip_special_tokens=True)

    # ---- Generate captions for real images ----
    real_captions = []
    for img in real_imgs:
        inputs = _BLIP_PROC(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            out = _BLIP_MODEL.generate(
                **inputs,
                max_length=30,
                num_return_sequences=1
            )
        real_captions.append(
            _BLIP_PROC.batch_decode(out, skip_special_tokens=True)[0]
        )

    # ---- Encode captions ----
    gen_tokens = clip.tokenize(gen_captions).to(device)
    real_tokens = clip.tokenize(real_captions).to(device)

    with torch.no_grad():
        gen_emb = normalize_embeddings(
            _CLIP_L14.encode_text(gen_tokens).float()
        )
        real_emb = normalize_embeddings(
            _CLIP_L14.encode_text(real_tokens).float()
        )

    # (C, R)
    sims = gen_emb @ real_emb.T

    # Best generated caption per real image
    best_per_real = sims.max(dim=0).values  # (R,)
    scores = torch.clamp(best_per_real, min=1e-6)
    gm_ccs = torch.exp(torch.mean(torch.log(scores)))

    return {
        "generated_captions": gen_captions,
        "real_captions": real_captions,
        "per_real_similarity": best_per_real.cpu().tolist(),
        "gm": gm_ccs.item()
    }


def compute_wav2clip_ais_sample(image_embedding, list_audio_embeddings_batch, correct_indices, random_audio_catalog):
    img = ensure_tensor(image_embedding, device_arg=device).unsqueeze(0)
    img = normalize_embeddings(img)
    random_cat = ensure_tensor(random_audio_catalog, device_arg=device)
    random_cat = normalize_embeddings(random_cat)
    random_scores = (
        pairwise_cos_sim(img, random_cat).cpu().numpy().squeeze()
        if random_cat.shape[0] > 0 else np.array([])
    )

    K = random_scores.shape[0]
    N = len(list_audio_embeddings_batch)
    total_wins = 0
    total_comparisons = N * K

    for idx in range(N):
        audios = ensure_tensor(list_audio_embeddings_batch[idx], device_arg=device)
        audios = normalize_embeddings(audios)
        
        correct_idx = correct_indices[idx]
        correct_audio_emb = audios[correct_idx].unsqueeze(0)
        
        correct_score = float(pairwise_cos_sim(img, correct_audio_emb).item())
        if K > 0:
            total_wins += int((correct_score > random_scores).sum())

    ais = float(total_wins / total_comparisons) if total_comparisons > 0 else 0.0
    return {"ais": ais, "wins": total_wins}


def compute_iis_sample(image_embedding, list_image_embeddings_batch, correct_indices, random_image_catalog):
    img = ensure_tensor(image_embedding, device_arg=device).unsqueeze(0)
    img = normalize_embeddings(img)
    random_cat = ensure_tensor(random_image_catalog, device_arg=device)
    random_cat = normalize_embeddings(random_cat)
    random_scores = (
        pairwise_cos_sim(img, random_cat).cpu().numpy().squeeze()
        if random_cat.shape[0] > 0 else np.array([])
    )
    K = random_scores.shape[0]
    N = len(list_image_embeddings_batch)
    total_wins = 0
    total_comparisons = N * K

    for i in range(N):
        dataset_emb = ensure_tensor(list_image_embeddings_batch[i], device_arg=device)
        dataset_emb = normalize_embeddings(dataset_emb)

        correct_idx = correct_indices[i]
        correct_emb = dataset_emb[correct_idx].unsqueeze(0)

        correct_score = float(pairwise_cos_sim(img, correct_emb).item())

        if K > 0:
            total_wins += int((correct_score > random_scores).sum())

    iss = float(total_wins / total_comparisons) if total_comparisons > 0 else 0.0

    return {
        "iss": iss,
        "wins": total_wins,
        "K": K,
        "N": N
    }

def compute_aic_sample(image_embedding, text_label_embeddings, correct_label_list=None):
    if correct_label_list is None:
        correct_label_list = []

    img = ensure_tensor(image_embedding, device_arg=device).unsqueeze(0)
    text_emb = ensure_tensor(text_label_embeddings, device_arg=device)
    img = normalize_embeddings(img)
    text_emb = normalize_embeddings(text_emb)

    # cosine similarities
    sims = pairwise_cos_sim(img, text_emb).cpu().numpy().squeeze()  # (num_classes,)

    # Top-K where K = number of correct labels
    K = max(1, len(correct_label_list))
    topk_idx = np.argsort(-sims)[:K]
    topk_labels = [class_names[i] for i in topk_idx]

    accuracy = 0
    for label in correct_label_list:
        if label in topk_labels:
            accuracy += 1
    accuracy = accuracy / len(correct_label_list) if len(correct_label_list) > 0 else 0.0
    return { "topk_idx": topk_idx.tolist(), "accuracy": accuracy}


def evaluate_all(generated_image, list_batch_images, list_batch_audios, list_label, current_audio_idx_in_batch):
    _load_models_once()
    results: Dict[str, Dict] = {}
    
    generated_image = _to_pil(generated_image)
    
    current_idx = current_audio_idx_in_batch
    current_labels = list_label[current_idx]
    
    # Only iterate over the number of objects present in the specific scene we are evaluating
    num_objects_in_current_scene = len(list_batch_images[current_idx])
    
    if num_objects_in_current_scene == 0:
        return results # Failsafe against empty scenes

    B = len(list_batch_images)

    with torch.no_grad():
        img_l14_in = _PRE_L14(generated_image).unsqueeze(0).to(
            device=device, dtype=next(_CLIP_L14.parameters()).dtype
        )
        img_b32_in = _PRE_B32(generated_image).unsqueeze(0).to(
            device=device, dtype=next(_CLIP_B32.parameters()).dtype
        )

        img_b32_gen = normalize_embeddings(_CLIP_B32.encode_image(img_b32_in).float())
        img_l14_gen = normalize_embeddings(_CLIP_L14.encode_image(img_l14_in).float())

    list_audio_embeddings_batch = []
    list_image_embeddings_batch = []
    random_audio_cat = []
    random_img_cat = []
    
    list_correct_indices = []

    # Dynamically gather matching objects across the batch
    for o in range(num_objects_in_current_scene):
        obj_images = []
        obj_audios = []
        correct_idx_for_this_obj = -1
        
        for b in range(B):
            # Only append if this batch item actually has an object at index 'o'
            if o < len(list_batch_images[b]):
                if b == current_idx:
                    correct_idx_for_this_obj = len(obj_images)
                obj_images.append(list_batch_images[b][o])
                obj_audios.append(list_batch_audios[b][o])
                
        list_correct_indices.append(correct_idx_for_this_obj)
        
        batch_images_t = torch.cat(obj_images, dim=0)
        batch_audios_t = torch.cat(obj_audios, dim=0)

        # -------- IMAGE EMBEDDINGS --------
        with torch.no_grad():
            batch_images_pil = torch.stack([
                _PRE_L14(to_pil_image(img).convert("RGB")) for img in batch_images_t
            ]).to(device=device, dtype=next(_CLIP_L14.parameters()).dtype)

            img_l14_batch = normalize_embeddings(
                _CLIP_L14.encode_image(batch_images_pil).float()
            )

        list_image_embeddings_batch.append(img_l14_batch)

        # collect image negatives (skipping the correct one)
        if img_l14_batch.shape[0] > 1:
            neg = torch.cat([
                img_l14_batch[:correct_idx_for_this_obj],
                img_l14_batch[correct_idx_for_this_obj + 1:]
            ], dim=0)
            random_img_cat.append(neg[:MAX_NEG])

        # -------- AUDIO EMBEDDINGS (CPU SAFE) --------
        with torch.no_grad():
            audio_np = batch_audios_t.cpu().numpy()
            audio_emb_np = wav2clip.embed_audio(audio_np, _WAV2C)

        audio_w2c_batch = normalize_embeddings(
            torch.from_numpy(audio_emb_np).float().to(device)
        )

        list_audio_embeddings_batch.append(audio_w2c_batch)

        # collect audio negatives (skipping the correct one)
        if audio_w2c_batch.shape[0] > 1:
            neg = torch.cat([
                audio_w2c_batch[:correct_idx_for_this_obj],
                audio_w2c_batch[correct_idx_for_this_obj + 1:]
            ], dim=0)
            random_audio_cat.append(neg[:MAX_NEG])

        # ---- IMMEDIATE CLEANUP ----
        del batch_images_pil, img_l14_batch, audio_w2c_batch
        torch.cuda.empty_cache()

    # Stack negatives safely
    random_audio_cat = torch.cat(random_audio_cat, dim=0) if random_audio_cat else torch.zeros((0, 512), device=device)
    random_img_cat = torch.cat(random_img_cat, dim=0) if random_img_cat else torch.zeros((0, img_l14_gen.shape[1]), device=device)

    # -------- METRICS --------
    results["ais"] = compute_wav2clip_ais_sample(
        img_b32_gen, list_audio_embeddings_batch, list_correct_indices, random_audio_cat
    )

    text_label_embs = get_text_embeddings(_CLIP_L14)
    results["aic"] = compute_aic_sample(img_l14_gen, text_label_embs, current_labels)

    results["iis"] = compute_iis_sample(
        img_l14_gen, list_image_embeddings_batch, list_correct_indices, random_img_cat
    )
    
    res_cls = caption_label_similarity(generated_image, current_labels, 1)
    results['cls'] = res_cls
    
    # Real images for the current scene are simply the list of images at the current index
    res_ccs = caption_caption_similarity(generated_image, list_batch_images[current_idx], 1)
    results['ccs'] = res_ccs

    # FINAL CLEANUP
    del random_audio_cat, random_img_cat, list_audio_embeddings_batch, list_image_embeddings_batch
    torch.cuda.empty_cache()

    return results