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
    "accordion", "acousticguitar", "airplane", "alarmclock", "alligator", "ambulance", "baby", "babylaughter", 
    "banjo", "bassoon", "bird", "blowtorch", "bull", "bus", "canary", "capgun", "car", "cat", "cattle", "cello",
    "chainsaw", "cheetah", "chicken", "child", "chimpanzee", "chinchilla", "chipmunk", "chips", "churchbell", 
    "computerkeyboard", "cornet", "cow", "cricket", "crocodile", "crow", "cymbal", "dinosaur", "dog", "donkey", "drum", 
    "eagle", "electricblender", "electricgrinder", "electricguitar", "electronicorgan", "elephant", "elk", "female", 
    "firetruck", "firework", "flute", "fox", "frenchhorn", "gibbon", "goat", "guitar", "hairdryer", "harp", "harpsichord", 
    "hedgetrimmer", "helicopter", "horse", "icecreamtruck", "lathe", "lawnmower", "lion", "machinegun", "male", "mandolin", 
    "marching", "mice", "missile", "motorboat", "motorcycle", "mouse", "oboe", "ocean", "orchestra", "owl", "parrot",
    "penguin", "people", "piano", "pigeon", "policecar", "popcorn", "popcornmachine", "racecar", "river", "saxophone", "sealion",
    "sheep", "shofar", "skateboard", "slotmachine", "snake", "snowmobile", "stream", "subway", "tapdancing", "telephone",
    "timbale", "tractor", "train", "trumpet", "turkey", "ukulele", "vacuumcleaner", "waterfall", "whale", "windchime", "woodpecker"
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


# def _load_models_once():
#     global _CLIP_B32, _PRE_B32, _CLIP_L14, _PRE_L14, _WAV2C
#     if _WAV2C is None:
#         _WAV2C = wav2clip.get_model().to(device).eval()
#     if _CLIP_B32 is None or _CLIP_L14 is None:
#         _CLIP_B32, _PRE_B32 = clip.load("ViT-B/32", device=_DEVICE_STR)
#         _CLIP_L14, _PRE_L14 = clip.load("ViT-L/14", device=_DEVICE_STR)
#         _CLIP_B32 = _CLIP_B32.eval().to(device)
#         _CLIP_L14 = _CLIP_L14.eval().to(device)

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

def _to_pil(image):
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, str):
        return Image.open(image).convert("RGB")
    if isinstance(image, torch.Tensor):
        # tensor expected in C,H,W or H,W,[C]
        t = image.detach().cpu()
        if t.ndim == 3 and t.shape[0] in (1,3):  # C,H,W
            t = t.permute(1, 2, 0).numpy()
        else:
            t = t.numpy()
        # scale float tensors to 0-255
        if np.issubdtype(t.dtype, np.floating):
            t = (np.clip(t, 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(t).convert("RGB")
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 3 and arr.shape[2] == 3:
            pass
        elif arr.ndim == 3 and arr.shape[0] == 3:  # C,H,W
            arr = np.transpose(arr, (1, 2, 0))
        # float -> uint8
        if np.issubdtype(arr.dtype, np.floating):
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")
    raise TypeError("Unsupported image type. Provide PIL.Image, file path, torch tensor or numpy array.")


def get_text_embeddings(model, template="a photo of a {}"):
    prompts = [template.format(c) for c in class_names]
    tokens = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        text_emb = model.encode_text(tokens)  # (N, D)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    return text_emb 


def classify_image(image, model_choice="L14", template="a photo of a {}", top_k=1):
    _load_models_once()
    if model_choice == "B32":
        model, preprocess = _CLIP_B32, _PRE_B32
    else:
        model, preprocess = _CLIP_L14, _PRE_L14

    # prepare image
    pil = _to_pil(image)
    x = preprocess(pil).unsqueeze(0).to(device=device, dtype=next(model.parameters()).dtype)

    # prepare text embeddings if not provided
    text_embeddings = get_text_embeddings(model)

    with torch.no_grad():
        img_emb = model.encode_image(x)  # (1, D)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)  # normalized

        # text_embeddings shape (N, D); compute similarity
        sims = (img_emb @ text_embeddings.T).squeeze(0)  # (N,)
        # sims are cosine similarities in [-1, 1]
        topk_vals, topk_idx = sims.topk(k=top_k)

    results = [(class_names[i], float(topk_vals[idx].item())) for idx, i in enumerate(topk_idx.cpu().numpy())]

    if top_k == 1:
        return results[0]  # (class, score)
    return results


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




def fid(x, y):
    metric = FID().to(device)
    metric.update(x, is_real=False)
    metric.update(y, is_real=True)
    return metric.compute().item()

def get_fid(generated_images, target_images):
    generated_images = generated_images.float()
    target_images = target_images.float()

    # Ensure both are in [0, 1]
    if generated_images.min() < 0 or generated_images.max() > 1:
        generated_images = (generated_images + 1) / 2
    if target_images.min() < 0 or target_images.max() > 1:
        target_images = (target_images + 1) / 2

    fid_score = fid(generated_images, target_images)
    return fid_score



def compute_wav2clip_ais_sample( image_embedding, list_audio_embeddings_batch, correct_audio_idx_in_batch, random_audio_catalog):
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
        correct_audio_emb = audios[correct_audio_idx_in_batch].unsqueeze(0)
        correct_score = float(pairwise_cos_sim(img, correct_audio_emb).item())
        if K > 0:
            total_wins += int((correct_score > random_scores).sum())

    ais = float(total_wins / total_comparisons) if total_comparisons > 0 else 0.0
    return {"ais": ais, "wins": total_wins}



def compute_iis_sample( image_embedding, list_image_embeddings_batch, correct_image_idx_in_batch, random_image_catalog):
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

        correct_idx = correct_image_idx_in_batch
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
    N = len(list_batch_images)

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

    for i in range(N):
        batch_images = list_batch_images[i]
        batch_audios = list_batch_audios[i]

        # -------- IMAGE EMBEDDINGS --------
        with torch.no_grad():
            batch_images_pil = torch.stack([
                _PRE_L14(to_pil_image(img).convert("RGB")) for img in batch_images
            ]).to(device=device, dtype=next(_CLIP_L14.parameters()).dtype)

            img_l14_batch = normalize_embeddings(
                _CLIP_L14.encode_image(batch_images_pil).float()
            )

        list_image_embeddings_batch.append(img_l14_batch)

        # collect image negatives
        if img_l14_batch.shape[0] > 1:
            neg = torch.cat([
                img_l14_batch[:current_audio_idx_in_batch],
                img_l14_batch[current_audio_idx_in_batch + 1:]
            ], dim=0)
            random_img_cat.append(neg[:MAX_NEG])

        # -------- AUDIO EMBEDDINGS (CPU SAFE) --------
        with torch.no_grad():
            audio_np = batch_audios.cpu().numpy()
            audio_emb_np = wav2clip.embed_audio(audio_np, _WAV2C)

        audio_w2c_batch = normalize_embeddings(
            torch.from_numpy(audio_emb_np).float().to(device)
        )

        list_audio_embeddings_batch.append(audio_w2c_batch)

        if audio_w2c_batch.shape[0] > 1:
            neg = torch.cat([
                audio_w2c_batch[:current_audio_idx_in_batch],
                audio_w2c_batch[current_audio_idx_in_batch + 1:]
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
        img_b32_gen, list_audio_embeddings_batch, current_audio_idx_in_batch, random_audio_cat
    )

    text_label_embs = get_text_embeddings(_CLIP_L14)
    results["aic"] = compute_aic_sample(img_l14_gen, text_label_embs, list_label)

    results["iis"] = compute_iis_sample(
        img_l14_gen, list_image_embeddings_batch, current_audio_idx_in_batch, random_img_cat
    )
    
    res_cls = caption_label_similarity(generated_image, list_label, 1)
    results['cls'] = res_cls
    list_images = []
    for i in range(len(list_batch_images)):
        list_images.append(list_batch_images[i][current_audio_idx_in_batch])
    res_ccs = caption_caption_similarity(generated_image, list_images, 1)
    results['ccs'] = res_ccs

    # FINAL CLEANUP
    del random_audio_cat, random_img_cat, list_audio_embeddings_batch, list_image_embeddings_batch
    torch.cuda.empty_cache()

    return results






