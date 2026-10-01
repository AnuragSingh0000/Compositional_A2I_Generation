import numpy as np
import torch
from typing import Dict, List, Optional, Union
import PIL
from PIL import Image
import clip
import wav2clip
from torcheval.metrics import FrechetInceptionDistance as FID
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import to_pil_image
from transformers import BlipProcessor, BlipForConditionalGeneration

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

def _load_models_once():
    global _CLIP_B32, _PRE_B32, _CLIP_L14, _PRE_L14, _WAV2C, _BLIP_MODEL, _BLIP_PROC
    # if _WAV2C is None:
    #     _WAV2C = wav2clip.get_model().to(device).eval()
    if _WAV2C is None:
        _WAV2C = wav2clip.get_model().cpu().eval()
    if _CLIP_B32 is None or _CLIP_L14 is None:
        _CLIP_B32, _PRE_B32 = clip.load("ViT-B/32", device=_DEVICE_STR)
        _CLIP_L14, _PRE_L14 = clip.load("ViT-L/14", device=_DEVICE_STR)
        _CLIP_B32 = _CLIP_B32.eval().to(device)
        _CLIP_L14 = _CLIP_L14.eval().to(device)
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


def caption_label_similarity(image, labels: List[str], num_captions: int = 1) -> Dict:
    image = _to_pil(image)

    # ---- Caption generation ----
    inputs = _BLIP_PROC(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = _BLIP_MODEL.generate(
            **inputs, max_length=30, num_return_sequences=num_captions
        )
    captions = _BLIP_PROC.batch_decode(out, skip_special_tokens=True)

    # ---- Encode captions + labels ----
    cap_tokens = clip.tokenize(captions).to(device)
    lbl_tokens = clip.tokenize(labels).to(device)

    with torch.no_grad():
        cap_emb = normalize_embeddings(_CLIP_L14.encode_text(cap_tokens).float())
        lbl_emb = normalize_embeddings(_CLIP_L14.encode_text(lbl_tokens).float())

    sims = (cap_emb @ lbl_emb.T).cpu().numpy()  # (C, L)

    best_scores = sims.max(axis=0)

    return {
        "captions": captions,
        "similarity": float(best_scores[0])
    }


def caption_caption_similarity(gen_img, real_img, num_captions) -> Dict:
    gen_img = _to_pil(gen_img)
    real_img = _to_pil(real_img)

    # ---- Caption generation ----
    gen_inputs = _BLIP_PROC(images=gen_img, return_tensors="pt").to(device)
    with torch.no_grad():
        gen_out = _BLIP_MODEL.generate(
            **gen_inputs,
            max_length=30,
            num_return_sequences=num_captions
        )
    gen_captions = _BLIP_PROC.batch_decode(gen_out, skip_special_tokens=True)

    real_inputs = _BLIP_PROC(images=real_img, return_tensors="pt").to(device)
    with torch.no_grad():
        real_out = _BLIP_MODEL.generate(
            **real_inputs,
            max_length=30,
            num_return_sequences=1
        )
    real_captions = _BLIP_PROC.batch_decode(real_out, skip_special_tokens=True)

    # ---- Encode captions ----
    gen_tokens = clip.tokenize(gen_captions).to(device)
    real_tokens = clip.tokenize(real_captions).to(device)

    with torch.no_grad():
        gen_emb = normalize_embeddings(_CLIP_L14.encode_text(gen_tokens).float())
        real_emb = normalize_embeddings(_CLIP_L14.encode_text(real_tokens).float())

    sims = gen_emb @ real_emb.T  # (C, 1)
    score = sims.max().item()

    return {
        "generated_captions": gen_captions,
        "real_caption": real_captions[0],
        "similarity": score
    }


def fid(metric, x, y):
    metric.update(x, is_real=False)
    metric.update(y, is_real=True)

def process_image(fid_metric, generated_images, target_images):
    if isinstance(generated_images, list):
        generated_images = torch.stack(generated_images, dim=0)

    if isinstance(target_images, list):
        target_images = torch.stack(target_images, dim=0)
    generated_images = generated_images.float()
    target_images = target_images.float()

    # Ensure both are in [0, 1]
    if generated_images.min() < 0 or generated_images.max() > 1:
        generated_images = (generated_images + 1) / 2
    if target_images.min() < 0 or target_images.max() > 1:
        target_images = (target_images + 1) / 2
    fid(fid_metric, generated_images, target_images)


def compute_wav2clip_ais_sample(image_embedding, audio_embeddings_batch, correct_audio_idx_in_batch, random_audio_catalog):
    img = ensure_tensor(image_embedding, device_arg=device).unsqueeze(0)
    audios = ensure_tensor(audio_embeddings_batch, device_arg=device)
    random_cat = ensure_tensor(random_audio_catalog, device_arg=device)
    img = normalize_embeddings(img)
    audios = normalize_embeddings(audios)
    random_cat = normalize_embeddings(random_cat)
    correct_audio_emb = audios[correct_audio_idx_in_batch].unsqueeze(0)
    correct_score = float(pairwise_cos_sim(img, correct_audio_emb).item())
    random_scores = pairwise_cos_sim(img, random_cat).cpu().numpy().squeeze() if random_cat.shape[0] > 0 else np.array([])
    wins = int((correct_score > random_scores).sum()) if random_scores.size > 0 else 0
    K = random_scores.shape[0] if random_scores.size > 0 else 0
    ais = float(wins / K) if K > 0 else 0.0
    return {"ais": ais, "correct_score": correct_score, "random_scores": random_scores, "wins": wins, "K": K}

def compute_iis_sample(image_embedding, dataset_image_embeddings_batch, correct_image_idx_in_batch, random_image_catalog):
    img = ensure_tensor(image_embedding, device_arg=device).unsqueeze(0)
    dataset_emb = ensure_tensor(dataset_image_embeddings_batch, device_arg=device)
    random_cat = ensure_tensor(random_image_catalog, device_arg=device)
    img = normalize_embeddings(img)
    dataset_emb = normalize_embeddings(dataset_emb)
    random_cat = normalize_embeddings(random_cat)
    correct_emb = dataset_emb[correct_image_idx_in_batch].unsqueeze(0)
    correct_score = float(pairwise_cos_sim(img, correct_emb).item())
    random_scores = pairwise_cos_sim(img, random_cat).cpu().numpy().squeeze() if random_cat.shape[0] > 0 else np.array([])
    wins = int((correct_score > random_scores).sum()) if random_scores.size > 0 else 0
    K = random_scores.shape[0] if random_scores.size > 0 else 0
    iss = float(wins / K) if K > 0 else 0.0
    return {"iss": iss, "correct_score": correct_score, "random_scores": random_scores, "wins": wins, "K": K}

def compute_aic_sample(image_embedding, text_label_embeddings, correct_label=None):
    img = ensure_tensor(image_embedding, device_arg=device).unsqueeze(0)
    text_emb = ensure_tensor(text_label_embeddings, device_arg=device)
    img = normalize_embeddings(img)
    text_emb = normalize_embeddings(text_emb)
    sims = pairwise_cos_sim(img, text_emb).cpu().numpy().squeeze()
    pred = int(np.argmax(sims))
    pred_score = float(sims[pred])
    # print(pred, class_names[pred], correct_label)
    if (correct_label is not None and class_names[pred] == correct_label):
        accuracy = 1
    else:
        accuracy = 0
    return {"pred": pred, "pred_score": pred_score, "accuracy": accuracy}

def evaluate_all(generated_image, batch_images, batch_audios, label, current_audio_idx_in_batch):
    _load_models_once()
    global _CLIP_B32, _PRE_B32, _CLIP_L14, _PRE_L14, _WAV2C
    results: Dict[str, Dict] = {}

    img_l14_in = _PRE_L14(generated_image).unsqueeze(0).to(device=device, dtype=next(_CLIP_L14.parameters()).dtype)
    img_b32_in = _PRE_B32(generated_image).unsqueeze(0).to(device=device, dtype=next(_CLIP_B32.parameters()).dtype)
    batch_images_pil = torch.stack([_PRE_L14(to_pil_image(img).convert("RGB")) 
                                for img in batch_images], dim=0).to(device=device, dtype=next(_CLIP_L14.parameters()).dtype)

    with torch.no_grad():
        img_b32_gen = _CLIP_B32.encode_image(img_b32_in).float()
        img_l14_gen = _CLIP_L14.encode_image(img_l14_in).float()
    img_b32_gen = normalize_embeddings(img_b32_gen)
    img_l14_gen = normalize_embeddings(img_l14_gen)
    batch_images_tensor = batch_images_pil.unsqueeze(0).to(device)
    with torch.no_grad():
        img_l14_batch = _CLIP_L14.encode_image(batch_images_tensor[0]).float()
    
    img_l14_batch = normalize_embeddings(img_l14_batch)
    audio_np = batch_audios.cpu().numpy()
    audio_emb_np = wav2clip.embed_audio(audio_np, _WAV2C)
    audio_w2c_batch = torch.from_numpy(audio_emb_np).to(device).float()
    audio_w2c_batch = normalize_embeddings(audio_w2c_batch)
    if audio_w2c_batch is not None:
        if audio_w2c_batch.shape[0] > 1:
            random_audio_cat = torch.cat([audio_w2c_batch[:current_audio_idx_in_batch], audio_w2c_batch[current_audio_idx_in_batch+1:]], dim=0)
        else:
            random_audio_cat = torch.zeros((0, audio_w2c_batch.shape[1]), device=device)
        res_wav2 = compute_wav2clip_ais_sample(img_b32_gen, audio_w2c_batch, current_audio_idx_in_batch, random_audio_cat)
        results['ais'] = res_wav2
        text_label_embs = get_text_embeddings(_CLIP_L14)
        res_audioimage = compute_aic_sample(img_l14_gen, text_label_embs, label)
        results['aic'] = res_audioimage
    if (img_l14_gen is not None) and (img_l14_batch is not None):
        if img_l14_batch.shape[0] > 1:
            random_img_cat = torch.cat([img_l14_batch[:current_audio_idx_in_batch], img_l14_batch[current_audio_idx_in_batch+1:]], dim=0)
        else:
            random_img_cat = torch.zeros((0, img_l14_batch.shape[1]), device=device)
        res_iis = compute_iis_sample(img_l14_gen, img_l14_batch, current_audio_idx_in_batch, random_img_cat)
        results['iis'] = res_iis
    res_cls = caption_label_similarity(generated_image, [label], 1)
    results['cls'] = res_cls
    res_ccs = caption_caption_similarity(generated_image, batch_images[current_audio_idx_in_batch], 1)
    results['ccs'] = res_ccs
        
    return results