import numpy as np
import torch
from typing import Dict, List, Optional, Union
from PIL import Image
import clip
import wav2clip
from torcheval.metrics import FrechetInceptionDistance as FID
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import to_pil_image
import PIL
import torchaudio
from torch.utils.data import Dataset
import cv2
import PIL
import random
import numpy as np
from packaging import version
from PIL import Image
import os
import torch
import sys
import pandas as pd
import pickle
import json
import subprocess
import itertools
from collections import defaultdict  

import random
random_seed=45
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

TensorLike = Union[torch.Tensor, np.ndarray, List[float]]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_DEVICE_STR = "cuda" if device.type == "cuda" else "cpu"
_CLIP_L14 = None
_PRE_L14 = None
_BLIP_MODEL = None
_BLIP_PROC = None

MAX_NEG = 256   # cap negatives to avoid unbounded growth

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }
# ------------

def _load_models_once():
    from transformers import BlipProcessor, BlipForConditionalGeneration
    global _CLIP_L14, _PRE_L14, _BLIP_MODEL, _BLIP_PROC

    if _CLIP_L14 is None:
        _CLIP_L14, _PRE_L14 = clip.load("ViT-L/14", device=_DEVICE_STR)
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




class VGG_Sound_dataset2(Dataset):
    def __init__(
            self,
            data_dir,
            input_length=10,
            interpolation='bicubic'
    ):

        self.data = data_dir
        n_pairs = len(os.listdir(self.data)) // 4
        # idxs = range(n_pairs)
        self.input_length = input_length
        self.image_path = defaultdict(list)
        self.audio_path = defaultdict(list)
        for f in os.listdir(self.data):
            idx = int(f.split('_')[0])
            if f.endswith('.jpg'):
                self.image_path[idx].append(os.path.join(self.data, f))
            else:
                self.audio_path[idx].append(os.path.join(self.data, f))

        self.num_samples = len(self.image_path)
        self._length = self.num_samples

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]
        self.center_crop = True
        # self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length


    def img_proc(self, img):
        img = Image.open(img).convert("RGB")
        img = np.array(img).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            (
                h,
                w,
            ) = (
                img.shape[0],
                img.shape[1],
            )
            img = img[(h - crop) // 2: (h + crop) // 2, (w - crop) // 2: (w + crop) // 2]

        image = Image.fromarray(img)
        image = image.resize((512, 512), resample=self.interpolation)

        # image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = (image / 127.5 - 1.0).astype(np.float32)
        return torch.from_numpy(image).permute(2, 0, 1)


    def aud_proc_beats(self, aud, rand_sec=0):
        target_sr = getattr(self, "target_sr", 32000)
        try:
            waveform, sr = torchaudio.load(aud)
        except Exception as e:
            print(f"[AUDIO LOAD ERROR] {aud}")
            raise e
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # resample if needed
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            waveform = resampler(waveform)
            sr = target_sr

        waveform = waveform.float()  # ensure float dtype

        desired_len = int(target_sr * self.input_length)  # number of samples we want (int)
        cur_len = waveform.shape[1]

        # If shorter -> pad with zeros on the right
        if cur_len < desired_len:
            pad_amount = desired_len - cur_len
            waveform = F.pad(waveform, (0, pad_amount))  # pad on right

        # Now it's >= desired_len: compute start (clamped) and slice
        max_start = waveform.shape[1] - desired_len
        start = int(round(rand_sec * target_sr))
        if start < 0:
            start = 0
        if start > max_start:
            # clamp to valid range
            start = max(0, max_start)

        end = start + desired_len
        segment = waveform[:, start:end]  # shape [1, desired_len]
        return segment.squeeze(0)   


    def __getitem__(self, idx):
        try:
            path_a1 = self.audio_path[idx][0]
            path_a2 = self.audio_path[idx][1]
            path_b1 = self.image_path[idx][0]
            path_b2 = self.image_path[idx][1]
        except:
            print("----------")
            print(idx)
            print("----------")
        example = {}
        # name_a = os.path.splitext(os.path.basename(path_a))[0]
        # name_b = os.path.splitext(os.path.basename(path_b))[0]
        name_a1 = os.path.basename(path_a1)[:-4]
        name_a2 = os.path.basename(path_a2)[:-4].split('_')[-1]
        example['full_name'] = f"{name_a1}_{name_a2}"

        # load each audio separately
        # (you can pass different rand_secs if you like)
        example["audio_values1"] = self.aud_proc_beats(path_a1)
        example["image_values1"] = self.img_proc(path_b1)
        example["audio_values2"] = self.aud_proc_beats(path_a2)
        example["image_values2"] = self.img_proc(path_b2)
        return example






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



def compute_iis_sample(image_embedding, list_image_embeddings_batch, correct_image_idx_in_batch, random_image_catalog):
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



def evaluate_all(generated_image, list_batch_images, list_label, current_audio_idx_in_batch):
    _load_models_once()
    results: Dict[str, Dict] = {}
    N = len(list_batch_images)

    with torch.no_grad():
        img_l14_in = _PRE_L14(generated_image).unsqueeze(0).to(
            device=device, dtype=next(_CLIP_L14.parameters()).dtype
        )
        img_l14_gen = normalize_embeddings(_CLIP_L14.encode_image(img_l14_in).float())

    list_image_embeddings_batch = []
    random_img_cat = []

    for i in range(N):
        batch_images = list_batch_images[i]

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

    # Stack negatives safely
    random_img_cat = torch.cat(random_img_cat, dim=0) if random_img_cat else torch.zeros((0, img_l14_gen.shape[1]), device=device)

    # -------- METRICS --------
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
    return results






