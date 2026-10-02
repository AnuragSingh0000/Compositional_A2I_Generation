# ============================================================
# VGG-Sound: Audio → Image Generation + Evaluation
# ============================================================
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "CLAP", "src"))
import torch
from PIL import Image
from tqdm import tqdm

import json
import gradio as gr
from glob import glob
from omegaconf import OmegaConf
from datetime import datetime
from safetensors import safe_open

from PIL import Image
from unet2d_custom import UNet2DConditionModel
from pipeline_stable_diffusion_custom import StableDiffusionPipeline
from ldm.modules.encoders.audio_projector_res import Adapter
from utils.metrics import evaluate_all
from diffusers import logging
logging.set_verbosity_error()

import random
from torchvision.transforms.functional import to_tensor
import PIL
import torchaudio
import numpy as np
from torch.utils.data import Dataset
import PIL
import cv2
from packaging import version
from utils.metrics import evaluate_all

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
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }

seed_n = 45
random.seed(seed_n)
np.random.seed(seed_n)
torch.manual_seed(seed_n)
torch.cuda.manual_seed(seed_n)
torch.cuda.manual_seed_all(seed_n)


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

# DATASET_DIR = "/home/singh_prajwal/dataset/VGG_Sound_extracted/temp_data_wav"
DATASET_DIR = sys.argv[1]
OUTPUT_DIR = "../output_sd"
DEVICE = "cuda"
CFG_SCALE = 7.5
NUM_STEPS = 50
SR = 44100
BATCH_SIZE = 8

MODEL_ID = "CompVis/stable-diffusion-v1-4"
CLAP_PATH = "CLAP/msclap"
CLAP_WEIGHTS = "ckpts/CLAP_weights_2022.pth"
ADAPTER_CKPT = "ckpts/audio_projector_landscape.pth"
GATE_DICT = "ckpts/landscape.pt"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/imgs', exist_ok=True)



def get_pipe(gate_dict_path="ckpts/landscape.pt", 
            clap_path = "CLAP/msclap",
                clap_weights = 'ckpts/CLAP_weights_2022.pth',
             adapter_ckpt_path="ckpts/audio_projector_landscape.pth",
            ):
    
    model_id = "CompVis/stable-diffusion-v1-4"
    unet = UNet2DConditionModel.from_pretrained(
        model_id,
        subfolder="unet",
        use_adapter_list=[False, True, True],
        low_cpu_mem_usage=False,
        device_map=None
    ).to("cuda")
    
    
    gate_dict = torch.load(gate_dict_path)
    for name, param in unet.named_parameters():
        if "adapter" in name:    
            param.data = gate_dict[name]
        
    unet.to("cuda")


    pipeline = StableDiffusionPipeline.from_pretrained(model_id, use_safetensors=True)
    pipeline.to("cuda")

    pipeline.unet = unet
    
    
    import sys
    sys.path.append(clap_path)
    from CLAPWrapper import CLAPWrapper

    audio_encoder = CLAPWrapper(clap_weights, use_cuda=True)     
    audio_projector = Adapter(audio_token_count=77, transformer_layer_count=4).cuda()
    audio_projector.load_state_dict(torch.load(adapter_ckpt_path))
    audio_projector.eval()    
    

    return audio_encoder, audio_projector, pipeline


# ------------------------------------------------------------
# GENERATION
# ------------------------------------------------------------
@torch.no_grad()
def generate(pipeline, audio_encoder, audio_projector, audio=None, prompt=None, cfg_scale=5, num_inference_steps=50):
    with torch.no_grad():
        audio_emb, _ = audio_encoder.get_audio_embeddings([audio], resample = 44100)
        audio_proj = audio_projector(audio_emb.unsqueeze(1))

        audio_emb = torch.zeros(1, 1024).cuda()
        audio_uc = audio_projector(audio_emb.unsqueeze(1))    

        audio_context = torch.cat([audio_uc, audio_proj]).cuda()
        image = pipeline(
                    prompt=prompt, 
                    audio_context=audio_context,
                    guidance_scale=cfg_scale,
                    num_inference_steps=num_inference_steps)
        return image.images[0]



# def img_proc(img):
#     img = Image.open(img).convert("RGB")
#     img = np.array(img).astype(np.uint8)
#     image = Image.fromarray(img)
#     image = image.resize((512, 512), resample=PIL_INTERPOLATION["bicubic"])

#     image = np.array(image).astype(np.uint8)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image = (image / 127.5 - 1.0).astype(np.float32)
#     return torch.from_numpy(image).permute(2, 0, 1)


# def aud_proc_beats(aud, rand_sec=0):
#     wav, sr = torchaudio.load(aud)
#     wav = torch.tile(wav, (1, 10))
#     wav = wav[:, :sr*10]
#     start = rand_sec * sr
#     end = (rand_sec+10) * sr
#     wav = wav[:, start:end]
#     return wav[0]

def img_proc(img, center_crop=True, interpolation=PIL_INTERPOLATION["bicubic"]):
    img = Image.open(img).convert("RGB")
    img = np.array(img).astype(np.uint8)

    # Center crop (square)
    if center_crop:
        crop = min(img.shape[0], img.shape[1])
        h, w = img.shape[0], img.shape[1]
        img = img[(h - crop)//2:(h + crop)//2,
                  (w - crop)//2:(w + crop)//2]

    image = Image.fromarray(img)
    image = image.resize((512, 512), resample=interpolation)

    image = np.array(image).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Normalize to [-1, 1]
    image = (image / 127.5 - 1.0).astype(np.float32)

    return torch.from_numpy(image).permute(2, 0, 1)



def aud_proc_beats(aud, rand_sec=0, target_sr=32000, input_length=10):
    try:
        waveform, sr = torchaudio.load(aud)
    except Exception as e:
        print(f"[AUDIO LOAD ERROR] {aud}")
        raise e

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
        sr = target_sr

    waveform = waveform.float()

    desired_len = int(target_sr * input_length)
    cur_len = waveform.shape[1]

    # Pad if too short
    if cur_len < desired_len:
        pad_amount = desired_len - cur_len
        waveform = F.pad(waveform, (0, pad_amount))

    # Clamp start index
    max_start = waveform.shape[1] - desired_len
    start = int(round(rand_sec * target_sr))

    if start < 0:
        start = 0
    if start > max_start:
        start = max(0, max_start)

    end = start + desired_len
    segment = waveform[:, start:end]

    return segment.squeeze(0)



def parse_filename(filename):
    base = os.path.basename(filename)
    base = base[:-4]
    parts = base.split("_")
    idx = parts[0]
    label = parts[-1]
    return base, idx, label

def main():
    print("Loading SonicDiffusion pipeline...")
    audio_encoder, audio_projector, pipeline = get_pipe()
    audio_files = [f for f in os.listdir(DATASET_DIR) if f.endswith(".wav")]
    random.shuffle(audio_files)  # randomize order
    print(f"Found {len(audio_files)} samples")

    # Temp storage for batch metrics
    batch_gen_imgs = []
    batch_gt_imgs = []
    batch_audio_feats = []
    batch_labels = []
    batch_inds = []

    iis = 0
    ais = 0
    aic = 0
    ccs = 0
    cts = 0
    cnt = 0

    with torch.no_grad():
        for audio_file in audio_files:
            audio_path = os.path.join(DATASET_DIR, audio_file)
            _, idx, label = parse_filename(audio_file)
            prompt = f"generate an image"
            # prompt += f" of a {label}"
            # print(f"Processing {audio_file} with prompt: '{prompt}'")

            # GT image path
            gt_img_path = os.path.join(DATASET_DIR, audio_file.replace(".wav", ".jpg"))

            # Generate image for a single audio
            gen_img = generate(pipeline, audio_encoder, audio_projector, audio=audio_path, prompt=prompt)

            gen_img.save(f'{OUTPUT_DIR}/imgs/{idx}_{label}.jpg')
            ind = cnt % BATCH_SIZE
            # Store for batch evaluation
            batch_gen_imgs.append(gen_img)
            batch_gt_imgs.append(img_proc(gt_img_path))
            batch_audio_feats.append(aud_proc_beats(audio_path))
            batch_labels.append(label)
            batch_inds.append(ind)

            
            # Evaluate when batch is full
            if len(batch_gen_imgs) == BATCH_SIZE:
                batch_gen_imgs_tensor = torch.stack([
                    to_tensor(img) for img in batch_gen_imgs
                ])
                batch_gt_imgs = torch.stack([
                    img for img in batch_gt_imgs
                ], dim=0)
                batch_audio_feats = torch.stack([
                    feat for feat in batch_audio_feats
                ], dim=0)
                for j in range(BATCH_SIZE):
                    metrics = evaluate_all(
                        batch_gen_imgs[j], batch_gt_imgs, batch_audio_feats, batch_labels[j], current_audio_idx_in_batch=batch_inds[j]
                    )
                    iis += metrics["iis"]["iss"]
                    ais += metrics["ais"]["ais"]
                    aic += metrics["aic"]["accuracy"]
                    cts += metrics["cls"]["similarity"]
                    ccs += metrics["ccs"]["similarity"]
                    cnt += 1
                # Write batch metrics to file
                with open(os.path.join(OUTPUT_DIR, "sonicdiff_metrics.txt"), "a") as f:
                    f.write(f"IIS={iis:.2f}, AIS={(ais):.2f}, AIC={(aic):.2f}, CLS={(cts):.2f}, CCS={(ccs):.2f}, count={cnt}\n")

                # Clear batch lists
                batch_gen_imgs = []
                batch_gt_imgs = []
                batch_audio_feats = []
                batch_labels = []
                batch_inds = []

        batch_gen_imgs_tensor = torch.stack([
            to_tensor(img) for img in batch_gen_imgs
        ])
        batch_gt_imgs = torch.stack([
            img for img in batch_gt_imgs
        ], dim=0)
        batch_audio_feats = torch.stack([
            feat for feat in batch_audio_feats
        ], dim=0)
        # Process remaining samples < BATCH_SIZE
        for j in range(len(batch_gen_imgs)):
            metrics = evaluate_all(
                batch_gen_imgs[j], batch_gt_imgs, batch_audio_feats, batch_labels[j], current_audio_idx_in_batch=batch_inds[j]
            )
            iis += metrics["iis"]["iss"]
            ais += metrics["ais"]["ais"]
            aic += metrics["aic"]["accuracy"]
            cts += metrics["cls"]["similarity"]
            ccs += metrics["ccs"]["similarity"]
            cnt += 1
        # Write final metrics
        with open(os.path.join(OUTPUT_DIR, "sonicdiff_metrics.txt"), "a") as f:
            f.write(f"IIS={iis:.2f}, AIS={(ais):.2f}, AIC={(aic):.2f}, CLS={(cts):.2f}, CCS={(ccs):.2f}, count={cnt}\n")

        print("Done.")




# ------------------------------------------------------------
if __name__ == "__main__":
    main()
