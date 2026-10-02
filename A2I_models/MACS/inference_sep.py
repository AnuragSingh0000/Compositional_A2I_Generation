import argparse
import logging
import os.path
import warnings

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from modules import StableDiffusion
from modules import evaluate_all
from dataloader import VGG_Sound_dataset2, VGG_Sound_datasetn

warnings.filterwarnings('ignore')
# Placeholder for logger (assuming it's defined elsewhere in your utils)
# logger = get_logger(__name__) 

def parse_args():
    ps = argparse.ArgumentParser()
    # model hyperparams
    ps.add_argument('--M', type=int, default=6)

    # functional params
    ps.add_argument('--prompt', type=str, default='')
    ps.add_argument('--negative_prompt', type=str, default='')
    ps.add_argument('--cfg', action='store_true')

    # model dirs
    ps.add_argument('--pt_path', type=str, default='weights/', help='Path to the pretrained weights. (Must be a directory)')
    ps.add_argument('--sd_path', type=str, default='CompVis/stable-diffusion-v1-4', help='Hugging Face repo or local path')

    # others
    ps.add_argument('--batch_size', type=int, default=1)
    ps.add_argument('--data_dir', type=str)
    ps.add_argument('--data_set', type=str, default='test')
    ps.add_argument('--output_dir', type=str, default='./output')
    ps.add_argument('--input_length', default=8)
    ps.add_argument('--sr', type=int, default=16000)
    ps.add_argument('--mixed_precision', default='fp16')
    ps.add_argument('--dataset_name', default='Landscape')
    ps.add_argument('--weights_name', default='AudioSet')
    ps.add_argument('--training', default=False)
    ps.add_argument('--run_name', default='macs_eval', type=str) # Added for metrics saving
    args = ps.parse_args()

    args.output_dir = os.path.join(args.output_dir, args.dataset_name)
    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
    if not os.path.exists(f'{args.output_dir}/imgs'): os.makedirs(f'{args.output_dir}/imgs')
    return args


args = parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Clear previous outputs
for fn in os.listdir(args.output_dir):
    file = os.path.join(args.output_dir, fn)
    if os.path.isfile(file):
        os.remove(file)

# Initialize Dataset and Dataloader
# Using logger=None as a placeholder, adjust if you have a real logger
dataset = VGG_Sound_datasetn(args, logger=None) 
print("Length of test data: ", dataset.__len__())
batch_size = 1 # Keep at 1 for MACS base inference, we will batch metrics manually
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True)

model = StableDiffusion(pt_path=args.pt_path, sd_path=args.sd_path, train=False, dataset=args.weights_name,
                        M=args.M, cfg=args.cfg).to(device)

generated = []

# --- METRICS INITIALIZATION ---
iis, ais, aic, cts, ccs, cnt = 0, 0, 0, 0, 0, 0
B = 4  # Evaluation batch size
gen_img_batch = []
labels = [[] for _ in range(B)]
ivs = [[] for _ in range(B)]
avs = [[] for _ in range(B)]

for step, batch in enumerate(tqdm(dataloader)):
    # Get ID or fallback to step count
    ytid = batch.get('ytid', batch.get('full_name', [f'sample_{step}']))[0]
    if ytid in generated: continue
    generated.append(ytid)

    # 1. Extract separated audio waveforms dynamically based on num_samples
    num = batch["num_samples"][0] 
    separated_audios = []
    
    for i in range(num):
        # Fetch individual audio/image and move to device
        av = batch[f"audio_values{i+1}"].to(device)
        iv = batch[f"image_values{i+1}"]
        
        ivs[step % B].append(iv)
        avs[step % B].append(av)
        separated_audios.append(av)
        
        # Extract label dynamically if using VGG_Sound naming convention
        if "full_name" in batch:
            # label = batch["full_name"][0].split("_")[-(i+1)]
            label = batch["labels"][i][0]
            labels[step % B].append(label)

    # 2. Generation using the updated infer() method
    with torch.no_grad():
         image = model.infer(separated_audios[0], prompt=args.prompt, negative_prompt=args.negative_prompt)
        # image = model.infer(separated_audios, prompt=args.prompt, negative_prompt=args.negative_prompt)
    
    # 3. Store for Metrics (evaluate_all expects [C, H, W] tensors on device)
    # MACS infer output shape is [B, H, W, C]. We take index [0] to remove the batch dim.
    # Resulting shape becomes [C, H, W] (e.g., [3, 512, 512])
    img_tensor = image[0].permute(2, 0, 1).to(device) 
    gen_img_batch.append(img_tensor)

    # 4. Save to Disk
    image_np = (image.cpu().numpy() * 255).astype(np.uint8)
    image_bgr = cv2.cvtColor(image_np[0], cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'{args.output_dir}/imgs/{ytid}.jpg', image_bgr)

    # 5. Calculate Metrics every B steps
    if (step + 1) % B == 0:
        for i in range(B):
            metrics = evaluate_all(gen_img_batch[i], ivs, avs, labels, current_audio_idx_in_batch=i)
            iis += metrics["iis"]["iss"]
            ais += metrics["ais"]["ais"]
            aic += metrics["aic"]["accuracy"]
            cts += metrics["cls"]["gm"]
            ccs += metrics["ccs"]["gm"]
            cnt += 1
        
        # Reset evaluation buffers
        gen_img_batch = []
        labels = [[] for _ in range(B)]
        ivs = [[] for _ in range(B)]
        avs = [[] for _ in range(B)]

        # Print and write to file
        print(f"Step {step}: IIS={iis:.2f}, AIS={ais:.2f}, AIC={aic:.2f}, CLS={cts:.2f}, CCS={ccs:.2f}")
        
        metrics_file = os.path.join(args.output_dir, f"{args.run_name}_metrics2.txt")
        with open(metrics_file, "a") as f:
            f.write(f"Step {step}: IIS={iis:.2f}, AIS={ais:.2f}, AIC={aic:.2f}, CLS={cts:.2f}, CCS={ccs:.2f}, count={cnt}\n")

print("Inference and evaluation complete!")