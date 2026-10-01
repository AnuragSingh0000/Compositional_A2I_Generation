import argparse
import logging
import os
import torch
from PIL import Image
from typing import List
import numpy as np
import random
from tqdm import tqdm  # <-- Added tqdm import

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import CLIPTokenizer
from torchvision.transforms.functional import to_tensor
import datasets
import transformers

# Import your custom datasets and evaluators
from AudioToken.data.dataloader import VGG_Sound_datasetn
# from AudioToken.data.dataloader import IS3  # Uncomment if needed

logger = get_logger(__name__)


import random
random_seed=45
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate pre-generated images.")
    
    parser.add_argument("--generated_imgs_dir", type=str, required=True,
                        help="Path to the directory containing pre-generated .jpg images.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default='stabilityai/stable-diffusion-2',
                        help="Path to pretrained model or model identifier from huggingface.co/models (used for tokenizer).")
    parser.add_argument("--tokenizer_name", type=str, default=None,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="A folder containing the training/testing data.")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="The output directory where the model metrics will be written.")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--run_name", type=str, default='Audio_Eval_Run',
                        help="Insert run name for the metrics output file.")
    parser.add_argument("--resolution", type=int, default=768,
                        help="The resolution for input images.")
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=45,
                        help="A seed for reproducible evaluation.")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    parser.add_argument("--data_set", type=str, default='test', choices=['train', 'test'])
    parser.add_argument("--input_length", type=int, default=10)
    parser.add_argument("--split", type=bool, default=True)
    parser.add_argument("--caption", type=bool, default=False)
    
    args = parser.parse_args()
    return args

def evaluate_images(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device using: ", device)

    # Setup Logging & Accelerator
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        project_dir=logging_dir,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if args.seed is not None:
        set_seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    # Import Evaluator based on data_dir
    if "is3" in args.data_dir.lower():
        from AudioToken.evaluate_is3 import evaluate_all
    else:
        from AudioToken.evaluate_vgg import evaluate_all

    # Load tokenizer
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    else:
        tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")

    # Add placeholder tokens
    for i in range(1, 6):
        tokenizer.add_tokens(f"<s{i}>")

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Initialize Dataset and Dataloader
    test_dataset = VGG_Sound_datasetn(args, tokenizer, logger, args.resolution)
    print("Length of test data: ", test_dataset.__len__())

    batch_size = 1
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=args.dataloader_num_workers
    )

    test_dataloader = accelerator.prepare(test_dataloader)

    # Tracking Variables
    iis = 0
    ais = 0
    aic = 0
    cnt = 0
    cts = 0
    ccs = 0

    B = 4
    gen_img_batch = []
    labels = [[] for _ in range(B)]
    ivs = [[] for _ in range(B)]
    avs = [[] for _ in range(B)]

    metrics_file = os.path.join(args.output_dir, f"{args.run_name}_metrics2.txt")

    with torch.no_grad():
        # <-- Wrapped test_dataloader with tqdm here
        for step, batch in enumerate(tqdm(test_dataloader, desc="Evaluating Images")):
            
            num = batch["num_samples"][0]
            
            for i in range(num):
                av = batch[f"audio_values{i+1}"].to(dtype=weight_dtype)
                iv = batch[f"image_values{i+1}"].to(dtype=weight_dtype)
                ivs[step % B].append(iv)
                avs[step % B].append(av)

                label = batch["full_name"][0].split("_")[-(i+1)]
                labels[step % B].append(label)

            full_name = batch["full_name"][0]
            
            img_path = os.path.join(args.generated_imgs_dir, f"{full_name}.jpg")
            
            if not os.path.exists(img_path):
                img_path = os.path.join(args.generated_imgs_dir, f"{full_name}.png")
            
            if not os.path.exists(img_path):
                # Using tqdm.write instead of print prevents the progress bar from breaking
                tqdm.write(f"WARNING: Image not found at {img_path}. Skipping batch evaluation alignment.")
                continue
                
            gen_img = Image.open(img_path).convert("RGB")
            gen_img_tensor = to_tensor(gen_img).to(device)
            gen_img_batch.append(gen_img_tensor)

            # Evaluate Batch
            if (step + 1) % B == 0:
                for i in range(B):
                    metrics = evaluate_all(gen_img_batch[i], ivs, avs, labels, current_audio_idx_in_batch=i)
                    
                    iis += metrics["iis"]["iss"]
                    ais += metrics["ais"]["ais"]
                    aic += metrics["aic"]["accuracy"]
                    cts += metrics["cls"]["gm"]
                    ccs += metrics["ccs"]["gm"]
                    cnt += 1
                
                gen_img_batch = []
                labels = [[] for _ in range(B)]
                ivs = [[] for _ in range(B)]
                avs = [[] for _ in range(B)]

                if (step + 1) % (B*10) == 0:
                    with open(metrics_file, "a") as f:
                        f.write(f"Step {step}: IIS={iis:.2f}, AIS={(ais):.2f}, AIC={(aic):.2f}, CLS={cts:.2f}, CCS={ccs:.2f}, count={cnt}\n")
        
        # Final Write
        if cnt > 0:
            with open(metrics_file, "a") as f:
                f.write(f"FINAL: IIS={iis:.2f}, AIS={(ais):.2f}, AIC={(aic):.2f}, CLS={cts:.2f}, CCS={ccs:.2f}, count={cnt}\n")

if __name__ == "__main__":
    args = parse_args()
    evaluate_images(args)