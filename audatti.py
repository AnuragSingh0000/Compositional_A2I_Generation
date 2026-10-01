import argparse
import logging
import os
from Compositional_A2I_Generation.Attend_Excite.utils import ptp_utils
import torch
import pprint
from typing import List
from PIL import Image, ImageDraw, ImageFont   # ← ImageDraw and ImageFont were missing

import pyrallis
import torch
import os
import torch.utils.checkpoint
from torch.utils.data import Dataset
import datasets
import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import StableDiffusionPipeline
from diffusers.utils import check_min_version
from transformers import CLIPTokenizer
from AudioToken.data.dataloader import VGG_Sound_datasetn, VGG_Sound_dataset
from AudioToken.cluster import visualize_tokens
from torchvision.transforms.functional import to_tensor

from Compositional_A2I_Generation.Attend_Excite.utils.ptp_utils import AttentionStore
from Compositional_A2I_Generation.Attend_Excite.config import RunConfig
from Compositional_A2I_Generation.Attend_Excite.utils import vis_utils
torch.cuda.empty_cache()

check_min_version("0.12.0")
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device using: ", device)
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")

logger = get_logger(__name__)

import random
random_seed = 45
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--learned_embeds", type=str, default='output/embedder_learned_embeds.bin',
                        help="Path to pretrained embedder")
    parser.add_argument("--learned_embeds_lora", type=str, default='output/lora_layers_learned_embeds.bin',
                        help="Path to pretrained embedder")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default='stabilityai/stable-diffusion-2',
                        help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--revision", type=str, default=None, required=False)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--data_dir", type=str, help="A folder containing the training data.")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--data_set", type=str, default='test', choices=['train', 'test'])
    parser.add_argument("--generation_steps", type=int, default=50)
    parser.add_argument("--run_name", type=str, default='Audio Attend Excite2')
    parser.add_argument("--pipeline", type=str, default='attend_and_excite')
    parser.add_argument("--audio_encoder", type=str, default='Beats', choices=['Clap', 'Beats'])
    parser.add_argument("--set_size", type=str, default='full')
    parser.add_argument("--prompt", type=str,
                        default='a photo of a <s1> and a <s2> and a <s3> and a <s4> and a <s5>')
    parser.add_argument("--use_label", type=str, default='No')
    parser.add_argument("--show_attn", type=str, default='No')
    parser.add_argument("--input_length", type=int, default=10)
    parser.add_argument("--lora", type=bool, default=False)
    parser.add_argument("--split", type=bool, default=True)
    parser.add_argument("--caption", type=bool, default=False)

    args = parser.parse_args()

    if args.data_dir is None:
        raise ValueError("You must specify a train data directory.")

    return args


def get_indices_to_alter(stable, prompt: str) -> List[int]:
    token_idx_to_word = {idx: stable.tokenizer.decode(t)
                         for idx, t in enumerate(stable.tokenizer(prompt)['input_ids'])
                         if 0 < idx < len(stable.tokenizer(prompt)['input_ids']) - 1}
    words = prompt.split()
    tokens = len(words)
    if tokens == 8:
        token_indices = "5,8"
    elif tokens < 8:
        token_indices = "5"
    else:
        token_indices = "5"
        for i in range(5, tokens):
            if words[i] != "and" and words[i] != "a":
                token_indices += f",{i+1}"

    token_indices = [int(i) for i in token_indices.split(",")]
    return token_indices


def run_on_prompt(prompt: List[str],
                  model: 'AttendAndExcitePipeline',
                  controller: AttentionStore,
                  token_indices: List[int],
                  seed: torch.Generator,
                  config: RunConfig,
                  show_attn: bool = False,
                  attn_save_path: str = None) -> Image.Image:
    if controller is not None:
        ptp_utils.register_attention_control(model, controller)
    outputs = model(prompt=prompt,
                    attention_store=controller,
                    indices_to_alter=token_indices,
                    attention_res=config.attention_res,
                    guidance_scale=config.guidance_scale,
                    generator=seed,
                    num_inference_steps=config.n_inference_steps,
                    max_iter_to_alter=config.max_iter_to_alter,
                    run_standard_sd=config.run_standard_sd,
                    thresholds=config.thresholds,
                    scale_factor=config.scale_factor,
                    scale_range=config.scale_range,
                    smooth_attentions=config.smooth_attentions,
                    sigma=config.sigma,
                    kernel_size=config.kernel_size,
                    sd_2_1=config.sd_2_1,
                    show_attn=show_attn,
                    attn_save_path=attn_save_path)
    image = outputs.images[0]
    return image


def inference(args):

    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    for folder_path in [
        f"{args.output_dir}/",
        f"{args.output_dir}/imgs/",
        f"{args.output_dir}/imgs/{args.run_name}/",
    ]:
        os.makedirs(folder_path, exist_ok=True)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if args.pipeline.lower() == "astar":
        from Compositional_A2I_Generation.Attend_Excite.pipeline_astar import AttendAndExcitePipeline
    elif args.pipeline.lower() == "at":
        from Compositional_A2I_Generation.Attend_Excite.pipeline_sd import AttendAndExcitePipeline
    elif args.pipeline.lower() == "mixed":
        from Compositional_A2I_Generation.Attend_Excite.pipeline_mixed import AttendAndExcitePipeline
    else:
        from Compositional_A2I_Generation.Attend_Excite.pipeline_attend_and_excite import AttendAndExcitePipeline

    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer"
        )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if (args.data_dir is not None) and (args.data_dir != ""):
        if "is3" in args.data_dir.lower():
            from AudioToken.evaluate_is3 import evaluate_all
        else:
            from AudioToken.evaluate_vgg import evaluate_all

    if args.audio_encoder == 'Clap':
        from AudioToken.modules.AudioToken.AudioToken_clap import AudioTokenWrapper
    else:
        from AudioToken.modules.AudioToken.AudioToken import AudioTokenWrapper

    model = AudioTokenWrapper(args, accelerator).to(weight_dtype).eval()

    num_added_tokens = 0
    for i in range(1, 6):
        num_added_tokens += tokenizer.add_tokens(f"<s{i}>")
    if num_added_tokens != 5:
        raise ValueError(
            f"The tokenizer already contains the token {args.prompt}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    test_dataset = VGG_Sound_datasetn(args, tokenizer, logger, args.resolution)
    print("Length of test data: ", test_dataset.__len__())

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=args.dataloader_num_workers
    )

    model, test_dataloader = accelerator.prepare(model, test_dataloader)

    obj_token_ids = []
    for i in range(1, 6):
        obj_token_ids.append(tokenizer.convert_tokens_to_ids(f"<s{i}>"))

    accelerator.unwrap_model(model).text_encoder.resize_token_embeddings(len(tokenizer))

    config = RunConfig()
    config.prompt = args.prompt

    pipeline = AttendAndExcitePipeline.from_pretrained(
        config.sd_model,
        tokenizer=tokenizer,
        text_encoder=accelerator.unwrap_model(model).text_encoder,
        vae=accelerator.unwrap_model(model).vae,
        unet=accelerator.unwrap_model(model).unet,
    ).to(device)

    full_words = args.prompt.split(" ")
    # full_words for default prompt:
    # ['a','photo','of','a','<s1>','and','a','<s2>','and','a','<s3>','and','a','<s4>','and','a','<s5>']
    # i=0 → words[0:5]   = "a photo of a <s1>"
    # i=1 → words[0:8]   = "a photo of a <s1> and a <s2>"
    # i=k → words[0:5+k*3]

    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            print(f"Step {step+1}/{len(test_dataloader)}")

            num = batch["num_samples"][0]
            print(f"  num_samples in this batch: {num}")

            if num < 2:
                # Only one sound — nothing to make progressive, skip or save single
                print("  Skipping: need at least 2 sounds for a progressive row.")
                continue

            # Re-seed every sample so all columns start from identical noise,
            # making the progression visually comparable.
            g = torch.Generator(device).manual_seed(42)

            # Get the live embedding table — writes here are reflected in the
            # pipeline's text encoder immediately (shared reference via .data).
            token_embeds = (
                accelerator.unwrap_model(model)
                .text_encoder.get_input_embeddings()
                .weight.data
            )

            row_images     = []
            current_labels = []

            for i in range(num):
                av = batch[f"audio_values{i+1}"].to(dtype=weight_dtype)

                # ── Audio features ─────────────────────────────────────────
                if args.audio_encoder == 'Clap':
                    feats = accelerator.unwrap_model(model).aud_encoder \
                                .get_audio_embedding_from_data(av.cpu().float().numpy())
                    feats = torch.tensor(feats).to(
                        dtype=weight_dtype, device=accelerator.device
                    )
                    feats.unsqueeze_(1)   # [B, 512] → [B, 1, 512]
                else:
                    feats = accelerator.unwrap_model(model).aud_encoder \
                                .extract_features(av)[1]

                # ── Embed → inject into slot i of the token table ──────────
                token    = accelerator.unwrap_model(model).embedder(feats)
                token_1d = token.squeeze()          # [B,1,D] or [B,D] → [D]
                token_embeds[obj_token_ids[i]] = token_1d.clone()

                # ── Progressive prompt: slots 0..i are now filled ──────────
                end_idx = 5 + i * 3                 # see table above
                prompt  = " ".join(full_words[:end_idx])

                label = batch['labels'][i][0]
                current_labels.append(label.replace('_', ' '))

                if args.use_label == 'Yes':
                    prompt = "a photo of a " + " and a ".join(current_labels)

                token_indices = (
                    get_indices_to_alter(pipeline, prompt)
                    if config.token_indices is None
                    else config.token_indices
                )

                # Fresh controller per generation — don't accumulate stale maps
                step_controller = AttentionStore()

                # Re-seed so every column uses the same starting noise
                g = torch.Generator(device).manual_seed(42)

                img = run_on_prompt(
                    prompt=prompt,
                    model=pipeline,
                    controller=step_controller,
                    token_indices=token_indices,
                    seed=g,
                    config=config,
                    show_attn=(args.show_attn == 'Yes'),
                    attn_save_path=None,
                )
                row_images.append(img)

            # ── Assemble the progressive row image ─────────────────────────
            if len(row_images) > 1:
                w, h        = row_images[0].size
                text_height = 60

                # Each column is one image wide → total width = w * num_images
                total_width  = w * len(row_images)
                total_height = h + text_height

                grid_img = Image.new('RGB', (total_width, total_height), color='white')
                draw     = ImageDraw.Draw(grid_img)

                try:
                    font = ImageFont.truetype("arial.ttf", 24)
                except IOError:
                    font = ImageFont.load_default()

                for idx, img in enumerate(row_images):
                    # Paste image into its column slot
                    grid_img.paste(img, (idx * w, 0))

                    # Cumulative label below each column
                    label_text = " + ".join(current_labels[:idx + 1])

                    text_bbox = draw.textbbox((0, 0), label_text, font=font)
                    text_w    = text_bbox[2] - text_bbox[0]
                    text_h    = text_bbox[3] - text_bbox[1]

                    text_x = (idx * w) + (w - text_w) // 2
                    text_y = h + (text_height - text_h) // 2

                    draw.text((text_x, text_y), label_text, fill="black", font=font)

                save_dir  = f'{args.output_dir}/imgs/{args.run_name}'
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f'{batch["full_name"][0]}.jpg')
                grid_img.save(save_path)


if __name__ == "__main__":
    args = parse_args()
    inference(args)