import argparse
import logging
import os
from Compositional_A2I_Generation.Attend_Excite.utils import ptp_utils
import torch
import pprint
from typing import List

import pyrallis
import torch
import os
from PIL import Image
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
from AudioToken.visualise_utils import visualize_tokens
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

    # ── Embedder weights — separate paths for each audio encoder ──────────
    parser.add_argument("--learned_embeds", type=str,
                        default='output_train/learned_embeds.bin',
                        help="Path to pretrained BEATs embedder weights.")
    parser.add_argument("--learned_embeds_clap", type=str,
                        default='output_train_claps/learned_embeds.bin',
                        help="Path to pretrained CLAP embedder weights.")
    # ───────────────────────────────────────────────────────────────────────

    parser.add_argument("--learned_embeds_lora", type=str,
                        default='output/lora_layers_learned_embeds.bin',
                        help="Path to pretrained LoRA embedder")
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default='stabilityai/stable-diffusion-2',
                        help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--revision", type=str, default=None, required=False)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--data_dir", type=str,
                        help="A folder containing the training data.")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--mixed_precision", type=str, default="no",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--data_set", type=str, default='test',
                        choices=['train', 'test'])
    parser.add_argument("--generation_steps", type=int, default=50)
    parser.add_argument("--run_name", type=str, default='Audio Attend Excite2')
    parser.add_argument("--pipeline", type=str, default='attend_and_excite')
    parser.add_argument("--audio_encoder", type=str, default='Beats',
                        choices=['Clap', 'Beats'],
                        help="Primary audio encoder (used for single-encoder paths).")
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
                  show_attn: bool,
                  attn_save_path: str) -> Image.Image:
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

    # Load tokenizer
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

    from AudioToken.modules.AudioToken.AudioToken_clap import AudioTokenWrapper as AudioTokenWrapper_claps
    from AudioToken.modules.AudioToken.AudioToken import AudioTokenWrapper as AudioTokenWrapper_beats

    # ── BEATs wrapper: args.learned_embeds → BEATs checkpoint ─────────────
    model_beats = AudioTokenWrapper_beats(args, accelerator).to(weight_dtype).eval()

    # ── CLAP wrapper: temporarily swap to CLAP checkpoint, then restore ────
    _saved_beats_embeds  = args.learned_embeds
    args.learned_embeds  = args.learned_embeds_clap
    model_claps = AudioTokenWrapper_claps(args, accelerator).to(weight_dtype).to(accelerator.device).eval()
    args.learned_embeds  = _saved_beats_embeds   # restore for anything downstream
    # ───────────────────────────────────────────────────────────────────────

    # Add placeholder tokens
    num_added_tokens = 0
    for i in range(1, 6):
        num_added_tokens += tokenizer.add_tokens(f"<s{i}>")
    if num_added_tokens != 5:
        raise ValueError(
            f"The tokenizer already contains some placeholder tokens. "
            "Please pass a different `placeholder_token`."
        )

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    test_dataset = VGG_Sound_datasetn(args, tokenizer, logger, args.resolution)
    print("Length of test data: ", test_dataset.__len__())

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=args.dataloader_num_workers
    )

    # Prepare the BEATs model + dataloader with accelerator
    # (model_claps runs on accelerator.device but is not DDP-wrapped)
    model_beats, test_dataloader = accelerator.prepare(model_beats, test_dataloader)

    for i in range(1, 6):
        tokenizer.convert_tokens_to_ids(f"<s{i}>")   # validate tokens exist

    accelerator.unwrap_model(model_beats).text_encoder.resize_token_embeddings(len(tokenizer))

    config = RunConfig()
    config.prompt = args.prompt

    pipeline = AttendAndExcitePipeline.from_pretrained(
        config.sd_model,
        tokenizer=tokenizer,
        text_encoder=accelerator.unwrap_model(model_beats).text_encoder,
        vae=accelerator.unwrap_model(model_beats).vae,
        unet=accelerator.unwrap_model(model_beats).unet,
    ).to(device)

    all_tokens_beats = []
    all_tokens_claps = []
    labels = []

    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            print(f"Step {step+1}/{len(test_dataloader)}")

            av = batch["audio_values1"].to(dtype=weight_dtype)
            iv = batch["image_values1"].to(dtype=weight_dtype)

            # ── CLAP features ──────────────────────────────────────────────
            feats_claps = accelerator.unwrap_model(model_claps) \
                              .aud_encoder.get_audio_embedding_from_data(
                                  av.cpu().float().numpy()
                              )
            # get_audio_embedding_from_data returns a numpy array / CPU tensor;
            # explicitly place on the correct device before passing to the embedder.
            feats_claps = torch.tensor(feats_claps).to(dtype=weight_dtype, device=accelerator.device)
            feats_claps.unsqueeze_(1)   # [B, 512] → [B, 1, 512]

            # ── BEATs features ─────────────────────────────────────────────
            feats_beats = accelerator.unwrap_model(model_beats) \
                              .aud_encoder.extract_features(av)[1]

            # ── Project through each model's own embedder ──────────────────
            token_claps = accelerator.unwrap_model(model_claps).embedder(feats_claps)
            token_beats = accelerator.unwrap_model(model_beats).embedder(feats_beats)

            label = batch["full_name"][0].split("_")[-1]

            all_tokens_beats.append((label, token_beats.clone()))
            all_tokens_claps.append((label, token_claps.clone()))
            labels.append(label)

    visualize_tokens(
        all_tokens_beats,
        all_tokens_claps,
        reduce_algo="umap",
        save_dir=f'{args.output_dir}/plots/{args.run_name}/Tokens/',
        plot_labels=True,
        model=model_beats,        # text_encoder lives here
        tokenizer=tokenizer,
    )


if __name__ == "__main__":
    args = parse_args()
    inference(args)