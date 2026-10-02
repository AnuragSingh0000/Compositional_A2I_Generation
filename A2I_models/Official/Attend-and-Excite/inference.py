import pprint
from typing import List

import pyrallis
import torch
from PIL import Image

from config import RunConfig
# from pipeline_attend_and_excite import AttendAndExcitePipeline
# from pipeline_mixed import AttendAndExcitePipeline
from pipeline_astar_test import AttendAndExcitePipeline
# from pipeline_astar import AttendAndExcitePipeline
from utils import ptp_utils, vis_utils
from utils.ptp_utils import AttentionStore

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import os
import argparse
from torch import autocast
from diffusers import StableDiffusionPipeline
from evaluate2 import VGG_Sound_dataset2, evaluate_all

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT_DIR = "../output/atest_is2"
IMG_DIR = os.path.join(OUT_DIR, "images")
DATA_DIR = "../../../dataset/IS2"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

import sys

log_file = open("output.txt", "w")
sys.stdout = log_file

print("This will go into the file")
print("All prints now go to output.txt")



def load_model(config: RunConfig):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    if config.sd_2_1:
        stable_diffusion_version = "stabilityai/stable-diffusion-2-1-base"
    else:
        stable_diffusion_version = "CompVis/stable-diffusion-v1-4"
    stable = AttendAndExcitePipeline.from_pretrained(stable_diffusion_version).to(device)
    return stable


# def get_indices_to_alter(stable, prompt: str) -> List[int]:
#     token_idx_to_word = {idx: stable.tokenizer.decode(t)
#                          for idx, t in enumerate(stable.tokenizer(prompt)['input_ids'])
#                          if 0 < idx < len(stable.tokenizer(prompt)['input_ids']) - 1}
#     pprint.pprint(token_idx_to_word)
#     token_indices = input("Please enter the a comma-separated list indices of the tokens you wish to "
#                           "alter (e.g., 2,5): ")
#     token_indices = [int(i) for i in token_indices.split(",")]
#     print(f"Altering tokens: {[token_idx_to_word[i] for i in token_indices]}")
#     return token_indices

def get_indices_to_alter(stable, prompt: str) -> List[int]:
    token_idx_to_word = {idx: stable.tokenizer.decode(t)
                        for idx, t in enumerate(stable.tokenizer(prompt)['input_ids'])
                        if 0 < idx < len(stable.tokenizer(prompt)['input_ids']) - 1}
    # pprint.pprint(token_idx_to_word)
    # token_indices = input("Please enter the a comma-separated list indices of the tokens you wish to "
    #                     "alter (e.g., 2,5): ")
    words = prompt.split()
    tokens = len(words)
    if (tokens == 8):
        token_indices = "5,8"
    elif (tokens < 8):
        token_indices = "5"
    else:
        token_indices = "5"
        for i in range(5, tokens):
            if words[i] != "and" and words[i] != "a":
                token_indices += f",{i+1}"

    token_indices = [int(i) for i in token_indices.split(",")]
    # print("-----------")
    # print(words)
    # print(prompt, token_indices)
    # print("-----------")
    # print(f"Altering tokens: {[token_idx_to_word[i] for i in token_indices]}")
    return token_indices





def run_on_prompt(prompt: List[str],
                  model: AttendAndExcitePipeline,
                  controller: AttentionStore,
                  token_indices: List[int],
                  seed: torch.Generator,
                  config: RunConfig) -> Image.Image:
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
                    sd_2_1=config.sd_2_1)
    image = outputs.images[0]
    return image


@pyrallis.wrap()
def main(config: RunConfig):
    stable = load_model(config)
    test_dataset = VGG_Sound_dataset2(DATA_DIR, input_length=10, interpolation='bicubic')
    # test_dataset = VGG_Sound_dataset2(args, tokenizer, logger, args.resolution)

    print("Length of test data: ", test_dataset.__len__())


    batch_size = 8
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=6
    )

    iis = 0
    cts = 0
    ccs = 0
    cnt = 0
    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            print("------------")
            print(f"Step {step+1}/{len(test_dataloader)}")
            print("------------")
            iv1 = batch["image_values1"].to(device)
            iv2 = batch["image_values2"].to(device)

            for ind in range(batch_size):
                label1 = batch["full_name"][ind].split("_")[-2]
                label2 = batch["full_name"][ind].split("_")[-1]

                prompti = f"a photo of a {label1} and a {label2}"
                token_indices = get_indices_to_alter(stable, prompti) if config.token_indices is None else config.token_indices
                # images = []
                g = torch.Generator('cuda').manual_seed(config.seeds[0])
                controller = AttentionStore()
                image = run_on_prompt(prompt=prompti,
                                    model=stable,
                                    controller=controller,
                                    token_indices=token_indices,
                                    seed=g,
                                    config=config)

                metrics = evaluate_all(image, [iv1, iv2], [label1, label2], current_audio_idx_in_batch=ind)
                iis += metrics["iis"]["iss"]
                cts += metrics["cls"]["gm"]
                ccs += metrics["ccs"]["gm"]
                cnt += 1
                image.save(os.path.join(IMG_DIR, f"{batch['full_name'][ind]}.png"))
                
            if (step + 1) % 10 == 0:
                print(iis, cts, ccs, cnt)
                with open(os.path.join(OUT_DIR, f"metrics.txt"), "a") as f:
                        f.write(f"Step {step}: IIS={iis:.2f}, CLS={cts:.2f}, CCS={ccs:.2f}, count={cnt}\n")

        with open(os.path.join(OUT_DIR, f"metrics.txt"), "a") as f:
            f.write(f"Final Step: IIS={iis/cnt:.4f}, CLS={cts/cnt:.4f}, CCS={ccs/cnt:.4f}, count={cnt}\n")
        


if __name__ == '__main__':
    main()
    print("Evaluation completed.")
