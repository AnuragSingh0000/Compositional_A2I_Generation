import argparse
import logging
import os
import torch
import torch.utils.checkpoint
from torch.utils.data import Dataset
import datasets
import diffusers
import transformers
from PIL import Image
import numpy as np
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import StableDiffusionPipeline
from diffusers.utils import check_min_version
from transformers import CLIPTokenizer, CLIPTextModel
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
from data.dataloader import VGG_Sound_datasetn
from visualise_utils import run_kmeans_tsne 
# from cluster import cluster_tokens
from cluster import visualize_tokens

torch.cuda.empty_cache()

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.12.0")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device using: ", device)
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    
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
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--learned_embeds", type=str, default='output/embedder_learned_embeds.bin',
                        help="Path to pretrained embedder")
    parser.add_argument("--learned_embeds_lora", type=str, default='output/lora_layers_learned_embeds.bin',
                        help="Path to pretrained embedder")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default='stabilityai/stable-diffusion-2',
                        help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--revision", type=str, default=None, required=False,
                        help="Revision of pretrained model identifier from huggingface.co/models.")
    parser.add_argument("--tokenizer_name", type=str, default=None,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--data_dir", type=str,
                        help="A folder containing the training data.")
    parser.add_argument("--obj1", type=str, default="<*>",
                        help="A token to use as a placeholder for the audio.",)
    parser.add_argument("--obj2", type=str, default="<^>",
                        help="A token to use as a placeholder for the audio.",)
    parser.add_argument("--obj3", type=str, default="<!>",
                        help="A token to use as a placeholder for the audio.",)
    parser.add_argument("--obj4", type=str, default="<@>",
                        help="A token to use as a placeholder for the audio.",)
    parser.add_argument("--obj5", type=str, default="<$>",
                        help="A token to use as a placeholder for the audio.",)
    parser.add_argument("--output_dir", type=str, default="output",
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--seed", type=int, default=None,
                        help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=768,
                        help="The resolution for input images, all the images in the train/validation"
                            " dataset will be resized to this resolution")
    parser.add_argument("--dataloader_num_workers", type=int, default=6,
                        help="Number of subprocesses to use for data loading."
                            " 0 means that the data will be loaded in the main process.")
    parser.add_argument("--logging_dir", type=str, default="logs",
                        help="[TensorBoard](https://www.tensorflow.org/tensorboard) log directory."
                             " Will default to *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***.")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"],
                        help="Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16)."
                            " Bf16 requires PyTorch >= 1.10. and an Nvidia Ampere GPU.")
    parser.add_argument("--allow_tf32", action="store_true",
                        help="Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training."
                            " For more information, see"
                            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices")
    parser.add_argument("--report_to", type=str, default="tensorboard",
                        help='The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
                            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.')
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--data_set", type=str, default='test', choices=['train', 'test'],
                        help="Whether use train or test set")
    parser.add_argument("--generation_steps", type=int, default=50)
    parser.add_argument("--run_name", type=str, default='AudioToken2',
                        help="Insert run name")    
    parser.add_argument("--audio_encoder", type=str, default='Beats',
                        help="Select the audio encoder to use, either 'Beats' or 'Clap'")
    parser.add_argument("--set_size", type=str, default='full')
    parser.add_argument("--prompt", type=str, default='a photo of a <*> and a <^> and a <!> and a <@> and a <$>')
    parser.add_argument("--use_label", type=str, default='No',
                        help="Whether to use the label in the prompt or not")
    parser.add_argument("--input_length", type=int, default=10,
                        help="Select the number of seconds of audio you want in each test-sample.")
    parser.add_argument("--lora", type=bool, default=False,
                        help="Whether load Lora layers or not")
    parser.add_argument("--split", type=bool, default=True,
                        help="Whether split the audio or not")
    parser.add_argument("--caption", type=bool, default=False,
                        help="Whether split the audio or not")

    args = parser.parse_args()
    # env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    # if env_local_rank != -1 and env_local_rank != args.local_rank:
    #     args.local_rank = env_local_rank

    if args.data_dir is None:
        raise ValueError("You must specify a train data directory.")

    return args


def inference(args):
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    folder_path = f"{args.output_dir}/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    folder_path = f"{args.output_dir}/imgs/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    folder_path = f"{args.output_dir}/imgs/{args.run_name}/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
    )

    # Make one log on every process with the configuration for debugging.
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

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Load tokenizer
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        
    if (args.data_dir is not None) and (args.data_dir != ""):
        if "is3" in args.data_dir.lower():
            from evaluate6 import evaluate_all
            print("-------------")
            print("Using evaluate6 for IS3 dataset")
            print("-------------")
        else:
            from evaluate5 import evaluate_all

    if args.audio_encoder == 'Clap':
        from modules.AudioToken.AudioToken_clap import AudioTokenWrapper
    else:
        from modules.AudioToken.AudioToken import AudioTokenWrapper
    
    model = AudioTokenWrapper(args, accelerator).to(weight_dtype).eval()

    # Add the placeholder token in tokenizer
    num_added_tokens = 0
    num_added_tokens += tokenizer.add_tokens(args.obj1)
    num_added_tokens += tokenizer.add_tokens(args.obj2)
    num_added_tokens += tokenizer.add_tokens(args.obj3)
    num_added_tokens += tokenizer.add_tokens(args.obj4)
    num_added_tokens += tokenizer.add_tokens(args.obj5)
    if num_added_tokens != 5:
        raise ValueError(
            f"The tokenizer already contains the token {args.prompt}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # test_dataset = VGGSound(
    #     args=args,
    #     tokenizer=tokenizer,
    #     logger=logger,
    #     size=args.resolution,
    # )

    test_dataset = VGG_Sound_datasetn(args, tokenizer, logger, args.resolution)
    # test_dataset = VGGSound(args, tokenizer, logger, args.resolution)

    print("Length of test data: ", test_dataset.__len__())


    batch_size = 1
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=args.dataloader_num_workers
    )

    # Prepare everything with our `accelerator`.
    model, test_dataloader = accelerator.prepare(
        model, test_dataloader
    )

    obj_token_ids = []
    obj_token_id1 = tokenizer.convert_tokens_to_ids(args.obj1)
    obj_token_ids.append(obj_token_id1)
    obj2_token_id2 = tokenizer.convert_tokens_to_ids(args.obj2)
    obj_token_ids.append(obj2_token_id2)
    obj3_token_id3 = tokenizer.convert_tokens_to_ids(args.obj3)
    obj_token_ids.append(obj3_token_id3)
    obj4_token_id4 = tokenizer.convert_tokens_to_ids(args.obj4)
    obj_token_ids.append(obj4_token_id4)
    obj5_token_id5 = tokenizer.convert_tokens_to_ids(args.obj5)
    obj_token_ids.append(obj5_token_id5)
    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    accelerator.unwrap_model(model).text_encoder.resize_token_embeddings(len(tokenizer))

    prompt = args.prompt
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        tokenizer=tokenizer,
        text_encoder=accelerator.unwrap_model(model).text_encoder,
        vae=accelerator.unwrap_model(model).vae,
        unet=accelerator.unwrap_model(model).unet,
        
    ).to(device)
    
    # pipeline = StableDiffusionPipeline.from_pretrained(
    #     "stabilityai/stable-diffusion-2-1-base",
    #     torch_dtype=torch.float16 if device=="cuda" else torch.float32
    # ).to(device)

    all_tokens = []
    iis = 0
    ais = 0
    aic = 0
    cts = 0
    ccs = 0
    cnt = 0

    B = 4
    gen_img_batch = []
    g = torch.Generator(device).manual_seed(42)
    labels = [[] for _ in range(B)]
    ivs = [[] for _ in range(B)]
    avs = [[] for _ in range(B)]

    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            # if step >= args.generation_steps:
            #     break
            print(f"Step {step+1}/{len(test_dataloader)}")

            num = batch["num_samples"][0]
            token_embeds = model.text_encoder.get_input_embeddings().weight.data
            for i in range(num):
                av = batch[f"audio_values{i+1}"].to(dtype=weight_dtype)
                iv = batch[f"image_values{i+1}"].to(dtype=weight_dtype)
                ivs[step % B].append(iv)
                avs[step % B].append(av)

                if args.audio_encoder == 'Clap':
                    feats = accelerator.unwrap_model(model).aud_encoder.get_audio_embedding_from_data(
                        av.cpu().float().numpy()
                    )
                    feats = torch.tensor(feats).to(accelerator.device)
                    # add a dimension for the number of audio tokens (in this case, 1) 
                    # [8, 512] -> [8, 1, 512]
                    feats.unsqueeze_(1) 
                else:
                    feats = accelerator.unwrap_model(model).aud_encoder.extract_features(av)[1]

                token = accelerator.unwrap_model(model).embedder(feats)
                token_embeds[obj_token_ids[i]] = token.clone()
                
                label = batch["full_name"][0].split("_")[-(i+1)]
                labels[step % B].append(label.replace(' ', '_'))

            if args.use_label == 'Yes':
                prompt = "a photo of a " + " and a ".join(labels[step % B])
            else:
                words = args.prompt.split(" ")
                prompt = " ".join(words[:5 + (num-1)*3])
            
            out = pipeline(prompt, num_inference_steps=args.num_inference_steps, generator=g, guidance_scale=7.5)
            gen_img = out.images[0]
            gen_img_batch.append(to_tensor(gen_img).to(device))
            gen_img.save(f'{args.output_dir}/imgs/{args.run_name}/{batch["full_name"][0]}.jpg')

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
        
                print(iis, ais, aic, cts, ccs)
                with open(os.path.join(args.output_dir, f"{args.run_name}_metrics2.txt"), "a") as f:
                    f.write(f"Step {step}: IIS={iis:.2f}, AIS={(ais):.2f}, AIC={(aic):.2f}, CLS={cts:.2f}, CCS={ccs:.2f}, count={cnt}\n")
        
        # visualize_tokens(all_tokens, reduce_algo="umap", save_path=f'{args.output_dir}/plots/{args.run_name}/umap_final.png', show=True)
        

if __name__ == "__main__":
    args = parse_args()
    inference(args)
