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
from data.dataloader import VGG_Sound_dataset, VGGSound
from visualise_utils import run_kmeans_tsne 
from evaluate import evaluate_all, process_image
from torcheval.metrics import FrechetInceptionDistance as FID
# from cluster import cluster_tokens
from cluster import visualize_tokens

torch.cuda.empty_cache()

import random
random_seed=45
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.12.0")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device using: ", device)
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    
logger = get_logger(__name__)

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
    parser.add_argument("--run_name", type=str, default='AudioToken',
                        help="Insert run name")
    parser.add_argument("--audio_encoder", type=str, default='Beats',
                        help="Select the audio encoder to use, either 'Beats' or 'Clap'")
    parser.add_argument("--set_size", type=str, default='full')
    parser.add_argument("--prompt", type=str, default='a photo of a <*>')
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

    if args.audio_encoder == 'Clap':
        from modules.AudioToken.AudioToken_clap import AudioTokenWrapper
    else:
        from modules.AudioToken.AudioToken import AudioTokenWrapper

    model = AudioTokenWrapper(args, accelerator).to(weight_dtype).eval()

    # Add the placeholder token in tokenizer
    num_added_tokens = tokenizer.add_tokens(args.obj1)
    if num_added_tokens == 0:
        raise ValueError(
            f"The tokenizer already contains the token {args.placeholder_token}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True


    test_dataset = VGG_Sound_dataset(args, tokenizer, logger, args.resolution)
    # test_dataset = VGGSound(args, tokenizer, logger, args.resolution)

    print("Length of test data: ", test_dataset.__len__())


    batch_size = 8
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=args.dataloader_num_workers
    )

    # Prepare everything with our `accelerator`.
    model, test_dataloader = accelerator.prepare(
        model, test_dataloader
    )

    obj1_token_id = tokenizer.convert_tokens_to_ids(args.obj1)
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

    print(args.pretrained_model_name_or_path)
    all_tokens = []
    iis = 0
    ais = 0
    aic = 0
    ccs = 0
    cts = 0
    # fid_metric = FID().to(device)
    cnt = 0
    g = torch.Generator(device).manual_seed(42)

    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            # if step >= args.generation_steps:
            #     break
            print(f"Step {step+1}/{len(test_dataloader)}")
            av = batch["audio_values"].to(dtype=weight_dtype)
            iv = batch["image_values"].to(dtype=weight_dtype)

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
            token_embeds = model.text_encoder.get_input_embeddings().weight.data
            gen_img_batch = []
            for ind in range(token.shape[0]):
                token_embeds[obj1_token_id] = token[ind].clone()

                label = batch["full_name"][ind].split("_")[-1]
                all_tokens.append((label, feats[ind].detach().cpu()))

                if args.use_label == 'Yes':
                    prompt = "a photo of a " + label.replace(' ', '_')
                # print(prompt)
                
                out = pipeline(prompt, num_inference_steps=args.num_inference_steps, generator=g, guidance_scale=7.5)
                gen_img = out.images[0]
                gen_img_batch.append(to_tensor(gen_img).to(device))
                gen_img.save(f'{args.output_dir}/imgs/{args.run_name}/{batch["full_name"][ind]}.png')

                metrics = evaluate_all(gen_img, iv, av, label, current_audio_idx_in_batch=ind)
                iis += metrics["iis"]["iss"]
                ais += metrics["ais"]["ais"]
                aic += metrics["aic"]["accuracy"]
                cts += metrics["cls"]["similarity"]
                ccs += metrics["ccs"]["similarity"]
                cnt += 1
            
            print(iis, ais, aic, cts, ccs)
            if (step + 1) % 10 == 0:
                print(iis, ais, aic, cts, ccs)
                with open(os.path.join(args.output_dir, f"{args.run_name}_metrics.txt"), "a") as f:
                    f.write(f"Step {step}: IIS={iis:.2f}, AIS={(ais):.2f}, AIC={(aic):.2f}, CLS={(cts):.2f}, CCS={(ccs):.2f}, count={cnt}\n")
            
        visualize_tokens(all_tokens, reduce_algo="umap", save_dir=f'{args.output_dir}/plots/{args.run_name}', show=True)
        print(iis, ais, aic, cts, ccs)
        with open(os.path.join(args.output_dir, f"{args.run_name}_metrics.txt"), "a") as f:
            f.write(f"Step {step}: IIS={iis:.2f}, AIS={(ais):.2f}, AIC={(aic):.2f}, CLS={(cts):.2f}, CCS={(ccs):.2f}, count={cnt}\n")
        

if __name__ == "__main__":
    args = parse_args()
    inference(args)



# import argparse
# import logging
# import os
# import torch
# import torch.utils.checkpoint
# from torch.utils.data import Dataset
# import datasets
# import diffusers
# import transformers
# from PIL import Image
# import numpy as np
# from accelerate import Accelerator
# from accelerate.logging import get_logger
# from accelerate.utils import set_seed
# from diffusers import StableDiffusionPipeline
# from diffusers.utils import check_min_version
# from transformers import CLIPTokenizer, CLIPTextModel
# from torchvision import transforms
# from torchvision.transforms.functional import to_tensor
# from data.dataloader import VGG_Sound_dataset, VGGSound
# from modules.AudioToken.AudioToken import AudioTokenWrapper
# from visualise_utils import run_kmeans_tsne 
# from evaluate import evaluate_all, process_image
# from torcheval.metrics import FrechetInceptionDistance as FID
# # from cluster import cluster_tokens
# from cluster import visualize_tokens

# torch.cuda.empty_cache()

# import random
# random_seed=45
# torch.manual_seed(random_seed)
# torch.cuda.manual_seed(random_seed)
# torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# np.random.seed(random_seed)
# random.seed(random_seed)

# # Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.12.0")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Device using: ", device)
# print("CUDA available:", torch.cuda.is_available())
# print("Device count:", torch.cuda.device_count())
# for i in range(torch.cuda.device_count()):
#     print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    
# logger = get_logger(__name__)

# def parse_args():
#     parser = argparse.ArgumentParser(description="Simple example of a training script.")
#     parser.add_argument("--learned_embeds", type=str, default='output/embedder_learned_embeds.bin',
#                         help="Path to pretrained embedder")
#     parser.add_argument("--learned_embeds_lora", type=str, default='output/lora_layers_learned_embeds.bin',
#                         help="Path to pretrained embedder")
#     parser.add_argument("--pretrained_model_name_or_path", type=str, default='stabilityai/stable-diffusion-2',
#                         help="Path to pretrained model or model identifier from huggingface.co/models.")
#     parser.add_argument("--revision", type=str, default=None, required=False,
#                         help="Revision of pretrained model identifier from huggingface.co/models.")
#     parser.add_argument("--tokenizer_name", type=str, default=None,
#                         help="Pretrained tokenizer name or path if not the same as model_name")
#     parser.add_argument("--data_dir", type=str,
#                         help="A folder containing the training data.")
#     parser.add_argument("--obj1", type=str, default="<*>",
#                         help="A token to use as a placeholder for the audio.",)
#     parser.add_argument("--obj2", type=str, default="<^>",
#                         help="A token to use as a placeholder for the audio.",)
#     parser.add_argument("--output_dir", type=str, default="output",
#                         help="The output directory where the model predictions and checkpoints will be written.")
#     parser.add_argument("--seed", type=int, default=None,
#                         help="A seed for reproducible training.")
#     parser.add_argument("--resolution", type=int, default=768,
#                         help="The resolution for input images, all the images in the train/validation"
#                             " dataset will be resized to this resolution")
#     parser.add_argument("--dataloader_num_workers", type=int, default=6,
#                         help="Number of subprocesses to use for data loading."
#                             " 0 means that the data will be loaded in the main process.")
#     parser.add_argument("--logging_dir", type=str, default="logs",
#                         help="[TensorBoard](https://www.tensorflow.org/tensorboard) log directory."
#                              " Will default to *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***.")
#     parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"],
#                         help="Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16)."
#                             " Bf16 requires PyTorch >= 1.10. and an Nvidia Ampere GPU.")
#     parser.add_argument("--allow_tf32", action="store_true",
#                         help="Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training."
#                             " For more information, see"
#                             " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices")
#     parser.add_argument("--report_to", type=str, default="tensorboard",
#                         help='The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
#                             ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.')
#     parser.add_argument("--num_inference_steps", type=int, default=50)
#     parser.add_argument("--data_set", type=str, default='test', choices=['train', 'test'],
#                         help="Whether use train or test set")
#     parser.add_argument("--generation_steps", type=int, default=50)
#     parser.add_argument("--run_name", type=str, default='AudioToken',
#                         help="Insert run name")
#     parser.add_argument("--set_size", type=str, default='full')
#     parser.add_argument("--prompt", type=str, default='a photo of a <*>')
#     parser.add_argument("--input_length", type=int, default=10,
#                         help="Select the number of seconds of audio you want in each test-sample.")
#     parser.add_argument("--lora", type=bool, default=False,
#                         help="Whether load Lora layers or not")
#     parser.add_argument("--split", type=bool, default=True,
#                         help="Whether split the audio or not")
#     parser.add_argument("--caption", type=bool, default=False,
#                         help="Whether split the audio or not")

#     args = parser.parse_args()
#     # env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
#     # if env_local_rank != -1 and env_local_rank != args.local_rank:
#     #     args.local_rank = env_local_rank

#     if args.data_dir is None:
#         raise ValueError("You must specify a train data directory.")

#     return args


# def inference(args):
#     logging_dir = os.path.join(args.output_dir, args.logging_dir)

#     folder_path = "output/"
#     if not os.path.exists(folder_path):
#         os.makedirs(folder_path)

#     folder_path = "output/imgs/"
#     if not os.path.exists(folder_path):
#         os.makedirs(folder_path)

#     folder_path = f"output/imgs/{args.run_name}/"
#     if not os.path.exists(folder_path):
#         os.makedirs(folder_path)

#     accelerator = Accelerator(
#         mixed_precision=args.mixed_precision,
#         log_with=args.report_to,
#         project_dir=logging_dir,
#     )

#     # Make one log on every process with the configuration for debugging.
#     logging.basicConfig(
#         format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
#         datefmt="%m/%d/%Y %H:%M:%S",
#         level=logging.INFO,
#     )
#     logger.info(accelerator.state, main_process_only=False)
#     if accelerator.is_local_main_process:
#         datasets.utils.logging.set_verbosity_warning()
#         transformers.utils.logging.set_verbosity_warning()
#         diffusers.utils.logging.set_verbosity_info()
#     else:
#         datasets.utils.logging.set_verbosity_error()
#         transformers.utils.logging.set_verbosity_error()
#         diffusers.utils.logging.set_verbosity_error()

#     # If passed along, set the training seed now.
#     if args.seed is not None:
#         set_seed(args.seed)

#     # Load tokenizer
#     if args.tokenizer_name:
#         tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
#     elif args.pretrained_model_name_or_path:
#         tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")

#     weight_dtype = torch.float32
#     if accelerator.mixed_precision == "fp16":
#         weight_dtype = torch.float16
#     elif accelerator.mixed_precision == "bf16":
#         weight_dtype = torch.bfloat16

#     model = AudioTokenWrapper(args, accelerator).to(weight_dtype).eval()

#     # Add the placeholder token in tokenizer
#     num_added_tokens = tokenizer.add_tokens(args.obj1)
#     if num_added_tokens == 0:
#         raise ValueError(
#             f"The tokenizer already contains the token {args.placeholder_token}. Please pass a different"
#             " `placeholder_token` that is not already in the tokenizer."
#         )

#     # Enable TF32 for faster training on Ampere GPUs,
#     # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
#     if args.allow_tf32:
#         torch.backends.cuda.matmul.allow_tf32 = True


#     test_dataset = VGG_Sound_dataset(args, tokenizer, logger, args.resolution)
#     # test_dataset = VGGSound(args, tokenizer, logger, args.resolution)

#     print("Length of test data: ", test_dataset.__len__())


#     batch_size = 8
#     test_dataloader = torch.utils.data.DataLoader(
#         test_dataset, batch_size=batch_size, shuffle=True, num_workers=args.dataloader_num_workers
#     )

#     # Prepare everything with our `accelerator`.
#     model, test_dataloader = accelerator.prepare(
#         model, test_dataloader
#     )

#     obj1_token_id = tokenizer.convert_tokens_to_ids(args.obj1)
#     # Resize the token embeddings as we are adding new special tokens to the tokenizer
#     accelerator.unwrap_model(model).text_encoder.resize_token_embeddings(len(tokenizer))

#     prompt = args.prompt

#     pipeline = StableDiffusionPipeline.from_pretrained(
#         args.pretrained_model_name_or_path,
#         tokenizer=tokenizer,
#         text_encoder=accelerator.unwrap_model(model).text_encoder,
#         vae=accelerator.unwrap_model(model).vae,
#         unet=accelerator.unwrap_model(model).unet,
#     ).to(device)
    
#     # pipeline = StableDiffusionPipeline.from_pretrained(
#     #     "stabilityai/stable-diffusion-2-1-base",
#     #     torch_dtype=torch.float16 if device=="cuda" else torch.float32
#     # ).to(device)

#     print(args.pretrained_model_name_or_path)
#     all_tokens = []
#     iis = 0
#     ais = 0
#     aic = 0
#     ccs = 0
#     cts = 0
#     # fid_metric = FID().to(device)
#     cnt = 0

#     with torch.no_grad():
#         for step, batch in enumerate(test_dataloader):
#             # if step >= args.generation_steps:
#             #     break
#             print(f"Step {step+1}/{len(test_dataloader)}")
#             av = batch["audio_values"].to(dtype=weight_dtype)
#             iv = batch["image_values"].to(dtype=weight_dtype)

#             feats = accelerator.unwrap_model(model).aud_encoder.extract_features(av)[1]
#             token = accelerator.unwrap_model(model).embedder(feats)
#             token_embeds = model.text_encoder.get_input_embeddings().weight.data
#             g = torch.Generator(device).manual_seed(42)
#             gen_img_batch = []
#             for ind in range(token.shape[0]):
#                 token_embeds[obj1_token_id] = token[ind].clone()

#                 # cls = classify_image(iv[ind].detach().cpu(), model_choice="L14", top_k=1)
#                 label = batch["full_name"][ind].split("_")[-1]
#                 all_tokens.append((label, feats[ind].detach().cpu()))

#                 # prompt = "a photo of a " + label
#                 # print(prompt)
#                 out = pipeline(prompt, num_inference_steps=args.num_inference_steps, generator=g, guidance_scale=7.5)
#                 gen_img = out.images[0]
#                 gen_img_batch.append(to_tensor(gen_img).to(device))
#                 gen_img.save(f'output/imgs/{args.run_name}/{batch["full_name"][ind]}.png')

#                 metrics = evaluate_all(gen_img, iv, av, label, current_audio_idx_in_batch=ind)
#                 iis += metrics["iis"]["iss"]
#                 ais += metrics["ais"]["ais"]
#                 aic += metrics["aic"]["accuracy"]
#                 cts += metrics["cls"]["similarity"]
#                 ccs += metrics["ccs"]["similarity"]
#                 cnt += 1
            
#             # process_image(fid_metric, gen_img_batch, iv)
#             print(iis, ais, aic, cts, ccs)
#             if (step + 1) % 10 == 0:
#                 # visualize_tokens(all_tokens, reduce_algo="umap", save_path=f'output/plots/{args.run_name}/umap_{step}.png', show=True)
#                 # fid = fid_metric.compute().item()
#                 print(iis, ais, aic, cts, ccs)
#                 with open(os.path.join(args.output_dir, f"{args.run_name}_metrics.txt"), "a") as f:
#                     f.write(f"Step {step}: IIS={iis:.2f}, AIS={(ais):.2f}, AIC={(aic):.2f}, CLS={(cts):.2f}, CCS={(ccs):.2f}, count={cnt}\n")
            
#         visualize_tokens(all_tokens, reduce_algo="umap", save_dir=f'output/plots/{args.run_name}', show=True)
#         # fid = fid_metric.compute().item()
#         print(iis, ais, aic, cts, ccs)
#         with open(os.path.join(args.output_dir, f"{args.run_name}_metrics.txt"), "a") as f:
#             f.write(f"Step {step}: IIS={iis:.2f}, AIS={(ais):.2f}, AIC={(aic):.2f}, CLS={(cts):.2f}, CCS={(ccs):.2f}, count={cnt}\n")
        

# if __name__ == "__main__":
#     args = parse_args()
#     inference(args)

