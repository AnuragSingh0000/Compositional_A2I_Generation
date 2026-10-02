import torch
import os
import argparse
from torch import autocast
from diffusers import StableDiffusionPipeline
from evaluate2 import VGG_Sound_dataset2, evaluate_all

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT_DIR = "output/sd_is3"
IMG_DIR = os.path.join(OUT_DIR, "images")
DATA_DIR = "../../dataset/IS3"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)


test_dataset = VGG_Sound_dataset2(DATA_DIR, input_length=10, interpolation='bicubic')
# test_dataset = VGG_Sound_dataset2(args, tokenizer, logger, args.resolution)

print("Length of test data: ", test_dataset.__len__())


batch_size = 8
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=6
)

pipe = StableDiffusionPipeline.from_pretrained(
	"CompVis/stable-diffusion-v1-4", 
	use_auth_token=True
).to(device)

prompt = "a photo of a "
iis = 0
cts = 0
ccs = 0
cnt = 0
with torch.no_grad():
    for step, batch in enumerate(test_dataloader):
        print(f"Step {step+1}/{len(test_dataloader)}")
        num_samples = batch["num_samples"][0].item()

        for ind in range(batch_size):
            labels = []
            iv = []
            for i in range(num_samples):
                label = batch["full_name"][ind].split("_")[-(i+1)]
                label = label.replace(' ', '_')
                iv.append(batch[f'image_values{i+1}'][ind])
                labels.append(label)
                
            prompti = f"a photo of a " + " and a ".join(labels)
            with autocast(device.type):
                image = pipe(prompti).images[0]  
            metrics = evaluate_all(image, iv, labels, current_audio_idx_in_batch=ind)
            iis += metrics["iis"]["iss"]
            cts += metrics["cls"]["gm"]
            ccs += metrics["ccs"]["gm"]
            cnt += 1
            image.save(os.path.join(IMG_DIR, f"{batch['full_name'][ind]}.png"))
            
        if (step + 1) % 10 == 0:
            with open(os.path.join(OUT_DIR, f"metrics.txt"), "a") as f:
                    f.write(f"Step {step}: IIS={iis:.2f}, CLS={cts:.2f}, CCS={ccs:.2f}, count={cnt}\n")

    with open(os.path.join(OUT_DIR, f"metrics.txt"), "a") as f:
        f.write(f"Final Step: IIS={iis/cnt:.4f}, CLS={cts/cnt:.4f}, CCS={ccs/cnt:.4f}, count={cnt}\n")

        
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

            words = args.prompt.split(" ")
            prompt = " ".join(words[:5 + (num-1)*3])

            controller = AttentionStore()
            token_indices = get_indices_to_alter(pipeline, prompt) if config.token_indices is None else config.token_indices
            
            if args.use_label == "Yes":
                prompt = "a photo of a " + " and a ".join(labels[step % B])
            

            img = run_on_prompt(prompt,
                model=pipeline,
                controller=controller,
                token_indices=token_indices,
                seed=g,
                config=config)
            # img = pipeline(config.prompt, num_inference_steps=args.num_inference_steps, guidance_scale=7.5).images[0]

            gen_img = img
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
        
print("Evaluation completed.")
            