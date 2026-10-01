# import os
# import argparse
# import torch
# from torchvision.io import read_image
# from torcheval.metrics import FrechetInceptionDistance as FID

# # Device setup
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def process_image(fid_metric, generated_image, target_image):
#     """
#     Processes a single generated image and a single real image.
#     """

#     # Ensure shape: [1, C, H, W]
#     if generated_image.ndim == 3:
#         generated_image = generated_image.unsqueeze(0)
#     if target_image.ndim == 3:
#         target_image = target_image.unsqueeze(0)

#     generated_image = generated_image.float()
#     target_image = target_image.float()

#     # Normalize to [0, 1]
#     if generated_image.min() < 0 or generated_image.max() > 1:
#         generated_image = (generated_image + 1) / 2
#     if target_image.min() < 0 or target_image.max() > 1:
#         target_image = (target_image + 1) / 2

#     fid_metric.update(generated_image.to(device), is_real=False)
#     fid_metric.update(target_image.to(device), is_real=True)


# def calculate_fid(gen_folder, real_folder, n):
#     fid_metric = FID().to(device)
#     valid_exts = ('.jpg', '.png')

#     gen_files = [f for f in os.listdir(gen_folder) if f.lower().endswith(valid_exts)]
#     print("    Lenght of generated folder: ", len(gen_files))

#     for gen_file in gen_files:
#         gen_path = os.path.join(gen_folder, gen_file)
#         gen_img = read_image(gen_path).float() / 255.0
#         gen_img = gen_img.to(device)

#         base_name, _ = os.path.splitext(gen_file)
#         target_images = []

#         if n == 1:
#             for ext in valid_exts:
#                 real_path = os.path.join(real_folder, f"{base_name}{ext}")
#                 if os.path.exists(real_path):
#                     real_img = read_image(real_path).float() / 255.0
#                     target_images.append(real_img.to(device))
#                     break

#         elif n == 2:
#             parts = base_name.split('_')
#             if len(parts) == 3:
#                 idx, label1, label2 = parts
#                 real_names = [f"{idx}_{label1}", f"{idx}_{label2}"]

#                 for r_name in real_names:
#                     for ext in valid_exts:
#                         real_path = os.path.join(real_folder, f"{r_name}{ext}")
#                         if os.path.exists(real_path):
#                             real_img = read_image(real_path).float() / 255.0
#                             target_images.append(real_img.to(device))
#                             break
        
#         for real_img in target_images:
#             process_image(fid_metric, gen_img, real_img)
        

#     return fid_metric.compute().item()


# def main():
#     parser = argparse.ArgumentParser(description="Compute FID score")
#     parser.add_argument("--n", type=int, required=True, help="Number of labels (1 or 2)")
#     parser.add_argument("--gen_folder", type=str, required=True, help="Generated images folder")
#     parser.add_argument("--real_folder", type=str, required=True, help="Real images folder")
#     parser.add_argument("--run_name", type=str, required=True, help="Run identifier")

#     args = parser.parse_args()

#     print(f"    Calculating FID for run: {args.run_name} with n={args.n}")
#     print(f"    Generated images folder: {args.gen_folder}")
#     print(f"    Real images folder: {args.real_folder}")

#     fid_score = calculate_fid(args.gen_folder, args.real_folder, args.n)

#     print(f"    Final FID Score (run={args.run_name}): {fid_score}")

#     # Save to file
#     output_file = "output_fid.txt"
#     with open(output_file, "a") as f:
#         f.write(f"{args.run_name}: {fid_score}\n")


# if __name__ == "__main__":
#     main()




import os
import argparse
import torch
from tqdm import tqdm
from torchvision.io import read_image
from torcheval.metrics import FrechetInceptionDistance as FID

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def process_image(fid_metric, generated_image, target_image):
    """
    Processes a single generated image and a single real image.
    """

    # Ensure shape: [1, C, H, W]
    if generated_image.ndim == 3:
        generated_image = generated_image.unsqueeze(0)
    if target_image.ndim == 3:
        target_image = target_image.unsqueeze(0)

    generated_image = generated_image.float()
    target_image = target_image.float()

    # Normalize to [0, 1]
    if generated_image.min() < 0 or generated_image.max() > 1:
        generated_image = (generated_image + 1) / 2
    if target_image.min() < 0 or target_image.max() > 1:
        target_image = (target_image + 1) / 2

    fid_metric.update(generated_image.to(device), is_real=False)
    fid_metric.update(target_image.to(device), is_real=True)


def calculate_fid(gen_folder, real_folder, n_val):
    fid_metric = FID().to(device)
    valid_exts = ('.jpg', '.png')

    gen_files = [f for f in os.listdir(gen_folder) if f.lower().endswith(valid_exts)]
    print("    Length of generated folder: ", len(gen_files))

    for gen_file in tqdm(gen_files, desc="Processing"):
        gen_path = os.path.join(gen_folder, gen_file)
        gen_img = read_image(gen_path).float() / 255.0
        gen_img = gen_img.to(device)

        base_name, _ = os.path.splitext(gen_file)
        target_images = []

        # Parse target names dynamically based on the filename
        real_names = []
        if str(n_val) == "1":
            real_names = [base_name]
        else:
            # Splits idx_label1_label2... into parts
            parts = base_name.split('_')
            
            if len(parts) >= 2:
                idx = parts[0]
                labels = parts[1:]
                # Creates a list of [idx_label1, idx_label2, ..., idx_labelN]
                real_names = [f"{idx}_{label}" for label in labels]
            else:
                # Fallback if no underscores are present in the filename
                real_names = [base_name]

        # Find matching target images
        for r_name in real_names:
            for ext in valid_exts:
                real_path = os.path.join(real_folder, f"{r_name}{ext}")
                if os.path.exists(real_path):
                    real_img = read_image(real_path).float() / 255.0
                    target_images.append(real_img.to(device))
                    break
        
        # Calculate FID for each generated vs real pair
        for real_img in target_images:
            process_image(fid_metric, gen_img, real_img)
        
    return fid_metric.compute().item()


def main():
    parser = argparse.ArgumentParser(description="Compute FID score")
    # Changed type to str to accept integers or words like "variable"
    parser.add_argument("--n", type=str, required=True, help="Number of labels (1 to 5, or 'variable')")
    parser.add_argument("--gen_folder", type=str, required=True, help="Generated images folder")
    parser.add_argument("--real_folder", type=str, required=True, help="Real images folder")
    parser.add_argument("--run_name", type=str, required=True, help="Run identifier")

    args = parser.parse_args()

    print(f"    Calculating FID for run: {args.run_name} with n={args.n}")
    print(f"    Generated images folder: {args.gen_folder}")
    print(f"    Real images folder: {args.real_folder}")

    fid_score = calculate_fid(args.gen_folder, args.real_folder, args.n)

    print(f"    Final FID Score (run={args.run_name}): {fid_score}")

    # Save to file
    output_file = "output_fid.txt"
    with open(output_file, "a") as f:
        f.write(f"{args.run_name}: {fid_score}\n")


if __name__ == "__main__":
    main()