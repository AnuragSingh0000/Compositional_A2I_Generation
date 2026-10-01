import os
import shutil

folder1 = "../dataset/LLP/image"
folder2 = "../dataset/LLP/audio"
output_folder = "../dataset/LLP/all"

os.makedirs(output_folder, exist_ok=True)


def copy_files(src_folder):
    for file in sorted(os.listdir(src_folder)):
        print(file)
        src_path = os.path.join(src_folder, file)
        dst_path = os.path.join(output_folder, file)

        if os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)

copy_files(folder1)
copy_files(folder2)

print("Files copied successfully!")
