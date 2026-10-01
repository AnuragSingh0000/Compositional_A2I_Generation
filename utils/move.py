import os
import shutil
from collections import defaultdict

src_dir = "../dataset/VGG_Sound_extracted/test_data_paired"
parts = 5
sz = len(os.listdir(src_dir)) // 4
part_sz = sz // parts + 1
cnt = 0
idx_label = defaultdict(list)

for filename in os.listdir(src_dir):
    if (filename.split(".")[-1] != "jpg"):
        continue
    name = filename.split(".")[0]
    idx = name.split("_")[0]
    idx_label[idx].append(name)

for key in idx_label:
    if (cnt % part_sz == 0):
        dst_dir = f"../dataset/VGG_Sound_extracted/test_data_paired_{cnt//part_sz}"
        os.makedirs(dst_dir, exist_ok=True)
    for i in range(2):
        name = idx_label[key][i]
        label = name.split("_")[-1]
        dst_name = f"{cnt%part_sz}_{label}"

        shutil.copy(
            os.path.join(src_dir, name+".jpg"),
            os.path.join(dst_dir, dst_name+".jpg")
        )
        shutil.copy(
            os.path.join(src_dir, name+".mp3"),
            os.path.join(dst_dir, dst_name+".mp3")
        )
    cnt += 1

print("Files moved successfully.")
print(f"Files moved to destination directory: {sz}")
