import os
import re
import torchaudio
from collections import defaultdict

DATASET_DIR = "/home/singh_prajwal/dataset/VGG_Sound_extracted/test_data_paired"   # CHANGE THIS
grp_sz = 2

# regex: captures index, label, and extension
# examples matched:
# 12_dog.jpg, 12_cat.mp3
pattern = re.compile(r"^(\d+)_.*\.(jpg|mp3)$")

files = os.listdir(DATASET_DIR)

index_to_files = defaultdict(list)
all_indices = set()

# -------------------------
# 1. Parse files by index
# -------------------------
for f in files:
    match = pattern.match(f)
    if match:
        idx = int(match.group(1))
        index_to_files[idx].append(f)
        all_indices.add(idx)

# -------------------------
# 2. Check index continuity
# -------------------------
print("\n===== INDEX CONTINUITY CHECK =====")

if not all_indices:
    print("❌ No valid indexed files found.")
    exit()

min_idx = min(all_indices)
max_idx = max(all_indices)

if min_idx != 0:
    print(f"❌ Dataset does not start from index 0 (starts at {min_idx})")

missing_indices = [i for i in range(0, max_idx + 1) if i not in all_indices]

if missing_indices:
    print(f"❌ Missing indices ({len(missing_indices)}):")
    print(missing_indices[:20], "..." if len(missing_indices) > 20 else "")
else:
    print("✅ No missing indices")

# -------------------------
# 3. Check group completeness
# -------------------------
print("\n===== GROUP COMPLETENESS CHECK =====")

incomplete_groups = []

for idx, flist in index_to_files.items():
    jpgs = [f for f in flist if f.endswith(".jpg")]
    mp3s = [f for f in flist if f.endswith(".mp3")]

    if len(jpgs) != grp_sz or len(mp3s) != grp_sz:
        incomplete_groups.append((idx, jpgs, mp3s))

if incomplete_groups:
    print(f"❌ Incomplete groups found: {len(incomplete_groups)}")
    for idx, jpgs, mp3s in incomplete_groups[:20]:
        print(f"Index {idx}: jpgs={len(jpgs)}, mp3s={len(mp3s)}")
else:
    print("✅ All groups have JPGs + MP3s")

# -------------------------
# 4. Validate audio files
# -------------------------
print("\n===== AUDIO VALIDITY CHECK =====")

bad_audio = []

for idx, flist in index_to_files.items():
    for f in flist:
        if f.endswith(".mp3"):
            path = os.path.join(DATASET_DIR, f)
            try:
                torchaudio.load(path)
            except Exception as e:
                bad_audio.append((idx, f, str(e)))

if bad_audio:
    print(f"❌ Corrupted / unreadable audio files: {len(bad_audio)}")
    for idx, f, err in bad_audio[:20]:
        print(f"Index {idx} | {f} | {err}")
else:
    print("✅ All audio files load correctly")

# -------------------------
# 5. Final summary
# -------------------------
print("\n===== SUMMARY =====")
print(f"Total indices found : {len(all_indices)}")
print(f"Missing indices     : {len(missing_indices)}")
print(f"Incomplete groups   : {len(incomplete_groups)}")
print(f"Bad audio files     : {len(bad_audio)}")
