import os
import shutil
import random
from collections import defaultdict
import soundfile as sf

# ---------------- CONFIG ----------------
SRC_DIR = "../dataset/VGG_Sound_extracted/test_data"
OUT_DIR = "../dataset/VGG_Sound_extracted/test_data_triples"
SEED = 42
MIN_GROUP_SIZE = 3
MAX_GROUP_SIZE = 3
MAX_GROUPS = 1000 # safety cap to avoid infinite loops; adjust if needed

random.seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------- AUDIO VALIDATION ----------------
def is_audio_valid(path):
    try:
        with sf.SoundFile(path) as f:
            return f.frames > 0
    except Exception:
        return False


# ---------------- BUILD CLASS → FILE PAIRS ----------------
class_map = defaultdict(list)

for file in os.listdir(SRC_DIR):
    name, ext = os.path.splitext(file)
    if ext.lower() not in [".jpg"]:
        continue

    try:
        ind, cls = name.split("_", 1)
    except ValueError:
        continue

    class_map[cls].append(ind)


# ---------------- FILTER INVALID AUDIO ONCE ----------------
valid_class_map = defaultdict(list)
bad_audio = 0

for cls, inds in class_map.items():
    for ind in inds:
        base = f"{ind}_{cls}"
        jpg = os.path.join(SRC_DIR, f"{base}.jpg")
        mp3 = os.path.join(SRC_DIR, f"{base}.mp3")

        if not (os.path.exists(jpg) and os.path.exists(mp3)):
            continue

        if not is_audio_valid(mp3):
            bad_audio += 1
            continue

        valid_class_map[cls].append(base)


print(f"Bad audio files removed: {bad_audio}")

# Remove empty classes
class_map = {cls: files for cls, files in valid_class_map.items() if files}
print(f"Total classes: {len(class_map)}")

# ---------------- GROUPING ----------------
class_list = list(class_map.keys())
# counters: how many items already taken (0..len-1). We'll prefer classes with counter < len(...)
class_counters = {cls: 0 for cls in class_list}

group_id = 0

# Continue until every class has had all its indices used at least once
def all_covered(counters, cmap):
    return all(counters[cls] >= len(cmap[cls]) for cls in cmap)

grp_sz = [0,0,0,0]
while group_id < MAX_GROUPS:
    g = random.randint(MIN_GROUP_SIZE, MAX_GROUP_SIZE)
    grp_sz[g-2] += 1

    # Prefer classes that still have unused indices
    eligible = [c for c in class_list if class_counters[c] < len(class_map[c])]
    if not eligible:
        break

    # if not enough eligible classes, take all; else sample without replacement
    if len(eligible) <= g:
        selected = eligible
    else:
        selected = random.sample(eligible, g)

    # Copy selected classes' next file (sequentially, using counter) into one group
    for cls in selected:
        idx = class_counters[cls]  # 0-based index into class_map[cls]; ensures sequential usage
        fname = class_map[cls][idx]  # like "12345_classname"

        # copy jpg and mp3 to OUT_DIR; name files with group id and class to keep them identifiable
        shutil.copy(
            os.path.join(SRC_DIR, f"{fname}.jpg"),
            os.path.join(OUT_DIR, f"{group_id}_{cls}.jpg"),
        )
        shutil.copy(
            os.path.join(SRC_DIR, f"{fname}.mp3"),
            os.path.join(OUT_DIR, f"{group_id}_{cls}.mp3"),
        )

        # advance counter for this class so next time we use the next index
        class_counters[cls] = (class_counters[cls] + 1) % len(class_map[cls])

    group_id += 1
    # print("Group ID: ", group_id, " Group sz: ", g)

print("Grouping complete.")
print("Total groups created:", group_id)
print("Total files:", len(os.listdir(OUT_DIR)))
for i in range(4):
    print(f"Size of groups of {i+2} ", grp_sz[i])





# import os
# import shutil
# import random
# from collections import defaultdict
# import soundfile as sf

# # ---------------- CONFIG ----------------
# SRC_DIR = "../dataset/VGG_Sound_extracted/test_data"
# OUT_DIR = "../dataset/VGG_Sound_extracted/test_data_paired"
# SEED = 42
# MIN_GROUP_SIZE = 2
# MAX_GROUP_SIZE = 2
# MAX_GROUPS = 2500

# random.seed(SEED)
# os.makedirs(OUT_DIR, exist_ok=True)

# # ---------------- AUDIO VALIDATION ----------------
# def is_audio_valid(path):
#     try:
#         with sf.SoundFile(path) as f:
#             return f.frames > 0
#     except Exception:
#         return False

# # ---------------- BUILD CLASS → FILE PAIRS ----------------
# class_map = defaultdict(list)

# for file in os.listdir(SRC_DIR):
#     name, ext = os.path.splitext(file)
#     if ext.lower() != ".jpg":
#         continue

#     try:
#         ind, cls = name.split("_", 1)
#     except ValueError:
#         continue

#     class_map[cls].append(ind)

# # ---------------- FILTER INVALID AUDIO ONCE ----------------
# filtered_class_map = defaultdict(list)
# bad_audio = 0

# for cls, inds in class_map.items():
#     for ind in inds:
#         base = f"{ind}_{cls}"
#         jpg = os.path.join(SRC_DIR, f"{base}.jpg")
#         mp3 = os.path.join(SRC_DIR, f"{base}.mp3")

#         if not (os.path.exists(jpg) and os.path.exists(mp3)):
#             continue

#         if not is_audio_valid(mp3):
#             bad_audio += 1
#             continue

#         filtered_class_map[cls].append(base)

# # Remove empty classes
# class_map = {cls: files for cls, files in filtered_class_map.items() if files}

# print(f"Total classes after filtering: {len(class_map)}")
# print(f"Bad audio files removed: {bad_audio}")

# # ---------------- GROUPING ----------------
# class_list = list(class_map.keys())
# class_counters = {cls: 0 for cls in class_list}

# group_id = 0
# grp_sz = [0, 0, 0, 0]

# while group_id < MAX_GROUPS:
#     g = random.randint(MIN_GROUP_SIZE, MAX_GROUP_SIZE)

#     eligible = [c for c in class_list if class_counters[c] < len(class_map[c])]
#     if not eligible:
#         break

#     selected = (
#         eligible if len(eligible) <= g else random.sample(eligible, g)
#     )

#     for cls in selected:
#         idx = class_counters[cls]
#         fname = class_map[cls][idx]

#         shutil.copy(
#             os.path.join(SRC_DIR, f"{fname}.jpg"),
#             os.path.join(OUT_DIR, f"{group_id}_{cls}.jpg"),
#         )
#         shutil.copy(
#             os.path.join(SRC_DIR, f"{fname}.mp3"),
#             os.path.join(OUT_DIR, f"{group_id}_{cls}.mp3"),
#         )

#         class_counters[cls] += 1

#     grp_sz[g - 2] += 1
#     group_id += 1

# # ---------------- SUMMARY ----------------
# print("Grouping complete.")
# print("Total groups created:", group_id)
# print("Total files:", len(os.listdir(OUT_DIR)))
# for i in range(4):
#     print(f"Groups of size {i+2}: {grp_sz[i]}")
