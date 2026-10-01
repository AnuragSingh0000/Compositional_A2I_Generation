#!/usr/bin/env python3
"""
For each image in good_comparison/VGG2/, find the same-named image
(ignoring extension) in each model's source folder and copy it to
the corresponding output folder.
"""

import shutil
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────

SELECTED_FOLDER = "good_comparisons/VGG"

# Maps: source folder  →  destination folder
FOLDER_MAP = {
    "output_macs/VGG/imgs":             "good_comparison/VGG/MACS",
    "output_vgg/imgs/VGG_AT":          "good_comparison/VGG/AT",
    "output_vgg/imgs/VGG_AT_org":      "good_comparison/VGG/AT_org",
    "output_vgg/imgs/VGG_mixed":       "good_comparison/VGG/mixed",
    "output_vgg/imgs/VGG_mixed_claps": "good_comparison/VGG/mixed_claps",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}

# ── Main logic ────────────────────────────────────────────────────────────────

def main():
    selected_dir = Path(SELECTED_FOLDER)
    if not selected_dir.exists():
        raise FileNotFoundError(f"Selected folder not found: {selected_dir}")

    # Collect selected images (any image extension)
    selected_images = [
        f for f in selected_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not selected_images:
        print("No images found in selected folder.")
        return

    print(f"Found {len(selected_images)} selected image(s) in '{selected_dir}'.\n")

    for src_folder, dst_folder in FOLDER_MAP.items():
        src_dir = Path(src_folder)
        dst_dir = Path(dst_folder)

        print(f"── {src_folder}  →  {dst_folder}")

        if not src_dir.exists():
            print(f"   ⚠  Source folder not found, skipping.\n")
            continue

        dst_dir.mkdir(parents=True, exist_ok=True)

        # Build stem → Path lookup so extension differences don't matter
        src_lookup = {
            f.stem: f
            for f in src_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        }

        found = missing = 0
        for img in selected_images:
            src_file = src_lookup.get(img.stem)
            if src_file:
                shutil.copy2(src_file, dst_dir / src_file.name)
                ext_note = (f"  [{img.suffix} -> {src_file.suffix}]"
                            if img.suffix.lower() != src_file.suffix.lower() else "")
                print(f"   OK  {img.stem}{ext_note}")
                found += 1
            else:
                print(f"   XX  {img.stem}  (not found in source)")
                missing += 1

        print(f"   -> copied {found}, missing {missing}\n")

    print("Done.")


if __name__ == "__main__":
    main()