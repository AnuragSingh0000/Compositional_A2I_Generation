#!/usr/bin/env python3
"""
Remove all .png files from a folder and report how many were removed.
Usage: python remove_pngs.py <folder_path>
"""

import os
import sys
import glob


def remove_png_files(folder_path: str) -> None:
    if not os.path.isdir(folder_path):
        print(f"Error: '{folder_path}' is not a valid directory.")
        sys.exit(1)

    png_files = glob.glob(os.path.join(folder_path, "*.png"))

    if not png_files:
        print(f"No .png files found in '{folder_path}'.")
        return

    removed = 0
    failed = 0

    for png_path in png_files:
        try:
            os.remove(png_path)
            print(f"  Removed: {os.path.basename(png_path)}")
            removed += 1
        except OSError as e:
            print(f"  Failed to remove {os.path.basename(png_path)}: {e}")
            failed += 1

    print(f"\n--- Summary ---")
    print(f"Total .png files found : {len(png_files)}")
    print(f"Successfully removed   : {removed}")
    if failed:
        print(f"Failed to remove       : {failed}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python remove_pngs.py <folder_path>")
        sys.exit(1)

    remove_png_files(sys.argv[1]) 