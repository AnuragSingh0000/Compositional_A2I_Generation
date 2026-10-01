#!/usr/bin/env python3
from pathlib import Path
import sys


ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
IMAGE_EXTS = [".jpg", ".jpeg", ".png"]

try:
    import soundfile as sf
except Exception:
    sf = None

mp3_files = sorted(ROOT.glob("*.mp3"))
mp3_deleted = 0
img_deleted = 0

for mp3 in mp3_files:
    bad = False
    if sf:
        try:
            with sf.SoundFile(str(mp3), "r"):
                pass
        except Exception:
            bad = True
    else:
        # if soundfile not available, treat as unreadable (avoid accidental keep)
        bad = True

    if bad:
        print(f"Deleting: {mp3}", flush=True)
        try:
            mp3.unlink()
            mp3_deleted += 1
        except Exception as e:
            print(f"  failed to delete mp3: {e}", flush=True)
        stem = mp3.stem
        for ext in IMAGE_EXTS:
            img = ROOT / (stem + ext)
            if img.exists():
                print(f"  Deleting image: {img}", flush=True)
                try:
                    img.unlink()
                    img_deleted += 1
                except Exception as e:
                    print(f"    failed to delete image: {e}", flush=True)

print(f"Done. Deleted {mp3_deleted} mp3 files and {img_deleted} images.")
