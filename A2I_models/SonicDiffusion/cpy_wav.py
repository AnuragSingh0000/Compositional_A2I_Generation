import os
import shutil
import subprocess

SRC_DIR = "../../dataset/VGG_Sound_extracted/test_data3"
DST_DIR = "../../dataset/VGG_Sound_extracted/test_data_wav3"

os.makedirs(DST_DIR, exist_ok=True)

def convert_mp3_to_wav(mp3_path, wav_path):
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",
        "-i", mp3_path,
        "-ar", "44100",
        "-ac", "1",
        wav_path
    ]
    subprocess.run(cmd, check=True)

for file in os.listdir(SRC_DIR):
    src_path = os.path.join(SRC_DIR, file)

    # Copy JPG as-is
    if file.lower().endswith(".jpg"):
        dst_path = os.path.join(DST_DIR, file)
        shutil.copy2(src_path, dst_path)

    # Convert MP3 -> WAV
    elif file.lower().endswith(".mp3"):
        base_name = os.path.splitext(file)[0]
        wav_name = base_name + ".wav"

        wav_path = os.path.join(DST_DIR, wav_name)

        # print(f"Converting: {file} -> {wav_name}")
        convert_mp3_to_wav(src_path, wav_path)

print("✅ Done. Files saved in test_wav/")
