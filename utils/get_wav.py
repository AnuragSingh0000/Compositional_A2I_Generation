import os
import shutil
import librosa
import soundfile as sf
from tqdm import tqdm

input_dir = "../dataset/VGG_Sound_extracted/test_data"
output_dir = "../dataset/VGG_Sound_extracted/test_data_wav"

os.makedirs(output_dir, exist_ok=True)

files = os.listdir(input_dir)

for file in tqdm(files, desc="Processing files"):
    input_path = os.path.join(input_dir, file)

    if not os.path.isfile(input_path):
        continue

    filename, ext = os.path.splitext(file)

    # MP3 → WAV
    if ext.lower() == ".mp3":
        try:
            y, sr = librosa.load(input_path, sr=None)
            output_path = os.path.join(output_dir, filename + ".wav")
            sf.write(output_path, y, sr)
        except Exception as e:
            print(f"Error processing {file}: {e}")

    # Copy JPG
    elif ext.lower() == ".jpg":
        shutil.copy2(input_path, os.path.join(output_dir, file))

print("Done!")