import os
import shutil
import numpy as np
from collections import defaultdict
from scipy.io import wavfile
from pydub import AudioSegment
from tqdm import tqdm
import librosa

# -----------------------------
# 1. LOAD AUDIO (MP3 + WAV)
# -----------------------------

TARGET_SR = 44100  # choose one (very important)

def load_audio(path):
    ext = path.lower().split(".")[-1]

    if ext in ["wav", "mp3"]:
        # librosa handles BOTH mp3 + wav + resampling
        data, sr = librosa.load(path, sr=TARGET_SR, mono=False)

        # shape handling
        if data.ndim == 1:
            return TARGET_SR, data
        else:
            return TARGET_SR, data.T  # librosa gives (channels, samples)

    else:
        raise ValueError(f"Unsupported file format: {path}")


# -----------------------------
# 2. CHANNEL ALIGNMENT
# -----------------------------
def make_same_channels(arrays):
    max_channels = max([1 if a.ndim == 1 else a.shape[1] for a in arrays])

    out = []
    for a in arrays:
        if a.ndim == 1 and max_channels == 2:
            a = np.stack([a, a], axis=1)
        out.append(a)

    return out


# -----------------------------
# 3. MIX MULTIPLE AUDIOS
# -----------------------------
def mix_audios(paths, out_path, mode="longest", normalize_by_n=True):
    sample_rates = []
    audios = []

    for p in paths:
        sr, data = load_audio(p)
        sample_rates.append(sr)
        audios.append(data)

    if len(set(sample_rates)) != 1:
        raise ValueError(f"Sample rates mismatch: {sample_rates}")

    sr = sample_rates[0]

    audios = make_same_channels(audios)

    lengths = [a.shape[0] for a in audios]

    if mode == "longest":
        L = max(lengths)

        def pad(x):
            if x.shape[0] < L:
                if x.ndim == 1:
                    return np.concatenate([x, np.zeros(L - x.shape[0])])
                else:
                    return np.vstack([x, np.zeros((L - x.shape[0], x.shape[1]))])
            return x

        audios = [pad(a) for a in audios]

    elif mode == "shortest":
        L = min(lengths)
        audios = [a[:L] for a in audios]

    mix = np.sum(audios, axis=0)

    if normalize_by_n:
        mix = mix / len(audios)

    max_abs = np.max(np.abs(mix))
    if max_abs > 1.0:
        mix = mix / max_abs

    mix_int16 = (mix * 32767.0).astype(np.int16)

    wavfile.write(out_path, sr, mix_int16)

    # print(f"🎧 Saved audio: {out_path}")


# -----------------------------
# 4. PROCESS DATASET
# -----------------------------
def process_dataset(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    audio_groups = defaultdict(list)

    # Step 1: group audio files
    for fname in os.listdir(input_dir):
        if fname.endswith((".wav", ".mp3")):
            idx = fname.split("_")[0]
            audio_groups[idx].append(fname)

    print(f"Total audio groups: {len(audio_groups)}")

    # Step 2: mix audio with tqdm 🔥
    for idx, files in tqdm(audio_groups.items(), desc="🎧 Mixing audio", total=len(audio_groups)):
        paths = [os.path.join(input_dir, f) for f in files]

        labels = [f.split("_")[1].split(".")[0] for f in files]

        out_name = f"{idx}_{'_'.join(labels)}.wav"
        out_path = os.path.join(output_dir, out_name)

        try:
            mix_audios(paths, out_path)
        except Exception as e:
            print(f"\n❌ Audio error {idx}: {e}")

    # Step 3: copy images with tqdm 🔥
    image_files = [f for f in os.listdir(input_dir) if f.endswith((".jpg", ".png"))]

    for fname in tqdm(image_files, desc="🖼️ Copying images"):
        src = os.path.join(input_dir, fname)
        dst = os.path.join(output_dir, fname)

        try:
            shutil.copy2(src, dst)
        except Exception as e:
            print(f"\n❌ Image copy error {fname}: {e}")


# -----------------------------
# 5. RUN
# -----------------------------
if __name__ == "__main__":
    input_dir = "../dataset/VGG_Sound_extracted/test_data_paired"
    output_dir = "../dataset/Mixed/VGG2"

    process_dataset(input_dir, output_dir)