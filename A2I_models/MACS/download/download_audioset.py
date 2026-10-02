# import os
# import subprocess
# import signal
# import sys

# import moviepy.editor as mp
# import pandas as pd


# # 🔴 Handle Ctrl+C properly
# def signal_handler(sig, frame):
#     print("\nStopping all processes...")
#     sys.exit(0)

# signal.signal(signal.SIGINT, signal_handler)


# def run_command(cmd):
#     try:
#         subprocess.run(cmd, check=True)
#     except subprocess.CalledProcessError:
#         print("Command failed:", " ".join(cmd))
#         return False
#     return True


# def download(set, name, t_seg):
#     path_data = os.path.join(set, "video")
#     os.makedirs(path_data, exist_ok=True)

#     path_audio = os.path.join(set, "audio")
#     os.makedirs(path_audio, exist_ok=True)

#     link = f"https://www.youtube.com/watch?v={name}"

#     filename_template = os.path.join(path_data, name) + "_full_video.%(ext)s"
#     filename = os.path.join(path_data, name) + ".mp4"

#     if os.path.exists(filename):
#         print(f"{filename} already exists, skip")
#         return

#     t_start, t_end = t_seg
#     t_dur = t_end - t_start

#     print(f"Downloading: {name}")

#     # ✅ yt-dlp
#     success = run_command([
#         "yt-dlp",
#         "--ignore-config",
#         link,
#         "-o",
#         filename_template
#     ])
#     if not success:
#         return

#     base = filename_template[:-8]

#     if os.path.exists(base + ".mkv"):
#         full_video = base + ".mkv"
#     elif os.path.exists(base + ".webm"):
#         full_video = base + ".webm"
#     else:
#         full_video = base + ".mp4"

#     if not os.path.exists(full_video):
#         print("Download failed:", name)
#         return

#     print(f"Trimming [{t_start}-{t_end}]")

#     # ✅ ffmpeg
#     success = run_command([
#         "ffmpeg",
#         "-i", full_video,
#         "-ss", str(t_start),
#         "-t", str(t_dur),
#         "-vcodec", "libx264",
#         "-acodec", "aac",
#         "-y",
#         filename,
#         "-loglevel", "error"
#     ])
#     if not success:
#         return

#     os.remove(full_video)

#     if os.path.exists(filename):
#         try:
#             mp.VideoFileClip(filename).audio.write_audiofile(
#                 os.path.join(path_audio, name) + ".wav",
#                 fps=16000
#             )
#         except Exception as e:
#             print("Audio extraction failed:", e)

#     print("Finished:", name)


# # CSV
# segment = pd.read_csv(
#     "./eval_segments.csv",
#     skiprows=2,
#     quotechar='"',
#     skipinitialspace=True
# )

# output_dir = "../../../dataset/AudioSet"

# for _, it in segment.iterrows():
#     download(output_dir, it['YTID'], (it['start_seconds'], it['end_seconds']))






import os
import subprocess
import signal
import sys

import moviepy.editor as mp
import pandas as pd
import cv2

# 🔴 Global index
idx = 0

# 🔴 Handle Ctrl+C properly
def signal_handler(sig, frame):
    print("\nStopping all processes...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


def run_command(cmd):
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print("Command failed:", " ".join(cmd))
        return False
    return True


def download(set, name, t_seg):
    global idx

    path_data = os.path.join(set, "video")
    os.makedirs(path_data, exist_ok=True)

    path_audio = os.path.join(set, "audio")
    os.makedirs(path_audio, exist_ok=True)

    path_image = os.path.join(set, "image")
    os.makedirs(path_image, exist_ok=True)

    link = f"https://www.youtube.com/watch?v={name}"

    filename_template = os.path.join(path_data, name) + "_full_video.%(ext)s"
    filename = os.path.join(path_data, name) + ".mp4"

    t_start, t_end = t_seg
    t_dur = t_end - t_start

    print(f"Downloading: {name}")

    # ✅ yt-dlp
    success = run_command([
        "yt-dlp",
        "--ignore-config",
        link,
        "-o",
        filename_template
    ])
    if not success:
        return

    base = filename_template[:-8]

    if os.path.exists(base + ".mkv"):
        full_video = base + ".mkv"
    elif os.path.exists(base + ".webm"):
        full_video = base + ".webm"
    else:
        full_video = base + ".mp4"

    if not os.path.exists(full_video):
        print("Download failed:", name)
        return

    print(f"Trimming [{t_start}-{t_end}]")

    # ✅ ffmpeg trim
    success = run_command([
        "ffmpeg",
        "-i", full_video,
        "-ss", str(t_start),
        "-t", str(t_dur),
        "-vcodec", "libx264",
        "-acodec", "aac",
        "-y",
        filename,
        "-loglevel", "error"
    ])
    if not success:
        return

    os.remove(full_video)

    # ✅ Process trimmed video
    if not os.path.exists(filename):
        return

    try:
        # 🔹 Extract mid-frame
        cap = cv2.VideoCapture(filename)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if frame_count == 0:
            cap.release()
            print("Empty video:", name)
            return

        mid_frame = frame_count // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)

        ret, frame = cap.read()
        cap.release()

        if not ret:
            print("Frame extraction failed:", name)
            return

        img_path = os.path.join(path_image, f"{idx}.jpg")
        cv2.imwrite(img_path, frame)

        # 🔹 Extract audio
        audio_path = os.path.join(path_audio, f"{idx}.wav")

        success = run_command([
            "ffmpeg",
            "-stream_loop", "-1",
            "-i", filename,
            "-t", "10",
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            "-y",
            audio_path,
            "-loglevel", "error"
        ])

        if not success:
            print("Audio extraction failed:", name)
            return

        # 🔹 Delete trimmed video
        os.remove(filename)

        print(f"Finished: {name} -> idx {idx}")

        # ✅ Increment ONLY on success
        idx += 1

    except Exception as e:
        print("Processing failed:", e)
        return


# CSV
segment = pd.read_csv(
    "./eval_segments.csv",
    skiprows=2,
    quotechar='"',
    skipinitialspace=True
)

output_dir = "../../../dataset/AudioSet"

for _, it in segment.iterrows():
    download(output_dir, it['YTID'], (it['start_seconds'], it['end_seconds']))