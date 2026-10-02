# import os

# import moviepy.editor as mp
# import pandas as pd


# def download(output_dir, name, t_seg):
#     path_data = os.path.join(output_dir, "video")
#     if not os.path.exists(path_data):
#         os.makedirs(path_data)
#     link_prefix = "https://www.youtube.com/watch?v="

#     filename_full_video = os.path.join(path_data, name) + "_full_video.%(ext)s"
#     filename = os.path.join(path_data, name) + ".mp4"
#     link = link_prefix + name

#     if os.path.exists(filename) and os.path.exists(os.path.join(output_dir, 'image', name) + '.jpg'):
#         print("already exists, skip")
#         return

#     t_start, t_end = t_seg
#     print("download the whole video for: [%s] - [%s]" % (output_dir, name))
#     os.system(f'yt-dlp --ignore-config {link} -o {filename_full_video} ')

#     t_dur = t_end - t_start
#     filename_full_video = filename_full_video[:-8]
#     if os.path.exists(filename_full_video + '.mkv'):
#         filename_full_video += '.mkv'
#     elif os.path.exists(filename_full_video + '.webm'):
#         filename_full_video += '.webm'
#     else:
#         filename_full_video += '.mp4'
#     print(filename_full_video)
#     print("trim the video to [%.1f-%.1f]" % (t_start, t_end))
#     os.system(f'ffmpeg -i {filename_full_video} -ss {t_start} -t {t_dur}' +
#               f' -vcodec libx264 -acodec aac -strict -2 -y {filename}' +
#               f' -loglevel -8')
#     if os.path.exists(filename_full_video): os.remove(filename_full_video)

#     if os.path.exists(filename):
#         try:
#             # cap = cv2.VideoCapture(filename)
#             # frame_n = random.randint(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)
#             # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_n)
#             # _, frame = cap.read()
#             # cap.release()
#             # cv2.imwrite(os.path.join(output_dir, 'image', name) + '.jpg', frame)
#             mp.VideoFileClip(filename).audio.write_audiofile(os.path.join(output_dir, 'audio', name) + '.wav', fps=16000)
#         except:
#             print('error happened')
#     print("finish the video as: " + filename)


# filename_source = "./AVVP_dataset_full.csv"  #
# output_dir = "/path/to/download"
# df = pd.read_csv(filename_source, header=0, sep='\t')
# filenames = df["filename"]
# length = len(filenames)
# names = []
# segments = {}
# for i in range(length):
#     row = df.loc[i, :]
#     name = row.iloc[0][:11]
#     steps = row.iloc[0][11:].split("_")
#     labels = row.iloc[1]
#     if len(labels.split(',')) == 1: continue
#     t_start = float(steps[1])
#     t_end = t_start + 10
#     segments[name] = (t_start, t_end)
#     download(output_dir, name, segments[name])
#     names.append(name)
# print(len(segments))





import os
import subprocess
import signal
import sys
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


def sanitize_label(label):
    parts = [l.strip() for l in label.split(",")]
    label = "-".join(parts)
    return label.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")


def download(set, name, t_seg, label):
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

        safe_label = sanitize_label(label)

        img_path = os.path.join(path_image, f"{idx}_{safe_label}.jpg")
        cv2.imwrite(img_path, frame)

        # 🔹 Extract looped 10 sec audio
        audio_path = os.path.join(path_audio, f"{idx}_{safe_label}.wav")

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

        # ✅ increment only on success
        idx += 1

    except Exception as e:
        print("Processing failed:", e)
        return


# CSV
filename_source = "./AVVP_dataset_full.csv"
set = "../../../dataset/LLP"

df = pd.read_csv(filename_source, header=0, sep='\t')

for i in range(len(df)):
    row = df.loc[i, :]
    name = row.iloc[0][:11]
    steps = row.iloc[0][11:].split("_")
    labels = row.iloc[1]

    if len(labels.split(',')) == 1:
        continue

    t_start = float(steps[1])
    t_end = t_start + 10

    download(set, name, (t_start, t_end), labels)

print("Done")