# import pandas as pd
# import os
# import subprocess
# import json
# import cv2
# import time

# # Load train CSV file
# train_csv = "/home/singh_prajwal/anurag/train_final.csv"
# train_df = pd.read_csv(train_csv, header=None, names=["File_Name", "Label"])

# # Use train only
# dataset_df = train_df
# print("Size of dataset:", len(dataset_df))

# # Directories
# video_dir = "/home/singh_prajwal/dataset/VGG_Sound_videos/"
# os.makedirs(video_dir, exist_ok=True)
# output_dir = "/home/singh_prajwal/dataset/VGG_Sound_extracted/train_data"
# os.makedirs(output_dir, exist_ok=True)

# # Parameters
# duration = 10  # seconds
# fps = 10      # assumed
# counter = 0

# # Load annotations
# file_path = "/home/singh_prajwal/anurag/Cycle_Sound/top1_boxes_top10_moments.json"
# with open(file_path, "r") as f:
#     annotations = json.load(f)[1]
# print("Size of top10:", len(annotations))

# # Helpers
# def parse_filename(filename):
#     parts = filename.split("_")
#     youtube_id = parts[0]
#     for p in parts[1:-1]:
#         youtube_id += "_" + p
#     start_time = parts[-1].split(".")[0]
#     return youtube_id, start_time

# def download_video(youtube_id, output_path):
#     url = f"https://www.youtube.com/watch?v={youtube_id}"
#     if os.path.exists(output_path):
#         print(f"Skipping download, exists: {output_path}")
#         return output_path
#     cmd = ["yt-dlp", "-f", "mp4", "-o", output_path, url]
#     try:
#         subprocess.run(cmd, check=True)
#         print(f"Downloaded: {output_path}")
#         return output_path
#     except subprocess.CalledProcessError:
#         print(f"Download failed: {url}")
#         return None

# # Trim with ffmpeg
# def trim_video_ffmpeg(input_path, output_path, start, dur=10):
#     cmd = [
#         "ffmpeg", "-y",
#         "-ss", str(start),
#         "-i", input_path,
#         "-t", str(dur),
#         "-c:v", "copy",
#         "-c:a", "copy",
#         output_path
#     ]
#     try:
#         subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#         print(f"Trimmed video: {output_path}")
#         return True
#     except subprocess.CalledProcessError as e:
#         print(f"Trim failed: {input_path} -> {output_path}")
#         return False

# # Extract frames + audio via ffmpeg

# def extract_data(video_path, frame_numbers, fps):
#     global counter
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"Cannot open trimmed video: {video_path}")
#         return

#     for frame_num in frame_numbers:
#         # Frame extraction
#         cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
#         ret, frame = cap.read()
#         if not ret:
#             print(f"Frame {frame_num} fail in {video_path}")
#             continue
#         frame_file = os.path.join(output_dir, f"{counter}.jpg")
#         cv2.imwrite(frame_file, frame)
#         print(f"Saved frame: {frame_file}")

#         # Audio extraction + looping
#         start_time = frame_num / fps
#         audio_file = os.path.join(output_dir, f"{counter}.mp3")
#         cmd = [
#             "ffmpeg", "-y",
#             "-ss", str(start_time),
#             "-i", video_path,
#             "-t", str(duration),
#             "-filter_complex",
#             "[0:a]aloop=loop=-1:size=1e7,asetpts=PTS-STARTPTS,atrim=0:10[out]",
#             "-map", "[out]",
#             "-c:a", "libmp3lame",
#             audio_file
#         ]
#         try:
#             subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#             print(f"Saved audio: {audio_file}")
#         except subprocess.CalledProcessError:
#             print(f"Audio extract fail for frame {frame_num}")
#             os.remove(frame_file)
#             print(f"Deleted frame due to audio error: {frame_file}")
#             continue

#         counter += 1
#     cap.release()

# # Main loop
# for _, row in dataset_df.iloc[0:10].iterrows():
#     file_name = row["File_Name"]
#     youtube_id, start_str = parse_filename(file_name)
#     try:
#         start = float(start_str)
#     except:
#         print(f"Bad start time: {file_name}")
#         continue

#     start_ms = int(start * 1000)
#     end_ms = int((start + duration) * 1000)
#     annotated_id = f"{youtube_id}_{start_ms}_{end_ms}"

#     full_path = os.path.join(video_dir, f"{youtube_id}.mp4")
#     downloaded = download_video(youtube_id, full_path)
#     if not downloaded:
#         continue
#     trimmed = full_path.replace('.mp4', '_trimmed.mp4')
#     if trim_video_ffmpeg(downloaded, trimmed, start, duration):
#         annotation = annotations.get(annotated_id)
#         if annotation:
#             k = 2
#             print(annotation[:k])
#             extract_data(trimmed, annotation[:k], fps)
#         time.sleep(1)
#         os.remove(trimmed)
#         print(f"Removed trimmed: {trimmed}")
#     time.sleep(1)
#     os.remove(downloaded)
#     print(f"Removed original: {downloaded}")

# print("Processing complete!")













# import pandas as pd
# import os
# import subprocess
# import json
# import cv2
# import time

# # Load train CSV file
# file_type = "test"
# train_csv = f"/home/singh_prajwal/anurag/{file_type}_final.csv"
# train_df = pd.read_csv(train_csv, header=None, names=["File_Name", "Label"])

# # Use train only
# dataset_df = train_df
# print("Size of dataset:", len(dataset_df))

# # Directories
# video_dir = "/home/singh_prajwal/dataset/VGG_Sound_videos/"
# os.makedirs(video_dir, exist_ok=True)
# output_dir = f"/home/singh_prajwal/dataset/VGG_Sound_extracted/{file_type}_data"
# os.makedirs(output_dir, exist_ok=True)

# # Parameters
# duration = 10  # seconds
# fps = 10      # assumed
# counter = 0

# # Load annotations
# file_path = "/home/singh_prajwal/anurag/Cycle_Sound/top1_boxes_top10_moments.json"
# with open(file_path, "r") as f:
#     annotations = json.load(f)[1]
# print("Size of top10:", len(annotations))

# # Helpers
# def parse_filename(filename):
#     parts = filename.split("_")
#     youtube_id = parts[0]
#     for p in parts[1:-1]:
#         youtube_id += "_" + p
#     start_time = parts[-1].split(".")[0]
#     return youtube_id, start_time

# def download_video(youtube_id, output_path):
#     url = f"https://www.youtube.com/watch?v={youtube_id}"
#     if os.path.exists(output_path):
#         print(f"Skipping download, exists: {output_path}")
#         return output_path
#     cmd = ["yt-dlp", "-f", "mp4", "-o", output_path, url]
#     try:
#         subprocess.run(cmd, check=True)
#         print(f"Downloaded: {output_path}")
#         return output_path
#     except subprocess.CalledProcessError:
#         print(f"Download failed: {url}")
#         return None

# # Trim with ffmpeg
# def trim_video_ffmpeg(input_path, output_path, start, dur=10):
#     cmd = [
#         "ffmpeg", "-y",
#         "-ss", str(start),
#         "-i", input_path,
#         "-t", str(dur),
#         "-c:v", "copy",
#         "-c:a", "copy",
#         output_path
#     ]
#     try:
#         subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#         print(f"Trimmed video: {output_path}")
#         return True
#     except subprocess.CalledProcessError as e:
#         print(f"Trim failed: {input_path} -> {output_path}")
#         return False

# # Extract frames + audio via ffmpeg

# def extract_data(video_path, frame_numbers, fps):
#     global counter
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"Cannot open trimmed video: {video_path}")
#         return

#     for frame_num in frame_numbers:
#         # Frame extraction
#         cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
#         ret, frame = cap.read()
#         if not ret:
#             print(f"Frame {frame_num} fail in {video_path}")
#             continue
#         frame_file = os.path.join(output_dir, f"{counter}.jpg")
#         cv2.imwrite(frame_file, frame)
#         print(f"Saved frame: {frame_file}")

#         # Audio extraction + looping
#         start_time = frame_num / fps
#         audio_file = os.path.join(output_dir, f"{counter}.mp3")
#         cmd = [
#             "ffmpeg", "-y",
#             "-ss", str(start_time),
#             "-i", video_path,
#             "-t", str(duration),
#             "-filter_complex",
#             "[0:a]aloop=loop=-1:size=1e7,asetpts=PTS-STARTPTS,atrim=0:10[out]",
#             "-map", "[out]",
#             "-c:a", "libmp3lame",
#             audio_file
#         ]
#         try:
#             subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#             print(f"Saved audio: {audio_file}")
#         except subprocess.CalledProcessError:
#             print(f"Audio extract fail for frame {frame_num}")
#             os.remove(frame_file)
#             print(f"Deleted frame due to audio error: {frame_file}")
#             continue

#         counter += 1
#     cap.release()




# def extract_test_data(video_path, label, frame_numbers, fps):
#     global counter
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"Cannot open trimmed video: {video_path}")
#         return

#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
#     if frame_count <= 0:
#         cap.release()
#         try:
#             os.remove(video_path)
#         except Exception:
#             pass
#         return

#     mid_frame = frame_count // 2
#     frame_numbers = [mid_frame]
#     for frame_num in frame_numbers:
#         # Frame extraction
#         cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
#         ret, frame = cap.read()
#         if not ret:
#             print(f"Frame {frame_num} fail in {video_path}")
#             continue
#         frame_file = os.path.join(output_dir, f"{counter}_{label}.jpg")
#         cv2.imwrite(frame_file, frame)
#         print(f"Saved frame: {frame_file}")

#         # Audio extraction + looping
#         start_time = frame_num / fps
#         audio_file = os.path.join(output_dir, f"{counter}_{label}.mp3")
#         cmd = [
#             "ffmpeg", "-y",
#             "-ss", str(start_time),
#             "-i", video_path,
#             "-t", str(duration),
#             "-filter_complex",
#             "[0:a]aloop=loop=-1:size=1e7,asetpts=PTS-STARTPTS,atrim=0:10[out]",
#             "-map", "[out]",
#             "-c:a", "libmp3lame",
#             audio_file
#         ]
#         try:
#             subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#             print(f"Saved audio: {audio_file}")
#         except subprocess.CalledProcessError:
#             print(f"Audio extract fail for frame {frame_num}")
#             os.remove(frame_file)
#             print(f"Deleted frame due to audio error: {frame_file}")
#             continue

#         counter += 1
#     cap.release()




# # Main loop
# for _, row in dataset_df.iterrows():
#     file_name = row["File_Name"]
#     label = row["Label"]
#     youtube_id, start_str = parse_filename(file_name)
#     try:
#         start = float(start_str)
#     except:
#         print(f"Bad start time: {file_name}")
#         continue

#     start_ms = int(start * 1000)
#     end_ms = int((start + duration) * 1000)
#     annotated_id = f"{youtube_id}_{start_ms}_{end_ms}"

#     full_path = os.path.join(video_dir, f"{youtube_id}.mp4")
#     downloaded = download_video(youtube_id, full_path)
#     if not downloaded:
#         continue
#     trimmed = full_path.replace('.mp4', '_trimmed.mp4')
#     if trim_video_ffmpeg(downloaded, trimmed, start, duration):
#         if (file_type == "test"):
#             extract_test_data(trimmed, label, [], fps)
#         else:
#             annotation = annotations.get(annotated_id)
#             if annotation:
#                 k = 2
#                 print(annotation[:k])
#                 extract_data(trimmed, annotation[:k], fps)
#         time.sleep(1)
#         os.remove(trimmed)
#         print(f"Removed trimmed: {trimmed}")
#     time.sleep(1)
#     os.remove(downloaded)
#     print(f"Removed original: {downloaded}")

# print("Processing complete!")









# import pandas as pd
# import os
# import subprocess
# import json
# import cv2
# import time
# from concurrent.futures import ThreadPoolExecutor, as_completed

# # ------------ Config ------------
# file_type = "train"
# NUM_WORKERS = 6      # Number of threads
# DURATION = 10        # seconds
# FPS = 10             # assumed
# VIDEO_DIR = "/home/singh_prajwal/dataset/VGG_Sound_videos/"
# OUTPUT_DIR = "/home/singh_prajwal/dataset/VGG_Sound_extracted/train_data"
# TRAIN_CSV = "/home/singh_prajwal/anurag/train_final.csv"
# ANNOTATION_JSON = "/home/singh_prajwal/anurag/Cycle_Sound/top1_boxes_top10_moments.json"

# # create dirs
# os.makedirs(VIDEO_DIR, exist_ok=True)
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # Load data
# train_df = pd.read_csv(TRAIN_CSV, header=None, names=["File_Name", "Label"])
# with open(ANNOTATION_JSON, "r") as f:
#     annotations = json.load(f)[1]

# counter = 0
# def parse_filename(filename):
#     parts = filename.split("_")
#     youtube_id = parts[0]
#     for p in parts[1:-1]:
#         youtube_id += "_" + p
#     start_time = parts[-1].split(".")[0]
#     return youtube_id, start_time

# # Download video
# def download_video(youtube_id):
#     output_path = os.path.join(VIDEO_DIR, f"{youtube_id}.mp4")
#     if os.path.exists(output_path):
#         return output_path
#     cmd = ["yt-dlp", "-f", "mp4", "-o", output_path,
#            f"https://www.youtube.com/watch?v={youtube_id}"]
#     try:
#         subprocess.run(cmd, check=True)
#         return output_path
#     except subprocess.CalledProcessError:
#         return None

# # Trim video
# def trim_video(input_path, start):
#     output_path = input_path.replace('.mp4', '_trimmed.mp4')
#     cmd = ["ffmpeg", "-y", "-ss", str(start), "-i", input_path,
#            "-t", str(DURATION), "-c:v", "copy", "-c:a", "copy", output_path]
#     try:
#         subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#         return output_path
#     except subprocess.CalledProcessError:
#         return None

# # Extract frames and audio
# def extract_data_on_annotation(youtube_id, label, start, frame_numbers, lock):
#     global counter
#     input_video = download_video(youtube_id)
#     if not input_video:
#         return
#     trimmed = trim_video(input_video, start)
#     if not trimmed:
#         return

#     cap = cv2.VideoCapture(trimmed)
#     if not cap.isOpened():
#         return

#     for frame_num in frame_numbers:
#         cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
#         ret, frame = cap.read()
#         if not ret:
#             continue

#         # atomically assign counter
#         with lock:
#             idx = counter
#             counter += 1

#         frame_file = os.path.join(OUTPUT_DIR, f"{idx}_{label}.jpg")
#         cv2.imwrite(frame_file, frame)

#         audio_file = os.path.join(OUTPUT_DIR, f"{idx}_{label}.mp3")
#         start_time = frame_num / FPS
#         cmd = ["ffmpeg", "-y", "-ss", str(start_time), "-i", trimmed,
#                "-t", str(DURATION),
#                "-filter_complex", "[0:a]aloop=loop=-1:size=1e7,asetpts=PTS-STARTPTS,atrim=0:10[out]",
#                "-map", "[out]", "-c:a", "libmp3lame", audio_file]
#         subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

#     cap.release()
#     # os.remove(trimmed)
#     # os.remove(input_video)

# def extract_test_data(youtube_id, label, start, frame_numbers, lock):
#     # Based on middle frame extraction
#     global counter
#     input_video = download_video(youtube_id)
#     if not input_video:
#         return
#     trimmed = trim_video(input_video, start)
#     if not trimmed:
#         return

#     cap = cv2.VideoCapture(trimmed)
#     if not cap.isOpened():
#         return

#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
#     if frame_count <= 0:
#         cap.release()
#         try:
#             os.remove(trimmed)
#         except Exception:
#             pass
#         try:
#             os.remove(input_video)
#         except Exception:
#             pass
#         return

#     mid_frame = frame_count // 2
#     frame_numbers = [mid_frame]
#     print("\n\n\n------------\n", frame_count, mid_frame, "\n\n\n")
#     for frame_num in frame_numbers:
#         cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
#         ret, frame = cap.read()
#         if not ret:
#             continue

#         # atomically assign counter
#         with lock:
#             idx = counter
#             counter += 1

#         frame_file = os.path.join(OUTPUT_DIR, f"{idx}_{label}.jpg")
#         cv2.imwrite(frame_file, frame)

#         audio_file = os.path.join(OUTPUT_DIR, f"{idx}_{label}.mp3")
#         start_time = frame_num / FPS
#         cmd = ["ffmpeg", "-y", "-ss", str(start_time), "-i", trimmed,
#                "-t", str(DURATION),
#                "-filter_complex", "[0:a]aloop=loop=-1:size=1e7,asetpts=PTS-STARTPTS,atrim=0:10[out]",
#                "-map", "[out]", "-c:a", "libmp3lame", audio_file]
#         subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

#     cap.release()
#     # os.remove(trimmed)
#     # os.remove(input_video)



# # Worker wrapper
# from threading import Lock
# lock = Lock()

# def process_row(row):
#     file_name = row['File_Name']
#     label = row['Label']
#     youtube_id, start_str = parse_filename(file_name)
#     try:
#         start = float(start_str)
#     except ValueError:
#         return
#     start_ms = int(start * 1000)
#     end_ms = start_ms + DURATION * 1000
#     ann_id = f"{youtube_id}_{start_ms}_{end_ms}"
#     frame_nums = annotations.get(ann_id, [])[:1]
#     if file_type == "test":
#         extract_test_data(youtube_id, label, start, [], lock)
#     elif frame_nums:
#         extract_data_on_annotation(youtube_id, label, start, frame_nums, lock)
        

# # Run with ThreadPoolExecutor
# with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
#     futures = [executor.submit(process_row, row) for _, row in train_df.iterrows()]
#     for _ in as_completed(futures):
#         pass

# print("Processing complete!")




























import pandas as pd
import os
import subprocess
import json
import cv2
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ------------ Config ------------
file_type = "train"
NUM_WORKERS = 6      # Number of threads
DURATION = 10        # seconds
FPS = 10             # assumed
VIDEO_DIR = "/home/singh_prajwal/dataset/VGG_Sound_videos/"
# OUTPUT_DIR = "/home/singh_prajwal/dataset/VGG_Sound_extracted/temp_data"
OUTPUT_DIR = f"/home/singh_prajwal/dataset/VGG_Sound_extracted/{file_type}_data"
TEXT_DIR = f"/home/singh_prajwal/anurag/log_files/{file_type}_data"
TRAIN_CSV = f"/home/singh_prajwal/anurag/{file_type}_final.csv"
ANNOTATION_JSON = "/home/singh_prajwal/anurag/Cycle_Sound/top1_boxes_top10_moments.json"

# create dirs
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEXT_DIR, exist_ok=True)

# Load data
train_df = pd.read_csv(TRAIN_CSV, header=None, names=["File_Name", "Label"])
with open(ANNOTATION_JSON, "r") as f:
    annotations = json.load(f)[1]

counter = 0
def parse_filename(filename):
    parts = filename.split("_")
    youtube_id = parts[0]
    for p in parts[1:-1]:
        youtube_id += "_" + p
    start_time = parts[-1].split(".")[0]
    return youtube_id, start_time

# Download video
def download_video(youtube_id):
    output_path = os.path.join(VIDEO_DIR, f"{youtube_id}.mp4")
    if os.path.exists(output_path):
        return output_path
    cmd = ["yt-dlp", "-f", "mp4", "-o", output_path,
           f"https://www.youtube.com/watch?v={youtube_id}"]
    try:
        subprocess.run(cmd, check=True)
        return output_path
    except subprocess.CalledProcessError:
        return None

# Trim video
def trim_video(input_path, start):
    output_path = input_path.replace('.mp4', '_trimmed.mp4')
    cmd = ["ffmpeg", "-y", "-ss", str(start), "-i", input_path,
           "-t", str(DURATION), "-c:v", "copy", "-c:a", "copy", output_path]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return output_path
    except subprocess.CalledProcessError:
        return None

# Extract frames and audio
def extract_data_on_annotation(youtube_id, label, start, frame_numbers, lock):
    global counter
    input_video = download_video(youtube_id)
    if not input_video:
        return
    trimmed = trim_video(input_video, start)
    if not trimmed:
        return

    cap = cv2.VideoCapture(trimmed)
    if not cap.isOpened():
        return

    for frame_num in frame_numbers:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            continue

        # atomically assign counter
        with lock:
            idx = counter
            counter += 1

        frame_file = os.path.join(OUTPUT_DIR, f"{idx}_{label}.jpg")
        cv2.imwrite(frame_file, frame)

        audio_file = os.path.join(OUTPUT_DIR, f"{idx}_{label}.mp3")
        start_time = frame_num / FPS
        cmd = ["ffmpeg", "-y", "-ss", str(start_time), "-i", trimmed,
               "-t", str(DURATION),
               "-filter_complex", "[0:a]aloop=loop=-1:size=1e7,asetpts=PTS-STARTPTS,atrim=0:10[out]",
               "-map", "[out]", "-c:a", "libmp3lame", audio_file]
        subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    cap.release()
    # os.remove(trimmed)
    # os.remove(input_video)



def extract_test_data(youtube_id, label, start, frame_numbers, lock):
    # Based on middle frame extraction
    global counter
    input_video = download_video(youtube_id)
    if not input_video:
        return
    trimmed = trim_video(input_video, start)
    if not trimmed:
        return

    cap = cv2.VideoCapture(trimmed)
    if not cap.isOpened():
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        cap.release()
        try:
            os.remove(trimmed)
        except Exception:
            pass
        try:
            os.remove(input_video)
        except Exception:
            pass
        return

    # get actual fps from the trimmed video; fall back to global FPS if invalid
    try:
        video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    except Exception:
        video_fps = 0.0
    if not video_fps or video_fps <= 0:
        try:
            video_fps = float(FPS)
        except Exception:
            video_fps = 1.0  # last resort to avoid division by zero

    mid_frame = frame_count // 2
    frame_numbers = [mid_frame]

    # original print kept
    print("\n\n\n------------\n", frame_count, mid_frame, "\n\n\n")

    # # write the same info into a text file in OUTPUT_DIR
    # try:
    #     log_file = os.path.join(TEXT_DIR, f"{youtube_id}_{label}_frames.log")
    #     with open(log_file, "a") as lf:
    #         lf.write(f"VIDEO: {trimmed}\n")
    #         lf.write(f"frame_count: {frame_count}, mid_frame: {mid_frame}, fps: {video_fps}\n")
    # except Exception:
    #     # don't crash on logging errors
    #     pass

    for frame_num in frame_numbers:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            # # log skip to file
            # try:
            #     with open(log_file, "a") as lf:
            #         lf.write(f"frame {frame_num}: read failed\n")
            # except Exception:
            #     pass
            continue

        # atomically assign counter
        with lock:
            idx = counter
            counter += 1

        frame_file = os.path.join(OUTPUT_DIR, f"{idx}_{label}.jpg")
        cv2.imwrite(frame_file, frame)

        audio_file = os.path.join(OUTPUT_DIR, f"{idx}_{label}.mp3")
        # use actual video_fps to compute start time
        start_time = frame_num / video_fps
        # # also log the computed start_time
        # try:
        #     with open(log_file, "a") as lf:
        #         lf.write(f"extracting frame {frame_num}, start_time={start_time:.3f}s -> files: {frame_file}, {audio_file}\n")
        # except Exception:
        #     pass

        cmd = ["ffmpeg", "-y", "-ss", str(start_time), "-i", trimmed,
               "-t", str(DURATION),
               "-filter_complex", "[0:a]aloop=loop=-1:size=1e7,asetpts=PTS-STARTPTS,atrim=0:10[out]",
               "-map", "[out]", "-c:a", "libmp3lame", audio_file]
        subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    cap.release()
    # os.remove(trimmed)
    # os.remove(input_video)



# Worker wrapper
from threading import Lock
lock = Lock()

def process_row(row):
    file_name = row['File_Name']
    label = row['Label']
    youtube_id, start_str = parse_filename(file_name)
    print(f"Processing: {youtube_id}, label: {label}, start: {start_str}")
    try:
        start = float(start_str)
    except ValueError:
        return
    start_ms = int(start * 1000)
    end_ms = start_ms + DURATION * 1000
    ann_id = f"{youtube_id}_{start_ms}_{end_ms}"
    frame_nums = annotations.get(ann_id, [])[:1]
    if file_type == "test":
        extract_test_data(youtube_id, label, start, [], lock)
    elif frame_nums:
        extract_data_on_annotation(youtube_id, label, start, frame_nums, lock)
        

# Run with ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    futures = [executor.submit(process_row, row) for _, row in train_df.iterrows()]
    for _ in as_completed(futures):
        pass

print("Processing complete!")

