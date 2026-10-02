import torchaudio
from torch.utils.data import Dataset
import cv2
import PIL
import random
import numpy as np
from packaging import version
from PIL import Image
import os
import torch
import sys
import pandas as pd
import pickle
import json
import subprocess
import itertools

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }
# ------------------------------------------------------------------------------
imagenet_templates_small = [
    "a photo of {}"
]


class VGGSound(Dataset):
    def __init__(
            self,
            args,
            tokenizer,
            logger,
            size=512,
            interpolation='bicubic',
    ):
        video_lst = "video/"
        audio_lst = "audio/"

        self.video = args.data_dir + video_lst
        self.audio = args.data_dir + audio_lst
        # self.video = args.data_dir + "test/" + video_lst
        # self.audio = args.data_dir + "test/" + audio_lst
        
        self.vggsound = "AudioToken/data/VGGSound/vggsound.csv"
        self.video_path = list()
        self.audio_path = list()
        self.tokenizer = tokenizer
        self.size = size
        self.obj1 = args.obj1
        self.data_set = args.data_set
        self.df = pd.read_csv(self.vggsound)
        self.input_length = args.input_length
        if self.data_set == 'train':
            self.center_crop = args.center_crop
            self.filter_frames = args.filter_frames
            self.filter_unmatch_videos = args.filter_unmatch_videos
            self.filter_low_quality_imgs = args.filter_low_quality_imgs
        else:
            self.center_crop = False
            self.filter_frames = False
            self.filter_unmatch_videos = False
            self.filter_low_quality_imgs = False

        with open('AudioToken/constants/best_frames.json', 'r') as file:
            self.frames = json.load(file)

        videos = set([file_path[:-4] for file_path in os.listdir(self.video)])
        audios = set([file_path[:-4] for file_path in os.listdir(self.audio)])
        print(self.video, self.audio)
        print(videos, audios)
        samples = videos & audios
        print(samples)

        if self.data_set == 'train' and self.filter_unmatch_videos:
            with open("AudioToken/constants/unmatch_videos.pkl", "rb") as file:
                filtered_out = pickle.load(file)
                self.df = self.df[~self.df['ytid'].isin(filtered_out)]

        if self.data_set == 'train' and self.filter_low_quality_imgs:
            with open("AudioToken/constants/low_quality_videos.pkl", "rb") as file:
                filtered_out = pickle.load(file)
                self.df = self.df[~self.df['ytid'].isin(filtered_out)]

        self.label = list()
        self.prepare_dataset(samples)

        self.num_samples = len(self.video_path)

        self._length = self.num_samples

        logger.info(f"{args.data_set}, num samples: {self.num_samples}")

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]
        self.templates = imagenet_templates_small
        # self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length

    def prepare_dataset(self, samples):
        df = self.df[self.df["set"] == "train"]
        for vid in list(samples):
            video = vid[:11]
            df_vid = df[df.ytid == video]
            if df_vid.empty:
                continue
            label = df_vid["class"].unique()[0]
            self.video_path.append(os.path.join(self.video, vid + ".mp4"))
            self.audio_path.append(os.path.join(self.audio, vid + ".wav"))
            self.label.append(label)

    def sample_frame(self, video_path, rand_sec):
        # Open the speech_video file
        video = cv2.VideoCapture(video_path)
        # Get the total number of frames in the speech_video
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        low, high = (total_frames // 10) * rand_sec, (total_frames // 10) * (rand_sec+self.input_length)
        ytid = video_path.split('/')[-1][:-4]
        candidates = []

        if self.filter_frames:
            if ytid in self.frames:
                # TODO: fix the bug here
                try:
                    candidates = self.frames[ytid][0]
                    candidates = [frame for frame in candidates if low <= frame <= high]
                except:
                    candidates = []
        frame_num = random.choice(candidates) if candidates else random.randint(low, high)

        # Set the speech_video to the chosen frame
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        # Read the frame
        _, frame = video.read()
        # Release the speech_video file
        video.release()
        # Return the frame
        return frame

    def img_proc(self, vid, rand_sec):
        image = self.sample_frame(vid, rand_sec)
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            (
                h,
                w,
            ) = (
                img.shape[0],
                img.shape[1],
            )
            img = img[(h - crop) // 2: (h + crop) // 2, (w - crop) // 2: (w + crop) // 2]

        image = Image.fromarray(img)
        image = image.resize((512, 320), resample=self.interpolation)

        # image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = (image / 127.5 - 1.0).astype(np.float32)
        return torch.from_numpy(image).permute(2, 0, 1)

    def aud_proc_beats(self, aud, rand_sec=0):
        wav, sr = torchaudio.load(aud)
        wav = torch.tile(wav, (1, 10))
        wav = wav[:, :sr*10]
        start = rand_sec * sr
        end = (rand_sec+self.input_length) * sr
        wav = wav[:, start:end]
        return wav[0]

    def txt_proc(self):
        text = random.choice(self.templates).format(self.obj1)
        return self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

    def __getitem__(self, i):
        example = {}
        example["input_ids"] = self.txt_proc()
        ytid = self.audio_path[i % self.num_samples].split('/')[-1][:11]
        example["ytid"] = ytid
        example['label'] = self.label[i % self.num_samples]
        example['full_name'] = self.audio_path[i % self.num_samples].split('/')[-1][:-4]
        aud = self.audio_path[i % self.num_samples]
        if self.input_length == 10:
            rand_sec = 0
        else:
            rand_sec = np.random.randint(0, 10 - self.input_length)

        vid = self.video_path[i % self.num_samples]
        example["pixel_values"] = self.img_proc(vid, rand_sec)
        example["audio_values"] = self.aud_proc_beats(aud, rand_sec)
        return example





class VGG_Sound_dataset(Dataset):
    def __init__(
            self,
            args,
            tokenizer,
            logger,
            size=512,
            interpolation='bicubic',
    ):

        self.data = args.data_dir
        # n_pairs = len(os.listdir(self.data)) // 2
        # idxs = range(n_pairs)
        self.input_length = args.input_length
        self.image_path = list()
        self.audio_path = list()
        images = set([file_path[:-4] for file_path in os.listdir(self.data)])
        audios = set([file_path[:-4] for file_path in os.listdir(self.data)])
        samples = images & audios

        self.prepare_dataset(samples)
        self.num_samples = len(self.image_path)
        self._length = self.num_samples
        logger.info(f"{args.data_set}, num samples: {self.num_samples}")

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]
        self.center_crop = True
        self.templates = imagenet_templates_small
        # self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length

 
    def prepare_dataset(self, samples):
        for vid in list(samples):
            self.audio_path.append(os.path.join(self.data, vid + ".mp3"))
            self.image_path.append(os.path.join(self.data, vid + ".jpg"))

    def sample_frame(self, video_path, rand_sec):
        # Open the speech_video file
        video = cv2.VideoCapture(video_path)
        # Get the total number of frames in the speech_video
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        low, high = (total_frames // 10) * rand_sec, (total_frames // 10) * (rand_sec+self.input_length)
        ytid = video_path.split('/')[-1][:-4]
        candidates = []

        if self.filter_frames:
            if ytid in self.frames:
                # TODO: fix the bug here
                try:
                    candidates = self.frames[ytid][0]
                    candidates = [frame for frame in candidates if low <= frame <= high]
                except:
                    candidates = []
        frame_num = random.choice(candidates) if candidates else random.randint(low, high)

        # Set the speech_video to the chosen frame
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        # Read the frame
        _, frame = video.read()
        # Release the speech_video file
        video.release()
        # Return the frame
        return frame

    def img_proc(self, img):
        img = Image.open(img).convert("RGB")
        img = np.array(img).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            (
                h,
                w,
            ) = (
                img.shape[0],
                img.shape[1],
            )
            img = img[(h - crop) // 2: (h + crop) // 2, (w - crop) // 2: (w + crop) // 2]

        image = Image.fromarray(img)
        image = image.resize((512, 512), resample=self.interpolation)

        # image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = (image / 127.5 - 1.0).astype(np.float32)
        return torch.from_numpy(image).permute(2, 0, 1)

    def aud_proc_beats(self, aud, rand_sec=0):
        target_sr = getattr(self, "target_sr", 32000)
        waveform, sr = torchaudio.load(aud)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # resample if needed
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            waveform = resampler(waveform)
            sr = target_sr

        waveform = waveform.float()  # ensure float dtype

        desired_len = int(target_sr * self.input_length)  # number of samples we want (int)
        cur_len = waveform.shape[1]

        # If shorter -> pad with zeros on the right
        if cur_len < desired_len:
            pad_amount = desired_len - cur_len
            waveform = F.pad(waveform, (0, pad_amount))  # pad on right

        # Now it's >= desired_len: compute start (clamped) and slice
        max_start = waveform.shape[1] - desired_len
        start = int(round(rand_sec * target_sr))
        if start < 0:
            start = 0
        if start > max_start:
            # clamp to valid range
            start = max(0, max_start)

        end = start + desired_len
        segment = waveform[:, start:end]  # shape [1, desired_len]
        return segment.squeeze(0)   

    def txt_proc(self):
        text = random.choice(self.templates).format(self.obj1)
        return self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

    def __getitem__(self, idx):
        path_a = self.audio_path[idx]
        path_b = self.image_path[idx]
        example = {}
        # name_a = os.path.splitext(os.path.basename(path_a))[0]
        # name_b = os.path.splitext(os.path.basename(path_b))[0]
        name_a = os.path.basename(path_a)[:-4]
        name_b = os.path.basename(path_b)[:-4]
        example['full_name'] = f"{name_a}"

        # load each audio separately
        # (you can pass different rand_secs if you like)
        example["audio_values"] = self.aud_proc_beats(path_a)
        example["image_values"] = self.img_proc(path_b)
        return example




from collections import defaultdict  

class VGG_Sound_dataset2(Dataset):
    def __init__(
            self,
            args,
            tokenizer,
            logger,
            size=512,
            interpolation='bicubic',
    ):

        self.data = args.data_dir
        n_pairs = len(os.listdir(self.data)) // 4
        # idxs = range(n_pairs)
        self.input_length = args.input_length
        self.image_path = defaultdict(list)
        self.audio_path = defaultdict(list)
        for f in os.listdir(self.data):
            idx = int(f.split('_')[0])
            if f.endswith('.jpg'):
                self.image_path[idx].append(os.path.join(self.data, f))
            else:
                self.audio_path[idx].append(os.path.join(self.data, f))

        self.num_samples = len(self.image_path)
        self._length = self.num_samples
        logger.info(f"{args.data_set}, num samples: {self.num_samples}")

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]
        self.center_crop = True
        self.templates = imagenet_templates_small
        # self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length


    def img_proc(self, img):
        img = Image.open(img).convert("RGB")
        img = np.array(img).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            (
                h,
                w,
            ) = (
                img.shape[0],
                img.shape[1],
            )
            img = img[(h - crop) // 2: (h + crop) // 2, (w - crop) // 2: (w + crop) // 2]

        image = Image.fromarray(img)
        image = image.resize((512, 512), resample=self.interpolation)

        # image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = (image / 127.5 - 1.0).astype(np.float32)
        return torch.from_numpy(image).permute(2, 0, 1)


    def aud_proc_beats(self, aud, rand_sec=0):
        target_sr = getattr(self, "target_sr", 32000)
        try:
            waveform, sr = torchaudio.load(aud)
        except Exception as e:
            print(f"[AUDIO LOAD ERROR] {aud}")
            raise e
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # resample if needed
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            waveform = resampler(waveform)
            sr = target_sr

        waveform = waveform.float()  # ensure float dtype

        desired_len = int(target_sr * self.input_length)  # number of samples we want (int)
        cur_len = waveform.shape[1]

        # If shorter -> pad with zeros on the right
        if cur_len < desired_len:
            pad_amount = desired_len - cur_len
            waveform = F.pad(waveform, (0, pad_amount))  # pad on right

        # Now it's >= desired_len: compute start (clamped) and slice
        max_start = waveform.shape[1] - desired_len
        start = int(round(rand_sec * target_sr))
        if start < 0:
            start = 0
        if start > max_start:
            # clamp to valid range
            start = max(0, max_start)

        end = start + desired_len
        segment = waveform[:, start:end]  # shape [1, desired_len]
        return segment.squeeze(0)   

    def txt_proc(self):
        text = random.choice(self.templates).format(self.obj1)
        return self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

    def __getitem__(self, idx):
        try:
            path_a1 = self.audio_path[idx][0]
            path_a2 = self.audio_path[idx][1]
            path_b1 = self.image_path[idx][0]
            path_b2 = self.image_path[idx][1]
        except:
            print("----------")
            print(idx)
            print("----------")
        example = {}
        # name_a = os.path.splitext(os.path.basename(path_a))[0]
        # name_b = os.path.splitext(os.path.basename(path_b))[0]
        name_a1 = os.path.basename(path_a1)[:-4]
        name_a2 = os.path.basename(path_a2)[:-4].split('_')[-1]
        example['full_name'] = f"{name_a1}_{name_a2}"

        # load each audio separately
        # (you can pass different rand_secs if you like)
        example["audio_values1"] = self.aud_proc_beats(path_a1)
        example["image_values1"] = self.img_proc(path_b1)
        example["audio_values2"] = self.aud_proc_beats(path_a2)
        example["image_values2"] = self.img_proc(path_b2)
        return example









