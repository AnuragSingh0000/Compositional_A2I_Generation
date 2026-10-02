import os
import torch
from torch.optim import *
import torchvision
from torchvision.transforms import *
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import torchaudio
from model import AVENet
from torchvision.utils import save_image,make_grid
from inference.generate_images import get_model
from scipy import signal
import soundfile as sf
from torchvision.transforms.functional import to_tensor
import librosa
import cv2
from tqdm import tqdm
import PIL
from PIL import Image
# from pydub import AudioSegment
from torchvision.transforms.functional import to_pil_image
import pdb
import shutil
from packaging import version
from a2s_dataloader import GetVGGSound
from  utils.metrics import evaluate_all
# from ..AudioToken.cluster import visualize_tokens

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
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }



def get_arguments():
    parser = argparse.ArgumentParser()
    #Definine Sound2Scene model (audio encoder)
    parser.add_argument(
        '--pool',
        default="avgpool",
        type=str,
        help= 'either vlad or avgpool')
    parser.add_argument(
        '--batch_size',
        default=64,
        type=int,
        help='Batch Size')
    parser.add_argument(
        '--n_classes',
        default=2048,
        type=int,
        help=
        'Number of classes')
    parser.add_argument(
        '--model_depth',
        default=18,
        type=int,
        help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument(
        '--resnet_shortcut',
        default='B',
        type=str,
        help='Shortcut type of resnet (A | B)')
    parser.add_argument('--checkpoint_vggish', dest='checkpoint',help='Path of checkpoint file for load model')

    #Defining image encoder and image decoder
    parser.add_argument('--root_path', default="./checkpoints")
    parser.add_argument('--model', default="icgan")
    parser.add_argument('--model_backbone', default="biggan")
    parser.add_argument('--resolution', default=128)
    parser.add_argument("--z_var", type=float, default=1, help="Noise variance: %(default)s)")
    parser.add_argument(
        "--trained_dataset",
        type=str,
        default="imagenet",
        choices=["imagenet", "coco"],
        help="Dataset in which the model has been trained on.",
    )

    #Defining directories
    parser.add_argument("--dataset", type=str, default="vgg", choices=["vgg", "vegas"],help="Dataset in which the model has been trained on.", )
    parser.add_argument("--wav_path", type=str, default="./samples/inference")
    parser.add_argument("--out_path", type=str, default="./generated_images", help="where to dump the output images")



    return parser.parse_args()

def load_data(args, batch_size=None):
    if batch_size==None:
        batch_size=args.batch_size

    if args.dataset == 'vgg':
        aud_path = os.path.join(args.vgg_dir, "audios")
        vid_path = os.path.join(args.vgg_dir, "frames_10fps")
        train_dataset = GetVGGSound(args.data_txt, args.annotation, aud_path, vid_path, vid_path)
        test_dataset = GetVGGSound(args.data_t_txt, args.annotation, aud_path, vid_path, vid_path)

    elif args.dataset == "vegas":
        train_dataset = GetAudioVideoDataset(args.data_txt, args.aud_path, args.emb_path, args.img_path)
        test_dataset = GetAudioVideoDataset(args.data_t_txt, args.aud_t_path, args.emb_t_path, args.img_t_path)

    #train_loader=None
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader


# def showImage(args,generator,feature_extractor, emb):
#     # define noise_vectors
#     z = torch.empty(8,generator.dim_z).normal_(mean=0, std=args.z_var)

#     #normalize audio embedding
#     emb /= torch.linalg.norm(emb, dim=-1, keepdims=True)
#     emb_ = torch.tile(emb, (8,1))

#     gen_img = generator(z.cuda(), None, emb_)
#     gen_img = torch.clamp(gen_img,-1., 1.)

#     output = make_grid(gen_img, normalize=True, scale_each=True, nrow=8)

#     return output


def showImage(args, generator, feature_extractor, emb):
    # noise
    z = torch.empty(1, generator.dim_z).normal_(mean=0, std=args.z_var).cuda()

    # normalize embedding
    emb = emb / torch.linalg.norm(emb, dim=-1, keepdim=True)

    # FIX: ensure shape is (1, D), not (1,1,D)
    if emb.dim() == 1:
        emb_ = emb.unsqueeze(0)
    else:
        emb_ = emb

    emb_ = emb_.cuda()

    # generate
    gen_img = generator(z, None, emb_)
    gen_img = torch.clamp(gen_img, -1., 1.)

    return gen_img[0]

def gen_name(tuple):
    name = ''
    for i in tuple:
        name = name+i+"_"
    return name

def audio2spectrogra(wav_file):
    samples, samplerate = sf.read(wav_file)

    # repeat in case audio is too short
    resamples = np.tile(samples, 10)[:160000]
    resamples[resamples > 1.] = 1.
    resamples[resamples < -1.] = -1.
    frequencies, times, spectrogram = signal.spectrogram(resamples, samplerate, nperseg=512, noverlap=353)
    spectrogram = np.log(spectrogram + 1e-7)
    mean = np.mean(spectrogram)
    std = np.std(spectrogram)
    spectrogram = np.divide(spectrogram - mean, std + 1e-9)

    spectrogram =  torch.from_numpy(spectrogram).unsqueeze(0)
    return spectrogram

def generate_images(args, model, generator,feature_extractor,device):
    audio_paths = args.wav_path
    save_path = args.out_path + "/imgs"
    os.makedirs(save_path, exist_ok=True)

    for audio in os.listdir(audio_paths):
        audio = os.path.join(audio_paths,audio)
        # audio = "./samples/inference/chainsaw.wav"

        spectrogram = audio2spectrogra(audio)
        spectrogram = Variable(spectrogram).to(device)
        _,emb = model(spectrogram.unsqueeze(1).float())
        output = showImage(args, generator, feature_extractor, emb)

        save_name = audio.split("/")[-1].split(".")[0]

        save_final = os.path.join(save_path, save_name+".png")

        save_image(output.cpu(), save_final)


def img_proc(img, center_crop=True, interpolation=PIL_INTERPOLATION["bicubic"]):
    img = Image.open(img).convert("RGB")
    img = np.array(img).astype(np.uint8)

    # Center crop (square)
    if center_crop:
        crop = min(img.shape[0], img.shape[1])
        h, w = img.shape[0], img.shape[1]
        img = img[(h - crop)//2:(h + crop)//2,
                  (w - crop)//2:(w + crop)//2]

    image = Image.fromarray(img)
    image = image.resize((512, 512), resample=interpolation)

    image = np.array(image).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Normalize to [-1, 1]
    image = (image / 127.5 - 1.0).astype(np.float32)

    return torch.from_numpy(image).permute(2, 0, 1)



def aud_proc_beats(aud, rand_sec=0, target_sr=32000, input_length=10):
    try:
        waveform, sr = torchaudio.load(aud)
    except Exception as e:
        print(f"[AUDIO LOAD ERROR] {aud}")
        raise e

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
        sr = target_sr

    waveform = waveform.float()

    desired_len = int(target_sr * input_length)
    cur_len = waveform.shape[1]

    # Pad if too short
    if cur_len < desired_len:
        pad_amount = desired_len - cur_len
        waveform = F.pad(waveform, (0, pad_amount))

    # Clamp start index
    max_start = waveform.shape[1] - desired_len
    start = int(round(rand_sec * target_sr))

    if start < 0:
        start = 0
    if start > max_start:
        start = max(0, max_start)

    end = start + desired_len
    segment = waveform[:, start:end]

    return segment.squeeze(0)


def generate_images_visualise(args, model, generator,feature_extractor,device):
    audio_paths = args.wav_path
    save_path = args.out_path + "/imgs"
    os.makedirs(save_path, exist_ok=True)

    batch_gen_imgs = []
    batch_gt_imgs = []
    batch_audio_feats = []
    batch_labels = []
    batch_inds = []

    iis = 0
    ais = 0
    aic = 0
    cts = 0
    ccs = 0
    cnt = 0

    with torch.no_grad():
        audio_files = [f for f in os.listdir(audio_paths) if f.endswith(".wav")]
        for audio in tqdm(audio_files, desc="Generating images"):
            if (audio.split('.')[-1] != 'wav'):
                continue

            audio = os.path.join(audio_paths,audio)
            image = audio.split('.')[0] + ".jpg"
            iv = img_proc(image)
            # audio = "./samples/inference/chainsaw.wav"

            label = audio.split("_")[-1][:-4]
            spectrogram = audio2spectrogra(audio)
            spectrogram = Variable(spectrogram).to(device)
            av = aud_proc_beats(audio)

            _,emb = model(spectrogram.unsqueeze(1).float())
            output = showImage(args, generator, feature_extractor, emb)
            gen_img = output

            save_name = audio.split("/")[-1].split(".")[0]
            save_final = os.path.join(save_path, save_name+".jpg")
            save_image(output.cpu(), save_final)
            
            ind = cnt % 8
            # Store for batch evaluation
            batch_gen_imgs.append(to_pil_image(gen_img))
            batch_gt_imgs.append(iv)
            batch_audio_feats.append(av)
            batch_labels.append(label)
            batch_inds.append(ind)

            
            # Evaluate when batch is full
            if len(batch_gen_imgs) == 8:
                batch_gen_imgs_tensor = torch.stack([
                    to_tensor(img) for img in batch_gen_imgs
                ])
                batch_gt_imgs = torch.stack([
                    img for img in batch_gt_imgs
                ], dim=0)
                batch_audio_feats = torch.stack([
                    feat for feat in batch_audio_feats
                ], dim=0)
                for j in range(8):
                    metrics = evaluate_all(
                        batch_gen_imgs[j], batch_gt_imgs, batch_audio_feats, batch_labels[j], current_audio_idx_in_batch=batch_inds[j]
                    )
                    iis += metrics["iis"]["iss"]
                    ais += metrics["ais"]["ais"]
                    aic += metrics["aic"]["accuracy"]
                    cts += metrics["cls"]["similarity"]
                    ccs += metrics["ccs"]["similarity"]
                    cnt += 1
                # Write batch metrics to file
                with open(os.path.join(args.out_path, "sound2scene_metrics.txt"), "a") as f:
                    f.write(f"IIS={iis:.2f}, AIS={(ais):.2f}, AIC={(aic):.2f}, CLS={(cts):.2f}, CCS={(ccs):.2f}, count={cnt}\n")

                # Clear batch lists
                batch_gen_imgs = []
                batch_gt_imgs = []
                batch_audio_feats = []
                batch_labels = []
                batch_inds = []

        batch_gen_imgs_tensor = torch.stack([
            to_tensor(img) for img in batch_gen_imgs
        ])
        batch_gt_imgs = torch.stack([
            img for img in batch_gt_imgs
        ], dim=0)
        batch_audio_feats = torch.stack([
            feat for feat in batch_audio_feats
        ], dim=0)
        # Process remaining samples < BATCH_SIZE
        for j in range(len(batch_gen_imgs)):
            metrics = evaluate_all(
                batch_gen_imgs[j], batch_gt_imgs, batch_audio_feats, batch_labels[j], current_audio_idx_in_batch=batch_inds[j]
            )
            iis += metrics["iis"]["iss"]
            ais += metrics["ais"]["ais"]
            aic += metrics["aic"]["accuracy"]
            cts += metrics["cls"]["similarity"]
            ccs += metrics["ccs"]["similarity"]
            cnt += 1
        # Write final metrics
        with open(os.path.join(args.out_path, "sonicdiff_metrics.txt"), "a") as f:
            f.write(f"Final Step {cnt}: IIS={iis:.2f}, AIS={(ais):.2f}, AIC={(aic):.2f}, CLS={(cts):.2f}, CCS={(ccs):.2f}, count={cnt}\n")

        print("Done.")


def main():
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch.cuda.device_count():", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))

    import random
    random_seed=45
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    args = get_arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #load sound2scene model
    checkpoint = torch.load(args.checkpoint,map_location=device)
    model = AVENet(args).to(device)
    model.load_state_dict(checkpoint)
    model.eval()


    #load image decoder
    suffix = ( 
        "_nofeataug"
        if args.resolution == 256
        and args.trained_dataset == "imagenet"
        else ""
    )
    exp_name = "%s_%s_%s_res%i%s" % (
        args.model,
        args.model_backbone,
        args.trained_dataset,
        args.resolution,
        suffix,
    )
    generator, feature_extractor = get_model(
        exp_name, args.root_path, args.model_backbone, device=device
    )

    # generate_images(args,model, generator,feature_extractor,device)
    generate_images_visualise(args,model, generator,feature_extractor,device)


if __name__=='__main__':
    main()




