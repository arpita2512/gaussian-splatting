from glob import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode
from torchvision import transforms
from torchvision.transforms.functional import convert_image_dtype
from pathlib import PurePath

class CustomDataset(Dataset):
    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.all_files = glob(f"{self.input_dir}/images/*.png")
        self.videos = [PurePath(x).parts[-1][:3] for x in self.all_files]

    def __len__(self):
        return len(self.all_files)
    def __getitem__(self, idx):
        img = read_image(self.all_files[idx], mode=ImageReadMode.RGB)
        img = transforms.Resize((img.shape[1]//2, img.shape[2]//2))(img)
        img = convert_image_dtype(img)
        img_path = PurePath(self.all_files[idx])
        
        #emotion = img_path.parts[-5]
        #level = img_path.parts[-4]
        video = img_path.parts[-1][:3] 
        frame = img_path.parts[-1][4:-4]  # remove .png from string
        idx_frame = int(frame.lstrip("0")) - 1  # remove leading zeros, convert to int and subtract 1 for indexing
        
        aud_feats = np.load(f"{self.input_dir}/audio_features/{video}.npy")
        aud_feats = torch.from_numpy(aud_feats[idx_frame])

        emo_feats = np.load(f"{self.input_dir}/emo_features/{video}.npy")
        emo_feats = torch.from_numpy(emo_feats[idx_frame])
        return img, aud_feats, emo_feats, f"{video}_{str(frame)}"

class CustomDatasetOld(Dataset):
    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.all_files = glob(f"{self.input_dir}/neutral/level_1/colmap/images/*.png")
        self.videos = ["_".join([PurePath(x).parts[-5], PurePath(x).parts[-4], PurePath(x).parts[-1][:3]]) for x in self.all_files]

    def __len__(self):
        return len(self.all_files)
    def __getitem__(self, idx):
        img = read_image(self.all_files[idx], mode=ImageReadMode.RGB)
        img = transforms.Resize((img.shape[1]//2, img.shape[2]//2))(img)
        img = convert_image_dtype(img)
        img_path = PurePath(self.all_files[idx])
        
        emotion = img_path.parts[-5]
        level = img_path.parts[-4]
        video = img_path.parts[-1][:3] 
        frame = img_path.parts[-1][4:-4]  # remove .png from string
        idx_frame = int(frame.lstrip("0")) - 1  # remove leading zeros, convert to int and subtract 1 for indexing
        
        aud_feats = np.load(f"{self.input_dir}/{emotion}/{level}/audio_features/{video}.npy")
        aud_feats = torch.from_numpy(aud_feats[idx_frame])

        emo_feats = np.load(f"{self.input_dir}/{emotion}/{level}/emo_features/{video}.npy")
        emo_feats = torch.from_numpy(emo_feats[idx_frame])
        return img, aud_feats, emo_feats, f"{emotion}_{level}_{video}_{str(frame)}"