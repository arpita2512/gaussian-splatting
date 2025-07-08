from glob import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode
from torchvision import transforms
from torchvision.transforms.functional import convert_image_dtype
from pathlib import PurePath

from plyfile import PlyData

def get_ply_params(ply_path):
    
    plydata = PlyData.read(ply_path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
    
    features_dc = np.zeros((xyz.shape[0], 3), dtype=np.float32)
    features_dc[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])
    
    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)), dtype=np.float32)
    
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    #features_extra = features_extra.reshape((features_extra.shape[0], 3, 15))

    # combine features dc and rest
    features = np.hstack((features_dc, features_extra))
    
    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)), dtype=np.float32)
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)), dtype=np.float32)
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    return xyz, opacities, features, scales, rots 

class ViewDataset(Dataset):
    def __init__(self, input_dir, canon_ply_path):
        super().__init__()
        self.input_dir = input_dir
        self.all_files = glob(f"{self.input_dir}/per_cam_gs/*.ply")
        self.videos = [PurePath(x).parts[-1][:3] for x in self.all_files]
        self.canon_xyz, self.canon_opacities, self.canon_features, self.canon_scales, self.canon_rots = get_ply_params(canon_ply_path)
    
    def __len__(self):
        return len(self.all_files)
    
    def __getitem__(self, idx):
        file_path = PurePath(self.all_files[idx])
        xyz, opacities, features, scales, rots = get_ply_params(file_path)

        video = file_path.parts[-1][:3] 
        frame = file_path.parts[-1][4:-4]  # remove extension from string
        idx_frame = int(frame.lstrip("0")) - 1  # remove leading zeros, convert to int and subtract 1 for indexing
        
        aud_feats = np.load(f"{self.input_dir}/audio_features/{video}.npy")
        aud_feats = torch.from_numpy(aud_feats[idx_frame])
        aud_feats = torch.tile(aud_feats, (xyz.shape[0], 1))

        emo_feats = np.load(f"{self.input_dir}/emo_features/{video}.npy")
        emo_feats = torch.from_numpy(emo_feats[idx_frame])
        emo_feats = torch.tile(emo_feats, (xyz.shape[0], 1))

        return {
            'name': file_path.parts[-1][:-4],
            'aud_feats': aud_feats,
            'emo_feats': emo_feats,
            'delta_xyz': torch.from_numpy(xyz - self.canon_xyz),
            'delta_opa': torch.from_numpy(opacities - self.canon_opacities),
            'delta_ftr': torch.from_numpy(features - self.canon_features),
            'delta_sca': torch.from_numpy(scales - self.canon_scales),
            'delta_rot': torch.from_numpy(rots - self.canon_rots),
        }

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
        return img, aud_feats, emo_feats, img_path.parts[-1][:-4]
    
class CustomDataset_New(Dataset):
    def __init__(self, input_dir, n_gaussians):
        self.input_dir = input_dir
        self.n_gaussians = n_gaussians
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
        aud_feats = np.tile(aud_feats, (self.n_gaussians, 1))

        emo_feats = np.load(f"{self.input_dir}/emo_features/{video}.npy")
        emo_feats = torch.from_numpy(emo_feats[idx_frame])
        emo_feats = np.tile(emo_feats, (self.n_gaussians, 1))
        return img, aud_feats, emo_feats, img_path.parts[-1][:-4]