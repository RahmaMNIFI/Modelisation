import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
from torchvision import transforms
from tqdm import tqdm
import openslide
import pandas as pd
from models.resnet_custom import resnet50_baseline
from util import WSI_Dataset

# Argument parser
parser = argparse.ArgumentParser(description='Process patches and compute features.')

parser.add_argument('--wsi_csv', type=str, default='/home/rahma/CLAM/resultats/process_list_autogen.csv',
                    help='Path to the CSV file containing WSI information')
parser.add_argument('--wsi_path', type=str, default='/media/rahma/ESD-USB/',
                    help='Path to the directory containing WSI images')
parser.add_argument('--patches_path', type=str, default='/home/rahma/CLAM/resultats/patches/',
                    help='Path to the directory containing patch HDF5 files')
parser.add_argument('--output_path', type=str, default='/home/rahma/CLAM-/features/',
                    help='Path to the directory to save extracted features')

args = parser.parse_args()

# Transformer le ROI des patches
roi_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

class PatchesDataset(Dataset):
    def __init__(self, file_path, wsi, pretrained=False, custom_transforms=None, custom_downsample=1, target_patch_size=-1):
        self.pretrained = pretrained
        self.wsi = wsi
        self.roi_transforms = roi_transforms
        self.file_path = file_path
        
        with h5py.File(self.file_path, "r") as f:
            self.patch_level = f['coords'].attrs['patch_level']
            self.patch_size = f['coords'].attrs['patch_size']
            self.length = len(f['coords'])
            if target_patch_size > 0:
                self.target_patch_size = (target_patch_size, target_patch_size)
            elif custom_downsample > 1:
                self.target_patch_size = (self.patch_size // custom_downsample, self.patch_size // custom_downsample)
            else:
                self.target_patch_size = None

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as hdf5_file:
            coord = hdf5_file['coords'][idx]
        
        img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')

        if self.target_patch_size is not None:
            img = img.resize(self.target_patch_size)
        
        img = self.roi_transforms(img).unsqueeze(0)
        return img, coord

def collate_features(batch):
    imgs = torch.cat([item[0] for item in batch], dim=0)
    coords = np.vstack([item[1] for item in batch])
    return imgs, coords

def save_hdf5(output_path, asset_dict, attr_dict=None, mode='a'):
    with h5py.File(output_path, mode) as file:
        for key, val in asset_dict.items():
            data_shape = val.shape
            if key not in file:
                data_type = val.dtype
                chunk_shape = (1, ) + data_shape[1:]
                maxshape = (None, ) + data_shape[1:]
                dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
                dset[:] = val
                if attr_dict is not None and key in attr_dict:
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
            else:
                dset = file[key]
                dset.resize(len(dset) + data_shape[0], axis=0)
                dset[-data_shape[0]:] = val

    return output_path

# Initialisation des données et du modèle
wsi_bag = WSI_Dataset(args.wsi_csv)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = resnet50_baseline(pretrained=True).to(device)
mode = 'w'

# Parcours des WSI
for slide_id in wsi_bag.df['slide_id']:
    wsi = openslide.open_slide(args.wsi_path + slide_id[:-4] + '.tif')
    patches = PatchesDataset(args.patches_path + slide_id[:-4] + '.h5', wsi, True, roi_transforms, 1, 256)
    patches_loader = DataLoader(patches, batch_size=256, num_workers=4, pin_memory=True, collate_fn=collate_features)
    
    print(f'Processing {slide_id} - {len(patches_loader)} batches in total')
    
    for batch_images, batch_coords in tqdm(patches_loader):
        with torch.no_grad():
            batch_images = batch_images.to(device, non_blocking=True)
            features = model(batch_images)
            features = features.cpu().numpy()
            outputs = {'features': features, 'coords': batch_coords}
            save_hdf5(args.output_path + slide_id[:-4] + '_features.h5', outputs, None, mode=mode)
            mode = 'a'
    
    # Sauvegarde des features en format .pt
    file = h5py.File(args.output_path + slide_id[:-4] + '_features.h5', "r")
    features = file['features'][:]
    features = torch.from_numpy(features)
    torch.save(features, args.output_path + slide_id[:-4] + '_features.pt')

print("The features' computation is done.")
