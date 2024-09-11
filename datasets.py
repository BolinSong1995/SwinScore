import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob
import os
from math import ceil, floor
from medpy.io import load, header
from models import Model
import utils
import pandas as pd
import matplotlib.pyplot as plt


class RadDataset(Dataset):
    def __init__(self, df, root_data,train_flag=True, dim=[48, 48, 3], ring=15):
        self.df = df

        self.train_flag=train_flag
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomAffine(3, scale=(0.95, 1.05)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5)
            ])
        self.test_transforms = transforms.Compose([
            transforms.ToTensor(),
            ])
        self.y = np.array(df["DFS_3years"]).astype(np.float32)
        self.time = np.array(df["DFS"]).astype(np.float32)
        self.event = np.array(df["DFS_censor"]).astype(np.float32)
        self.ID = np.array(df["radiology_folder_name"])


        self.dim = dim
        self.ring = ring
        self.root_data = root_data

    def __len__(self):
        return len(self.y)

    def get_radiology(self, ct_image, index,train_flag):
        concat_vols = []

        torch.cuda.manual_seed_all(42)
        torch.manual_seed(42)
        np.random.seed(42)

        for location in ['tumor', 'lymph']:
            
            X_min, X_max, Y_min, Y_max, Z_min, Z_max = np.array(
                self.df["X_min_" + location][index]), np.array(
                self.df["X_max_" + location][index]), np.array(
                self.df["Y_min_" + location][index]), np.array(
                self.df["Y_max_" + location][index]), np.array(
                self.df["Z_min_" + location][index]), np.array(
                self.df["Z_max_" + location][index])
            X_min -= self.ring
            Y_min -= self.ring
            Z_min = max(3, Z_min - self.ring)
            X_max += self.ring
            Y_max += self.ring
            Z_max = min(ct_image.shape[-1]-1, Z_max+ self.ring)

            center_Y = int(ceil(int(Y_min+Y_max)/2))
            center_X = int(ceil(int(X_min+X_max)/2))

            Z_1, Z_2, Z_3 = Z_min+int((Z_max - Z_min)/4), Z_min + \
                int((Z_max - Z_min)/2), Z_min + \
                int(3*(Z_max - Z_min)/4)
            
            center_Z1 = int((Z_min+Z_1)/2)
            center_Z2 = int((Z_1+Z_2)/2)
            center_Z3 = Z_1
            center_Z4 = Z_3
            
            center1 = [center_Y, center_X, center_Z1]
            center2 = [center_Y, center_X, center_Z2]
            center3 = [center_Y, center_X, center_Z3]
            center4 = [center_Y, center_X, center_Z4]
            
            if train_flag:
                sub_vol1 = self.transforms(
                    utils.random_crop(ct_image, self.dim, center1))
                sub_vol2 = self.transforms(
                    utils.random_crop(ct_image, self.dim, center2))
                sub_vol3 = self.transforms(
                    utils.random_crop(ct_image, self.dim, center3))
                sub_vol4 = self.transforms(
                    utils.random_crop(ct_image, self.dim, center4))
                vol = torch.stack(
                    (sub_vol1, sub_vol2, sub_vol3, sub_vol4))
                concat_vols.append(vol)
            else:
                sub_vol1 = self.test_transforms(
                    utils.random_crop(ct_image, self.dim, center1))
                sub_vol2 = self.test_transforms(
                    utils.random_crop(ct_image, self.dim, center2))
                sub_vol3 = self.test_transforms(
                    utils.random_crop(ct_image, self.dim, center3))
                sub_vol4 = self.test_transforms(
                    utils.random_crop(ct_image, self.dim, center4))
                vol = torch.stack(
                    (sub_vol1, sub_vol2, sub_vol3, sub_vol4))
                concat_vols.append(vol)
        return concat_vols

    def __getitem__(self, index):

        ct_image, _ = load(os.path.join(self.root_data, self.df["radiology_folder_name"].iloc[index], "CT_img.nii.gz"))

        
        ct_image = utils.soft_tissue_window(ct_image)

        ct_vol = self.get_radiology(ct_image, index,self.train_flag)
        ct_tumor, ct_lymphnodes = ct_vol[0], ct_vol[1]

        return ct_tumor, ct_lymphnodes, self.y[index], self.time[index], self.event[index], self.ID[index]