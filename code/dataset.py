#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset
import numpy as np
from code.utils import data_provider
import h5py

class PointCLoudDataset(Dataset):
    def __init__(self, hdf5_file_path, transform=True):
        with h5py.File(hdf5_file_path, "r") as h5: 
            print(h5["poisson_4096"].shape) 
            gt = h5["poisson_4096"][:, :, :]

            if transform:
                centroid = np.mean(gt[:,:,0:3], axis=1, keepdims=True)
                gt[:,:,0:3] = gt[:,:,0:3] - centroid
                furthest_distance = np.amax(np.sqrt(np.sum(gt[:,:,0:3] ** 2, axis=-1)),axis=1,keepdims=True)
                gt[:, :, 0:3] = gt[:,:,0:3] / np.expand_dims(furthest_distance,axis=-1)
                
            self.gt_set = torch.from_numpy(gt)
            print(self.gt_set.shape)
        
    def __len__(self):
        return len(self.gt_set)


    def __getitem__(self, idx):
        gt = self.gt_set[idx]
        random_1024 = np.random.choice(4096, 1024) #random sampling
        ip = gt[random_1024]
        
        return (ip, gt) 
