import os
import torch
import numpy as np
from plyfile import PlyData
import pandas as pd
import glob
import random
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pickle
import joblib
import h5py
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class PointCloudDataset(Dataset):
    def __init__(self, filename, split='train'):
        self.point_cnt = 2048
        self.split = split
        self.filename = filename
        self.batch_count = 0
        self.h5_file = h5py.File(self.filename, 'r')
        
        self.batches = []
        self.total_len = 0

        if self.split not in self.h5_file:
            raise KeyError(f"The split '{self.split}' does not exist in the HDF5 file.")
        
        split_group = self.h5_file[self.split]
        
        for group_name in split_group.keys():
            group = split_group[group_name]
            datasets = [name for name in group.keys() if name.startswith('pts_') and not name.endswith('_gt')]
            if len(datasets) > 0:
                self.batch_count += 1
                self.total_len += len(datasets)
                self.batches.append((group_name, datasets))
        
    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        batch_idx = idx % len(self.batches)
        
        # file_idx를 ff_update와 ff_step으로 분리
        file_idx = idx // len(self.batches)
        ff_update = file_idx // 75  # ff_update 계산
        ff_step = file_idx % 75     # ff_step 계산
        
        group_name, datasets = self.batches[batch_idx]
        
        # 데이터셋 이름 생성
        pts_dataset_name = f'pts_{ff_update:02d}{ff_step:02d}'
        
        if pts_dataset_name not in datasets:
            raise IndexError(f"idx = {idx} : {pts_dataset_name} not found in datasets of group {group_name}")
        
        group = self.h5_file[f"{self.split}/{group_name}"]
        
        pts_data = group[pts_dataset_name][:].astype('float32')
        pts_gt_data = group[f'{pts_dataset_name}_gt'][:].astype('float32')
        position = group[f'position_{ff_update:02d}{ff_step:02d}'][:].astype('float32')
        quaternion = group[f'quaternion_{ff_update:02d}{ff_step:02d}'][:].astype('float32')
        
        return pts_data, pts_gt_data, position, quaternion, pts_dataset_name


    
    def close(self):
        self.h5_file.close()
class GetTarget(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        if os.path.exists(root_dir):
            self.file_paths = sorted(os.listdir(root_dir), key=lambda x: f"{int(os.path.splitext(x)[0]):03d}")
        else:
            self.file_paths = []
        # print(self.file_paths)
        
        
    def __len__(self):
        if len(self.file_paths) == 0:
            return 9999
        else:
            return len(self.file_paths)
    
    def __getitem__(self, idx):
        if idx >= len(self.file_paths):
            return []
        file_path = self.file_paths[idx]
        file_path = os.path.join(self.root_dir, file_path)
        if os.path.exists(file_path):
            occupancy_grids = joblib.load(file_path, mmap_mode='r')
            # print("File loaded successfully.")
        # else:
        #     print(f"File '{file_path}' does not exist.")
            
        return occupancy_grids