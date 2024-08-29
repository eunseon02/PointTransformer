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
    def __init__(self, filename, batch_dirs, split):
        self.point_cnt = 2048
        self.csv_files = []
        self.split = split
        self.file_len = 0
        self.filename = filename
        h5_file = h5py.File(self.filename, 'r')
        
        for group_name in h5_file.keys():
            print(f"Group: {group_name}")
            group = h5_file[group_name]
            for dataset_name in group.keys():
                print(f"  Dataset: {dataset_name}")
                
        if batch_dirs is None:
            batch_dirs = sorted(os.listdir(root_dir))
        self.batch_dirs = [os.path.join(root_dir, d) for d in batch_dirs]
        # print("batch_dirs", len(self.batch_dirs))
        self.total_len = 0
        self.batches = []
        for batch_dir in self.batch_dirs:
            # print("batch_dir", batch_dir)
            if os.path.isdir(batch_dir):
                csv_file_pattern = os.path.join(batch_dir, 'delta_pose*.csv')
                csv_files = sorted(glob.glob(csv_file_pattern))
                
                self.batch = []
                data_files = []
                gt_files = []
                for csv_file in csv_files:
                    gt_files = sorted([f for f in os.listdir(batch_dir) if f.endswith('_gt.ply')])
                    data_files = sorted([f for f in os.listdir(batch_dir) if f.endswith('.ply') and not f.endswith('_gt.ply')])

                    if len(gt_files) != len(data_files):
                        print(f"Warning: Mismatch in the number of GT files and data files in {batch_dir}")
                        continue
                    # print(len(gt_files))
                    if self.file_len != 0 and self.file_len != len(gt_files):
                        print(f"Warning :  file {len(gt_files)} have different length")
                    self.total_len +=len(gt_files)
                    self.batch.append((data_files[:], gt_files[:], csv_file, batch_dir))
                    # data_files, gt_files, csv_file, batch_dir = self.batch[0]
                    # print(len(data_files), len(gt_files), csv_file, batch_dir)
                self.batches.append(self.batch)
                # print(len(self.batch_dirs))
    def __len__(self):
        return self.total_len 

    def __getitem__(self, idx):
        batch_idx = idx % len(self.batch_dirs)
        file_idx = idx // len(self.batch_dirs)
        batch = self.batches[batch_idx]
        file_idx = ((file_idx%self.split)* (len(batch[0][0])//self.split)) + (idx // (len(self.batch_dirs)*self.split))        
        # print(idx // len(self.batch_dirs), file_idx)
        if len(batch) == 0 or len(batch[0]) == 0 or len(batch[0][0]) == 0:
            raise ValueError(f"Invalid dataset")
            
        if file_idx >= (len(batch[0][0])):
            raise IndexError(f"idx = {idx} : file_idx {file_idx} out of range for batch size {len(batch)}")

        data_files, gt_files, csv_file, batch_dir = batch[0]
        # print(len(data_files), len(gt_files), csv_file, batch_dir)
        
        data_file_path = os.path.join(batch_dir, data_files[file_idx])
        gt_file_path = os.path.join(batch_dir, gt_files[file_idx])

        data_pointcloud = self.read_ply(data_file_path)
        gt_pointcloud = self.read_ply(gt_file_path)
        lidar_pos, lidar_quat = self.read_csv(data_file_path, csv_file)
        return data_pointcloud, gt_pointcloud, lidar_pos, lidar_quat



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