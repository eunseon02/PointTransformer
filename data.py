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
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class PointCloudDataset(Dataset):
    def __init__(self, root_dir, batch_dirs):
        self.root_dir = root_dir
        self.point_cnt = 2048
        self.csv_files = []
        self.split = 4
        
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
                    print(len(gt_files))
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
        # print(len(self.batch_dirs))
        batch = self.batches[batch_idx]
        # print(len(batch[0][0]))
        file_idx = ((file_idx%self.split)* (len(batch[0][0])//self.split)) + (idx // (len(self.batch_dirs)*self.split))        

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

    def read_ply(self, file_path):
        # print("ply", file_path)
        try:
            plydata = PlyData.read(file_path)
            data = plydata['vertex'].data
            points = np.vstack([data['x'], data['y'], data['z']]).T
            if points.shape[0] < self.point_cnt:
                print("2048 in")
                padd = np.zeros((self.point_count, 3))
                padd[:points.shape[0], :] = points
            else:
                padd = points
            return torch.tensor(padd, dtype=torch.float32)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            
            # return an empty tensor or handle the error as appropriate
            return None

    def convert_to_float(self, x):
        try:
            if isinstance(x, str):
                return float(x.strip('[]'))
            return float(x)
        except ValueError:
            return np.nan

    def read_csv(self, file_path, csv_file_path):
        # print(f"reading {csv_file_path}")
        df = pd.read_csv(csv_file_path)
        file_path = file_path.replace("dataset/", '')
        file_path = file_path.replace("sample/", '')

        df['delta_quat_w'] = df['delta_quat_w'].apply(self.convert_to_float)
        df['delta_quat_x'] = df['delta_quat_x'].apply(self.convert_to_float)
        df['delta_quat_y'] = df['delta_quat_y'].apply(self.convert_to_float)
        df['delta_quat_z'] = df['delta_quat_z'].apply(self.convert_to_float)
        df['delta_pos_x'] = df['delta_pos_x'].apply(self.convert_to_float)
        df['delta_pos_y'] = df['delta_pos_y'].apply(self.convert_to_float)
        df['delta_pos_z'] = df['delta_pos_z'].apply(self.convert_to_float)

        row = df[df['filename'] == file_path]

        if row.empty:
            print(f"No data found for file name: {file_path}")
            return np.zeros(3, dtype=np.float32), np.array([1, 0, 0, 0], dtype=np.float32)
        else:
            # print(f"data found for file name: {file_path}")
            # print(row)
            delta_quat = row[['delta_quat_x', 'delta_quat_y', 'delta_quat_z', 'delta_quat_w']].values.flatten().astype(np.float32)
            delta_pos = row[['delta_pos_x', 'delta_pos_y', 'delta_pos_z']].values.flatten().astype(np.float32)
            # print(lidar_pos.shape)
            if delta_pos.shape[0] != 3:
                delta_pos = np.zeros(3, dtype=np.float32)
            if delta_quat.shape[0] != 4:
                delta_quat = np.array([1, 0, 0, 0], dtype=np.float32)
            return delta_pos, delta_quat


class GetTarget(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        if os.path.exists(root_dir):
            self.file_paths = sorted(os.listdir(root_dir), key=lambda x: f"{int(os.path.splitext(x)[0]):03d}")
        else:
            self.file_paths = []
        # print(self.file_paths)
        
        
    def __len__(self):
        return min(9999, len(self.file_paths)) 
    
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