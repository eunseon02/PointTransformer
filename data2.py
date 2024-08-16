import os
import torch
import numpy as np
from plyfile import PlyData
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import glob

class PointCloudDataset(Dataset):
    def __init__(self, root_dirs):
        if isinstance(root_dirs, str):
            root_dirs = [root_dirs]  # 하나의 경로만 주어진 경우 리스트로 변환

        self.root_dirs = root_dirs
        self.data_file_lists = []
        self.gt_file_lists = []
        self.csv_files = []

        for root_dir in root_dirs:
            data_file_list = []
            gt_file_list = []
            csv_file = None

            files = sorted(os.listdir(root_dir))
            data_files = [f for f in files if f.endswith('.ply') and not f.endswith('_gt.ply')]
            gt_files = [f for f in files if f.endswith('_gt.ply')]

            data_files.sort()
            gt_files.sort()

            for data_file in data_files:
                gt_file = data_file.replace('.ply', '_gt.ply')
                if gt_file in gt_files:
                    data_file_list.append(os.path.join(root_dir, data_file))
                    gt_file_list.append(os.path.join(root_dir, gt_file))
            
            csv_files = glob.glob(os.path.join(root_dir, 'delta_pose*.csv'))
            if csv_files:
                csv_file = csv_files[0]

            self.data_file_lists.append(data_file_list)
            self.gt_file_lists.append(gt_file_list)
            self.csv_files.append(csv_file)

            print(f"Directory {root_dir} - Total data files: {len(data_file_list)}")
            print(f"Directory {root_dir} - Total GT files: {len(gt_file_list)}")
            assert len(data_file_list) == len(gt_file_list), f"Data files and GT files list must be of the same length in {root_dir}."


    def __len__(self):
        return sum(len(file_list) for file_list in self.data_file_lists)

    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise IndexError("Index out of range")

        for data_file_list, gt_file_list, csv_file in zip(self.data_file_lists, self.gt_file_lists, self.csv_files):
            if idx < len(data_file_list):
                data_file_path = data_file_list[idx]
                gt_file_path = gt_file_list[idx]
                data_pointcloud = self.read_ply(data_file_path)
                gt_pointcloud = self.read_ply(gt_file_path)
                lidar_pos, lidar_quat = self.read_csv(data_file_path, csv_file)
                return data_pointcloud, gt_pointcloud, lidar_pos, lidar_quat
            idx -= len(data_file_list) 

    def read_ply(self, file_path):
        try:
            plydata = PlyData.read(file_path)
            data = plydata['vertex'].data
            points = np.vstack([data['x'], data['y'], data['z']]).T
            return torch.tensor(points, dtype=torch.float32)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return torch.zeros((2048, 3), dtype=torch.float32)

    def convert_to_float(self, x):
        try:
            if isinstance(x, str):
                return float(x.strip('[]'))
            return float(x)
        except ValueError:
            return np.nan

    def read_csv(self, file_path, csv_file):
        df = pd.read_csv(csv_file)
        file_path = os.path.basename(file_path)

        df['delta_quat_w'] = df['delta_quat_w'].apply(self.convert_to_float)
        df['delta_quat_x'] = df['delta_quat_x'].apply(self.convert_to_float)
        df['delta_quat_y'] = df['delta_quat_y'].apply(self.convert_to_float)
        df['delta_quat_z'] = df['delta_quat_z'].apply(self.convert_to_float)
        df['delta_pos_x'] = df['delta_pos_x'].apply(self.convert_to_float)
        df['delta_pos_y'] = df['delta_pos_y'].apply(self.convert_to_float)
        df['delta_pos_z'] = df['delta_pos_z'].apply(self.convert_to_float)

        row = df[df['filename'] == file_path]

        if row.empty:
            return np.zeros(3, dtype=np.float32), np.array([1, 0, 0, 0], dtype=np.float32)
        else:
            lidar_quat = row[['delta_quat_x', 'delta_quat_y', 'delta_quat_z', 'delta_quat_w']].values.flatten().astype(np.float32)
            lidar_pos = row[['delta_pos_x', 'delta_pos_y', 'delta_pos_z']].values.flatten().astype(np.float32)
            return lidar_pos, lidar_quat

    def collate_fn(batch):
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return None
        return torch.utils.data.dataloader.default_collate(batch)