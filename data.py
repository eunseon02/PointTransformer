import os
import torch
import numpy as np
from plyfile import PlyData
import pandas as pd
import glob
import random
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class PointCloudDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.sub_region = 6.0
        self.point_cnt = 2048
        self.csv_files = []
        self.data_files = []
        self.gt_files = []
        self.csv_data_list = []
        self.batches = []
        self.batch_dir_list = []
        for batch_folder in sorted(os.listdir(root_dir)):
            print(batch_folder)
            batch_dir = os.path.join(root_dir, batch_folder)
            print(root_dir)
            # print(batch_dir)
            self.batch_dir_list.extend(batch_dir)
            if os.path.isdir(batch_dir):
                csv_file_pattern = os.path.join(batch_dir, 'delta_pose*.csv')
                csv_files = sorted(glob.glob(csv_file_pattern))
                
                for csv_file in csv_files:
                    csv_data = pd.read_csv(csv_file)
                    sorted_filenames = csv_data['filename'].tolist()

                    # 알파벳 순서대로 정렬
                    gt_files = sorted([f for f in os.listdir(batch_dir) if f.endswith('_gt.ply')])
                    data_files = sorted([f for f in os.listdir(batch_dir) if f.endswith('.ply') and not f.endswith('_gt.ply')])

                    if len(gt_files) != len(data_files):
                        print(f"Warning: Mismatch in the number of GT files and data files in {batch_dir}")
                        continue

                    self.gt_files.extend(gt_files)
                    self.data_files.extend(data_files)
                    self.batches.append((data_files, gt_files, csv_file, batch_dir))
    def __len__(self):
        return len(self.data_files)
        # return len(self.batches) * len(self.batches[0][0])

    def __getitem__(self, idx):
        if idx >= self.__len__():
            idx = idx % self.__len__()
        batch_size = len(self.batches[0][0])
        batch_idx = idx // batch_size
        file_idx = idx % batch_size
        data_files, gt_files, csv_file, batch_dir = self.batches[batch_idx]
        if file_idx >= len(data_files) or file_idx >= len(gt_files):
            return self.__getitem__(idx + 1)
        
        
        data_file_path = os.path.join(batch_dir, data_files[file_idx])
        gt_file_path = os.path.join(batch_dir, gt_files[file_idx])
        # print("check : ",data_file_path ,gt_file_path)
        data_pointcloud = self.read_ply(data_file_path)
        if data_pointcloud is None:
        # Skip to the next index if the current file is invalid
            if idx + 1 < self.__len__():
                return self.__getitem__(idx + 1)
            else:
                # Return some default value or handle the end of the dataset
                raise StopIteration("No more valid data files")
        gt_pointcloud = self.read_ply(gt_file_path)
             
        lidar_pos, lidar_quat = self.read_csv(data_file_path, csv_file)
        # print("getitem", data_pointcloud.shape)
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
            # print(f"Error reading {file_path}: {e}")
            
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

        df['delta_quat_w'] = df['delta_quat_w'].apply(self.convert_to_float)
        df['delta_quat_x'] = df['delta_quat_x'].apply(self.convert_to_float)
        df['delta_quat_y'] = df['delta_quat_y'].apply(self.convert_to_float)
        df['delta_quat_z'] = df['delta_quat_z'].apply(self.convert_to_float)
        df['delta_pos_x'] = df['delta_pos_x'].apply(self.convert_to_float)
        df['delta_pos_y'] = df['delta_pos_y'].apply(self.convert_to_float)
        df['delta_pos_z'] = df['delta_pos_z'].apply(self.convert_to_float)

        row = df[df['filename'] == file_path]

        if row.empty:
            # print(f"No data found for file name: {file_path}")
            return np.zeros(3, dtype=np.float32), np.array([1, 0, 0, 0], dtype=np.float32)
        else:
            # print(row)
            delta_quat = row[['delta_quat_x', 'delta_quat_y', 'delta_quat_z', 'delta_quat_w']].values.flatten().astype(np.float32)
            delta_pos = row[['delta_pos_x', 'delta_pos_y', 'delta_pos_z']].values.flatten().astype(np.float32)
            # print(lidar_pos.shape)
            if delta_pos.shape[0] != 3:
                delta_pos = np.zeros(3, dtype=np.float32)
            if delta_quat.shape[0] != 4:
                delta_quat = np.array([1, 0, 0, 0], dtype=np.float32)
            return delta_pos, delta_quat

