import numpy as np
import pandas as pd
import time

class DataReader:
    def __init__(self, point_count=2048):
        self.point_count = point_count

    def convert_to_float(self, x):
        try:
            return float(x)
        except ValueError:
            return np.nan

    def read_csv(self, file_path, csv_file_path):
        start_time = time.time()
        
        df = pd.read_csv(csv_file_path)
        file_path = file_path.replace("../../env/envs/rsg_raibo_rough_terrain/", '')

        df['delta_quat_w'] = df['delta_quat_w'].apply(self.convert_to_float)
        df['delta_quat_x'] = df['delta_quat_x'].apply(self.convert_to_float)
        df['delta_quat_y'] = df['delta_quat_y'].apply(self.convert_to_float)
        df['delta_quat_z'] = df['delta_quat_z'].apply(self.convert_to_float)
        df['delta_pos_x'] = df['delta_pos_x'].apply(self.convert_to_float)
        df['delta_pos_y'] = df['delta_pos_y'].apply(self.convert_to_float)
        df['delta_pos_z'] = df['delta_pos_z'].apply(self.convert_to_float)

        row = df[df['filename'] == file_path]

        if row.empty:
            delta_pos = np.zeros(3, dtype=np.float32)
            delta_quat = np.array([1, 0, 0, 0], dtype=np.float32)
        else:
            delta_quat = row[['delta_quat_x', 'delta_quat_y', 'delta_quat_z', 'delta_quat_w']].values.flatten().astype(np.float32)
            delta_pos = row[['delta_pos_x', 'delta_pos_y', 'delta_pos_z']].values.flatten().astype(np.float32)
            if delta_pos.shape[0] != 3:
                delta_pos = np.zeros(3, dtype=np.float32)
            if delta_quat.shape[0] != 4:
                delta_quat = np.array([1, 0, 0, 0], dtype=np.float32)

        print(f"Time taken by read_csv: {time.time() - start_time:.6f} seconds")
        return delta_pos, delta_quat



# Example usage
reader = DataReader()
file_path = "your_file_path_here"
csv_file_path = "dataset/train/batch_0/delta_pose_0.csv"

# Test pandas read_csv
delta_pos_csv, delta_quat_csv = reader.read_csv(file_path, csv_file_path)

# Test numpy genfromtxt
delta_pos_gen, delta_quat_gen = reader.read_genfromtxt(csv_file_path)
