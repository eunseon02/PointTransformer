import time
import numpy as np
import torch
from plyfile import PlyData

class PointCloudDataset:
    def __init__(self, point_cnt=2048):
        self.point_cnt = point_cnt

    def read_ply_plyfile(self, file_path):
        try:
            plydata = PlyData.read(file_path)
            data = plydata['vertex'].data
            points = np.vstack([data['x'], data['y'], data['z']]).T
            if points.shape[0] < self.point_cnt:
                padd = np.zeros((self.point_cnt, 3))
                padd[:points.shape[0], :] = points
            else:
                padd = points
            return torch.tensor(padd, dtype=torch.float32)
        except Exception as e:
            return None

    def read_ply_genfromtxt(self, file_path):
        try:
            points = np.genfromtxt(file_path, skip_header=10, usecols=(0, 1, 2), dtype=np.float32)
            if points.shape[0] < self.point_cnt:
                padd = np.zeros((self.point_cnt, 3), dtype=np.float32)
                padd[:points.shape[0], :] = points
                print(padd.shape)
            else:
                padd = points
            return torch.tensor(padd, dtype=torch.float32)
        except Exception as e:
            print("err")
            return None

# 테스트할 파일 경로
file_path = 'dataset/train/batch_0/pts_0000_gt.ply'

dataset = PointCloudDataset()

# 첫 번째 방법 테스트
start_time = time.time()
for _ in range(100):
    dataset.read_ply_plyfile(file_path)
plyfile_time = (time.time() - start_time) / 100

# 두 번째 방법 테스트
start_time = time.time()
for _ in range(100):
    dataset.read_ply_genfromtxt(file_path)
genfromtxt_time = (time.time() - start_time) / 100

print(f"PlyFile method average time: {plyfile_time * 1000:.3f} ms")
print(f"Genfromtxt method average time: {genfromtxt_time * 1000:.3f} ms")
