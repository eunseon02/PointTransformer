import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
from config import config as cfg
from loss import ChamferLoss
from data import PointCloudDataset
import open3d as o3d
import os
import time
import argparse
from feature_model import PointTransformerV3ForGlobalFeature
from torch.utils.data import Dataset, DataLoader

def show_point_clouds(ply_file_1, ply_file_2):
    # 첫 번째 PLY 파일 읽기
    pcd1 = o3d.io.read_point_cloud(ply_file_1)
    pcd1.paint_uniform_color([1, 0, 0])  # 첫 번째 파일은 빨간색으로 표시

    # 두 번째 PLY 파일 읽기
    pcd2 = o3d.io.read_point_cloud(ply_file_2)
    pcd2.paint_uniform_color([0, 1, 0])  # 두 번째 파일은 녹색으로 표시

    # 두 개의 포인트 클라우드 시각화
    o3d.visualization.draw_geometries([pcd1, pcd2])
torch.cuda.set_device(0)
# check if gpu is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
point_cnt = 2048
def read_ply(file_path):
    # print("ply", file_path)
    try:
        # Using genfromtxt to read the point cloud data
        points = np.genfromtxt(file_path, skip_header=10, usecols=(0, 1, 2), dtype=np.float32)            
        if points.shape[0] < point_cnt:
            padd = np.zeros((point_cnt, 3), dtype=np.float32)
            padd[:points.shape[0], :] = points
        else:
            padd = points

        return torch.tensor(padd, dtype=torch.float32)
    except Exception as e:
        # print(f"Error reading {file_path}: {e}")
        
        # return an empty tensor or handle the error as appropriate
        return torch.zeros((point_cnt, 3), dtype=torch.float32)


def load_model(model_path):
    model = torch.load(model_path)
    model.eval() 
    return model


def get_parser():
    parser = argparse.ArgumentParser(description='Unsupervised Point Cloud Feature Learning')
    parser.add_argument('--model', type=str, default='weight/model_epoch_best_139.pth', metavar='N',
                        help='Path to load model')
    parser.add_argument('--data', type=str, default="/root/raibo_arm/raisimGymTorch/algo/PointTransFormer/dataset/train/batch_0/pts_1007.ply", metavar='N',
                        help='Path to load data')
    args = parser.parse_args()
    return args




# def voxelize(coord):
#         coord_no_nan = coord.masked_fill(torch.isnan(coord), float('inf'))
#         grid_coord = torch.div(
#             coord - coord_no_nan.min(0)[0], torch.tensor([0.05]).to(coord.device), rounding_mode="trunc"
#         ).int()
#         return grid_coord

def tensor_to_ply(points_tensor, file_path= 'output.ply'):
    points = points_tensor.cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(file_path, pcd)
    print(f"Point cloud saved to {file_path}")



if __name__ == "__main__":
    args = get_parser()
    points = read_ply(args.data)
    val_path = 'dataset/valid'
    val_dataset = PointCloudDataset(val_path)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=8, pin_memory=True)

    for iter, batch  in enumerate(val_loader):

        points, gt_pts, lidar_pos, lidar_quat = batch

        model = PointTransformerV3ForGlobalFeature(1).to(device)
        checkpoint = torch.load(args.model)
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {}

        for key, value in state_dict.items():
            new_key = key.replace("module.", "")  
            new_state_dict[new_key] = value

        model.load_state_dict(new_state_dict)
        
        model.eval()
        points = points.view(-1, 3)
        pts = torch.nan_to_num(points, nan=0.0)
        print("pts shape", pts.shape)

        data_dict = {
            'feat': pts.to(device),
            'coord': pts.to(device),
            'grid_size': torch.tensor([0.1]).to(device),
            'offset': torch.arange(0, pts.size(0) + 1, 2048, device=device)
        }


        with torch.no_grad():
            output = model(data_dict)

        tensor_to_ply(output, 'output.ply')
        tensor_to_ply(gt_pts.squeeze(0), 'gts.ply')
        tensor_to_ply(points.squeeze(0), 'points.ply')
        
        show_point_clouds('output.ply', 'gts.ply')
        