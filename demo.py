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
from model_spconv import PointCloud3DCNN
from torch.utils.data import Dataset, DataLoader
import cumm.tensorview as tv
import spconv.pytorch as spconv
from collections import OrderedDict

def show_point_clouds(ply_file_1, ply_file_2):
    # 첫 번째 PLY 파일 읽기
    pcd1 = o3d.io.read_point_cloud(ply_file_1)
    pcd1.paint_uniform_color([1, 0, 0])  # 첫 번째 파일은 빨간색으로 표시

    # 두 번째 PLY 파일 읽기
    pcd2 = o3d.io.read_point_cloud(ply_file_2)
    pcd2.paint_uniform_color([0, 1, 0])  # 두 번째 파일은 녹색으로 표시

    # 두 개의 포인트 클라우드 시각화
    o3d.visualization.draw_geometries([pcd1, pcd2])


def _load_pretrain(model, pretrain):
    # Load checkpoint
    checkpoint = torch.load(pretrain, map_location='cpu')
    # Extract the model's state dictionary from the checkpoint
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    new_state_dict = OrderedDict()
    for key, val in state_dict.items():
        if key.startswith('module.'):
            name = key[len('module.'):]  # remove 'module.' prefix
        else:
            name = key
        new_state_dict[name] = val
    # Load the state dictionary into the model
    model.load_state_dict(new_state_dict)
    print(f"Model loaded from {pretrain}")

def get_parser():
    parser = argparse.ArgumentParser(description='Unsupervised Point Cloud Feature Learning')
    parser.add_argument('--model', type=str, default='check/model_epoch_best_279.pth', metavar='N',
                        help='Path to load model')
    parser.add_argument('--data', type=str, default="/root/raibo_arm/raisimGymTorch/algo/PointTransFormer/dataset/train/batch_0/pts_1007.ply", metavar='N',
                        help='Path to load data')
    args = parser.parse_args()
    return args




def preprocess(pc):
    from spconv.utils import Point2VoxelGPU3d
    from spconv.pytorch.utils import PointToVoxel

    # Voxel generator
    gen = Point2VoxelGPU3d(
        vsize_xyz=[0.05, 0.05, 0.05],
        coors_range_xyz=[-3, -3, -1, 3, 3, 1.5],
        num_point_features=3,
        max_num_voxels=600000,
        max_num_points_per_voxel=3
        )
    
    batch_size = pc.shape[0]
    all_voxels, all_indices = [], []
    tensors = []

    for batch_idx in range(batch_size):
        pc_single = pc[batch_idx]
        pc_single = tv.from_numpy(pc_single.cpu().numpy())
        voxels_tv, indices_tv, num_p_in_vx_tv = gen.point_to_voxel_hash(pc_single.cuda())
        # print(voxels_tv)

        voxels_torch = torch.tensor(voxels_tv.cpu().numpy(), dtype=torch.float32).to(device)
        indices_torch = torch.tensor(indices_tv.cpu().numpy(), dtype=torch.int32).to(device)
        # print(voxels_torch)
        # mean = voxels_torch.mean()
        # std = voxels_torch.std()
        # voxels_torch = (voxels_torch - mean) / std
        # print(voxels_torch)
        mean = voxels_torch.mean(dim=1, keepdim=True)  # (batch, 1, 3)
        voxels_torch = voxels_torch - mean
        # print(voxels_torch)
        # valid = num_p_in_vx_tv.cpu().numpy() > 0
        # voxels_flatten = voxels_torch.view(-1, self.model.num_point_features * self.model.max_num_points_per_voxel)[valid]
        # indices_torch = indices_torch[valid]
        voxels_flatten = torch.abs(voxels_torch.view(-1, 3 * 3))
        # tensor_to_ply(indices_torch, "indices_torch.ply")

        batch_indices = torch.full((indices_torch.shape[0], 1), batch_idx, dtype=torch.int32).to(device)
        indices_combined = torch.cat([batch_indices, indices_torch], dim=1)
        # tensor = spconv.SparseConvTensor(voxels_flatten, indices_combined, self.input_shape, self.batch_size)
        all_voxels.append(voxels_flatten)
        all_indices.append(indices_combined.int())
        # tensors.append(tensor)

    all_voxels = torch.cat(all_voxels, dim=0)
    all_indices = torch.cat(all_indices, dim=0)
    sparse_tensor = spconv.SparseConvTensor(all_voxels, all_indices, input_shape, batch_size)

    # dense_tensor = sparse_tensor.dense()
    return sparse_tensor
def occupancy_grid(pc):
    from spconv.utils import Point2VoxelGPU3d
    from spconv.pytorch.utils import PointToVoxel

    # Voxel generator
    gen = Point2VoxelGPU3d(
        vsize_xyz=[0.05, 0.05, 0.05],
        coors_range_xyz=[-3, -3, -1, 3, 3, 1.5],
        num_point_features=3,
        max_num_voxels=600000,
        max_num_points_per_voxel=3 
    )
    
    batch_size = pc.shape[0]
    all_voxels, all_indices = [], []

    for batch_idx in range(batch_size):
        pc_single = pc[batch_idx]
        pc_single = tv.from_numpy(pc_single.cpu().numpy())
        voxels_tv, indices_tv, num_p_in_vx_tv = gen.point_to_voxel_hash(pc_single.cuda())

        # Check if each voxel has any points, if yes, mark it as occupied (1), otherwise leave it empty (0)
        occupancy = (num_p_in_vx_tv.cpu().numpy() > 0).astype(float)
        occupancy = torch.tensor(occupancy, dtype=torch.float32).to(device).view(-1, 1)  # shape [N, 1]
        
        indices_torch = torch.tensor(indices_tv.cpu().numpy(), dtype=torch.int32).to(device)

        batch_indices = torch.full((indices_torch.shape[0], 1), batch_idx, dtype=torch.int32).to(device)
        indices_combined = torch.cat([batch_indices, indices_torch], dim=1)

        all_voxels.append(occupancy)
        all_indices.append(indices_combined.int())

    all_voxels = torch.cat(all_voxels, dim=0)
    all_indices = torch.cat(all_indices, dim=0)
    
    # Create SparseConvTensor with occupancy as features
    sparse_tensor = spconv.SparseConvTensor(all_voxels, all_indices, input_shape, batch_size)
    
    return sparse_tensor    
def save_single_occupancy_grid_as_ply(occupancy_grid, file_name="occupancy_grid.ply"):
    # Assume occupancy_grid is of shape (batch_size, 1, H, W, D)
    _, _, H, W, D = occupancy_grid.shape
    
    # Extract the occupancy grid for the first batch item
    grid = occupancy_grid[0, 0]  # shape (H, W, D)

    # Get the indices of the occupied voxels
    occupied_voxels = torch.nonzero(grid > 0, as_tuple=False).cpu().numpy()

    # Convert to float32 if necessary
    occupied_voxels = occupied_voxels.astype(np.float32)

    # Create a point cloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(occupied_voxels)

    # Save the point cloud as a PLY file
    o3d.io.write_point_cloud(file_name, point_cloud)
    print(f"Saved occupancy grid to {file_name}")
def tensor_to_ply(points_tensor, file_path= 'output.ply'):
    points = points_tensor.cpu().numpy()
    
    print("points",points.shape)
    if points.shape[1] != 3:
        print("points should have a shape of (N, 3)")
        points = points.squeeze(0)
        
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(file_path, pcd)
    print(f"Point cloud saved to {file_path}")


if __name__ == "__main__":
    args = get_parser()
    # points = read_ply(args.data)
    torch.cuda.set_device(0)
    # check if gpu is available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    val_path = 'dataset/valid'
    val_dataset = PointCloudDataset(val_path)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=8, pin_memory=True)
    model = PointCloud3DCNN(1).to(device)
    _load_pretrain(model, args.model)
    input_shape = (50, 120, 120)
with torch.no_grad():   
    for iter, batch  in enumerate(val_loader):
            model.train()
            pts, gt_pts, lidar_pos, lidar_quat = batch
            pts = pts.to(device)
            gt_pts = gt_pts.to(device)
            lidar_pos = lidar_pos.to(device)
            lidar_quat = lidar_quat.to(device)
            gt_pts_np = gt_pts.squeeze(0).cpu().numpy() if hasattr(gt_pts, 'numpy') else gt_pts
            pcd = o3d.geometry.PointCloud()

            # gt_pts를 open3d PointCloud 데이터로 변환
            pcd.points = o3d.utility.Vector3dVector(gt_pts_np)

            # 시각화
            o3d.visualization.draw_geometries([pcd], window_name=f"Point Cloud at Iteration {iter}")

            pts = torch.nan_to_num(pts, nan=0.0)
            sptensor = preprocess(pts)
            gt_occu = occupancy_grid(gt_pts)

            output, occu = model(sptensor)

            save_single_occupancy_grid_as_ply(occu, 'occu.ply')
            save_single_occupancy_grid_as_ply(gt_occu.dense(), 'gt_occu.ply')
            tensor_to_ply(output, 'output.ply')

            tensor_to_ply(gt_pts.squeeze(0), 'gts.ply')
            tensor_to_ply(pts.squeeze(0), 'points.ply')
            
            show_point_clouds('output.ply', 'gts.ply')
            