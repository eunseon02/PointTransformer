
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
from config import config as cfg
from data import PointCloudDataset
# import open3d as o3d
import os
import time
import argparse
import open3d as o3d
import spconv.pytorch as spconv
import cumm.tensorview as tv
import sys
from model_spconv import PointCloud3DCNN
from torch.utils.data import Dataset, DataLoader
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import cProfile
import pstats
import io
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import gc
import logging
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
from os.path import join
# import tensorflow as tf
import open3d as o3d
from open3d.visualization.tensorboard_plugin import summary
from open3d.visualization.tensorboard_plugin.util import to_dict_batch
from torch.multiprocessing import Process

BASE_LOGDIR = "./logs" 
writer = SummaryWriter(join(BASE_LOGDIR, "visualize"))


def tensor_to_ply(tensor, filename):
    # print("tensor", tensor.shape)
    points = tensor.cpu().detach().numpy()
    points = points.astype(np.float64)
    # points=  points[0]
    if points.shape[1] != 3:
        raise ValueError(f"Expected point cloud data with shape (n, 3), but got {points.shape}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pcd)
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
def profileit(func):
    def wrapper(*args, **kwargs):
        datafn = func.__name__ + ".profile" # Name the data file sensibly
        prof = cProfile.Profile()
        retval = prof.runcall(func, *args, **kwargs)
        prof.dump_stats(datafn)
        return retval

    return wrapper
def occupancy_grid_to_coords(occupancy_grid):
    _, _, H, W, D = occupancy_grid.shape
    occupancy_grid = occupancy_grid[0, 0]
    indices = torch.nonzero(occupancy_grid > 0, as_tuple=False) 
    # print(indices.dtype) 
    return indices

class Train():
    def __init__(self, args):
        self.epochs = 300
        self.snapshot_interval = 10
        self.batch_size = 1
        self.split = 1
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(self.device)
        self.model = PointCloud3DCNN(self.batch_size).to(self.device)
        self.model_path = args.model_path
        if self.model_path != '':
            self._load_pretrain(args.model_path)
        
        self.ply_file = "preds.ply"
        self.val_path = 'dataset/train'
        self.val_paths = ['batch_2']
        self.val_dataset = PointCloudDataset(self.val_path, self.val_paths, self.split)
        print(f"Total valid dataset length: {len(self.val_dataset)}")
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,pin_memory=True)
        if len(self.val_dataset.batch_dirs) != self.batch_size:
            print(len(self.val_dataset.batch_dirs))
            raise RuntimeError('Wrong batch_size')
        self.parameter = self.model.parameters()
        self.input_shape = (50, 120, 120)
        
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        

    def run(self):
        print('start!!')

        self.model.train()
        prev_preds_val = None

        # start_epoch = 0
        # for epoch in range(start_epoch, self.epochs):
        prev_preds_val = self.demo(0, prev_preds_val)

    def tensorboard_launcher(self, points, step, color, tag):
        points = occupancy_grid_to_coords(points)
        num_points = points.shape[0]
        colors = torch.tensor(color).repeat(num_points, 1)
        writer.add_3d(
        tag,
        {
            "vertex_positions": points.float(), # (N, 3)
            "vertex_colors": colors.float()  # (N, 3)
        },
        step)
        # print("save")
    def transform_point_cloud(self, point_cloud, pos, quat):
        """
        Transform point cloud to world frame using position and quaternion.
        """
        quat = quat / np.linalg.norm(quat)
        r = R.from_quat(quat)
        rotation_matrix = r.as_matrix()

        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = pos.flatten()


        ones = torch.ones((point_cloud.shape[0], 1), dtype=torch.float32)
        pc_homo = torch.cat([point_cloud, ones], dim=1)
        transformation_matrix_torch = torch.from_numpy(transformation_matrix).float()
        transformed_pc_homo = torch.matmul(transformation_matrix_torch, pc_homo.T).T

        transformed_pc = transformed_pc_homo[:, :3]
        del point_cloud, ones, pc_homo, transformation_matrix_torch, transformed_pc_homo
        return transformed_pc
    def preprocess(self, pc):
        from spconv.utils import Point2VoxelGPU3d
        from spconv.pytorch.utils import PointToVoxel

        # Voxel generator
        gen = Point2VoxelGPU3d(
            vsize_xyz=[0.05, 0.05, 0.05],
            coors_range_xyz=[-3, -3, -1, 3, 3, 1.5],
            num_point_features=self.model.num_point_features,
            max_num_voxels=600000,
            max_num_points_per_voxel=self.model.max_num_points_per_voxel
            )
        
        batch_size = pc.shape[0]
        all_voxels, all_indices = [], []

        for batch_idx in range(batch_size):
            pc_single = pc[batch_idx]
            pc_single = tv.from_numpy(pc_single.cpu().numpy())
            voxels_tv, indices_tv, num_p_in_vx_tv = gen.point_to_voxel_hash(pc_single.cuda())

            voxels_torch = torch.tensor(voxels_tv.cpu().numpy(), dtype=torch.float32).to(self.device)
            indices_torch = torch.tensor(indices_tv.cpu().numpy(), dtype=torch.int32).to(self.device)
            mean = voxels_torch.mean(dim=1, keepdim=True)  # (batch, 1, 3)
            voxels_torch = voxels_torch - mean
            valid = num_p_in_vx_tv.cpu().numpy() > 0
            voxels_flatten = voxels_torch.view(-1, self.model.num_point_features * self.model.max_num_points_per_voxel)[valid]
            indices_torch = indices_torch[valid]
            voxels_flatten = torch.abs(voxels_torch.view(-1, self.model.num_point_features * self.model.max_num_points_per_voxel))

            batch_indices = torch.full((indices_torch.shape[0], 1), batch_idx, dtype=torch.int32).to(self.device)
            indices_combined = torch.cat([batch_indices, indices_torch], dim=1)
            all_voxels.append(voxels_flatten)
            all_indices.append(indices_combined.int())

        all_voxels = torch.cat(all_voxels, dim=0)
        all_indices = torch.cat(all_indices, dim=0)
        sparse_tensor = spconv.SparseConvTensor(all_voxels, all_indices, self.input_shape, self.batch_size)

        return sparse_tensor
    def occupancy_grid_(self, pc):
        from spconv.utils import Point2VoxelGPU3d
        from spconv.pytorch.utils import PointToVoxel
        # Voxel generator
        gen = Point2VoxelGPU3d(
            vsize_xyz=[0.05, 0.05, 0.05],
            coors_range_xyz=[-3, -3, -1, 3, 3, 1.5],
            num_point_features=self.model.num_point_features,
            max_num_voxels=600000,
            max_num_points_per_voxel=self.model.max_num_points_per_voxel
        )
        
        batch_size = pc.shape[0]
        all_voxels, all_indices = [], []

        for batch_idx in range(batch_size):
            pc_single = pc[batch_idx]
            pc_single = tv.from_numpy(pc_single.cpu().numpy())
            voxels_tv, indices_tv, num_p_in_vx_tv = gen.point_to_voxel_hash(pc_single.cuda())

            occupancy = (num_p_in_vx_tv.cpu().numpy() > 0).astype(float)
            occupancy = torch.tensor(occupancy, dtype=torch.float32).to(self.device).view(-1, 1)  # shape [N, 1]
            
            indices_torch = torch.tensor(indices_tv.cpu().numpy(), dtype=torch.int32).to(self.device)

            batch_indices = torch.full((indices_torch.shape[0], 1), batch_idx, dtype=torch.int32).to(self.device)
            indices_combined = torch.cat([batch_indices, indices_torch], dim=1)

            all_voxels.append(occupancy)
            all_indices.append(indices_combined.int())

        all_voxels = torch.cat(all_voxels, dim=0)
        all_indices = torch.cat(all_indices, dim=0)
        
        sparse_tensor = spconv.SparseConvTensor(all_voxels, all_indices, self.input_shape, self.batch_size)
        
        return sparse_tensor    

    def demo(self, epoch, prev_preds):
        epoch_start_time = time.time()
        loss_buf = []
        # self.model.eval()
        preds = None
        transformed_preds = []
        with torch.no_grad():
            print("len:", len((self.val_loader)))
            for iter, batch  in enumerate(self.val_loader):
                print(f"{iter}/{len((self.val_loader))}")
                # print("iter", iter)
                pts, gt_pts, lidar_pos, lidar_quat, _ = batch
    
                if batch is None:
                    print(f"Skipping batch {iter} because it is None")
                    continue
                
                if gt_pts.shape[0] != self.batch_size:
                    # print(f"Skipping batch {iter} because gt_pts first dimension {gt_pts.shape[0]} does not match batch size {self.batch_size}")
                    continue

                pts = pts.to(self.device)
                gt_pts = gt_pts.to(self.device)
                lidar_pos = lidar_pos.to(self.device)
                lidar_quat = lidar_quat.to(self.device)
                
                # concat
                if prev_preds is not None:
                    prev_preds = [torch.as_tensor(p) for p in prev_preds]
                    prev_preds_tensor = torch.stack(prev_preds).to(self.device)
                    pts = torch.cat((prev_preds_tensor, pts), dim=1)
                    del prev_preds_tensor
                else:
                    pts = pts.repeat_interleave(2, dim=0)
                    pts = pts.view(self.batch_size, -1, 3)
                    
                pts = torch.nan_to_num(pts, nan=0.0)
                sptensor = self.preprocess(pts)
                gt_occu = self.occupancy_grid_(gt_pts)
                pts_occu = self.occupancy_grid_(pts)

                preds, occu, probs, cm = self.model(sptensor)
                # if iter ==200:
                #     print("save tensorboard")
                
                self.tensorboard_launcher(occu, iter, [1.0, 0.0, 0.0], "Reconstrunction")
                self.tensorboard_launcher(gt_occu.dense(), iter, [0.0, 0.0, 1.0], "GT")
                self.tensorboard_launcher(pts_occu.dense(), iter, [0.0, 1.0, 0.0], "pts")
                # writer.add_scalar("example_scalar", 0.5, 0)

                # save_single_occupancy_grid_as_ply(gt_occu.dense(), 'gt_occu.ply')
                save_single_occupancy_grid_as_ply(occu, 'occu.ply')
                # save_single_occupancy_grid_as_ply(pts_occu.dense(), 'pts_occu.ply')
                # tensor_to_ply(pts[0], "pts.ply")
                # tensor_to_ply(preds[0], "preds.ply")


                # transform
                transformed_preds = []
                if preds is not None and not np.array_equal(lidar_pos, np.zeros(3, dtype=np.float32)) and not np.array_equal(lidar_quat, np.array([1, 0, 0, 0], dtype=np.float32)):
                    for i in range(min(self.batch_size, preds.size(0))):
                        transformed_pred = self.transform_point_cloud(preds[i].cpu(), lidar_pos[i].cpu(), lidar_quat[i].cpu())
                        transformed_preds.append(transformed_pred.tolist())
                        del transformed_pred

                del pts, gt_pts, lidar_pos, lidar_quat, batch, preds            
        return transformed_preds
        
    def _load_pretrain(self, pretrain):
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
        self.model.load_state_dict(new_state_dict)
        print(f"Model loaded from {pretrain}")
        
    def log(self, message):
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')
        print(message)
        

def get_parser():
    parser = argparse.ArgumentParser(description='Unsupervised Point Cloud Feature Learning')
    parser.add_argument('--model_path', type=str, default='weight2/model_epoch_best_39.pth', metavar='N',
                        help='Path to load model')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_parser()
    trainer = Train(args)  
    trainer.run()