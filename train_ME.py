
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
from config import config as cfg
from data2 import PointCloudDataset
import os
import time
import argparse
import open3d as o3d
import spconv.pytorch as spconv
import cumm.tensorview as tv
import sys
from model_ME import PointCloud3DCNN
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import cProfile
import pstats
import io
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from loss_ME import NSLoss
import gc
import logging
from collections import OrderedDict
import pickle
from os.path import join
from torch.utils.tensorboard import SummaryWriter
from open3d.visualization.tensorboard_plugin import summary
from torch.multiprocessing import Process
import joblib
import h5py
from data2 import GetTarget
import random
import MinkowskiEngine as ME
from debug import occupancy_grid_to_coords, tensor_to_ply, save_single_occupancy_grid_as_ply, profileit

BASE_LOGDIR = "./train_logs2" 
writer = SummaryWriter(join(BASE_LOGDIR, "occu"))
writer2 = SummaryWriter(join(BASE_LOGDIR, "pred"))
writer3 = SummaryWriter(join(BASE_LOGDIR, "prob"))

def pad_or_trim_cloud(pc, target_size=3000):
    n = pc.size(0)
    if n < target_size:
        padding = torch.zeros((target_size - n, 3))
        pc = torch.cat([pc, padding], dim=0) 
    elif n > target_size:
        pc = pc[:target_size, :]  
    return pc


class Train():
    def __init__(self, args):
        self.epochs = 300
        self.snapshot_interval = 10
        self.batch_size = 2
        self.device = cfg.device
        torch.cuda.set_device(self.device)
        self.model = PointCloud3DCNN(self.batch_size, in_channels=12, out_channels=12, dimension=4, n_depth=4).to(self.device)
        self.model_path = args.model_path
        if self.model_path != '':
            self._load_pretrain(args.model_path)
        
        self.h5_file_path = "lidar_data.h5"
        self.train_dataset = PointCloudDataset(self.h5_file_path, self.batch_size, 'train')
        print(f"Total valid dataset length: {len(self.train_dataset)}")
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False,pin_memory=True)
        
        self.val_dataset = PointCloudDataset(self.h5_file_path, self.batch_size, 'valid')
        print(f"Total valid dataset length: {len(self.val_dataset)}")
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,pin_memory=True)
        
        self.parameter = self.model.parameters()
        self.criterion = NSLoss().to(self.device)
        self.optimizer = optim.Adam(self.parameter, lr=0.001, betas=(0.9, 0.999), weight_decay=1e-6)
        self.weight_folder = "weight2"
        self.log_file = args.log_file if hasattr(args, 'log_file') else 'train_log2.txt'
        
        self.min_coord_range_zyx = torch.tensor([-1.0, -3.0, -3.0])
        self.max_coord_range_zyx = torch.tensor([1.5, 3.0, 3.0])
        
        self.voxel_size = torch.tensor([0.05, 0.05, 0.05]).to(self.device)
        self.vsize_xyz=[0.05, 0.05, 0.05]
        self.coors_range_xyz=[-3, -3, -1, 3, 3, 1.5]
        self.input_shape = (50, 120, 120, 2)
        
        self.train_target_dir = "train_"
        self.train_get_target = GetTarget(self.train_target_dir)
        self.valid_target_dir = "valid_"
        self.valid_get_target = GetTarget(self.valid_target_dir)
        self.train_taget_loader = torch.utils.data.DataLoader(self.train_get_target, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
        self.val_taget_loader = torch.utils.data.DataLoader(self.valid_get_target, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

        self.teacher_forcing_ratio = 1.0
        self.decay_rate = 0.01
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

    def tensorboard_launcher(self, points, step, color, tag, writer):
        # points = occupancy_grid_to_coords(points)
        num_points = points.shape[0]
        colors = torch.tensor(color).repeat(num_points, 1)
        if num_points == 0:
            print(f"Warning: num_points is 0 at step {step}, skipping add_3d")
            # return
        else:
            writer.add_3d(
            tag,
            {
                "vertex_positions": points.float(), # (N, 3)
                "vertex_colors": colors.float()  # (N, 3)
            },
            step)
        del points, colors
        torch.cuda.empty_cache()
    def run(self):
        self.train_hist = {
            'train_loss': [],
            'val_loss': [],
            'per_epoch_time': [],
            'total_time': []
        }
        self.val_hist = {'per_epoch_time': [], 'val_loss': []}
        best_loss = 1000000000
        print('Training start!!')
        start_time = time.time()

        self.model.train()

        start_epoch = 0
        for epoch in range(start_epoch, self.epochs):
            train_loss, epoch_time = self.train_epoch(epoch)
            writer2.add_scalar("Loss/train", train_loss, epoch)
            # val_loss = self.validation_epoch(epoch)
            # writer2.add_scalar("Loss/valid", val_loss, epoch)

            if len(self.train_taget_loader) != len(self.train_loader):
                print("Regenerate train loader")
                self.train_get_target = GetTarget(self.train_target_dir)
                self.train_taget_loader = torch.utils.data.DataLoader(self.train_get_target, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
            # if len(self.val_taget_loader) != len(self.val_loader):
            #     print("Regenerate valid loader")
            #     self.valid_get_target = GetTarget(self.valid_target_dir)
            #     self.val_taget_loader = torch.utils.data.DataLoader(self.valid_get_target, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

            if (epoch+1) % 30 == 0:
                self.teacher_forcing_ratio = max(0.0, self.teacher_forcing_ratio - self.decay_rate)

            # save snapeshot
            if (epoch + 1) % self.snapshot_interval == 0:
                self._snapshot(epoch + 1)
                if train_loss < best_loss:
                    best_loss = train_loss
                    self._snapshot('best_{}'.format(epoch))
            log_message = f"Epoch [{epoch + 1}/{self.epochs}] - Train Loss: {train_loss:.4f}, Time: {epoch_time:.4f}s"
            self.log(log_message)
        # finish all epoch
        self._snapshot(epoch + 1)
        if train_loss < best_loss:
            best_loss = train_loss
            self._snapshot('best')
        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.epochs, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")
        

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
            vsize_xyz=self.vsize_xyz,
            coors_range_xyz=self.coors_range_xyz,
            num_point_features=self.model.num_point_features,
            max_num_voxels=600000,
            max_num_points_per_voxel=self.model.max_num_points_per_voxel
            )

        batch_size = pc.shape[0]
        all_voxels, all_indices = [], []

        for batch_idx in range(batch_size):
            pc_single = pc[batch_idx]
            pc_single = tv.from_numpy(pc_single.detach().cpu().numpy())

            voxels_tv, indices_tv, num_p_in_vx_tv = gen.point_to_voxel_hash(pc_single.cuda())
            voxels_torch = torch.tensor(voxels_tv.cpu().numpy(), dtype=torch.float32).to(self.device)
            indices_torch = torch.tensor(indices_tv.cpu().numpy(), dtype=torch.int32).to(self.device)
            ## sub-voxel feature
            indices_torch_trans = indices_torch[:, [2, 1, 0]] 
            voxel_centers = (indices_torch_trans.float() * torch.tensor([0.05, 0.05, 0.05]).to(self.device)) + torch.tensor([-3.0, -3.0, -1.0]).to(self.device)
            # tensor_to_ply(voxel_centers[0].view(-1, 3), "voxel_centers.ply")
            t_values = voxels_torch[:, :, 3] 
            voxels_torch = voxels_torch[:, :, :3]
            relative_pose = torch.where(voxels_torch == 0, torch.tensor(0.0).to(voxels_torch.device), (voxels_torch - voxel_centers.unsqueeze(1)) / self.voxel_size)
            relative_pose = torch.cat([relative_pose, t_values.unsqueeze(-1)], dim=2)
            voxels_flatten = relative_pose.view(-1, 4 * self.model.max_num_points_per_voxel)

            valid = num_p_in_vx_tv.cpu().numpy() > 0
            indices_torch = indices_torch[valid]
            ## not using abs -> only half of lidar remain     
            
            mask_0 = (t_values == 0).any(dim=1)
            mask_1 = (t_values == 1).any(dim=1)

            indices_0 = indices_torch[mask_0].to(self.device)
            indices_1 = indices_torch[mask_1].to(self.device)
            
            t0 = torch.zeros((indices_0.shape[0], 1), dtype=torch.int32).to(self.device)
            t1 = torch.ones((indices_1.shape[0], 1), dtype=torch.int32).to(self.device)
            batch_indices_0 = torch.full((indices_0.shape[0], 1), batch_idx, dtype=torch.int32).to(self.device)
            batch_indices_1 = torch.full((indices_1.shape[0], 1), batch_idx, dtype=torch.int32).to(self.device)
            indices_combined_0 = torch.cat([batch_indices_0, indices_0, t0], dim=1)
            indices_combined_1 = torch.cat([batch_indices_1, indices_1, t1], dim=1)
            indices_combined = torch.cat([indices_combined_0, indices_combined_1], dim=0)  # [N_total, 4]
            voxels_flatten = torch.cat([voxels_flatten[mask_0], voxels_flatten[mask_1]], dim=0)  
            del indices_combined_0, indices_combined_1,batch_indices_0, batch_indices_1, t0, t1, indices_0, indices_1
            # batch_indices = torch.full((indices_torch.shape[0], 1), batch_idx, dtype=torch.int32).to(self.device)
            # indices_combined = torch.cat([batch_indices, indices_torch], dim=1)


            all_voxels.append(voxels_flatten) # N X (self.model.num_point_features X self.model.max_num_points_per_voxel)
            all_indices.append(indices_combined.int()) # N x (batch, D, W, H, t)
            
        features_tc = torch.cat(all_voxels, dim=0)
        indices_tc = torch.cat(all_indices, dim=0)
        sparse_tensor = ME.SparseTensor(features=features_tc,
                                        coordinates=indices_tc,
                                        quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)
        
        
        del voxels_torch, indices_torch, relative_pose, voxel_centers, all_voxels, all_indices
        return sparse_tensor
    
    def coords_to_seq(self, coord):
        batch_size = coord.shape[0]
        all_indices = []

        for batch_idx in range(batch_size):
            
            pc_single = coord[batch_idx]
            all_indices.append(pc_single)
            
            
        return all_indices

    def occupancy_grid_(self, pc):
        from spconv.utils import Point2VoxelGPU3d
        from spconv.pytorch.utils import PointToVoxel
        # Voxel generator
        gen = Point2VoxelGPU3d(
            vsize_xyz=self.vsize_xyz,
            coors_range_xyz=self.coors_range_xyz,
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

        features_tc = torch.cat(all_voxels, dim=0)
        indices_tc = torch.cat(all_indices, dim=0)
        
        sparse_tensor = ME.SparseTensor(features=features_tc,
                                        coordinates=indices_tc,
                                        quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)
        del indices_torch, all_voxels, all_indices

        return sparse_tensor , indices_tc
    

    def occupancy_grid(self, pc, grid_size, voxel_size, device='cpu'):

        batch_size = pc.shape[0]

        # Initialize the occupancy grid with zeros
        occupancy_grid = torch.zeros((batch_size, *grid_size), dtype=torch.float32)

        for batch_idx in range(batch_size):

            pc_single = pc[batch_idx]
            pc_single = pc_single[:, [2, 1, 0]]

            # Filtering point clouds (remove O.O.D points)
            filtered_pc = self.filter_point_cloud(pc_single, self.min_coord_range_zyx, self.max_coord_range_zyx)

            # Consider the minimum value as 0 index.
            voxel_indices = torch.div(filtered_pc - self.min_coord_range_zyx.to(filtered_pc.device) - 1e-4, voxel_size.to(filtered_pc.device)).long()
            voxel_indices[:, 0] = voxel_indices[:, 0].clamp(0, grid_size[0] - 1)
            voxel_indices[:, 1] = voxel_indices[:, 1].clamp(0, grid_size[1] - 1)
            voxel_indices[:, 2] = voxel_indices[:, 2].clamp(0, grid_size[2] - 1)
            # Increment the occupancy grid at the corresponding indices
            occupancy_grid[batch_idx, voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = 1.0

        occupancy_grid = occupancy_grid.unsqueeze(-1).to(device)
        occupancy_grid = occupancy_grid.permute(0, 4, 1, 2, 3)
        # tensor_to_ply(occupancy_grid_to_coords(occupancy_grid), "occupancy.ply")

        return occupancy_grid

    def filter_point_cloud(self, pc, min_coord_range_zyx, max_coord_range_zyx):
        """
        Filter out points from the point cloud that are outside the specified coordinate range.

        Args:
            pc (torch.Tensor): Point cloud data of shape (N, 3).
            min_coord_range_zyx (torch.Tensor): Minimum coordinate values for x, y, z.
            max_coord_range_zyx (torch.Tensor): Maximum coordinate values for x, y, z.

        Returns:
            torch.Tensor: Filtered point cloud with points within the specified range.
        """
        # Create masks for each coordinate
        mask_x = (pc[..., 0] >= min_coord_range_zyx[0]) & (pc[..., 0] <= max_coord_range_zyx[0])
        mask_y = (pc[..., 1] >= min_coord_range_zyx[1]) & (pc[..., 1] <= max_coord_range_zyx[1])
        mask_z = (pc[..., 2] >= min_coord_range_zyx[2]) & (pc[..., 2] <= max_coord_range_zyx[2])

        # filter redundant points which allocated in the (0,0,0) coordinate
        mask_center = torch.logical_not((pc[:] == torch.Tensor([0, 0, 0]).to(pc.device)).all(dim=1))


        # Combine masks to get the final mask
        mask = mask_x & mask_y & mask_z & mask_center

        # Apply the mask to filter out points that are out of range
        filtered_pc = pc[mask]

        return filtered_pc
        
    def get_target(self, occupancy_grid, cm, idx):
        batch_size = occupancy_grid.size(0)
        n = cm.size(1)

        x_indices = cm[:, :, 0].to(occupancy_grid.device)
        y_indices = cm[:, :, 1].to(occupancy_grid.device)
        z_indices = cm[:, :, 2].to(occupancy_grid.device)

        X, Y, Z = occupancy_grid.shape[2], occupancy_grid.shape[3], occupancy_grid.shape[4]

        x_indices = torch.clamp(x_indices, min=0, max=X-1)
        y_indices = torch.clamp(y_indices, min=0, max=Y-1)
        z_indices = torch.clamp(z_indices, min=0, max=Z-1)

        batch_indices = torch.arange(batch_size).view(-1, 1).expand(-1, n).to(occupancy_grid.device)
        occupancy_values = occupancy_grid[batch_indices, 0, x_indices, y_indices, z_indices]
        return occupancy_values
                
    # @profileit
    def train_epoch(self, epoch):
        epoch_start_time = time.time()
        loss_buf = []
        self.model.train()
        preds = None
        prev_preds = []
        cham_loss_buf, occu_loss_buf, cls_losses_buf = [], [], []
        with tqdm(total=len(self.train_loader), desc=f"Epoch {epoch + 1}/{self.epochs}", unit="batch") as pbar:
            for iter, (batch) in enumerate(self.train_loader):
                # print("1", len(self.train_loader))
                # print("2", len(self.train_taget_loader))
                if batch is None:
                    print(f"Skipping batch {iter} because it is None")
                    pbar.update(1)
                    continue
                
                pts, gt_pts, lidar_pos, lidar_quat, data_file_path = batch
                if gt_pts.shape[0] != self.batch_size:
                    print(f"Skipping batch {iter} because gt_pts first dimension {gt_pts.shape[0]} does not match batch size {self.batch_size}")
                    pbar.update(1)
                    continue
                
                pts = pts.to(self.device)
                gt_pts = gt_pts.to(self.device)
                lidar_pos = lidar_pos.to(self.device)
                lidar_quat = lidar_quat.to(self.device)
                pts_occu, _ = self.occupancy_grid_(pts)
                # self.tensorboard_launcher(pts[0], iter, [1.0, 0.0, 1.0], "pts_iter")

                # if len(self.train_taget_loader) != len(self.train_loader):
                #     print(f"calculate : not matching {len(self.train_taget_loader)} & {len(self.train_loader)}")
                #     output_directory = "train_"
                #     file_path = os.path.join(output_directory, f'{iter}.joblib')
                #     occupancy_grids = []
                #     occupancy_grids.append(self.occupancy_grid(gt_pts, (7, 15, 15), (self.max_coord_range_zyx - self.min_coord_range_zyx) / torch.tensor([5, 14, 14], dtype=torch.float32)))
                #     occupancy_grids.append(self.occupancy_grid(gt_pts, (13, 30, 30), (self.max_coord_range_zyx - self.min_coord_range_zyx) / torch.tensor([11, 29, 29], dtype=torch.float32)))
                #     occupancy_grids.append(self.occupancy_grid(gt_pts, (25, 60, 60), (self.max_coord_range_zyx - self.min_coord_range_zyx) / torch.tensor([24, 59, 59], dtype=torch.float32)))
                #     occupancy_grids.append(self.occupancy_grid(gt_pts, (50, 120, 120), (self.max_coord_range_zyx - self.min_coord_range_zyx) / torch.tensor([50, 120, 120], dtype=torch.float32)))
                #     os.makedirs(output_directory, exist_ok=True)
                #     joblib.dump(occupancy_grids, file_path)

                # concat
                if len(prev_preds) > 0:
                    prev_preds = [torch.as_tensor(p) for p in prev_preds]
                    prev_preds_tensor = torch.stack(prev_preds).to(self.device)
                    ## 4D Convolution
                    batch_size, n, _ = pts.shape
                    zeros = torch.zeros((batch_size, n, 1), device=pts.device)
                    pts = torch.cat([pts, zeros], dim=2)
                    batch_size, n, _ = prev_preds_tensor.shape
                    ones = torch.ones((batch_size, n, 1), device=pts.device)
                    prev_preds_tensor = torch.cat([prev_preds_tensor, ones], dim=2)
                    pts = torch.cat((prev_preds_tensor, pts), dim=1)
                    # tensor_to_ply(prev_preds_tensor[0], f"transformed_pred_{iter}.ply")
                    # self.tensorboard_launcher(prev_preds_tensor[0], iter, [1.0, 0.0, 0.0], "transformed_pts")
                    # tensor_to_ply(gt_pts[0], f"pts_{iter}.ply")
                    # self.tensorboard_launcher(gt_pts[0], iter, [1.0, 0.0, 1.0], "gt_pts")
                    del prev_preds, prev_preds_tensor
                    prev_preds = []
                else:
                    # pts = pts.repeat_interleave(2, dim=0)
                    pts = pts.view(self.batch_size, -1, 3)
                    ## 4D Convolution
                    batch_size, n, _ = pts.shape
                    zeros = torch.zeros((batch_size, n, 1), device=pts.device)
                    pts = torch.cat([pts, zeros], dim=2)
                    
                pts = torch.nan_to_num(pts, nan=0.0)
                sptensor = self.preprocess(pts)
                _,indices= self.occupancy_grid_(gt_pts)
                
                cm = sptensor.coordinate_manager
                zeros = torch.zeros((indices.size(0), 1), device=pts.device)
                gt_pts_with_t = torch.cat([indices, zeros], dim=1)
                target_key, _ = cm.insert_and_map(
                    gt_pts_with_t.int(),
                    string_id="target",
                )
                self.optimizer.zero_grad()
                preds, occu, gt_occu, out = self.model(sptensor, target_key, True)
                # self.tensorboard_launcher(occu[0], iter, [1.0, 0.0, 0.0], "Reconstrunction_iter", writer)
                # self.tensorboard_launcher(gt_occu[0], iter, [0.0, 0.0, 1.0], "pts_iter", writer)

                # self.tensorboard_launcher(occupancy_grid_to_coords(occu), iter, [1.0, 0.0, 0.0], "Reconstrunction_iter", writer)
                # self.tensorboard_launcher(occupancy_grid_to_coords(pts_occu.dense()), iter, [0.0, 0.0, 1.0], "pts_iter", writer)

                # ## check preprocess & occupancy grid
                # # self.tensorboard_launcher(occupancy_grid_to_coords(sptensor.dense()[..., 0]), iter, [1.0, 0.0, 0.0], "sptensor")
                # # self.tensorboard_launcher(occupancy_grid_to_coords(gt_occu.dense()), iter, [0.0, 0.0, 1.0], "GT_iter")
                # # self.tensorboard_launcher(occupancy_grid_to_coords(occupancy_grids[0].squeeze(0)), iter, [1.0, 0.0, 0.0], "target")
                # # self.tensorboard_launcher(occupancy_grid_to_coords(occupancy_grids[1].squeeze(0)), iter, [1.0, 1.0, 0.0], "target2")
                # # self.tensorboard_launcher(occupancy_grid_to_coords(occupancy_grids[2].squeeze(0)), iter, [1.0, 1.0, 0.0], "target3")
                # # self.tensorboard_launcher(occupancy_grid_to_coords(occupancy_grids[3].squeeze(0)), iter, [1.0, 1.0, 0.0], "target4")
                if iter == 1:
                    print("tensorboard_launcher")
                    # min_coord = torch.tensor([0, 0, 0, 0], dtype=torch.int32)
                    # dense_tensor = out.dense(min_coordinate=min_coord)
                    self.tensorboard_launcher(occupancy_grid_to_coords(out), epoch, [1.0, 0.0, 0.0], "Reconstrunction_train", writer)
                #     self.tensorboard_launcher(occupancy_grid_to_coords(decoding), epoch, [1.0, 0.0, 0.0], "decoding", writer)
                #     self.tensorboard_launcher(occupancy_grid_to_coords(gt_occu.dense()), epoch, [0.0, 0.0, 1.0], "GT_train", writer)
                #     self.tensorboard_launcher(occupancy_grid_to_coords(pts_occu.dense()), epoch, [0.0, 1.0, 1.0], "pts_train", writer)
                #     self.tensorboard_launcher(preds[0].float(), epoch, [1.0, 0.0, 0.0], "preds", writer2)
                #     self.tensorboard_launcher(gt_pts[0].float(), epoch, [0.0, 0.0, 1.0], "gt_pts", writer2)
                
                loss = self.criterion(occu, gt_occu, preds, gt_pts)
                # cham_loss_buf.append(cham_loss.item())
                # occu_loss_buf.append(occu_loss.item())
                # cls_losses_buf.append(cls_losses.item())
                loss.backward()

                # for name, param in self.model.named_parameters():
                #     print(f"Layer: {name} | requires_grad: {param.requires_grad}")
                #     if param.grad is not None:
                #         print(f"Layer: {name} | Gradient mean: {param.grad.mean()}")
                #     else:
                #         print(f"Layer: {name} | No gradient calculated!")
                self.optimizer.step()
                loss_buf.append(loss.item())
                
                # transform
                if preds is not None and not np.array_equal(lidar_pos, np.zeros(3, dtype=np.float32)) and not np.array_equal(lidar_quat, np.array([1, 0, 0, 0], dtype=np.float32)):
                    for i in range(min(self.batch_size, preds.size(0))):
                        if random.random() < self.teacher_forcing_ratio:
                            input_data = gt_pts[i].cpu()
                        else:
                            input_data = preds[i].cpu()
                        transformed_pred = self.transform_point_cloud(input_data, lidar_pos[i].cpu(), lidar_quat[i].cpu())
                        transformed_pred = pad_or_trim_cloud(transformed_pred, target_size=3000)
                        prev_preds.append(transformed_pred)
                        del transformed_pred
                # if epoch % 20 == 0:
                #     self.teacher_forcing_ratio = max(0.0, self.teacher_forcing_ratio - self.decay_rate)

                # empty memory
                del pts, gt_pts, lidar_pos, lidar_quat, batch, preds, loss, occu, sptensor, gt_occu
                torch.cuda.empty_cache()
                pbar.set_postfix(train_loss=np.mean(loss_buf) if loss_buf else 0)
                pbar.update(1)
        torch.cuda.synchronize()
        epoch_time = time.time() - epoch_start_time
        self.train_hist['per_epoch_time'].append(epoch_time)
        self.train_hist['train_loss'].append(np.mean(loss_buf))
        # return np.mean(loss_buf), epoch_time, np.mean(cham_loss_buf), np.mean(occu_loss_buf), np.mean(cls_losses)
        return np.mean(loss_buf), epoch_time
    # @profileit
    def validation_epoch(self, epoch):
        epoch_start_time = time.time()
        loss_buf = []
        self.model.eval()
        preds = None
        prev_preds = []
        with torch.no_grad():
            with tqdm(total=len(self.val_loader), desc=f"Validation {epoch + 1}/{self.epochs}", unit="batch") as pbar:
                # print("1", len(self.val_taget_loader))
                # print("2", len(self.val_loader))
                for iter, (batch) in enumerate((self.val_loader)):
                    if batch is None:
                        print(f"Skipping batch {iter} because it is None")
                        pbar.update(1)
                        continue

                    pts, gt_pts, lidar_pos, lidar_quat, _ = batch
                    if gt_pts.shape[0] != self.batch_size:
                        print(f"Skipping batch {iter} because gt_pts first dimension {gt_pts.shape[0]} does not match batch size {self.batch_size}")
                        pbar.update(1)
                        continue
                        
                    pts = pts.to(self.device)
                    gt_pts = gt_pts.to(self.device)
                    lidar_pos = lidar_pos.to(self.device)
                    lidar_quat = lidar_quat.to(self.device)
           
                    # if len(self.val_taget_loader) != len(self.val_loader):
                    #     print(f"calculate : not matching {len(self.val_taget_loader)} & {len(self.val_loader)}")
                    #     output_directory = "valid_"
                    #     file_path = os.path.join(output_directory, f'{iter}.joblib')
                    #     occupancy_grids = []
                    #     occupancy_grids.append(self.occupancy_grid(gt_pts, (7, 15, 15), (self.max_coord_range_zyx - self.min_coord_range_zyx) / torch.tensor([5, 14, 14], dtype=torch.float32)))
                    #     occupancy_grids.append(self.occupancy_grid(gt_pts, (13, 30, 30), (self.max_coord_range_zyx - self.min_coord_range_zyx) / torch.tensor([11, 29, 29], dtype=torch.float32)))
                    #     occupancy_grids.append(self.occupancy_grid(gt_pts, (25, 60, 60), (self.max_coord_range_zyx - self.min_coord_range_zyx) / torch.tensor([24, 59, 59], dtype=torch.float32)))
                    #     occupancy_grids.append(self.occupancy_grid(gt_pts, (50, 120, 120), (self.max_coord_range_zyx - self.min_coord_range_zyx) / torch.tensor([50, 120, 120], dtype=torch.float32)))
                    #     os.makedirs(output_directory, exist_ok=True)
                    #     joblib.dump(occupancy_grids, file_path)


                    # concat
                    if len(prev_preds) > 0:
                        prev_preds = [torch.as_tensor(p) for p in prev_preds]
                        prev_preds_tensor = torch.stack(prev_preds).to(self.device)
                        ## 4D Convolution
                        batch_size, n, _ = pts.shape
                        zeros = torch.zeros((batch_size, n, 1), device=pts.device)
                        pts = torch.cat([pts, zeros], dim=2)
                        batch_size, n, _ = prev_preds_tensor.shape
                        ones = torch.ones((batch_size, n, 1), device=pts.device)
                        prev_preds_tensor = torch.cat([prev_preds_tensor, ones], dim=2)
                        pts = torch.cat((prev_preds_tensor, pts), dim=1)
                        del prev_preds, prev_preds_tensor
                        prev_preds = []
                    else:
                        # pts = pts.repeat_interleave(2, dim=0)
                        pts = pts.view(self.batch_size, -1, 3)
                        ## 4D Convolution
                        batch_size, n, _ = pts.shape
                        zeros = torch.zeros((batch_size, n, 1), device=pts.device)
                        pts = torch.cat([pts, zeros], dim=2)
                        
                        
                    pts = torch.nan_to_num(pts, nan=0.0)
                    sptensor = self.preprocess(pts)
                    _,indices = self.occupancy_grid_(gt_pts)
                    
                    cm = sptensor.coordinate_manager
                    zeros = torch.zeros((indices.size(0), 1), device=pts.device)
                    gt_pts_with_t = torch.cat([indices, zeros], dim=1)
                    target_key, _ = cm.insert_and_map(
                        gt_pts_with_t.int(),
                        string_id="target",
                    )

                    
                    preds, occu, gt_occu, _ = self.model(sptensor, target_key, False)              
                    
                    # self.tensorboard_launcher(occupancy_grid_to_coords(occu), iter, [1.0, 0.0, 0.0], "Reconstrunction_iter")
                    # self.tensorboard_launcher(occupancy_grid_to_coords(gt_occu.dense()), iter, [0.0, 0.0, 1.0], "GT_iter")

                    # if iter == 120:
                    #     print("tensorboard_launcher")
                    #     self.tensorboard_launcher(occupancy_grid_to_coords(occu), epoch, [1.0, 0.0, 0.0], "Reconstrunction_valid", writer)
                    #     self.tensorboard_launcher(occupancy_grid_to_coords(gt_occu.dense()), epoch, [0.0, 0.0, 1.0], "GT_valid", writer)
                        
                    #     self.tensorboard_launcher(occupancy_grid_to_coords(sptensor.dense()), epoch, [0.0, 1.0, 1.0], "pts_valid")


                    # ## get_target
                    # idx = 0
                    # gt_probs = []
                    # for idx in range(len(probs)):
                    #     gt_prob = self.get_target(occupancy_grids[idx].squeeze(0), cm[idx], idx)
                    #     gt_probs.append(gt_prob)
                    # del gt_prob

                    
                    loss = self.criterion(occu, gt_occu, preds, gt_pts)               
                    loss_buf.append(loss.item())
                    
                    # transform
                    if preds is not None and not np.array_equal(lidar_pos, np.zeros(3, dtype=np.float32)) and not np.array_equal(lidar_quat, np.array([1, 0, 0, 0], dtype=np.float32)):
                        for i in range(min(self.batch_size, preds.size(0))):
                            transformed_pred = self.transform_point_cloud(preds[i].cpu(), lidar_pos[i].cpu(), lidar_quat[i].cpu())
                            prev_preds.append(transformed_pred)
                            del transformed_pred

                    # empty memory
                    del pts, gt_pts, lidar_pos, lidar_quat, batch, preds, loss, occu, sptensor, gt_occu
                    torch.cuda.empty_cache()
                    pbar.set_postfix(val_loss=np.mean(loss_buf) if loss_buf else 0)
                    pbar.update(1)                
            torch.cuda.synchronize()
            epoch_time = time.time() - epoch_start_time
            self.val_hist['per_epoch_time'].append(epoch_time)
            self.val_hist['val_loss'].append(np.mean(loss_buf))
            val_loss = np.nanmean(loss_buf) if loss_buf else 0
            return val_loss


    def _snapshot(self, epoch):
        if not os.path.exists(self.weight_folder):
            os.makedirs(self.weight_folder) 
        snapshot_filename = os.path.join(self.weight_folder, f"model_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, snapshot_filename)
        print(f"Snapshot saved to {snapshot_filename}")


        
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
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Path to load model')
    args = parser.parse_args()
    return args

def main_worker(args):
    trainer = Train(args)  
    trainer.run()

def main():
    args = get_parser()
    main_worker(args) 
if __name__ == "__main__":
    main()
