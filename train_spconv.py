
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
from config import config as cfg
from data import PointCloudDataset
import os
import time
import argparse
import open3d as o3d
import spconv.pytorch as spconv
import cumm.tensorview as tv
import sys
from model_spconv import PointCloud3DCNN
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
from loss import NSLoss
import gc
import logging
from collections import OrderedDict
import pickle
from os.path import join
from torch.utils.tensorboard import SummaryWriter
from open3d.visualization.tensorboard_plugin import summary
from torch.multiprocessing import Process

BASE_LOGDIR = "./train_logs" 
writer = SummaryWriter(join(BASE_LOGDIR, "visualize"))

def occupancy_grid_to_coords(occupancy_grid):
    _, H, W, D = occupancy_grid.shape
    occupancy_grid = occupancy_grid.squeeze(0)
    indices = torch.nonzero(occupancy_grid, as_tuple=False)  
    return indices
def tensor_to_ply(tensor, filename):
    print("tensor", tensor.shape)
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

class Train():
    def __init__(self, args):
        self.epochs = 300
        self.snapshot_interval = 10
        self.batch_size = 16
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(self.device)
        self.model = PointCloud3DCNN(self.batch_size).to(self.device)
        self.model_path = args.model_path
        if self.model_path != '':
            self._load_pretrain(args.model_path)
        
        self.train_path = 'dataset/train'
        self.train_dataset = PointCloudDataset(self.train_path)
        print(f"Total train dataset length: {len(self.train_dataset)}")
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8, pin_memory=True)
        
        self.val_path = 'dataset/valid'
        self.val_dataset = PointCloudDataset(self.val_path)
        print(f"Total valid dataset length: {len(self.val_dataset)}")
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8, pin_memory=True)
        
        self.parameter = self.model.parameters()
        self.criterion = NSLoss().to(self.device)
        self.optimizer = optim.Adam(self.parameter, lr=0.001, betas=(0.9, 0.999), weight_decay=1e-6)
        self.weight_folder = "weight2"
        self.log_file = args.log_file if hasattr(args, 'log_file') else 'train_log2.txt'
        self.input_shape = (50, 120, 120)
        
        self.min_coord_range_xyz = torch.tensor([-3.0, -3.0, -3.0])
        self.max_coord_range_xyz = torch.tensor([3.0, 3.0, 3.0])
        
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

    def tensorboard_launcher(self, points, step, color, tag):
        points = occupancy_grid_to_coords(points[0])
        num_points = points.shape[0]
        colors = torch.tensor(color).repeat(num_points, 1)
        writer.add_3d(
        tag,
        {
            "vertex_positions": points.float(), # (N, 3)
            "vertex_colors": colors.float()  # (N, 3)
        },
        step)
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
        prev_preds = None
        prev_preds_val = None

        start_epoch = 0
        for epoch in range(start_epoch, self.epochs):
            train_loss, epoch_time, prev_preds = self.train_epoch(epoch, prev_preds)
            writer.add_scalar("Loss/train", train_loss, epoch)
            val_loss, prev_preds_val = self.validation_epoch(epoch, prev_preds_val)
            writer.add_scalar("Loss/valid", train_loss, epoch)

            # save snapeshot
            if (epoch + 1) % self.snapshot_interval == 0:
                self._snapshot(epoch + 1)
                if train_loss < best_loss:
                    best_loss = train_loss
                    self._snapshot('best_{}'.format(epoch))
            log_message = f"Epoch [{epoch + 1}/{self.epochs}] - Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Time: {epoch_time:.4f}s"
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
    

    def occupancy_grid(self, pc, grid_size, voxel_size, device='cpu'):

        batch_size = pc.shape[0]

        # Initialize the occupancy grid with zeros

        occupancy_grid = torch.zeros((batch_size, *grid_size), dtype=torch.float32)

        for batch_idx in range(batch_size):

            pc_single = pc[batch_idx]
            pc_single = pc_single[:, [2, 1, 0]]


            # Filtering point clouds (remove O.O.D points)
            filtered_pc = self.filter_point_cloud(pc_single, self.min_coord_range_xyz, self.max_coord_range_xyz)

            # Consider the minimum value as 0 index.
            voxel_indices = torch.div(filtered_pc - self.min_coord_range_xyz.to(filtered_pc.device) - 1e-4, voxel_size.to(filtered_pc.device)).long()

            # Increment the occupancy grid at the corresponding indices
            occupancy_grid[batch_idx, voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = 1.0

        occupancy_grid = occupancy_grid.unsqueeze(-1).to(device)
        occupancy_grid = occupancy_grid.permute(0, 4, 1, 2, 3)
        # print("occupancy_grid", occupancy_grid.shape)
        # save_single_occupancy_grid_as_ply(occupancy_grid, 'occupancy_grid.ply')

        return occupancy_grid

    def filter_point_cloud(self, pc, min_coord_range_xyz, max_coord_range_xyz):
        """
        Filter out points from the point cloud that are outside the specified coordinate range.

        Args:
            pc (torch.Tensor): Point cloud data of shape (N, 3).
            min_coord_range_xyz (torch.Tensor): Minimum coordinate values for x, y, z.
            max_coord_range_xyz (torch.Tensor): Maximum coordinate values for x, y, z.

        Returns:
            torch.Tensor: Filtered point cloud with points within the specified range.
        """
        # Create masks for each coordinate
        mask_x = (pc[..., 0] >= min_coord_range_xyz[0]) & (pc[..., 0] <= max_coord_range_xyz[0])
        mask_y = (pc[..., 1] >= min_coord_range_xyz[1]) & (pc[..., 1] <= max_coord_range_xyz[1])
        mask_z = (pc[..., 2] >= min_coord_range_xyz[2]) & (pc[..., 2] <= max_coord_range_xyz[2])

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
                

    @profileit
    def train_epoch(self, epoch, prev_preds):
        epoch_start_time = time.time()
        loss_buf = []
        self.model.train()
        preds = None
        transformed_preds = []
        with tqdm(total=len(self.train_loader), desc=f"Epoch {epoch + 1}/{self.epochs}", unit="batch") as pbar:
            for iter, batch  in enumerate(self.train_loader):
                if batch is None:
                    print(f"Skipping batch {iter} because it is None")
                    pbar.update(1)
                    continue
                
                pts, gt_pts, lidar_pos, lidar_quat = batch
                # tensor_to_ply(pts, f"pts_{iter}.ply")
                # tensor_to_ply(gt_pts, f"gt_{iter}.ply")

                if gt_pts.shape[0] != self.batch_size:
                    print(f"Skipping batch {iter} because gt_pts first dimension {gt_pts.shape[0]} does not match batch size {self.batch_size}")
                    pbar.update(1)
                    continue
                
                pts = pts.to(self.device)
                gt_pts = gt_pts.to(self.device)
                lidar_pos = lidar_pos.to(self.device)
                lidar_quat = lidar_quat.to(self.device)
                
                output_directory = "train_"
                file_path = os.path.join(output_directory, f'{iter}.pkl')

                ## get_target
                if os.path.exists(file_path):
                    with open(file_path, 'rb') as f:
                        occupancy_grids = pickle.load(f)
                    # print("File loaded successfully.")
                else:
                    # print(f"File '{file_path}' does not exist.")
                    occupancy_grids = []
                    torch.tensor([5, 14, 14], dtype=torch.float32)
                    occupancy_grids.append(self.occupancy_grid(gt_pts, (5, 14, 14), (self.max_coord_range_xyz - self.min_coord_range_xyz) / torch.tensor([5, 14, 14], dtype=torch.float32)))
                    occupancy_grids.append(self.occupancy_grid(gt_pts, (11, 29, 29), (self.max_coord_range_xyz - self.min_coord_range_xyz) / torch.tensor([11, 29, 29], dtype=torch.float32)))
                    occupancy_grids.append(self.occupancy_grid(gt_pts, (24, 59, 59), (self.max_coord_range_xyz - self.min_coord_range_xyz) / torch.tensor([24, 59, 59], dtype=torch.float32)))
                    occupancy_grids.append(self.occupancy_grid(gt_pts, (50, 120, 120), (self.max_coord_range_xyz - self.min_coord_range_xyz) / torch.tensor([50, 120, 120], dtype=torch.float32)))
                    os.makedirs(output_directory, exist_ok=True)
                    with open(file_path, 'wb') as f:
                        pickle.dump(occupancy_grids, f)
                
                # print("occupancy_grids", occupancy_grids[0].shape)
                # concat
                if prev_preds is not None:
                    prev_preds = [torch.as_tensor(p) for p in prev_preds]
                    # print(f"prev_preds-before cat : ", prev_preds[:5])
                    prev_preds_tensor = torch.stack(prev_preds).to(self.device)
                    pts = torch.cat((prev_preds_tensor, pts), dim=1)
                    # print(f"prev_preds-before cat : ", prev_preds[:5])
                    del prev_preds_tensor
                else:
                    pts = pts.repeat_interleave(2, dim=0)
                    pts = pts.view(self.batch_size, -1, 3)
                pts = torch.nan_to_num(pts, nan=0.0)
                sptensor = self.preprocess(pts)
                gt_occu = self.occupancy_grid_(gt_pts)

                self.optimizer.zero_grad()
                preds, occu, probs, cm = self.model(sptensor)
                self.tensorboard_launcher(occu, iter, [1.0, 0.0, 0.0], "reconstrunction")
                self.tensorboard_launcher(gt_occu.dense(), iter, [0.0, 0.0, 1.0], "GT")

                # save_single_occupancy_grid_as_ply(gt_occu.dense(), 'gt_occu.ply')
                # save_single_occupancy_grid_as_ply(occu, 'occu.ply')
                
                ## get_target
                idx = 0
                gt_probs = []
                for idx in range(len(probs)):
                    # tensor_to_ply(cm[idx][0], "cm.ply")
                    gt_prob = self.get_target(occupancy_grids[idx], cm[idx], idx)
                    gt_probs.append(gt_prob)
                
                loss = self.criterion(preds, occu, gt_pts, gt_occu.dense(), probs, gt_probs)

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
                transformed_preds = []
                if preds is not None and not np.array_equal(lidar_pos, np.zeros(3, dtype=np.float32)) and not np.array_equal(lidar_quat, np.array([1, 0, 0, 0], dtype=np.float32)):
                    for i in range(min(self.batch_size, gt_pts.size(0))):
                        transformed_pred = self.transform_point_cloud(gt_pts[i].cpu(), lidar_pos[i].cpu(), lidar_quat[i].cpu())
                        transformed_preds.append(transformed_pred)   
                        
                        del transformed_pred
                        
                # # for debugging
                # pts = pts.view(self.batch_size, -1, 3)
                # if not np.array_equal(lidar_pos, np.zeros(3, dtype=np.float32)) and not np.array_equal(lidar_quat, np.array([1, 0, 0, 0], dtype=np.float32)):
                #     for i in range(min(self.batch_size, pts.size(0))):
                #         transformed_pred = self.transform_point_cloud(pts[i].cpu(), lidar_pos[i].cpu(), lidar_quat[i].cpu())
                #         transformed_preds.append(transformed_pred.tolist())   
                #         # tensor_to_ply(transformed_preds, f"transformed_{iter}")
                #         del transformed_pred
                #         # gc.collect()
                #         torch.cuda.empty_cache()
                # transformed_preds = torch.tensor(transformed_preds).to(self.device)  
                # tensor_to_ply(transformed_preds, f"transformed_{iter}.ply")

                                
                # empty memory
                del pts, gt_pts, lidar_pos, lidar_quat, batch, preds, loss
                pbar.set_postfix(train_loss=np.mean(loss_buf) if loss_buf else 0)
                pbar.update(1)
        torch.cuda.synchronize()
        # memory logging
        allocated_final = torch.cuda.memory_allocated()
        reserved_final = torch.cuda.memory_reserved()
        logging.info(f"Epoch {epoch}:")
        logging.info(f"train -Memory allocated after deleting tensor and emptying cache: {allocated_final / (1024 ** 2):.2f} MB")
        logging.info(f"train - Reserved after deleting tensor and emptying cache: {reserved_final / (1024 ** 2):.2f} MB")

        epoch_time = time.time() - epoch_start_time
        self.train_hist['per_epoch_time'].append(epoch_time)
        self.train_hist['train_loss'].append(np.mean(loss_buf))
        return np.mean(loss_buf), epoch_time, transformed_preds
    
    def validation_epoch(self, epoch,prev_preds):
        epoch_start_time = time.time()
        loss_buf = []
        self.model.eval()
        preds = None
        transformed_preds = []
        with torch.no_grad():
            with tqdm(total=len(self.val_loader), desc=f"Validation {epoch + 1}/{self.epochs}", unit="batch") as pbar:
                for iter, batch  in enumerate(self.val_loader):
                    if batch is None:
                        print(f"Skipping batch {iter} because it is None")
                        pbar.update(1)
                        continue

                    pts, gt_pts, lidar_pos, lidar_quat = batch
                    if gt_pts.shape[0] != self.batch_size:
                        # print(f"Skipping batch {iter} because gt_pts first dimension {gt_pts.shape[0]} does not match batch size {self.batch_size}")
                        pbar.update(1)
                        continue
                        
                    pts = pts.to(self.device)
                    gt_pts = gt_pts.to(self.device)
                    lidar_pos = lidar_pos.to(self.device)
                    lidar_quat = lidar_quat.to(self.device)
                    
                    
                    output_directory = "valid_"
                    file_path = os.path.join(output_directory, f'{iter}.pkl')
                    
                    ## get_target
                    if os.path.exists(file_path):
                        with open(file_path, 'rb') as f:
                            occupancy_grids = pickle.load(f)
                        # print("File loaded successfully.")
                    else:
                        # print(f"File '{file_path}' does not exist.")
                        occupancy_grids = []
                        torch.tensor([5, 14, 14], dtype=torch.float32)
                        occupancy_grids.append(self.occupancy_grid(gt_pts, (5, 14, 14), (self.max_coord_range_xyz - self.min_coord_range_xyz) / torch.tensor([5, 14, 14], dtype=torch.float32)))
                        occupancy_grids.append(self.occupancy_grid(gt_pts, (11, 29, 29), (self.max_coord_range_xyz - self.min_coord_range_xyz) / torch.tensor([11, 29, 29], dtype=torch.float32)))
                        occupancy_grids.append(self.occupancy_grid(gt_pts, (24, 59, 59), (self.max_coord_range_xyz - self.min_coord_range_xyz) / torch.tensor([24, 59, 59], dtype=torch.float32)))
                        occupancy_grids.append(self.occupancy_grid(gt_pts, (50, 120, 120), (self.max_coord_range_xyz - self.min_coord_range_xyz) / torch.tensor([50, 120, 120], dtype=torch.float32)))
                        os.makedirs(output_directory, exist_ok=True)
                        with open(file_path, 'wb') as f:
                            pickle.dump(occupancy_grids, f)
                    
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
                    preds, occu, probs, cm = self.model(sptensor)

                    ## get_target
                    idx = 0
                    gt_probs = []
                    for idx in range(len(probs)):
                        # tensor_to_ply(cm[idx][0], "cm.ply")
                        gt_prob = self.get_target(occupancy_grids[idx], cm[idx], idx)
                        gt_probs.append(gt_prob)
                    
                    loss = self.criterion(preds, occu, gt_pts, gt_occu.dense(), probs, gt_probs)                    
                    
                    # transform
                    transformed_preds = []
                    if preds is not None and not np.array_equal(lidar_pos, np.zeros(3, dtype=np.float32)) and not np.array_equal(lidar_quat, np.array([1, 0, 0, 0], dtype=np.float32)):
                        for i in range(min(self.batch_size, preds.size(0))):
                            transformed_pred = self.transform_point_cloud(preds[i].cpu(), lidar_pos[i].cpu(), lidar_quat[i].cpu())
                            transformed_preds.append(transformed_pred)
                            del transformed_pred

                    loss_buf.append(loss.item())
                    
                    # empty memory
                    del pts, gt_pts, lidar_pos, lidar_quat, batch, preds, loss
                    pbar.set_postfix(val_loss=np.mean(loss_buf) if loss_buf else 0)
                    pbar.update(1)                
            torch.cuda.synchronize()
            allocated_final = torch.cuda.memory_allocated()
            reserved_final = torch.cuda.memory_reserved()
            logging.info(f"valid -Memory allocated after deleting tensor and emptying cache: {allocated_final / (1024 ** 2):.2f} MB")
            logging.info(f"valid - Reserved after deleting tensor and emptying cache: {reserved_final / (1024 ** 2):.2f} MB")

            epoch_time = time.time() - epoch_start_time
            self.val_hist['per_epoch_time'].append(epoch_time)
            self.val_hist['val_loss'].append(np.mean(loss_buf))
            val_loss = np.mean(loss_buf) if loss_buf else 0
            return val_loss, transformed_preds, 


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
