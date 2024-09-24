import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from config import config as cfg
from data2 import PointCloudDataset
import os
import time
import argparse
import open3d as o3d
from model_ME import PointCloud3DCNN
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.transform import Rotation as R
import cProfile
from loss_ME import NSLoss
from collections import OrderedDict
from os.path import join
from torch.utils.tensorboard import SummaryWriter
import joblib
from data import GetTarget
import MinkowskiEngine as ME

BASE_LOGDIR = "./train_logs" 
writer = SummaryWriter(join(BASE_LOGDIR, "check_teacher"))
def pad_or_trim_cloud(cloud, target_size=3000):
    n = cloud.size(0)  # 현재 포인트 클라우드의 개수 (n x 3에서 n)
    
    if n < target_size:
        # n이 target_size보다 작으면 패딩 (0으로 채움)
        padding = torch.zeros((target_size - n, 3))  # (target_size - n)개의 0으로 채워진 3D 좌표 생성
        cloud = torch.cat([cloud, padding], dim=0)  # 원본 포인트 클라우드에 패딩 추가
    elif n > target_size:
        # n이 target_size보다 크면 자름
        cloud = cloud[:target_size, :]  # 첫 target_size개의 포인트만 사용

    return cloud
def occupancy_grid_to_coords(occupancy_grid):
    _, _, H, W, D = occupancy_grid.shape
    occupancy_grid = occupancy_grid[0, 0]
    indices = torch.nonzero(occupancy_grid > 0, as_tuple=False) 
    return indices
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

class Train():
    def __init__(self, args):
        self.epochs = 300
        self.snapshot_interval = 10
        self.batch_size = 64
        self.split = 1
        self.device = cfg.device
        torch.cuda.set_device(self.device)
        self.model = PointCloud3DCNN(self.batch_size, in_channels=3, out_channels=3, dimension=3, n_depth=4).to(self.device)
        self.model_path = args.model_path
        if self.model_path != '':
            self._load_pretrain(args.model_path)
        

        self.h5_file_path = "lidar_data.h5"
        self.train_dataset = PointCloudDataset(self.h5_file_path, 'train')
        print(f"Total valid dataset length: {len(self.train_dataset)}")
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False,pin_memory=True)
        if self.train_dataset.batch_count != self.batch_size:
            print(self.train_dataset.batch_count)
            raise RuntimeError('Wrong batch_size')        
        
        
        self.val_dataset = PointCloudDataset(self.h5_file_path, 'valid')
        print(f"Total valid dataset length: {len(self.val_dataset)}")
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,pin_memory=True)
        if self.val_dataset.batch_count != self.batch_size:
            print(self.val_dataset.batch_count)
            raise RuntimeError('Wrong batch_size')
        
        self.parameter = self.model.parameters()
        self.criterion = NSLoss().to(self.device)
        self.optimizer = optim.Adam(self.parameter, lr=0.001, betas=(0.9, 0.999), weight_decay=1e-6)
        self.weight_folder = "weight_teacher"
        self.log_file = args.log_file if hasattr(args, 'log_file') else 'train_log_teacher.txt'
        self.input_shape = (50, 120, 120)

        # Voxelize
        self.grid_size = 64
        self.min_coord_range_xyz = torch.tensor([-3.0, -3.0, -3.0])
        self.max_coord_range_xyz = torch.tensor([3.0, 3.0, 3.0])
        self.voxel_size = (self.max_coord_range_xyz - self.min_coord_range_xyz) / self.grid_size

        self.train_occu = []
        self.valid_occu = []

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

    def tensorboard_launcher(self, points, step, color, tag):
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

        start_epoch = 150
        for epoch in range(start_epoch, self.epochs):
            train_loss, epoch_time = self.train_epoch(epoch)
            writer.add_scalar("Loss/train", train_loss, epoch)
            # writer.add_scalar("Loss/cham_loss", cham_loss, epoch)
            # writer.add_scalar("Loss/occu_loss", occu_loss, epoch)
            # writer.add_scalar("Loss/cls_losses", cls_losses, epoch)
            # val_loss = self.validation_epoch(epoch)
            val_loss = 0
            writer.add_scalar("Loss/valid", val_loss, epoch)

            if len(self.train_taget_loader) != len(self.train_loader):
                print("Regenerate train loader")
                self.train_get_target = GetTarget(self.train_target_dir)
                self.train_taget_loader = torch.utils.data.DataLoader(self.train_get_target, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
                # self.val_taget_loader = torch.utils.data.DataLoader(self.valid_get_target, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
            if len(self.val_taget_loader) != len(self.val_loader):
                print("Regenerate valid loader")
                self.valid_get_target = GetTarget(self.valid_target_dir)
                self.val_taget_loader = torch.utils.data.DataLoader(self.valid_get_target, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

            
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

    def preprocess(self, pc, device='cpu'):

        batch_size = pc.shape[0]

        features_list = []
        indices_list = []

        for batch_idx in range(batch_size):
            pc_single = pc[batch_idx]

            # Filtering point clouds (remove O.O.D points)
            filtered_pc = self.filter_point_cloud(pc_single, self.min_coord_range_xyz, self.max_coord_range_xyz)

            # Consider the minimum value as 0 index.
            voxel_indices = torch.div(filtered_pc - self.min_coord_range_xyz - 1e-4, self.voxel_size).long()

            # voxel center
            voxel_center = self.min_coord_range_xyz + (voxel_indices + 0.5) * self.voxel_size

            # feature generation (normalized distance w.r.t. points in each voxel grid coordinate)
            normalized_features = (filtered_pc - voxel_center) / self.voxel_size + 0.5

            # batch_indices
            batch_indices = torch.ones((normalized_features.shape[0], 1), dtype=torch.int32) * batch_idx

            # Concat voxel indices & batch indices in proper format
            sparse_tensor_indices = torch.cat((voxel_indices, batch_indices), dim=1).int().contiguous()

            # append into the batches
            features_list.append(normalized_features)
            indices_list.append(sparse_tensor_indices)

        features_tc = torch.cat(features_list, dim=0).to(device)
        indices_tc = torch.cat(indices_list, dim=0).to(device)

        sparse_tensor = ME.SparseTensor(features=features_tc,
                                        coordinates=indices_tc,
                                        quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)
        return sparse_tensor

    def occupancy_grid(self, pc, device='cpu'):

        batch_size = pc.shape[0]

        # Initialize the occupancy grid with zeros

        occupancy_grid = torch.zeros((batch_size, self.grid_size, self.grid_size, self.grid_size), dtype=torch.float32)

        for batch_idx in range(batch_size):

            pc_single = pc[batch_idx]

            # Filtering point clouds (remove O.O.D points)
            filtered_pc = self.filter_point_cloud(pc_single, self.min_coord_range_xyz, self.max_coord_range_xyz)

            # Consider the minimum value as 0 index.
            voxel_indices = torch.div(filtered_pc - self.min_coord_range_xyz - 1e-4, self.voxel_size).long()

            # Increment the occupancy grid at the corresponding indices
            occupancy_grid[batch_idx, voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = 1.0

        occupancy_grid = occupancy_grid.unsqueeze(-1).to(device)

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

    def postprocess(self, preds, device='cpu'):
        pc_batch = []

        predicted_coords = preds.coordinates
        predicted_feats = preds.features

        # Extract batch indices and voxel indices
        voxel_indices = preds.coordinates[:, :3]
        batch_indices = predicted_coords[:, 3]

        # Calculate the voxel centers in the original coordinate space
        voxel_centers = self.min_coord_range_xyz.to(device) + (voxel_indices + 0.5) * self.voxel_size.to(device)

        # Denormalize the features (reverse the normalization applied in preprocess)
        denormalized_features = (predicted_feats - 0.5) * self.voxel_size.to(device) + voxel_centers

        # Append denormalized features to pc_batch for each batch
        unique_batches = torch.unique(batch_indices)
        for batch_idx in unique_batches:
            mask = (batch_indices == batch_idx)
            batch_pc = denormalized_features[mask]
            pc_batch.append(batch_pc)

        return pc_batch
                
    @profileit
    def train_epoch(self, epoch):
        epoch_start_time = time.time()
        loss_buf = []
        self.model.train()
        preds = None
        prev_preds = []
        cham_loss_buf, occu_loss_buf, cls_losses_buf = [], [], []
        with tqdm(total=len(self.train_loader), desc=f"Epoch {epoch + 1}/{self.epochs}", unit="batch") as pbar:
            for iter, (batch, occupancy_grids) in enumerate(zip(self.train_loader, self.train_taget_loader)):
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
                
                # pts = pts.to(self.device)
                # gt_pts = gt_pts.to(self.device)
                # lidar_pos = lidar_pos.to(self.device)
                # lidar_quat = lidar_quat.to(self.device)
                # self.tensorboard_launcher(pts[0], iter, [1.0, 0.0, 1.0], "pts_iter")

                if len(self.train_taget_loader) != len(self.train_loader):
                    print(f"calculate : not matching {len(self.train_taget_loader)} & {len(self.train_loader)}")
                    # output_directory = "train_"
                    # file_path = os.path.join(output_directory, f'{iter}.joblib')
                    # occupancy_grids = []
                    # occupancy_grids.append(self.occupancy_grid(gt_pts, (5, 14, 14), (self.max_coord_range_xyz - self.min_coord_range_xyz) / torch.tensor([5, 14, 14], dtype=torch.float32)))
                    # occupancy_grids.append(self.occupancy_grid(gt_pts, (11, 29, 29), (self.max_coord_range_xyz - self.min_coord_range_xyz) / torch.tensor([11, 29, 29], dtype=torch.float32)))
                    # occupancy_grids.append(self.occupancy_grid(gt_pts, (24, 59, 59), (self.max_coord_range_xyz - self.min_coord_range_xyz) / torch.tensor([24, 59, 59], dtype=torch.float32)))
                    # occupancy_grids.append(self.occupancy_grid(gt_pts, (50, 120, 120), (self.max_coord_range_xyz - self.min_coord_range_xyz) / torch.tensor([50, 120, 120], dtype=torch.float32)))
                    # os.makedirs(output_directory, exist_ok=True)
                    # joblib.dump(occupancy_grids, file_path)
                    
                # concat
                if len(prev_preds) > 0:
                    prev_preds = [torch.as_tensor(p) for p in prev_preds]
                    prev_preds_tensor = torch.stack(prev_preds).to(self.device)
                    pts = torch.cat((prev_preds_tensor, pts), dim=1)
                    # tensor_to_ply(prev_preds_tensor[0], f"transformed_pred_{iter}.ply")
                    # self.tensorboard_launcher(prev_preds_tensor[0], iter, [1.0, 0.0, 0.0], "transformed_pts")
                    # tensor_to_ply(gt_pts[0], f"pts_{iter}.ply")
                    # self.tensorboard_launcher(gt_pts[0], iter, [1.0, 0.0, 1.0], "gt_pts")
                    del prev_preds, prev_preds_tensor
                    prev_preds = []
                else:
                    pts = pts.repeat_interleave(2, dim=0)
                    pts = pts.view(self.batch_size, -1, 3)
                pts = torch.nan_to_num(pts, nan=0.0)
                sptensor = self.preprocess(pts, device=self.device)
                gt_occu = self.occupancy_grid(gt_pts, device=self.device)
                gt_pts_filtered = []
                for i in range(gt_pts.shape[0]):
                    gt_pts_filtered.append(self.filter_point_cloud(gt_pts[i], self.min_coord_range_xyz, self.max_coord_range_xyz).to(self.device))
                # gt_occu = self.occupancy_grid_(gt_pts)

                self.optimizer.zero_grad()
                preds, occu = self.model(sptensor)
                pred_pc = self.postprocess(preds, device=self.device)
                loss = self.criterion(pred_pc, occu, gt_pts_filtered, gt_occu)

                loss.backward()

                # if iter == 490:
                    # print("tensorboard_launcher")
                    # self.tensorboard_launcher(occupancy_grid_to_coords(occu), epoch, [1.0, 0.0, 0.0], "Reconstrunction_train")
                    # self.tensorboard_launcher(occupancy_grid_to_coords(gt_occu.dense()), epoch, [0.0, 0.0, 1.0], "GT_train")
                    # self.tensorboard_launcher(occupancy_grid_to_coords(sptensor.dense()), epoch, [0.0, 1.0, 1.0], "pts_train")

                ## get_target
                # idx = 0
                # gt_probs = []
                # for idx in range(len(probs)):
                #     gt_prob = self.get_target(occupancy_grids[idx].squeeze(0), cm[idx], idx)
                #     gt_probs.append(gt_prob)
                # occupancy_grids.clear()
                
                # loss, cham_loss, occu_loss, cls_losses = self.criterion(preds, occu, gt_pts, gt_occu.dense(), probs, gt_probs)
                # cham_loss_buf.append(cham_loss.item())
                # occu_loss_buf.append(occu_loss.item())
                # cls_losses_buf.append(cls_losses.item())
                # loss.backward()

                # for name, param in self.model.named_parameters():
                #     print(f"Layer: {name} | requires_grad: {param.requires_grad}")
                #     if param.grad is not None:
                #         print(f"Layer: {name} | Gradient mean: {param.grad.mean()}")
                #     else:
                #         print(f"Layer: {name} | No gradient calculated!")
                self.optimizer.step()

                loss_buf.append(loss.item())
                # loss_buf.append(cls_losses.item())
                
                # transform
                if preds is not None and not np.array_equal(lidar_pos, np.zeros(3, dtype=np.float32)) and not np.array_equal(lidar_quat, np.array([1, 0, 0, 0], dtype=np.float32)):
                    for i in range(min(self.batch_size, gt_pts.size(0))):
                        # if random.random() < self.teacher_forcing_ratio:
                        #     input_data = gt_pts[i].cpu()
                        # else:
                        #     input_data = preds[i].cpu()
                        transformed_pred = self.transform_point_cloud(gt_pts[i].cpu(), lidar_pos[i].cpu(), lidar_quat[i].cpu())
                        transformed_pred = pad_or_trim_cloud(transformed_pred, target_size=3000)
                        prev_preds.append(transformed_pred)
                        del transformed_pred
                    
                self.teacher_forcing_ratio = max(0.0, self.teacher_forcing_ratio - self.decay_rate)

                # empty memory
                del pts, gt_pts, lidar_pos, lidar_quat, batch, preds, loss, occu
                pbar.set_postfix(train_loss=np.mean(loss_buf) if loss_buf else 0)
                pbar.update(1)
        torch.cuda.synchronize()
        epoch_time = time.time() - epoch_start_time
        self.train_hist['per_epoch_time'].append(epoch_time)
        self.train_hist['train_loss'].append(np.mean(loss_buf))
        # return np.mean(loss_buf), epoch_time, np.mean(cham_loss_buf), np.mean(occu_loss_buf), np.mean(cls_losses)
        return np.mean(loss_buf), epoch_time

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
                for iter, (batch, occupancy_grids) in enumerate(zip(self.val_loader, self.val_taget_loader)):
                    if batch is None:
                        print(f"Skipping batch {iter} because it is None")
                        pbar.update(1)
                        continue

                    pts, gt_pts, lidar_pos, lidar_quat, _ = batch
                    if gt_pts.shape[0] != self.batch_size:
                        print(f"Skipping batch {iter} because gt_pts first dimension {gt_pts.shape[0]} does not match batch size {self.batch_size}")
                        pbar.update(1)
                        continue
                        
                    # pts = pts.to(self.device)
                    # gt_pts = gt_pts.to(self.device)
                    # lidar_pos = lidar_pos.to(self.device)
                    # lidar_quat = lidar_quat.to(self.device)


           
                    if len(self.val_taget_loader) != len(self.val_loader):
                        print(f"calculate : not matcing {len(self.val_taget_loader)} and {len(self.val_loader)}")
                        # output_directory = "valid_"
                        # file_path = os.path.join(output_directory, f'{iter}.joblib')
                        # occupancy_grids = []
                        # occupancy_grids.append(self.occupancy_grid(gt_pts, (5, 14, 14), (self.max_coord_range_xyz - self.min_coord_range_xyz) / torch.tensor([5, 14, 14], dtype=torch.float32)))
                        # occupancy_grids.append(self.occupancy_grid(gt_pts, (11, 29, 29), (self.max_coord_range_xyz - self.min_coord_range_xyz) / torch.tensor([11, 29, 29], dtype=torch.float32)))
                        # occupancy_grids.append(self.occupancy_grid(gt_pts, (24, 59, 59), (self.max_coord_range_xyz - self.min_coord_range_xyz) / torch.tensor([24, 59, 59], dtype=torch.float32)))
                        # occupancy_grids.append(self.occupancy_grid(gt_pts, (50, 120, 120), (self.max_coord_range_xyz - self.min_coord_range_xyz) / torch.tensor([50, 120, 120], dtype=torch.float32)))
                        # os.makedirs(output_directory, exist_ok=True)
                        # joblib.dump(occupancy_grids, file_path)
                        
                    # concat
                    if len(prev_preds) > 0:
                        prev_preds = [torch.as_tensor(p) for p in prev_preds]
                        prev_preds_tensor = torch.stack(prev_preds).to(self.device)
                        pts = torch.cat((prev_preds_tensor, pts), dim=1)
                        # tensor_to_ply(prev_preds_tensor[0], f"transformed_pred_{iter}.ply")
                        # self.tensorboard_launcher(prev_preds_tensor[0], iter, [1.0, 0.0, 0.0], "transformed_pts")
                        # tensor_to_ply(pts[0], f"pts_{iter}.ply")
                        # self.tensorboard_launcher(pts[0], iter, [1.0, 0.0, 1.0], "gt_pts")
                        del prev_preds_tensor, prev_preds
                        prev_preds = []
                    else:
                        pts = pts.repeat_interleave(2, dim=0)
                        pts = pts.view(self.batch_size, -1, 3)
                    # pts = torch.nan_to_num(pts, nan=0.0)
                    # sptensor = self.preprocess(pts)
                    # gt_occu = self.occupancy_grid_(gt_pts)
                    #
                    # preds, occu, probs, cm = self.model(sptensor)

                    pts = torch.nan_to_num(pts, nan=0.0)
                    sptensor = self.preprocess(pts, device=self.device)
                    gt_occu = self.occupancy_grid(gt_pts, device=self.device)
                    gt_pts_filtered = []
                    for i in range(gt_pts.shape[0]):
                        gt_pts_filtered.append(self.filter_point_cloud(gt_pts[i], self.min_coord_range_xyz, self.max_coord_range_xyz).to(self.device))

                    self.optimizer.zero_grad()
                    preds, occu = self.model(sptensor)
                    pred_pc = self.postprocess(preds, device=self.device)
                    loss = self.criterion(pred_pc, occu, gt_pts_filtered, gt_occu)
                    
                    # self.tensorboard_launcher(occupancy_grid_to_coords(occu), iter, [1.0, 0.0, 0.0], "Reconstrunction_iter")
                    # self.tensorboard_launcher(occupancy_grid_to_coords(gt_occu.dense()), iter, [0.0, 0.0, 1.0], "GT_iter")

                    if iter == 120:
                        print("tensorboard_launcher")
                        self.tensorboard_launcher(occupancy_grid_to_coords(occu), epoch, [1.0, 0.0, 0.0], "Reconstrunction_valid")
                        self.tensorboard_launcher(occupancy_grid_to_coords(gt_occu.dense()), epoch, [0.0, 0.0, 1.0], "GT_valid")
                    #     self.tensorboard_launcher(occupancy_grid_to_coords(sptensor.dense()), epoch, [0.0, 1.0, 1.0], "pts_valid")


                    ## get_target
                    # idx = 0
                    # gt_probs = []
                    # for idx in range(len(probs)):
                    #     gt_prob = self.get_target(occupancy_grids[idx].squeeze(0), cm[idx], idx)
                    #     gt_probs.append(gt_prob)
                    # occupancy_grids.clear()
                    #
                    #
                    # loss, cham_loss, occu_loss, cls_losses = self.criterion(preds, occu, gt_pts, gt_occu.dense(), probs, gt_probs)
                    loss_buf.append(loss.item())

                    # transform
                    if preds is not None and not np.array_equal(lidar_pos, np.zeros(3, dtype=np.float32)) and not np.array_equal(lidar_quat, np.array([1, 0, 0, 0], dtype=np.float32)):
                        for i in range(min(self.batch_size, gt_pts.size(0))):
                            # if random.random() < self.teacher_forcing_ratio:
                            #     input_data = gt_pts[i].cpu()
                            # else:
                            #     input_data = preds[i].cpu()
                            transformed_pred = self.transform_point_cloud(gt_pts[i].cpu(), lidar_pos[i].cpu(), lidar_quat[i].cpu())
                            transformed_pred = pad_or_trim_cloud(transformed_pred, target_size=3000)
                            prev_preds.append(transformed_pred)
                            del transformed_pred

                    # empty memory
                    del pts, gt_pts, lidar_pos, lidar_quat, batch, preds, loss, occu
                    pbar.set_postfix(val_loss=np.mean(loss_buf) if loss_buf else 0)
                    pbar.update(1)                
            torch.cuda.synchronize()
            epoch_time = time.time() - epoch_start_time
            self.val_hist['per_epoch_time'].append(epoch_time)
            self.val_hist['val_loss'].append(np.mean(loss_buf))
            val_loss = np.mean(loss_buf) if loss_buf else 0
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
