
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
# import open3d as o3d
import os
import time
import argparse
import open3d as o3d
from pointnet.dataloaders.shapenet_partseg import get_data_loaders
from feature_model import PointTransformerV3ForGlobalFeature
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
def save_occupancy_grid_to_ply(occupancy_grid, file_name="occupancy_grid.ply"):
    # occupancy_grid에서 점유된 위치(값이 1인 곳)를 추출합니다.
    occupied_indices = torch.nonzero(occupancy_grid> 0, as_tuple=False).cpu().numpy()
    
    # PLY 파일의 헤더를 작성합니다.
    num_vertices = occupied_indices.shape[0]
    header = f"""ply
format ascii 1.0
element vertex {num_vertices}
property float x
property float y
property float z
end_header
"""

    # 점 좌표를 PLY 파일로 작성합니다.
    with open(file_name, 'w') as f:
        f.write(header)
        for point in occupied_indices:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")

    print(f"PLY file saved as {file_name}")
def tensor_to_ply(tensor, filename):
    points = tensor.cpu().detach().numpy()
    points = points.astype(np.float64)
    points= points[0]
    if points.shape[1] != 3:
        raise ValueError(f"Expected point cloud data with shape (n, 3), but got {points.shape}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pcd)
# setting logging
logging.basicConfig(filename='memory_log.txt', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def create_ddp_model(model, *, fp16_compression=False, **kwargs):
    if dist.get_world_size() == 1:
        return model
    if "device_ids" not in kwargs:
        kwargs["device_ids"] = [dist.get_rank()]
        if "output_device" not in kwargs:
            kwargs["output_device"] = [dist.get_rank()]
    ddp = DDP(model, **kwargs)
    if fp16_compression:
        from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks
        ddp.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)
    return ddp

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
        self.epochs = 450
        self.snapshot_interval = 10
        self.batch_size = 32
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = PointTransformerV3ForGlobalFeature(self.batch_size).to(self.device)
        self.model_path = args.model_path
        if self.model_path != '':
            self._load_pretrain(args.model_path)        
        self.parameter = self.model.parameters()
        self.criterion = NSLoss().to(self.device)
        self.optimizer = optim.Adam(self.parameter, lr=0.0001*16/self.batch_size, betas=(0.9, 0.999), weight_decay=1e-6)
        self.weight_folder = "weight"
        self.log_file = args.log_file if hasattr(args, 'log_file') else 'train_log.txt'

        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

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
        
        (train_ds, val_ds, test_ds), (train_dl, val_dl, test_dl) = get_data_loaders(
        data_dir="./pointnet/data", batch_size=self.batch_size, phases=["train", "val", "test"]
        )

        start_epoch = 250
        for epoch in range(start_epoch, self.epochs):
            train_loss, epoch_time, prev_preds = self.train_epoch(epoch, train_dl, prev_preds)
            # gc.collect()
            torch.cuda.empty_cache()
            val_loss, prev_preds_val = self.validation_epoch(epoch, val_dl, prev_preds_val)
            # gc.collect()
            torch.cuda.empty_cache()
            self._snapshot(epoch + 1)
            # gc.collect()
            torch.cuda.empty_cache()
            # save snapeshot
            if (epoch + 1) % self.snapshot_interval == 0:
                self._snapshot(epoch + 1)
                if train_loss < best_loss:
                    best_loss = train_loss
                    self._snapshot('best_{}'.format(epoch))
            log_message = f"Epoch [{epoch + 1}/{self.epochs}] - Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Time: {epoch_time:.4f}s"
            self.log(log_message)
            # finish all epoch
            self.train_hist['total_time'].append(time.time() - start_time)
            print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                            self.epochs, self.train_hist['total_time'][0]))
            print("Training finish!... save training results")

    def preprocess(self, coord):
        coord_no_nan = coord.masked_fill(torch.isnan(coord), float('inf'))
        global_min = coord_no_nan.min(dim=1, keepdim=True)[0]
        grid_coord = (coord - global_min) / 0.1
        return grid_coord.int()


    def calculate_occupancy(self, voxel_coords):
        grid_size = (int(25.0 / 0.1), int(25.0 / 0.1), int(25.0 / 0.1))
        
        occupancy_grid = torch.zeros((self.batch_size, *grid_size), dtype=torch.float64, device=voxel_coords.device, requires_grad=True)
        x_coords, y_coords, z_coords = voxel_coords.unbind(-1)
        x_coords = torch.clamp(x_coords, 0, grid_size[0] - 1)
        y_coords = torch.clamp(y_coords, 0, grid_size[1] - 1)
        z_coords = torch.clamp(z_coords, 0, grid_size[2] - 1)

        indices = ((x_coords * grid_size[1] * grid_size[2]) + (y_coords * grid_size[2]) + z_coords).long()
        unique_indices, counts = indices[0].unique(return_counts=True)
        duplicates = unique_indices[counts > 1]
        # if len(duplicates) > 0:
        #     print(f"Found {len(duplicates)} duplicate indices.")
            
        #     # Get the indices in the original tensor where duplicates occur
        #     duplicate_mask = indices[0].unsqueeze(1) == duplicates.unsqueeze(0)
        #     duplicate_coords = voxel_coords[0][duplicate_mask.any(dim=1)]
            
        #     # Select the first duplicate index for further investigation
        #     selected_duplicate = duplicates[0]
            
        #     # Find the voxel coordinates corresponding to this duplicate index
        #     matching_coords_mask = indices[0] == selected_duplicate
        #     matching_coords = voxel_coords[0][matching_coords_mask]
            
        #     print(f"Duplicate index: {selected_duplicate.item()}")
        #     print("Voxel coordinates mapping to this index:")
        #     print(matching_coords)
        occupancy_grid_flat = occupancy_grid.view(self.batch_size, -1)
        ones = torch.ones(indices.size(), dtype=torch.float64, device=voxel_coords.device)
        occupancy_grid_flat = occupancy_grid_flat.scatter_add(1, indices, ones)
        # print("occupancy_grid_flat after scatter_add:", occupancy_grid_flat)
        occupancy_grid = occupancy_grid_flat.view(self.batch_size, *grid_size)
        return occupancy_grid


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
        torch.cuda.empty_cache()
        return transformed_pc

    def max_pool_with_offset(self, features, offset):
        print(features)
        print(offset)
        global_features = []
        for i in range(len(offset) - 1):
            start = offset[i]
            end = offset[i + 1]
            point_features = features[start:end]
            global_features.append(point_features)
        result = torch.stack(global_features)
        del global_features, features
        torch.cuda.empty_cache()
        return result
    

    def train_epoch(self, epoch, train_dl, prev_preds):
        epoch_start_time = time.time()
        loss_buf = []
        self.model.train()
        preds = None
        transformed_preds = []
        with tqdm(total=len(train_dl), desc=f"Epoch {epoch + 1}/{self.epochs}", unit="batch") as pbar:
            for points, pc_labels, class_labels in train_dl:
                # points = torch.tensor(points, dtype=torch.float32).to(self.device)
                # pc_labels = torch.tensor(pc_labels, dtype=torch.float32).to(self.device)
                # class_labels = torch.tensor(class_labels, dtype=torch.float32).to(self.device)
                points, pc_labels, class_labels = points.to(self.device).requires_grad_(True), pc_labels.to(self.device), class_labels.to(self.device)
                points = points.view(-1, 3)
                points = torch.nan_to_num(points, nan=0.0)
                data_dict = {
                    'feat': points,
                    'coord': points,
                    'grid_size': torch.tensor([0.1]).to(points.device),
                    'offset': torch.arange(0, points.size(0) + 1, 2048, device=points.device)
                }

                # forward
                self.optimizer.zero_grad()
                
                # backward
                # with torch.autograd.detect_anomaly():
                preds = self.model(data_dict)
                preds = preds.view(self.batch_size, -1, 3)
                points = points.view(self.batch_size, -1, 3)
                
                # print(f"points requires_grad: {points.requires_grad}")
                # print(f"points grad_fn: {points.grad_fn}")
                # print(f"preds requires_grad: {preds.requires_grad}")
                # print(f"preds grad_fn: {preds.grad_fn}")

                                
                # loss
                loss = self.criterion(preds, points)                
                loss.backward()
                self.optimizer.step()
                loss_buf.append(loss.item())
                
                # # transform
                # transformed_preds = []
                # if preds is not None and not np.array_equal(lidar_pos, np.zeros(3, dtype=np.float32)) and not np.array_equal(lidar_quat, np.array([1, 0, 0, 0], dtype=np.float32)):
                #     for i in range(min(self.batch_size, preds.size(0))):
                #         transformed_pred = self.transform_point_cloud(preds[i].cpu(), lidar_pos[i].cpu(), lidar_quat[i].cpu())
                #         transformed_preds.append(transformed_pred.tolist())   
                        
                #         del transformed_pred
                #         # gc.collect()
                #         torch.cuda.empty_cache()
                                
                # empty memory
                del preds, loss, data_dict
                # gc.collect()
                torch.cuda.empty_cache()
                pbar.set_postfix(train_loss=np.mean(loss_buf) if loss_buf else 0)
                pbar.update(1)
            # gc.collect()
            torch.cuda.empty_cache()
            
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
    
    def validation_epoch(self, epoch, val_dl, prev_preds):
        epoch_start_time = time.time()
        loss_buf = []
        self.model.eval()
        preds = None
        transformed_preds = []
        with torch.no_grad():
            with tqdm(total=len(val_dl), desc=f"Validation {epoch + 1}/{self.epochs}", unit="batch") as pbar:
                for points, pc_labels, class_labels in val_dl:
                    # points = torch.tensor(points, dtype=torch.float32).to(self.device)
                    # pc_labels = torch.tensor(pc_labels, dtype=torch.float32).to(self.device)
                    # class_labels = torch.tensor(class_labels, dtype=torch.float32).to(self.device)
                    points, pc_labels, class_labels = points.to(self.device).requires_grad_(True), pc_labels.to(self.device), class_labels.to(self.device) 
                    points = points.view(-1, 3)
                    points = torch.nan_to_num(points, nan=0.0)
                    data_dict = {
                        'feat': points,
                        'coord': points,
                        'grid_size': torch.tensor([0.1]).to(points.device),
                        'offset': torch.arange(0, points.size(0) + 1, 2048, device=points.device)
                    }

                    
                    # backward
                    # with torch.autograd.detect_anomaly():
                    preds = self.model(data_dict)
                    preds = preds.view(self.batch_size, -1, 3)
                    points = points.view(self.batch_size, -1, 3)

                    # loss
                    loss = self.criterion(preds, points)
                    loss_buf.append(loss.item())
                    
                    # # transform
                    # transformed_preds = []
                    # if preds is not None and not np.array_equal(lidar_pos, np.zeros(3, dtype=np.float32)) and not np.array_equal(lidar_quat, np.array([1, 0, 0, 0], dtype=np.float32)):
                    #     for i in range(min(self.batch_size, preds.size(0))):
                    #         transformed_pred = self.transform_point_cloud(preds[i].cpu(), lidar_pos[i].cpu(), lidar_quat[i].cpu())
                    #         transformed_preds.append(transformed_pred.tolist())
                    #         del transformed_pred
                    #         # gc.collect()
                    #         torch.cuda.empty_cache()

                    # loss_buf.append(loss.item())
                    
                    # empty memory
                    del preds, loss, data_dict
                    # gc.collect()
                    torch.cuda.empty_cache()
                    pbar.set_postfix(val_loss=np.mean(loss_buf) if loss_buf else 0)
                    pbar.update(1)
                # gc.collect()
                torch.cuda.empty_cache()
                
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
        # Check if distributed training is being used
        is_distributed = torch.distributed.is_initialized()
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
