
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
import spconv.pytorch as spconv
import cumm.tensorview as tv

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
from loss2 import NSLoss
import gc
import logging
from collections import OrderedDict

def tensor_to_ply(tensor, filename):
    print("tensor", tensor.shape)
    points = tensor.cpu().detach().numpy()
    points = points.astype(np.float64)
    points=  points[0]
    if points.shape[1] != 3:
        raise ValueError(f"Expected point cloud data with shape (n, 3), but got {points.shape}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pcd)

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
        self.batch_size = 256
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(self.device)
        self.model = PointCloud3DCNN(self.batch_size).to(self.device)
        self.model_path = args.model_path
        if self.model_path != '':
            self._load_pretrain(args.model_path)
        
        self.train_path = 'dataset/train'
        self.train_dataset = PointCloudDataset(self.train_path)
        print(f"Total train dataset length: {len(self.train_dataset)}")
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=8, pin_memory=True)
        
        self.val_path = 'dataset/valid'
        self.val_dataset = PointCloudDataset(self.val_path)
        print(f"Total valid dataset length: {len(self.val_dataset)}")
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=8, pin_memory=True)
        
        self.parameter = self.model.parameters()
        self.criterion = NSLoss().to(self.device)
        self.optimizer = optim.Adam(self.parameter, lr=0.001, betas=(0.9, 0.999), weight_decay=1e-6)
        self.weight_folder = "spconv_weight"
        self.log_file = args.log_file if hasattr(args, 'log_file') else 'train_log_spconv.txt'
        self.input_shape = (cfg.D, cfg.H, cfg.W)
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

        start_epoch = 0
        for epoch in range(start_epoch, self.epochs):
            train_loss, epoch_time, prev_preds = self.train_epoch(epoch, prev_preds)
            # gc.collect()
            torch.cuda.empty_cache()
            val_loss, prev_preds_val = self.validation_epoch(epoch, prev_preds_val)
            # gc.collect()
            torch.cuda.empty_cache()
            # self._snapshot(epoch + 1)
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
        torch.cuda.empty_cache()
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

            valid = num_p_in_vx_tv.cpu().numpy() > 0
            voxels_flatten = voxels_torch.view(-1, self.model.num_point_features * self.model.max_num_points_per_voxel)[valid]
            indices_torch = indices_torch[valid]

            batch_indices = torch.full((indices_torch.shape[0], 1), batch_idx, dtype=torch.int32).to(self.device)
            indices_combined = torch.cat([batch_indices, indices_torch], dim=1)

            all_voxels.append(voxels_flatten)
            all_indices.append(indices_combined)

        all_voxels = torch.cat(all_voxels, dim=0)
        all_indices = torch.cat(all_indices, dim=0)
        sparse_tensor = spconv.SparseConvTensor(all_voxels, all_indices, self.input_shape, self.batch_size)
        return sparse_tensor
    
    # @profileit
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
                
                # concat
                if prev_preds is not None:
                    prev_preds = [torch.as_tensor(p) for p in prev_preds]
                    # print(f"prev_preds-before cat : ", prev_preds[:5])
                    prev_preds_tensor = torch.stack(prev_preds).to(self.device)
                    pts = torch.cat((prev_preds_tensor, pts), dim=1)
                    # print(f"prev_preds-before cat : ", prev_preds[:5])
                    del prev_preds_tensor
                    # gc.collect()
                    torch.cuda.empty_cache()
                else:
                    pts = pts.repeat_interleave(2, dim=0)
                # nan_exists = torch.isnan(pts).any()
                # if nan_exists:
                #     print("pts 텐서에 NaN 값이 존재합니다.")
                # else:
                #     print("pts 텐서에 NaN 값이 없습니다.")
                pts = torch.nan_to_num(pts, nan=0.0)
                sptensor = self.preprocess(pts)
                self.optimizer.zero_grad()
                preds = self.model(sptensor)

                # loss = self.criterion(preds.unsqueeze(0), gt_pts.view(-1, 3).unsqueeze(0))
                loss = self.criterion(preds, gt_pts)


                loss.backward()
                # print(f"preds grad: {preds.grad}")
                # for name, param in self.model.named_parameters():
                #     print(f"Layer: {name} | requires_grad: {param.requires_grad}")
                #     if param.grad is not None:
                #         print(f"Layer: {name} | Gradient mean: {param.grad.mean()}")
                #     else:
                #         print(f"Layer: {name} | No gradient calculated!")
                # for name, param in self.model.named_parameters():
                #     if not param.requires_grad:
                #         print(f"Parameter {name} does not require grad!")
                #     else:
                #         print(f"Parameter {name} requires grad.")

                self.optimizer.step()
                loss_buf.append(loss.item())
                
                # transform
                transformed_preds = []
                if preds is not None and not np.array_equal(lidar_pos, np.zeros(3, dtype=np.float32)) and not np.array_equal(lidar_quat, np.array([1, 0, 0, 0], dtype=np.float32)):
                    for i in range(min(self.batch_size, preds.size(0))):
                        transformed_pred = self.transform_point_cloud(preds[i].cpu(), lidar_pos[i].cpu(), lidar_quat[i].cpu())
                        transformed_preds.append(transformed_pred.tolist())   
                        
                        del transformed_pred
                        # gc.collect()
                        torch.cuda.empty_cache()
                        
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
                    if gt_pts.shape[0] != self.batch_size or pts.shape[1] != 2048:
                        # print(f"Skipping batch {iter} because gt_pts first dimension {gt_pts.shape[0]} does not match batch size {self.batch_size}")
                        pbar.update(1)
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
                        # gc.collect()
                        torch.cuda.empty_cache()
                    else:
                        pts = pts.repeat_interleave(2, dim=0)
                        
                    pts = torch.nan_to_num(pts, nan=0.0)
                    sptensor = self.preprocess(pts)
                    preds = self.model(sptensor)
                    loss = self.criterion(preds, gt_pts)
                    
                    # transform
                    transformed_preds = []
                    if preds is not None and not np.array_equal(lidar_pos, np.zeros(3, dtype=np.float32)) and not np.array_equal(lidar_quat, np.array([1, 0, 0, 0], dtype=np.float32)):
                        for i in range(min(self.batch_size, preds.size(0))):
                            transformed_pred = self.transform_point_cloud(preds[i].cpu(), lidar_pos[i].cpu(), lidar_quat[i].cpu())
                            transformed_preds.append(transformed_pred.tolist())
                            del transformed_pred
                            # gc.collect()
                            torch.cuda.empty_cache()

                    loss_buf.append(loss.item())
                    
                    # empty memory
                    del pts, gt_pts, lidar_pos, lidar_quat, batch, preds, loss
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
