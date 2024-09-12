import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config as cfg
import numpy as np
from chamfer_distance import ChamferDistance as chamfer_dist
import open3d as o3d
import logging

import torch
import spconv.pytorch as spconv
import cumm.tensorview as tv
import ctypes
cuda_kernel = ctypes.CDLL("./get_target.so")

logging.basicConfig(filename='loss_log.txt', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')


def save_occupancy_grid_as_ply(occupancy_grid, filename="occupancy_grid.ply"):
    batch_size = occupancy_grid.size(0)
    
    for batch_idx in range(batch_size):
        # Get the occupied positions (where occupancy is greater than a threshold)
        occupied_positions = occupancy_grid[batch_idx].nonzero(as_tuple=True)
        
        # Extract the coordinates
        x_coords = occupied_positions[0].cpu().numpy()
        y_coords = occupied_positions[1].cpu().numpy()
        z_coords = occupied_positions[2].cpu().numpy()
        
        # Stack the coordinates
        points = np.vstack((x_coords, y_coords, z_coords)).T
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Save point cloud to a .ply file
        o3d.io.write_point_cloud(f"{filename}_{batch_idx}.ply", pcd)
        print(f"Saved {filename}_{batch_idx}.ply")
        
def get_voxel_coordinates(metadata):
    spatial_locations = metadata.getSpatialLocations().cpu().numpy()
    return spatial_locations[:, :3]

"""
train occupancy
"""
class CrossEntropyLoss(nn.Module):
    def __init__(self, smoothing=False):
        super(CrossEntropyLoss, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, gts, preds):
        # voxel_coords_after_network = get_voxel_coordinates(output_tensor.metadata)
        # print("loss", voxel_coords_after_network)

        gts = gts.contiguous().view(-1)

        if self.smoothing:
            eps = 0.2
            n_class = preds.size(1)
            one_hot = torch.zeros_like(preds).scatter(1, gts.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(preds, dim=1)

            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(preds, gts, reduction='mean')

        return loss


class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self, smoothing=False):
        super(BinaryCrossEntropyLoss, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, gts, preds):
        gts = gts.float()
        preds = preds.float()
        # print(f"gts: {gts}")

        if self.smoothing:
            eps = 0.2
            preds = torch.sigmoid(preds)
            one_hot = gts * (1 - eps) + (1 - gts) * eps
            loss = F.binary_cross_entropy(preds, one_hot, reduction='mean')
        else:
            # print(preds.min(), preds.max(), preds.mean())
            # preds = torch.sigmoid(preds)
            # gts = torch.sigmoid(gts)
            loss = F.binary_cross_entropy_with_logits(preds, gts, reduction='mean')
        # print(f"loss: {loss}")

        return loss

class NSLoss(nn.Module):
    def __init__(self):
        super(NSLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()
        self.occupancy_loss = BinaryCrossEntropyLoss()
        self.cls_loss = nn.BCEWithLogitsLoss()
        self.vsizes = []
        self.vsizes.append([0.428, 0.428, 0.428])
        self.vsizes.append([0.207, 0.207, 0.207])
        self.vsizes.append([0.1, 0.1, 0.1])
        self.vsizes.append([0.05, 0.05, 0.05])


    def tensor_to_ply(self, tensor, filename):
        points = tensor.detach().cpu().numpy()
        points = points[0]
        points = points.astype(np.float64)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(filename, pcd)
    
    def get_target(self, pc, cm, idx): ## target
        from spconv.utils import Point2VoxelGPU3d
        from spconv.pytorch.utils import PointToVoxel
        with torch.no_grad():
            # Voxel generator
            gen1 = Point2VoxelGPU3d(
                vsize_xyz=self.vsizes[idx],
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
                _, indices_tv, num_p_in_vx_tv = gen1.point_to_voxel_hash(pc_single.cuda())

                indices_torch = torch.tensor(indices_tv.cpu().numpy(), dtype=torch.int32).to(pc.device)
                valid = num_p_in_vx_tv.cpu().numpy() > 0
                indices_torch = indices_torch[valid]

                batch_indices = torch.full((indices_torch.shape[0], 1), batch_idx, dtype=torch.int32).to(pc.device)
                indices_combined = torch.cat([batch_indices, indices_torch], dim=1)
                all_indices.append(indices_combined.int())

            all_indices = torch.cat(all_indices, dim=0).cpu()
            target_coords = all_indices[:, 1:]
            output = torch.zeros(cm[idx].size(0), 1)

            # matching_indices = (cm[idx].cpu() == target_coords.unsqueeze(1)).all(dim=2).any(dim=0)
            # output[matching_indices]=1    
            threads_per_block = 256
            blocks_per_grid = (cm[idx].shape[0] + threads_per_block - 1) // threads_per_block

            cuda_kernel.get_target_kernel(
                ctypes.c_void_p(cm[idx].data_ptr()),
                ctypes.c_void_p(target_coords.data_ptr()),
                ctypes.c_void_p(output.data_ptr()),
                ctypes.c_int(cm[idx].shape[0]),
                ctypes.c_int(target_coords.shape[0]),
                ctypes.c_int(cm[idx].shape[1])
            )

        return output
   

    def forward(self, preds, pred_occu, gts, gt_occu, probs, gt_probs):
        chd = chamfer_dist()
        dist1, dist2, idx1, idx2 = chd(preds,gts)
        cham_loss = (torch.mean(dist1)) + (torch.mean(dist2))
        occu_loss = self.occupancy_loss(gt_occu, pred_occu)
        cls_losses = torch.tensor(0.0).to(preds.device)
        for idx in range(len(probs)):
            cls_loss = self.cls_loss(probs[idx].squeeze(-1), gt_probs[idx].to(preds.device))
            cls_losses = cls_losses + cls_loss
        cls_losses = cls_losses/ len(probs)

        total_loss = 0.1*cham_loss + 0.3*occu_loss + cls_losses
        return total_loss, cham_loss, occu_loss, cls_losses
    

        
