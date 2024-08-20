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
from spconv.pytorch.utils import PointToVoxel

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
            preds = torch.sigmoid(preds)
            gts = torch.sigmoid(gts)
            loss = F.binary_cross_entropy_with_logits(preds, gts, reduction='mean')
        # print(f"loss: {loss}")

        return loss

class ChamferLoss(nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()

    def batch_pairwise_dist(self, x, y):

        if x.dim() == 2:
            x = x.unsqueeze(0)
        if y.dim() == 2:
            y = y.unsqueeze(0)

        x = x.float()
        y = y.float()

        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = x.pow(2).sum(dim=-1)
        yy = y.pow(2).sum(dim=-1)
        zz = torch.bmm(x, y.transpose(2, 1))
        rx = xx.unsqueeze(1).expand_as(zz.transpose(2, 1))
        ry = yy.unsqueeze(1).expand_as(zz)
        P = (rx.transpose(2, 1) + ry - 2 * zz)
        return P

    def forward(self, preds, gts):
        # print("loss", preds.shape)
        # print("loss", gts.shape)
        P = self.batch_pairwise_dist(gts, preds)
        mins, _ = torch.min(P, 1)
        loss_1 = torch.sum(mins)
        mins, _ = torch.min(P, 2)
        loss_2 = torch.sum(mins)
        return loss_1 + loss_2



class NSLoss(nn.Module):
    def __init__(self):
        super(NSLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()
        self.occupancy_loss = BinaryCrossEntropyLoss()
        self.subvoxel_loss = ChamferLoss()

    def voxelize(self, coord):
        coord_no_nan = coord.masked_fill(torch.isnan(coord), float('inf'))
        global_min = coord_no_nan.min(dim=1, keepdim=True)[0]
        grid_coord = (coord - global_min) / 0.1
        return grid_coord.int()
    
    def voxelize2(self, coord):
        offsets = torch.tensor([-3.0, -3.0, -1.0], device=coord.device)  # X, Y, Z offsets
        grid_coord = torch.div(
            coord - offsets, torch.tensor([0.05]).to(coord.device), rounding_mode="trunc"
        )
        return grid_coord

    def calculate_occupancy(self, voxel_coords):
        grid_size = (120, 120, 70)
        batch_size = voxel_coords.size(0)
        
        occupancy_grid = torch.zeros((batch_size, *grid_size), dtype=torch.float64, device=voxel_coords.device, requires_grad=True)
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
        occupancy_grid_flat = occupancy_grid.view(batch_size, -1)
        ones = torch.ones(indices.size(), dtype=torch.float64, device=voxel_coords.device)
        occupancy_grid_flat = occupancy_grid_flat.scatter_add(1, indices, ones)
        # print("occupancy_grid_flat after scatter_add:", occupancy_grid_flat)
        occupancy_grid_flat = torch.clamp(occupancy_grid_flat, min=0, max=1)
        occupancy_grid = occupancy_grid_flat.view(batch_size, *grid_size)
        return occupancy_grid

    def create_occupancy_grid(self, pc):
        # Initialize PointToVoxel generator
        voxel_generator = PointToVoxel(
            vsize_xyz=[0.05, 0.05, 0.05],
            coors_range_xyz=[-3, -3, -1, 3, 3, 1.5],
            num_point_features=3,
            max_num_voxels=600000,
            max_num_points_per_voxel=5,
            device=pc.device
        )
        
        batch_size = pc.shape[0]  # Assume pc is of shape (batch_size, num_points, 3)
        grid_size = [50, 120, 120] 
        occupancy_grids = []

        for batch_idx in range(batch_size):
            pc_single = pc[batch_idx]
            voxels_tv, indices_tv, num_p_in_vx_tv, _ = voxel_generator.generate_voxel_with_id(pc_single)

            indices_torch = torch.tensor(indices_tv.cpu().numpy(), dtype=torch.int32).to(pc.device)
            # self.tensor_to_ply(indices_torch, "indices_torch.ply")
            # 유효한 인덱스 필터링
            valid = num_p_in_vx_tv.cpu().numpy() > 0
            indices_torch = indices_torch[valid]

            # Occupancy Grid 초기화
            occupancy_grid = torch.zeros(tuple(grid_size), dtype=torch.float32, device=pc.device, requires_grad=True)


            # Occupancy Grid에 점유된 위치 마킹
            for idx in indices_torch:
                occupancy_grid[idx[0], idx[1], idx[2]] = 1.0  # 점유된 보셀의 위치를 1로 설정

            # 결과를 리스트에 추가
            occupancy_grids.append(occupancy_grid)

        # 리스트를 배치 텐서로 변환
        occupancy_grids = torch.stack(occupancy_grids)
        return occupancy_grid




    def tensor_to_ply(self, tensor, filename):
        points = tensor.detach().cpu().numpy()
        points = points[0]
        points = points.astype(np.float64)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(filename, pcd)

    def forward(self, preds, gts):
        chd = chamfer_dist()
        dist1, dist2, idx1, idx2 = chd(preds,gts)
        loss2 = (torch.mean(dist1)) + (torch.mean(dist2))
        # loss2 = loss2 *0.001
        log_loss2 = torch.log(loss2)
        # print("loss2", loss2)
        
        # print(f"Preds requires_grad - before voxelize: {preds.requires_grad}")
        # print(f"Preds grad_fn - before voxelize: {preds.grad_fn}")

        gts_voxel = self.voxelize2(gts.float())
        preds_voxel = self.voxelize2(preds.float())
        # self.tensor_to_ply(gts_voxel, "gts_voxel.ply")
        
        # print(f"Preds requires_grad - after voxelize: {preds_voxel.requires_grad}")
        # print(f"Preds grad_fn - after voxelize: {preds_voxel.grad_fn}")

        pred_occu = self.calculate_occupancy(preds_voxel)
        gts_occu = self.calculate_occupancy(gts_voxel)
        # save_occupancy_grid_as_ply(gts_occu)
        # print(f"calculate_occupancy requires_grad: {pred_occu.requires_grad}")
        # print(f"calculate_occupancy grad_fn: {pred_occu.grad_fn}")
        # dist1, dist2, idx1, idx2 = chd(preds_voxel.float(),gts_voxel.float())
        # loss1 = (torch.mean(dist1)) + (torch.mean(dist2))
        # print("loss1", loss1)
        
        # print("pred_occu",pred_occu)

        loss1 = self.occupancy_loss(pred_occu, gts_occu).float()
        # print(f"Loss requires_grad: {loss1.requires_grad}")
        # print(f"Loss grad_fn: {loss1.grad_fn}")
        

        logging.info(f"loss1 {loss1}")
        logging.info(f"loss2 {log_loss2}")
        # print("loss1", loss1.grad_fn)
        # print("loss2", loss2.grad_fn)
        return loss2
    

        
