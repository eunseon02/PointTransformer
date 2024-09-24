import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch3d.loss.chamfer import chamfer_distance
import open3d as o3d
import logging

import torch

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

    def compute_occupancy_loss(self, pred_occu, gt_occu):
        num_depth = len(pred_occu)
        loss = 0

        # Each decoder depth, predict the occupancy probability
        for depth in range(num_depth):
            coordinates = pred_occu[depth].C.long()
            probs = pred_occu[depth].F
            gt_occu_part = gt_occu[coordinates[:, 3], coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]]
            loss += self.occupancy_loss(probs, gt_occu_part)

        loss /= num_depth

        return loss

    def compute_chamfer_loss(self, preds, gts):
        loss = 0
        batch_num = len(preds)
        for pred, gt in zip(preds, gts):
            # If the shape is (P, D), convert it to (1, P, D)
            if len(pred.shape) == 2:
                pred = pred.unsqueeze(0)
            if len(gt.shape) == 2:
                gt = gt.unsqueeze(0)
            # Compute chamfer distance
            loss += chamfer_distance(pred, gt)[0]

        loss /= batch_num
        return loss

    def forward(self, preds, pred_occu, gts, gt_occu):
        loss1 = self.compute_occupancy_loss(pred_occu, gt_occu)

        loss2 = self.compute_chamfer_loss(preds, gts)

        total_loss = loss1 + loss2

        logging.info(f"loss1 {loss1}")
        logging.info(f"loss2 {loss2}")
        print("loss1", loss1)
        print("loss2", loss2)
        return total_loss



