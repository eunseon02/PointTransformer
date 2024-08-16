import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config as cfg
import numpy as np
from chamfer_distance import ChamferDistance as chamfer_dist


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
        if self.smoothing:
            eps = 0.2
            preds = torch.sigmoid(preds)
            one_hot = gts * (1 - eps) + (1 - gts) * eps
            loss = F.binary_cross_entropy(preds, one_hot, reduction='mean')
        else:
            loss = F.binary_cross_entropy_with_logits(preds, gts, reduction='mean')

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
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.occupancy_loss = BinaryCrossEntropyLoss()
        self.subvoxel_loss = ChamferLoss()

    def voxelize(self, coord):
        coord_no_nan = coord.masked_fill(torch.isnan(coord), float('inf'))
        grid_coord = torch.div(
            coord - coord_no_nan.min(0)[0], torch.tensor([0.05]).to(coord.device), rounding_mode="trunc"
        ).int()
        grid_size = (grid_coord.max(0)[0] - grid_coord.min(0)[0] + 1).cpu().numpy()
        
        return grid_coord, grid_size
    
        
    def calculate_occupancy(self, voxel_coords, grid_size):
        
        grid_size = np.array(grid_size)  # Ensure grid_size is a numpy array
        if isinstance(voxel_coords, torch.Tensor):
            voxel_coords = voxel_coords.cpu().numpy()        # Initialize the occupancy grid
        occupancy_grid = np.zeros(grid_size, dtype=np.int32)
        # Ensure voxel_coords are within bounds
        # voxel_coords = voxel_coords.cpu().numpy()
        voxel_coords = np.clip(voxel_coords, 0, np.array(grid_size) - 1)

        # Update the occupancy grid
        occupancy_grid[tuple(voxel_coords.T)] = 1

        return occupancy_grid



    def forward(self, preds, gts_orgin):

        # assert preds.device == self.device, "preds is not on the correct device"
        # assert gts_orgin.device == self.device, "gts_orgin[0] is not on the correct device"
        # print("preds", preds.shape)
        # print("gts_orgin", gts_orgin.shape)
        # loss2 = self.subvoxel_loss(preds, gts_orgin)
        chd = chamfer_dist()
        # print("shape", preds.shape, gts_orgin.shape)

        dist1, dist2, idx1, idx2 = chd(preds,gts_orgin)
        loss2 = (torch.mean(dist1)) + (torch.mean(dist2))
        # print("loss : ", loss2)


        preds_voxel, _ = self.voxelize(preds)
        gts_voxel, _ = self.voxelize(preds)

        pred_occu = self.calculate_occupancy(preds_voxel.cpu().numpy(), (cfg.D, cfg.H, cfg.W))
        gt = self.calculate_occupancy(gts_voxel, (cfg.D, cfg.H, cfg.W))
        
        pred_occu = torch.tensor(pred_occu, dtype=torch.int64)
        gt = torch.tensor(gt, dtype=torch.int64)

        pred_occu = nn.Sigmoid()(pred_occu)
        gt = nn.Sigmoid()(gt)
        loss1 = self.occupancy_loss(pred_occu, gt)

        # print(loss1+loss2)
        return loss1+loss2
    

        
