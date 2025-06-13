import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch3d.loss.chamfer import chamfer_distance
import open3d as o3d
# import logging

import torch

class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self, smoothing=False):
        super(BinaryCrossEntropyLoss, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, preds, gts):
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
            loss = F.binary_cross_entropy(preds, gts, reduction='mean')
        # print(f"loss: {loss}")

        return loss

class NSLoss(nn.Module):
    def __init__(self, λ_keep=1.0, keep_ratio_thresh=0.6):
        super(NSLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()
        self.occupancy_loss = BinaryCrossEntropyLoss()
        self.λ_keep = λ_keep
        self.keep_ratio_thresh = keep_ratio_thresh
    def compute_occupancy_loss(self, pred_occu, gt_occu):
        num_depth = len(pred_occu)
        loss = 0
        weights = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float32)
        
        check = []
        
        # Each decoder depth, predict the occupancy probability
        for depth in range(num_depth):
            occu_loss = self.occupancy_loss(pred_occu[depth].squeeze(-1), gt_occu[depth])
            weighted_loss = occu_loss * weights[depth]
            loss += weighted_loss
            
            # print(f"depth : {depth}, loss : {occu_loss} -> {weighted_loss}, weight : {weights[depth]}")
            check.append(occu_loss)

        loss /= num_depth

        return loss, check
    
    def compute_occupancy_focal_loss(self, pred_occu, gt_occu):
        num_depth = len(pred_occu)
        loss = 0
        weights = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float32)
        
        check = []
        
        # Each decoder depth, predict the occupancy probability
        for depth in range(num_depth):
            pred = pred_occu[depth].squeeze(-1)
            target = gt_occu[depth]
            alpha, gamma = 0.25, 0.5
            
            occu_loss = self.occupancy_loss(pred, target)
            p_t = torch.where(target == 1, pred, 1 - pred)
            focal_loss = (alpha * (1 - p_t) ** gamma * occu_loss).mean()
             
            weighted_loss = focal_loss * weights[depth]
            loss += weighted_loss
            
            # print(f"depth : {depth}, loss : {occu_loss} -> {weighted_loss}, weight : {weights[depth]}")
            check.append(occu_loss)

        loss /= num_depth

        return loss, check

    def focal_loss_with_logits(self, logits, targets, alpha=0.25, gamma=2.0, reduction='mean'):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        loss = alpha * (1 - pt) ** gamma * bce

        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
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

    def forward(self, preds, gt_pts, pred_keep, keep):
        loss2, penalty_loss = 0.0, 0.0
        for depth in range(len(keep)):
            # print(pred_keep[depth].float().shape, keep[depth].float().shape)
            keep_loss = self.focal_loss_with_logits(
                    pred_keep[depth].unsqueeze(-1).float(),
                    keep[depth].float(),
                    alpha=0.25, 
                    gamma=2.0,
                    reduction='mean'
                )
            keep_penalty = torch.relu((keep[depth].sum() - pred_keep[depth].sum()) / torch.clamp(keep[depth].sum(), min=1.0))

            loss2 += keep_loss
            penalty_loss += keep_penalty
        loss2 /= len(keep)
        penalty_loss /= len(keep)




        total_loss = loss2 + self.λ_keep * penalty_loss

        return total_loss, loss2, penalty_loss, check
