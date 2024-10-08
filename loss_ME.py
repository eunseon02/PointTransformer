import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch3d.loss.chamfer import chamfer_distance
import open3d as o3d
import logging

import torch

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

    def compute_occupancy_loss(self, pred_occu, gt_occu):
        num_depth = len(pred_occu)
        loss = 0
        weights = torch.tensor([1, 2, 4, 16, 32], dtype=torch.float32)
        
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

    def forward(self, pred_occu, gt_occu, preds, gt_pts, pred_keep, keep):
        loss1, check = self.compute_occupancy_loss(pred_occu, gt_occu)
        loss2 = 0
        for depth in range(len(keep)):
            # print(pred_keep[depth].float().shape, keep[depth].float().shape)
            keep_loss = F.binary_cross_entropy_with_logits(pred_keep[depth].float(), keep[depth].float(), reduction='mean')
            loss2 += keep_loss
        loss2 /= len(keep)
        # print(loss1, loss2)

        # loss2 = self.compute_chamfer_loss(preds, gt_pts)

        total_loss = loss1 + 3*loss2

        # logging.info(f"loss1 {loss1}")
        # logging.info(f"loss2 {loss2}")
        # print("loss1", loss1)
        # print("loss2", loss2)
        return total_loss, loss1, loss2, check
