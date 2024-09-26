import spconv.pytorch as spconv
from spconv.pytorch import functional as Fsp
import torch
import torch.nn as nn
from spconv.pytorch.utils import PointToVoxel
from spconv.pytorch.hash import HashTable
import numpy as np
from cumm import tensorview as tv
import open3d as o3d
from config import config as cfg
import cumm
import torch.nn.functional as F

class SparseDecoder(spconv.SparseModule):
    def __init__(self, inplanes, planes, kernel, stride=(2, 2, 2, 1), indice_key=None):
        super(SparseDecoder, self).__init__()
        bias = nn.BatchNorm1d
        self.deconv = spconv.SparseSequential(
            spconv.SparseConvTranspose4d(inplanes, planes, kernel, stride, padding = (1, 1, 1, 0), bias=bias),
            nn.BatchNorm1d(planes),
            nn.ReLU(),
            spconv.SubMConv4d(planes, planes, kernel_size=3, stride=1, padding=1, bias=bias, indice_key=indice_key),
            nn.BatchNorm1d(planes),
            nn.ReLU(),
            spconv.SubMConv4d(planes, planes, kernel_size=3, stride=1, padding=1, bias=bias, indice_key=indice_key),
            nn.BatchNorm1d(planes),
            nn.ReLU(),
        )
        self.purning = spconv.SubMConv4d(planes, 1, kernel_size=1, stride=2, padding=0, indice_key=indice_key)

    def forward(self, x):
        out = self.deconv(x)
        mask = self.purning(out)

        # pred_prob = F.softmax(mask.features, 1)[:, 1]
        pred_prob = torch.sigmoid(mask.features)
        cm_, pred_prob = self.cls_postprocess(mask.indices, pred_prob)

        selected_features = out.features * mask.features
        selected_indices = out.indices

        out = out.replace_feature(selected_features)
        out.indices = selected_indices

        return out, cm_, pred_prob
    
    
    def cls_postprocess(self, feat_indices, pred_prob):
        batch_indices = feat_indices[:, 0]
        unique_batches = torch.unique(batch_indices)
        cm_batch = []
        pred_prob_batch = []

        for b in unique_batches:
            batch_mask = (batch_indices == b)
            cm_batch.append(feat_indices[batch_mask, 1:4])
            pred_prob_batch.append(pred_prob[batch_mask])
            
        max_points = max([t.size(0) for t in cm_batch])
        cm_batch_padded = []
        pred_prob_batch_padded = []
        for cm, prob in zip(cm_batch, pred_prob_batch):
            cm_padded = F.pad(cm, (0, 0, 0, max_points - cm.size(0)), value=0)  # [n, 3] -> [max_points, 3]
            prob_padded = F.pad(prob, (0, 0, 0, max_points - prob.size(0)), value=0)  # [n, 1] -> [max_points, 1]

            cm_batch_padded.append(cm_padded)
            pred_prob_batch_padded.append(prob_padded)

        cm_batch = torch.stack(cm_batch_padded)
        pred_prob_batch = torch.stack(pred_prob_batch_padded)

        return cm_batch, pred_prob_batch