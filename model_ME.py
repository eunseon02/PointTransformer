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
from os.path import join
from debug import tensor_to_ply
import MinkowskiEngine as ME

input_shape = (cfg.D, cfg.H, cfg.W)
class PointCloud3DCNN(nn.Module):
    ENC_CHANNELS = [16, 32, 64, 128, 256, 512, 1024]
    DEC_CHANNELS = [16, 32, 64, 128, 256, 512, 1024]
    def __init__(self, batch_size, in_channels, out_channels, dimension, n_depth=4):
        super(PointCloud3DCNN, self).__init__()
        enc_ch = self.ENC_CHANNELS
        dec_ch = self.DEC_CHANNELS
        
        self.D = dimension
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batch_size = batch_size
        
        self.device = cfg.device
        self.num_point_features = 4
        self.max_num_points_per_voxel = 3
        self.Encoder1 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=self.in_channels, out_channels=enc_ch[0], dimension=self.D),
            ME.MinkowskiBatchNorm(enc_ch[0]),
            ME.MinkowskiReLU()
        )
        self.Encoder2 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=enc_ch[0], out_channels=enc_ch[1], kernel_size=3, stride=2, dimension=self.D),
            ME.MinkowskiBatchNorm(enc_ch[1]),
            ME.MinkowskiReLU()
        )
        self.Encoder3 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=enc_ch[1], out_channels=enc_ch[2], kernel_size=3, stride=2, dimension=self.D),
            ME.MinkowskiBatchNorm(enc_ch[2]),
            ME.MinkowskiReLU()
        )

        self.Encoder4 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=enc_ch[2], out_channels=enc_ch[3], kernel_size=3, stride=2, dimension=self.D),
            ME.MinkowskiBatchNorm(enc_ch[3]),
            ME.MinkowskiReLU()
        )

        self.Decoder4 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(in_channels=dec_ch[3], out_channels=dec_ch[2], kernel_size=3, stride=2, dimension=self.D),
            ME.MinkowskiBatchNorm(dec_ch[2]),
            ME.MinkowskiReLU()
        )
        self.occu4 = nn.Sequential(
            ME.MinkowskiConvolution(dec_ch[2], 1, kernel_size=1, bias=True, dimension=self.D),
            ME.MinkowskiSigmoid()
        )
        self.Decoder3 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(in_channels=dec_ch[2], out_channels=dec_ch[1], kernel_size=3, stride=2, dimension=self.D),
            ME.MinkowskiBatchNorm(dec_ch[1]),
            ME.MinkowskiReLU()
        )
        self.occu3 = nn.Sequential(
            ME.MinkowskiConvolution(dec_ch[1], 1, kernel_size=1, bias=True, dimension=self.D),
            ME.MinkowskiSigmoid()
        )
        self.Decoder2 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(in_channels=dec_ch[1], out_channels=dec_ch[0], kernel_size=3, stride=2, dimension=self.D),
            ME.MinkowskiBatchNorm(dec_ch[0]),
            ME.MinkowskiReLU()
        )
        self.occu2 = nn.Sequential(
            ME.MinkowskiConvolution(dec_ch[0], 1, kernel_size=1, bias=True, dimension=self.D),
            ME.MinkowskiSigmoid()
        )

        self.Decoder1 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(in_channels=dec_ch[0], out_channels=self.out_channels, kernel_size=3, stride=2, dimension=self.D),
            ME.MinkowskiSigmoid()
            # ME.MinkowskiBatchNorm(self.in_channels),
            # ME.MinkowskiReLU()
        )

        self.occu1 = nn.Sequential(
            ME.MinkowskiConvolution(self.out_channels, 1, kernel_size=1, bias=True, dimension=self.D),
            ME.MinkowskiSigmoid()
        )

        # self.feat_nn = nn.Sequential(
        #     nn.Linear(self.in_channels, self.out_channels),
        #     nn.Sigmoid()
        # )

        self.weight_initialization()
        
    def forward(self, sparse_tensor):
        probs = []
        feat = sparse_tensor.features_at(batch_index=0)
        print(feat.shape)
        enc_0 = self.Encoder1(sparse_tensor)
        enc_1 = self.Encoder2(enc_0)
        enc_2 = self.Encoder3(enc_1)
        enc_3 = self.Encoder4(enc_2)
        enc_4 = self.Encoder5(enc_3)
        
        # enc_4_dense = enc_4.dense()
        # enc_4 = enc_4_dense[...,0] + enc_4_dense[...,1]
        # print(enc_4.shape)
        # sparse = spconv.SparseConvTensor.from_dense(enc_4.permute(0, 2, 3, 4, 1))
        dec_3 = self.Decoder4(enc_4)
        pred_prob = self.occu4(dec_3) # 7 x 15 x 15
        probs.append(pred_prob)

        dec_3 = dec_3 + enc_2
        dec_2 = self.Decoder3(dec_3) # 13 x 30 x 30
        pred_prob = self.occu3(dec_2)
        probs.append(pred_prob)
        
        dec_2 = dec_2 + enc_1
        dec_1 = self.Decoder2(dec_2) # 25 x 60 x 60
        pred_prob = self.occu2(dec_1)
        probs.append(pred_prob)

        dec_1 = dec_1 + enc_0
        dec_0 = self.Decoder1(dec_1dec_) # 50 x 120 x 120
        pred_prob = self.occu1(dec_0)
        probs.append(pred_prob)
        
        print(dec_0.dense()[0].shape)

            
        return preds, occu, probs,
    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

            if isinstance(m, nn.Linear):
                torch.nn.init.orthogonal_(m.weight, gain=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def cls_postprocess(self, feat_indices, pred_prob):
        batch_indices = feat_indices[:, 0]
        unique_batches = torch.unique(batch_indices)

        cm_batch = []
        pred_prob_batch = []

        for b in unique_batches:
            batch_mask = (batch_indices == b)
            cm_batch.append(feat_indices[batch_mask, 1:])
            pred_prob_batch.append(pred_prob[batch_mask].unsqueeze(-1))
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
        
    def feat_postprocess(self, preds):
        import torch.nn.functional as F

        batch_size = preds.batch_size
        output_coords = []
        
        batch_indices = preds.indices[:, 0]
        batch_counts = torch.zeros(batch_size, device=preds.indices.device)
        for batch_idx in range(batch_size):
            batch_counts[batch_idx] = (batch_indices == batch_idx).sum()

        max_num_points = batch_counts.max().int()
        for batch_idx in range(batch_size):
            batch_mask = (preds.indices[:, 0] == batch_idx)
            batch_coords = preds.features[batch_mask]            
            
            padding_size = max_num_points - batch_coords.shape[0]
            if padding_size > 0:
                padding = torch.zeros((padding_size, 3*self.max_num_points_per_voxel), dtype=batch_coords.dtype, device=batch_coords.device, requires_grad=True)
                padded_batch_coords = torch.cat([batch_coords, padding], dim=0)

            else:
                padded_batch_coords = batch_coords
            output_coords.append(padded_batch_coords)

        output = torch.stack(output_coords, dim=0)  # (batch_size, max_num_points, 3)
        if output.size(1)==0:
            raise Exception("no predicted points")
        return output
    
    def postprocess(self, feat, coords):
        batch_size = feat.shape[0]
        all_preds = []
        for batch_idx in range(batch_size):
            coords_batch = coords[batch_idx]
            feat_batch = feat[batch_idx]
            feat_batch = feat_batch.view(-1, self.max_num_points_per_voxel, 3)
            feat_batch = feat_batch[:, :, :3] ## remove t dim
            coords_batch = coords_batch[:, [2, 1, 0]]
            voxel_centers = (coords_batch.float() * torch.tensor([0.05, 0.05, 0.05])) + torch.tensor([-3.0, -3.0, -1.0])
            pred = torch.where(feat_batch == 0, torch.zeros_like(feat_batch), (voxel_centers.unsqueeze(1).to(self.device) + feat_batch*torch.tensor([0.05, 0.05, 0.05]).to(self.device)))
            preds = pred.view(-1, 3 * self.max_num_points_per_voxel)
            all_preds.append(preds)
        all_preds = torch.stack(all_preds, dim=0)    # (batch_size, max_num_points, 9) 
        return all_preds

    def indices_postprocess(self, preds):
        batch_size = preds.batch_size
        batch_indices = [[] for _ in range(batch_size)]
        for i in range(batch_size):
            batch_indices[i] = preds.indices[preds.indices[:, 0] == i][:, 1:]

        max_len = max(len(b) for b in batch_indices)
        padded_indices = torch.zeros((batch_size, max_len, 4), dtype=preds.indices.dtype)

        for i in range(batch_size):
            padded_indices[i, :batch_indices[i].shape[0], :] = batch_indices[i]
            
        return padded_indices



