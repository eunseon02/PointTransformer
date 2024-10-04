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
from os.path import join
from res import ResidualBlock

input_shape = (cfg.D, cfg.H, cfg.W)

def tensor_to_ply(tensor, filename):
    print("tensor", tensor.shape)
    points = tensor.cpu().detach().numpy()
    points = points.astype(np.float64)
    # points=  points[0]
    if points.shape[1] != 3:
        raise ValueError(f"Expected point cloud data with shape (n, 3), but got {points.shape}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pcd)



class PointCloud3DCNN(nn.Module):
    ENC_CHANNELS = [16, 32, 64, 128, 256, 512, 1024]
    DEC_CHANNELS = [16, 32, 64, 128, 256, 512, 1024]
    
    def tensorboard_launcher(self, points, step, color, tag):
        # points = occupancy_grid_to_coords(points)
        num_points = points.shape[0]
        colors = torch.tensor(color).repeat(num_points, 1)
        if num_points == 0:
            print(f"Warning: num_points is 0 at step {step}, skipping add_3d")
            # return
        else:
            writer.add_3d(
            tag,
            {
                "vertex_positions": points.float(), # (N, 3)
                "vertex_colors": colors.float()  # (N, 3)
            },
            step)
    
    def __init__(self, batch_size):
        super(PointCloud3DCNN, self).__init__()
        enc_ch = self.ENC_CHANNELS
        dec_ch = self.DEC_CHANNELS
        self.batch_size = batch_size
        self.device = cfg.device
        self.num_point_features = 4
        self.max_num_points_per_voxel = 3
        self.Encoder1 = spconv.SparseSequential(
            spconv.SubMConv4d(3*self.max_num_points_per_voxel, enc_ch[0], kernel_size=3, stride=1, indice_key="subm1"),
            nn.BatchNorm1d(enc_ch[0], momentum=0.1),
            nn.ReLU()
        )
        self.Encoder2 = spconv.SparseSequential(
            spconv.SparseConv4d(enc_ch[0], enc_ch[1], kernel_size=(3, 3, 3, 1), stride=(2, 2, 2, 1), padding=(1, 1, 1, 0),indice_key="spconv2"),
            nn.BatchNorm1d(enc_ch[1], momentum=0.1),
            nn.ReLU(),
            spconv.SubMConv4d(enc_ch[1], enc_ch[1], kernel_size=3, stride=1, padding=1, indice_key="subm2"),
            nn.BatchNorm1d(enc_ch[1], momentum=0.1),
            nn.ReLU()
        )
        self.Encoder3 = spconv.SparseSequential(
            spconv.SparseConv4d(enc_ch[1], enc_ch[2], kernel_size=(3, 3, 3, 1), stride=(2, 2, 2, 1), padding=(1, 1, 1, 0),indice_key="spconv3"), 
            nn.BatchNorm1d(enc_ch[2], momentum=0.1),
            nn.ReLU(),
            spconv.SubMConv4d(enc_ch[2], enc_ch[2], kernel_size=3, stride=1, padding=1,indice_key="subm3"),
            nn.BatchNorm1d(enc_ch[2], momentum=0.1),
            nn.ReLU()
        )
        self.Encoder4 = spconv.SparseSequential(
            spconv.SparseConv4d(enc_ch[2], enc_ch[3], kernel_size=(3, 3, 3, 1), stride=(2, 2, 2, 1), padding=(1, 1, 1, 0),indice_key="spconv4"),
            nn.BatchNorm1d(enc_ch[3], momentum=0.1),
            nn.ReLU(),
            spconv.SubMConv4d(enc_ch[3], enc_ch[3], kernel_size=3, stride=1, padding=1,indice_key="subm4"),
            nn.BatchNorm1d(enc_ch[3], momentum=0.1),
            nn.ReLU()
        )
        self.Encoder5 = spconv.SparseSequential(
            spconv.SparseConv4d(enc_ch[3], enc_ch[4], kernel_size=(3, 3, 3, 1), stride=(2, 2, 2, 1), padding=(1, 1, 1, 0),indice_key="spconv5"),
            nn.BatchNorm1d(enc_ch[4], momentum=0.1),
            nn.ReLU(),
            spconv.SubMConv4d(enc_ch[4], enc_ch[4], kernel_size=3, stride=1, padding=1,indice_key="subm5"),
            nn.BatchNorm1d(enc_ch[4], momentum=0.1),
            nn.ReLU()
        )
        ## decoder
        self.Decoder5 = spconv.SparseSequential(
            spconv.SparseInverseConv4d(dec_ch[4], dec_ch[3], kernel_size=(3, 3, 3, 1),indice_key="spconv5"),
            nn.BatchNorm1d(dec_ch[3], momentum=0.1),
            nn.ReLU(),
            spconv.SubMConv4d(dec_ch[3], dec_ch[3], kernel_size=3, stride=1, padding=1, indice_key="subm5d"), 
            nn.BatchNorm1d(dec_ch[3], momentum=0.1),
            nn.ReLU(),
        )
        self.cls5 = spconv.SparseSequential(
            spconv.SubMConv4d(dec_ch[3], 2, kernel_size=1, stride=2, padding=0, indice_key="subm5d"),  
        )
        self.Decoder4 = spconv.SparseSequential(
            spconv.SparseInverseConv4d(dec_ch[3], dec_ch[2], kernel_size=(3, 3, 3, 1),indice_key="spconv4"),
            nn.BatchNorm1d(dec_ch[2], momentum=0.1),
            nn.ReLU(),
            spconv.SubMConv4d(dec_ch[2], dec_ch[2], kernel_size=3, stride=1, padding=1, indice_key="subm4d"),  
            nn.BatchNorm1d(dec_ch[2], momentum=0.1),
            nn.ReLU()
        )
        self.cls4 = spconv.SparseSequential(
            spconv.SubMConv4d(dec_ch[2], 2, kernel_size=1, stride=2, padding=0, indice_key="subm4d"),  
        )
        self.Decoder3 = spconv.SparseSequential(
            spconv.SparseInverseConv4d(dec_ch[2], dec_ch[1], kernel_size=(3, 3, 3, 1),indice_key="spconv3"),
            nn.BatchNorm1d(dec_ch[1], momentum=0.1),
            nn.ReLU(),
            spconv.SubMConv4d(dec_ch[1], dec_ch[1], kernel_size=3, stride=1, padding=1, indice_key="subm3d"),  
            nn.BatchNorm1d(dec_ch[1], momentum=0.1),
            nn.ReLU()
        )
        self.cls3 = spconv.SparseSequential(
            spconv.SubMConv4d(dec_ch[1], 2, kernel_size=1, stride=2, padding=0, indice_key="subm3d"), 
        )
        self.Decoder2 = spconv.SparseSequential(
            spconv.SparseInverseConv4d(dec_ch[1], dec_ch[0], kernel_size=(3, 3, 3, 1),indice_key="spconv2"),
            nn.BatchNorm1d(dec_ch[0], momentum=0.1),
            nn.ReLU(),
            spconv.SubMConv4d(dec_ch[0], dec_ch[0], kernel_size=3, stride=1, padding=1, indice_key="subm2d"), 
            nn.BatchNorm1d(dec_ch[0], momentum=0.1),
            nn.ReLU()
        )
        self.Decoder1 = spconv.SparseSequential(
            spconv.SubMConv4d(dec_ch[0],3*self.max_num_points_per_voxel, kernel_size=3, stride=1, indice_key="subm1"),
            nn.BatchNorm1d(3*self.max_num_points_per_voxel, momentum=0.1),
            # nn.BatchNorm1d(1, momentum=0.1),

            nn.ReLU()
        )
        self.cls2 = spconv.SparseSequential(
            spconv.SubMConv4d(dec_ch[0], 2, kernel_size=1, stride=2, padding=0, indice_key="subm2d"), 
        )
        self.res_block1 = ResidualBlock(in_channels=9, out_channels=6)
        self.res_block2 = ResidualBlock(in_channels=6, out_channels=3)
        self.res_block3 = ResidualBlock(in_channels=3, out_channels=1)

        # self.conv1 = nn.Sequential(
        #     nn.Conv3d(3*self.max_num_points_per_voxel, 8, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.BatchNorm3d(8))
        # self.conv2 = nn.Sequential(
        #     nn.Conv3d(8, 4, kernel_size=5, padding=2),
        #     nn.ReLU(),
        #     nn.BatchNorm3d(4))
        # self.conv3 = nn.Sequential(
        #     nn.Conv3d(4, 1, kernel_size=7, padding=3),
        #     nn.ReLU(),
        #     nn.BatchNorm3d(1))
        # self.conv1d = nn.Conv1d(in_channels=dec_ch[0], out_channels=dec_ch[0], kernel_size=2, stride=1)
        # self.dense = spconv.ToDense()
        self.conv = nn.Conv3d(in_channels=3*self.max_num_points_per_voxel, out_channels=1, kernel_size=1)

    def forward(self, sparse_tensor):
        probs = []
        cm = []

        enc_0 = self.Encoder1(sparse_tensor)
        enc_1 = self.Encoder2(enc_0)
        enc_2 = self.Encoder3(enc_1)
        enc_3 = self.Encoder4(enc_2)
        enc_4 = self.Encoder5(enc_3)

        dec_3 = self.Decoder5(enc_4)
        feat_cls5 = self.cls5(dec_3) # 7 x 15 x 15
        pred_prob = F.softmax(feat_cls5.features, 1)[:, 1]
        cm_, pred_prob = self.cls_postprocess(feat_cls5.indices, pred_prob)
        probs.append(pred_prob)
        cm.append(cm_)

        dec_3 = dec_3 + enc_3
        dec_2 = self.Decoder4(dec_3) # 13 x 30 x 30
        feat_cls4 = self.cls4(dec_2)
        pred_prob = F.softmax(feat_cls4.features, 1)[:, 1]
        cm_, pred_prob = self.cls_postprocess(feat_cls4.indices, pred_prob)
        probs.append(pred_prob)
        cm.append(cm_)
        
        dec_2 = dec_2 + enc_2
        dec_1 = self.Decoder3(dec_2) # 25 x 60 x 60
        feat_cls3 = self.cls3(dec_1)
        pred_prob = F.softmax(feat_cls3.features, 1)[:, 1]
        cm_, pred_prob = self.cls_postprocess(feat_cls3.indices, pred_prob)
        probs.append(pred_prob)
        cm.append(cm_)

        dec_1 = dec_1 + enc_1
        dec_0 = self.Decoder2(dec_1) # 50 x 120 x 120
        feat_cls2 = self.cls2(dec_0)
        pred_prob = F.softmax(feat_cls2.features, 1)[:, 1]
        cm_, pred_prob = self.cls_postprocess(feat_cls2.indices, pred_prob)
        probs.append(pred_prob)
        cm.append(cm_)
        
        # dec_0 = self.Decoder1(dec_0)
        dec_0_dense = dec_0.dense()  # batch_size, channels, depth, height, width, time
        f_occu = dec_0_dense[...,0] + dec_0_dense[...,1] # batch_size, channels, depth, height, width, time
        # occu = self.conv1(f_occu)
        # occu = self.conv2(occu)
        # occu = self.conv3(occu) # batch_size, channels, depth, height, width
        # occu = torch.tanh(occu) # batch, 1, D, W, H
        occu, _ = torch.max(f_occu, dim=1, keepdim=True)

        preds = self.get_pointcloud(f_occu)
        
        
        # occu = self.res_block1(f_occu)
        # occu = self.res_block2(occu)
        # occu = self.res_block3(occu)

        # occu = self.conv(f_occu)
        # print(f_occu.shape)
        # occu = f_occu[:, 0, :, :, :] + f_occu[:, 1, :, :, :] + f_occu[:, 2, :, :, :]
        # occu = occu.unsqueeze(1)
        
        # feat = self.feat_postprocess(dec_0) # batch, n, 12
        # if feat.requires_grad:
        #     feat.retain_grad()
        # feat = torch.sigmoid(feat)
        # coords = self.indices_postprocess(dec_0) # batch, n, 4
        # preds = self.postprocess(feat, coords)
        # # print(preds.grad_fn)
        
        # preds = preds.view(self.batch_size, -1, 3)
            
        return preds, occu, probs, cm, f_occu
    
    def get_pointcloud(self, dense_tensor):
        batch_size, channels, depth, height, width = dense_tensor.shape
        coord = []
        feat = []
        
        max_points = 0
        for i in range(batch_size):
            occupied_voxels = (dense_tensor[i] > 0).any(dim=0).nonzero(as_tuple=False)
            max_points = max(max_points, occupied_voxels.shape[0])
        for i in range(batch_size):
            # occupied_voxels = (dense_tensor[i] > 0.0).all(dim=0).nonzero(as_tuple=False)
            
            voxel_coords = occupied_voxels[:, [2, 1, 0]].float()
            points = (voxel_coords * torch.tensor([0.05, 0.05, 0.05]).to(self.device)) + torch.tensor([-3.0, -3.0, -1.0]).to(self.device)
            
            # features = dense_tensor2[i, :, occupied_voxels[:, 0], occupied_voxels[:, 1], occupied_voxels[:, 2]]
            
            padded_coords = torch.zeros((max_points, 3), device=self.device)
            # padded_features = torch.zeros((max_points, features.shape[1]), device=self.device)
            
            num_points = points.shape[0]
            padded_coords[:num_points, :] = points
            # padded_features[:num_points, :] = features.T

            coord.append(padded_coords)
            # feat.append(padded_features)

        coord = torch.stack(coord)
        # feat = torch.stack(feat)
        # pred = self.postprocess(coord, feat)
        return coord

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

    def process_pointclouds(self, pc):
        np.random.seed(50051)
        from spconv.utils import Point2VoxelGPU3d
        from spconv.pytorch.utils import PointToVoxel
        pc = pc.to(device)

        # Voxel generator
        gen = PointToVoxel(
            vsize_xyz=[0.05, 0.05, 0.05],
            coors_range_xyz=[-3, -3, -1, 3, 3, 1.5],
            num_point_features=self.num_point_features,
            max_num_voxels=600000,
            max_num_points_per_voxel=self.max_num_points_per_voxel,
            device=device
        )

        # Convert torch.Tensor to numpy and then to tensorview Tensor

        # Generate voxels
        print("pc", pc.shape)
        pc = pc.view(-1, 3)
        voxels_tv, indices_tv, num_p_in_vx_tv, _ = gen.generate_voxel_with_id(pc)
        # Convert tensorview Tensors to PyTorch Tensors
        voxels_torch = torch.tensor(voxels_tv.cpu().numpy(), dtype=torch.float32).cuda()
        indices_torch = torch.tensor(indices_tv.cpu().numpy(), dtype=torch.int32).cuda()
        # tensor_to_ply(indices_torch, "indices.ply")
        print("indices", indices_tv.shape)
        # Process valid voxels only
        valid = num_p_in_vx_tv.cpu().numpy() > 0
        voxels_flatten = voxels_torch.view(-1, self.num_point_features * self.max_num_points_per_voxel)[valid]
        indices_combined = torch.cat([torch.zeros(indices_tv.shape[0], 1, dtype=torch.int32).cuda(), indices_torch], dim=1)[valid]

        # Create SparseConvTensor
        sparse_tensor = spconv.SparseConvTensor(voxels_flatten, indices_combined, input_shape, self.batch_size)

        # Forward pass through the network
        features = self.forward(sparse_tensor)
        print("feature", features.features.shape)

        # Extract and save occupied coordinates
        occupied_coords = features.indices[:, 1:]
        tensor_to_ply(occupied_coords, "occupied_coords.ply")

def tensor_to_ply(tensor, filename="features.ply"):
    tensor = tensor.cpu().numpy()
    points = tensor.astype(np.float64)
    # points=  points.squeeze(0)
    if points.shape[1] != 3:
        raise ValueError(f"Expected point cloud data with shape (n, 3), but got {points.shape}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pcd)
if __name__ == "__main__":
    pcd1 = o3d.io.read_point_cloud("dataset/train/batch_0/pts_0000_gt.ply")
    points1 = np.asarray(pcd1.points)
    tensor_to_ply(torch.Tensor(points1), "input.ply")
    PointCloud3DCNN(1).to(device).process_pointclouds(torch.tensor(points1, dtype=torch.float32))





