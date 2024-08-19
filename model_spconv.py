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
from loss import NSLoss
import cumm


input_shape = (cfg.D, cfg.H, cfg.W)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# def numpy_to_open3d_point_cloud(points):
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     o3d.visualization.draw_geometries([pcd])
class PointCloud3DCNN(nn.Module):
    ENC_CHANNELS = [16, 32, 64, 128, 256, 512, 1024]
    DEC_CHANNELS = [16, 32, 64, 128, 256, 512, 1024]
    
    def __init__(self, batch_size):
        super(PointCloud3DCNN, self).__init__()
        enc_ch = self.ENC_CHANNELS
        dec_ch = self.DEC_CHANNELS
        self.batch_size = batch_size
        self.num_point_features = 3
        self.max_num_points_per_voxel = 5
        self.Encoder1 = spconv.SparseSequential(
            spconv.SubMConv3d(self.num_point_features*self.max_num_points_per_voxel, enc_ch[0], kernel_size=3, stride=1, indice_key="subm1"),
            nn.BatchNorm1d(enc_ch[0], momentum=0.1),
            nn.ReLU()
        )
        self.Encoder2 = spconv.SparseSequential(
            spconv.SparseConv3d(enc_ch[0], enc_ch[1], kernel_size=3, stride=2, indice_key="spconv2"), # 1779 -> 2393
            nn.BatchNorm1d(enc_ch[1], momentum=0.1),
            nn.ReLU(),
            spconv.SubMConv3d(enc_ch[1], enc_ch[1], kernel_size=3, stride=1, indice_key="subm2"),
            nn.BatchNorm1d(enc_ch[1], momentum=0.1),
            nn.ReLU()
        )
        self.Encoder3 = spconv.SparseSequential(
            spconv.SparseConv3d(enc_ch[1], enc_ch[2], kernel_size=3, stride=2, indice_key="spconv3"), # 2393 ->713
            nn.BatchNorm1d(enc_ch[2], momentum=0.1),
            nn.ReLU(),
            spconv.SubMConv3d(enc_ch[2], enc_ch[2], kernel_size=3, stride=1, indice_key="subm3"),
            nn.BatchNorm1d(enc_ch[2], momentum=0.1),
            nn.ReLU()
        )
        self.Encoder4 = spconv.SparseSequential(
            spconv.SparseConv3d(enc_ch[2], enc_ch[3], kernel_size=3, stride=2, indice_key="spconv4"),
            nn.BatchNorm1d(enc_ch[3], momentum=0.1),
            nn.ReLU(),
            spconv.SubMConv3d(enc_ch[3], enc_ch[3], kernel_size=3, stride=1, indice_key="subm4"),
            nn.BatchNorm1d(enc_ch[3], momentum=0.1),
            nn.ReLU()
        )
        self.Encoder5 = spconv.SparseSequential(
            spconv.SparseConv3d(enc_ch[3], enc_ch[4], kernel_size=3, stride=2, indice_key="spconv5"),
            nn.BatchNorm1d(enc_ch[4], momentum=0.1),
            nn.ReLU(),
            spconv.SubMConv3d(enc_ch[4], enc_ch[4], kernel_size=3, stride=1, indice_key="subm5"),
            nn.BatchNorm1d(enc_ch[4], momentum=0.1),
            nn.ReLU()
        )
        ## decoder
        self.Decoder5 = spconv.SparseSequential(
            spconv.SparseInverseConv3d(dec_ch[4], dec_ch[3], kernel_size=3, indice_key="spconv5"), # 61 -> 648
            nn.BatchNorm1d(dec_ch[3], momentum=0.1),
            nn.ReLU(),
            spconv.SubMConv3d(dec_ch[3], dec_ch[3], kernel_size=3, stride=1, padding=1, indice_key="subm5d"), 
            nn.BatchNorm1d(dec_ch[3], momentum=0.1),
            nn.ReLU(),
        )
        self.cls5 = spconv.SparseSequential(
            spconv.SubMConv3d(dec_ch[3], 1, kernel_size=1, stride=2, padding=0, indice_key="subm5d"),  
        )
        self.Decoder4 = spconv.SparseSequential(
            spconv.SparseInverseConv3d(dec_ch[3], dec_ch[2], kernel_size=3, indice_key="spconv4"),
            nn.BatchNorm1d(dec_ch[2], momentum=0.1),
            nn.ReLU(),
            spconv.SubMConv3d(dec_ch[2], dec_ch[2], kernel_size=3, stride=1, padding=1, indice_key="subm4d"),  
            nn.BatchNorm1d(dec_ch[2], momentum=0.1),
            nn.ReLU()
        )
        self.cls4 = spconv.SparseSequential(
            spconv.SubMConv3d(dec_ch[2], 1, kernel_size=1, stride=2, padding=0, indice_key="subm4d"),  
        )
        self.Decoder3 = spconv.SparseSequential(
            spconv.SparseInverseConv3d(dec_ch[2], dec_ch[1], kernel_size=3, indice_key="spconv3"),
            nn.BatchNorm1d(dec_ch[1], momentum=0.1),
            nn.ReLU(),
            spconv.SubMConv3d(dec_ch[1], dec_ch[1], kernel_size=3, stride=1, padding=1, indice_key="subm3d"),  
            nn.BatchNorm1d(dec_ch[1], momentum=0.1),
            nn.ReLU()
        )
        self.cls3 = spconv.SparseSequential(
            spconv.SubMConv3d(dec_ch[1], 1, kernel_size=1, stride=2, padding=0, indice_key="subm3d"), 
        )
        self.Decoder2 = spconv.SparseSequential(
            spconv.SparseInverseConv3d(dec_ch[1], dec_ch[0], kernel_size=3, indice_key="spconv2"),
            nn.BatchNorm1d(dec_ch[0], momentum=0.1),
            nn.ReLU(),
            spconv.SubMConv3d(dec_ch[0], dec_ch[0], kernel_size=3, stride=1, padding=1, indice_key="subm2d"), 
            nn.BatchNorm1d(dec_ch[0], momentum=0.1),
            nn.ReLU()
        )
        self.cls2 = spconv.SparseSequential(
            spconv.SubMConv3d(dec_ch[0], 1, kernel_size=1, stride=2, padding=0, indice_key="subm2d"), 
        )
        self.dense = spconv.ToDense()
        self.fc1 = nn.Linear(16,3)

        self.loss = NSLoss()

    def forward(self, sparse_tensor):
        # print("input", sparse_tensor)
        enc_0 = self.Encoder1(sparse_tensor)
        # print("enc_0", enc_0)
        enc_1 = self.Encoder2(enc_0)
        # print("enc_1", enc_1)
        enc_2 = self.Encoder3(enc_1)
        # print("enc_2", enc_2)
        enc_3 = self.Encoder4(enc_2)
        # print("enc_3", enc_3)
        enc_4 = self.Encoder5(enc_3)
        # print("enc_4", enc_4)

        
        
        dec_3 = self.Decoder5(enc_4)
        # print("dec_3", dec_3)
        # dec_2 = torch.cat((dec_3 + enc_3), dim=0)
        dec_3 = dec_3 + enc_3
        dec_2 = self.Decoder4(dec_3)
        # print("dec_2", dec_2)
        dec_2 = dec_2 + enc_2
        dec_1 = self.Decoder3(dec_2)
        # print("dec_1", dec_1)

        dec_1 = dec_1 + enc_1
        dec_0 = self.Decoder2(dec_1)
        # x = self.dense(dec_0)
        output = self.fc1(dec_0.features)
        
        # batch_size, num_channels, depth, height, width = x.shape
        # x = x.permute(0, 2, 3, 4, 1).contiguous().view(-1, num_channels)
        # print("dense_tensor", x.shape)
        # x = self.fc1(x)
        # print("out", x.shape)
        # # print("dec_0", dec_0)
        
        # return x
        return dec_0
    

    def get_loss(self, pred, gt_process, gt_pts):
        # print("pred", pred.shape)
        # print("gt_process", gt_process.shape)
        # print("gt_pts", gt_pts.shape)
        return self.loss(pred, gt_process, gt_pts)

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





