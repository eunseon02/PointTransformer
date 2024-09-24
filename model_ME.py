import torch
import torch.nn as nn
import numpy as np
import open3d as o3d
from config import config as cfg
from loss import NSLoss
import torch.nn.functional as F
import MinkowskiEngine as ME

input_shape = (cfg.D, cfg.H, cfg.W)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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

    def __init__(self, batch_size, in_channels, out_channels, dimension, n_depth=4):
        super(PointCloud3DCNN, self).__init__()
        assert ME is not None, "Please install MinkowskiEngine.`"
        enc_ch = self.ENC_CHANNELS
        dec_ch = self.DEC_CHANNELS
        self.n_depth = n_depth
        self.D = dimension
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batch_size = batch_size


        self.Encoder1 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=self.in_channels, out_channels=enc_ch[0], kernel_size=3, stride=2, dimension=self.D),
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
        # print("input", sparse_tensor)
        enc_0 = self.Encoder1(sparse_tensor)
        # print("enc_0", enc_0)
        enc_1 = self.Encoder2(enc_0)
        # print("enc_1", enc_1)
        enc_2 = self.Encoder3(enc_1)
        # print("enc_2", enc_2)
        enc_3 = self.Encoder4(enc_2)
        # print("enc_3", enc_3)

        dec_3 = self.Decoder4(enc_3)
        pred_prob = self.occu4(dec_3) # 5 x 14 x 14
        probs.append(pred_prob)

        dec_3 = dec_3 + enc_2
        dec_2 = self.Decoder3(dec_3)
        pred_prob = self.occu3(dec_2) # 5 x 14 x 14
        probs.append(pred_prob)

        dec_2 = dec_2 + enc_1
        dec_1 = self.Decoder2(dec_2)
        pred_prob = self.occu2(dec_1) # 5 x 14 x 14
        probs.append(pred_prob)

        dec_1 = dec_1 + enc_0
        dec_0 = self.Decoder1(dec_1)
        pred_prob = self.occu1(dec_0)
        probs.append(pred_prob)

        # pred_pc = self.postprocess(dec_0.features, dec_0.coordinates)
        # feats = self.feat_nn(dec_0.features)
        return dec_0, probs

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
    # PointCloud3DCNN(1).to(device).process_pointclouds(torch.tensor(points1, dtype=torch.float32))





