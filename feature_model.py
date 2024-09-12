import torch
import torch.nn as nn
from model import PointTransformerV3, Point  # Adjust the import path if necessary
import torch.nn.functional as F
from loss import NSLoss

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


class PointTransformerV3ForGlobalFeature(PointTransformerV3):
    def __init__(self, batch_size,**kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.loss = NSLoss()
        self.fc1 = nn.Linear(64,3)
        

    def forward(self, data_dict):
        point = Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        # Pass through embedding and encoder
        point = self.embedding(point)
        point = self.enc(point)
        
        # If not using classification mode, pass through decoder
        # if not self.cls_mode:
        point = self.dec(point)
    
        # Extract features
        features = point.feat  # Shape: (N, C)
        features= F.max_pool1d(features.unsqueeze(0).transpose(1, 2), kernel_size=2, stride=2).squeeze().transpose(0, 1)
        features = self.fc1(features)
        
        return features
    
    
    # def get_loss(self, pred, gt_pts):
    #     return self.loss(pred, gt_pts)


    def process_pointclouds(self, data):
        # print("data",data.shape)
        # sequence_length, batch_size, history_num, num_points, _ = data.shape
        # merged_points = data.view(-1, num_points, 3)  # Flatten history, batch, and sequence dimensions
        # merged_points = merged_points.view(-1, 3)  # Flatten all dimensions into one point cloud

        # # Calculate the offset
        # offset = torch.arange(0, merged_points.size(0) + 1, num_points, device=merged_points.device)

        merged_points = torch.randn(64, 2048, 3)
        merged_points = merged_points.view(-1, 3).to(device)
        print("merged_points", merged_points.shape)
        # Create data_dict with offset
        data_dict = {
            'feat': merged_points,
            'coord': merged_points,
            'grid_size': torch.tensor([0.05]).to(merged_points.device),
            'offset': torch.arange(0, merged_points.size(0) + 1, 2048, device=merged_points.device)
        }

        # Pass through PointTransformer model
        features = self.forward(data_dict)
        # print("features". features.shape)
        # Slice features based on offset and apply max pooling
        # print(merged_points.size(0))
        # print(torch.arange(0, merged_points.size(0) + 1, 2048))
        global_features = self.max_pool_with_offset(features, torch.arange(0, features.size(0) + 1, 2048, device=merged_points.device))
        # print("global_features". global_features.shape)

        # # Reshape global features to match sequence x batch x history structure
        # global_features = global_features.view(sequence_length, batch_size, history_num, -1)

        return global_features

    def max_pool_with_offset(self, features, offset):
        global_features = []
        for i in range(len(offset) - 1):
            start = offset[i]
            end = offset[i + 1]
            point_features = features[start:end]
            global_feature = F.max_pool1d(point_features.unsqueeze(0).transpose(1, 2), kernel_size=point_features.size(0)).squeeze()
            global_features.append(global_feature)
        return torch.stack(global_features)
    
    
if __name__ == "__main__":
    obs_lidar_robot_tc = torch.load("../../env/envs/rsg_raibo_rough_terrain/lidar_data.pt")
    PointTransformerV3ForGlobalFeature(64, in_channels=3).to(device).process_pointclouds(obs_lidar_robot_tc)

