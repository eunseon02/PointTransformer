#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

import os
import torch
import numpy as np
import argparse
from model_ME import PointCloud3DCNN
from collections import OrderedDict

import wandb
wandb.init(project="Reconstruction_Result")
os.makedirs('logs', exist_ok=True)


class PointCloudProcessor(Node):
    def __init__(self, args):
        super().__init__('pointcloud_reconstruction')
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.model = PointCloud3DCNN(1, in_channels=12, out_channels=12, dimension=4, n_depth=4).to(self.device)
        self.model_path = args.model_path
        if self.model_path != '':
            self._load_pretrain(args.model_path)

        self.step = 0

        self.subscription = self.create_subscription(
            PointCloud2,
            '/zed2i/zed_node/point_cloud/cloud_registered',
            self.pc_callback,
            qos
        )
    

    def pc_callback(self, msg: PointCloud2):
        self.step += 1

        points_list = list(point_cloud2.read_points(
            msg,
            field_names=('x', 'y', 'z'),
            skip_nans=True
        ))
        if not points_list:
            self.get_logger().warn('Received empty point cloud.')
            return

        xyz_array = np.array([ [x, y, z] for x, y, z in points_list ], dtype=np.float32) 

        xyz_tensor = torch.from_numpy(xyz_array)  # shape: (N, 3)
        # wandb_log(xyz_tensor, )

        data = {
            "lidar" : xyz_tensor.unsqueeze(0)
        }

        self.get_logger().info(f'Point cloud tensor shape: {xyz_tensor.shape}')
        pred = self.model.process_pointclouds(data, self.step)


    def _load_pretrain(self, pretrain):
        # Load checkpoint
        checkpoint = torch.load(pretrain, map_location='cpu')
        # Extract the model's state dictionary from the checkpoint
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        new_state_dict = OrderedDict()
        for key, val in state_dict.items():
            if key.startswith('module.'):
                name = key[len('module.'):]  # remove 'module.' prefix
            else:
                name = key
            new_state_dict[name] = val
        # Load the state dictionary into the model
        self.model.load_state_dict(new_state_dict)
        print(f"Model loaded from {pretrain}")

def get_parser():
    parser = argparse.ArgumentParser(description='Unsupervised Point Cloud Feature Learning')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Path to load model')
    args = parser.parse_args()
    return args

def main():
    args = get_parser()
    rclpy.init()
    node = PointCloudProcessor(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
