import os
import glob
import open3d as o3d
import pandas as pd
import numpy as np
import time
class PointCloudDataset:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.sub_region = 6.0
        self.point_cnt = 2048
        self.csv_files = []
        self.data_files = []
        self.gt_files = []
        self.csv_data_list = []
        self.batches = []
        self.batch_dir_list = []
        for batch_folder in sorted(os.listdir(root_dir)):
            print(batch_folder)
            batch_dir = os.path.join(root_dir, batch_folder)
            print(root_dir)
            if os.path.isdir(batch_dir):
                csv_file_pattern = os.path.join(batch_dir, 'delta_pose*.csv')
                csv_files = sorted(glob.glob(csv_file_pattern))
                
                for csv_file in csv_files:
                    csv_data = pd.read_csv(csv_file)
                    sorted_filenames = csv_data['filename'].tolist()

                    # 알파벳 순서대로 정렬
                    gt_files = sorted([f for f in os.listdir(batch_dir) if f.endswith('_gt.ply')])
                    data_files = sorted([f for f in os.listdir(batch_dir) if f.endswith('.ply') and not f.endswith('_gt.ply')])

                    if len(gt_files) != len(data_files):
                        print(f"Warning: Mismatch in the number of GT files and data files in {batch_dir}")
                        continue

                    self.gt_files.extend(gt_files)
                    self.data_files.extend(data_files)
                    self.batches.append((data_files, gt_files, csv_file, batch_dir))

    def read_ply(self, file_path):
        # .ply 파일을 읽어와서 open3d의 PointCloud 객체로 반환
        pcd = o3d.io.read_point_cloud(file_path)
        geometry = o3d.geometry.PointCloud()
        geometry.points = o3d.utility.Vector3dVector(np.asarray(pcd.points))
        return geometry

    def load_all_pointclouds(self):
        pointclouds = []
        data_file_paths = []
        for idx in range(len(self.data_files)):
            batch_size = len(self.batches[0][0])
            batch_idx = idx // batch_size
            file_idx = idx % batch_size
            data_files, gt_files, csv_file, batch_dir = self.batches[batch_idx]
            
            if file_idx >= len(data_files) or file_idx >= len(gt_files):
                continue
            
            data_file_path = os.path.join(batch_dir, gt_files[file_idx])
            data_pointcloud = self.read_ply(data_file_path)
            pointclouds.append(data_pointcloud)
            data_file_paths.append(data_file_path)
        return pointclouds, data_file_paths




root_dir = 'dataset/valid' 
dataset = PointCloudDataset(root_dir)
all_pointclouds, data_file_paths = dataset.load_all_pointclouds()

print(f"Loaded {len(all_pointclouds)} point clouds.")
# print("all_pointclouds",all_pointclouds[0].points)
idx = 0  

vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(all_pointclouds[0])

for idx in range(len(all_pointclouds)):
    pointcloud = all_pointclouds[idx]

    print(data_file_paths[idx])

    vis.update_geometry(pointcloud)
    vis.poll_events()
    vis.update_renderer()
    
    time.sleep(1)
vis.destroy_window()
