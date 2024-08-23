import os
import glob
import open3d as o3d
import pandas as pd

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
        return pcd

    def load_all_pointclouds(self):
        pointclouds = []
        for idx in range(len(self.data_files)):
            batch_size = len(self.batches[0][0])
            batch_idx = idx // batch_size
            file_idx = idx % batch_size
            data_files, gt_files, csv_file, batch_dir = self.batches[batch_idx]
            
            if file_idx >= len(data_files) or file_idx >= len(gt_files):
                continue
            
            data_file_path = os.path.join(batch_dir, data_files[file_idx])
            data_pointcloud = self.read_ply(data_file_path)
            pointclouds.append(data_pointcloud)
        return pointclouds

root_dir = 'dataset/valid' 
dataset = PointCloudDataset(root_dir)
all_pointclouds = dataset.load_all_pointclouds()

print(f"Loaded {len(all_pointclouds)} point clouds.")


index = 0  
vis = o3d.visualization.Visualizer()
vis.create_window()

# geometry is the point cloud used in your animaiton
geometry = o3d.geometry.PointCloud()
vis.add_geometry(geometry)

for idx in range(len(all_pointclouds)):    # now modify the points of your geometry
    # you can use whatever method suits you best, this is just an example
    # gt_pointcloud = dataset[idx]
    pointcloud = all_pointclouds[idx]

    # 점 구름이 비어 있는지 확인
    if len(pointcloud.points) == 0:
        print(f"Skipping empty point cloud at index {idx}.")
        continue

    geometry.clear()
    geometry.points = pointcloud.points
    
    vis.update_geometry(geometry)
    vis.poll_events()
    vis.update_renderer()