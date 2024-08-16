import numpy as np
import open3d as o3d

def load_occupancy_grid(filename):
    return np.load(filename)

def occupancy_to_point_cloud(occupancy_grid):
    # Occupied voxels (i.e., where the grid value is 1)
    occupied_voxels = np.argwhere(occupancy_grid == 1)
    
    # Convert to Open3D PointCloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(occupied_voxels)
    
    return point_cloud

def visualize_point_clouds(*point_clouds):
    o3d.visualization.draw_geometries(point_clouds)

# 사용 예시
occupancy_grid1 = load_occupancy_grid('occupancy_grid_pred.npy')
occupancy_grid2 = load_occupancy_grid('occupancy_grid_gt.npy')
# Convert occupancy grids to point clouds
pcd1 = occupancy_to_point_cloud(occupancy_grid1)
pcd2 = occupancy_to_point_cloud(occupancy_grid2)

# Optional: Color the point clouds differently for distinction
pcd1.paint_uniform_color([1, 0, 0])  # Red
pcd2.paint_uniform_color([0, 0, 1])  # Blue

# Visualize both point clouds together
visualize_point_clouds(pcd1, pcd2)
