import torch
def tensor_to_ply(tensor, filename):
    # print("tensor", tensor.shape)
    points = tensor.cpu().detach().numpy()
    points = points.astype(np.float64)
    # points=  points[0]
    if points.shape[1] != 3:
        raise ValueError(f"Expected point cloud data with shape (n, 3), but got {points.shape}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pcd)

def save_single_occupancy_grid_as_ply(occupancy_grid, file_name="occupancy_grid.ply"):
    # Assume occupancy_grid is of shape (batch_size, 1, H, W, D)
    _, _, H, W, D = occupancy_grid.shape
    grid = occupancy_grid[0, 0]  # shape (H, W, D)
    occupied_voxels = torch.nonzero(grid > 0, as_tuple=False).cpu().numpy()
    occupied_voxels = occupied_voxels.astype(np.float32)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(occupied_voxels)
    o3d.io.write_point_cloud(file_name, point_cloud)
    print(f"Saved occupancy grid to {file_name}")
def occupancy_grid_to_coords(occupancy_grid):
    _, _, H, W, D = occupancy_grid.shape
    occupancy_grid = occupancy_grid[0, 0]
    indices = torch.nonzero(occupancy_grid > 0, as_tuple=False) 
    return indices

def profileit(func):
    def wrapper(*args, **kwargs):
        datafn = func.__name__ + ".profile" # Name the data file sensibly
        prof = cProfile.Profile()
        retval = prof.runcall(func, *args, **kwargs)
        prof.dump_stats(datafn)
        return retval

    return wrapper
