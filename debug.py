import torch
from torch.utils.tensorboard import SummaryWriter
from open3d.visualization.tensorboard_plugin import summary
from config import config as cfg

writer = cfg.writer

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

def occupancy_grid_to_coords(occupancy_grid):
    # occupancy_grid = occupancy_grid.permute(0, 4, 1, 2, 3)
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




def tensorboard_launcher(points, step, color, tag):
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
    del points, colors
    torch.cuda.empty_cache()