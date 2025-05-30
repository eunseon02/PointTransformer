import torch
from config import config as cfg
import numpy as np
import open3d as o3d
import wandb
import logging

def wandb_log(tensor, step_, tag, filename, last = False):
    points = tensor.cpu().detach().numpy()
    points = points.astype(np.float64)

    if points.shape[0] == 0:
        logging.warning(f"[WARNING] wandb_log: no points to write for tag={tag}, step={step_}")
    
    if points.shape[1] != 3:
        raise ValueError(f"Expected point cloud data with shape (n, 3), but got {points.shape}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pcd)
    del pcd

    # prefix = tag.rsplit('/', 1)[0]
    # step_tag = f"{prefix}/step" 
    # print(step_)

    wandb.log({tag: wandb.Object3D(points)}, step = step_)
    # if (step_ - 198) % 200 != 0:
    #     wandb.log({tag: wandb.Object3D(points)}, step = step_, commit=False)
    # else:
    #     wandb.log({tag: wandb.Object3D(points)}, step = step_, commit=True)
            # wandb.finish()

def occupancy_grid_to_coords(occupancy_grid):
    # occupancy_grid = occupancy_grid.permute(0, 4, 1, 2, 3)
    _, _, H, W, D = occupancy_grid.shape
    occupancy_grid = occupancy_grid[0, 0]
    indices = torch.nonzero(occupancy_grid > 0, as_tuple=False)
    # print(f"occupancy_grid shape: {occupancy_grid.shape}")
    # print(f"[min, max, mean = {occupancy_grid.min().item():.4f}, {occupancy_grid.max().item():.4f}, {occupancy_grid.mean().item():.4f}")
    # 몇 개의 포지티브가 있는지
    pos = (occupancy_grid > 0).sum().item()
    # print(f">0 개수 = {pos}")

    return indices

def profileit(func):
    def wrapper(*args, **kwargs):
        datafn = func.__name__ + ".profile" # Name the data file sensibly
        prof = cProfile.Profile()
        retval = prof.runcall(func, *args, **kwargs)
        prof.dump_stats(datafn)
        return retval
    return wrapper

def tensorboard_launcher(points, step, color, tag, writer=None):
    MAX_POINTS_FOR_LOG = 10000
    if points.shape[0] > MAX_POINTS_FOR_LOG:
        points = points[:MAX_POINTS_FOR_LOG]
    if writer is None:
        writer = cfg.writer

    points = points.detach().cpu().float()
    mask   = torch.isfinite(points).all(dim=1)
    points = points[mask]

    num_points = points.shape[0]
    colors = torch.tensor(color).repeat(num_points, 1)
    if num_points == 0:
        print(f"Warning: num_points is 0 : {tag}")
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
