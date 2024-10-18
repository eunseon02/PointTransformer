import torch 
from .config import config as cfg
import cumm.tensorview as tv
import MinkowskiEngine as ME
import numpy as np
from scipy.spatial.transform import Rotation as R

def occupancy_grid(pc):
    from spconv.utils import Point2VoxelGPU3d
    from spconv.pytorch.utils import PointToVoxel
    # Voxel generator
    gen = Point2VoxelGPU3d(
        vsize_xyz=cfg.vsize_xyz,
        coors_range_xyz=cfg.coors_range_xyz,
        num_point_features=cfg.num_point_features,
        max_num_voxels=600000,
        max_num_points_per_voxel=cfg.max_num_points_per_voxel
    )
    
    batch_size = pc.shape[0]
    all_voxels, all_indices = [], []

    for batch_idx in range(batch_size):
        pc_single = pc[batch_idx]
        pc_single = tv.from_numpy(pc_single.cpu().numpy())
        voxels_tv, indices_tv, num_p_in_vx_tv = gen.point_to_voxel_hash(pc_single.cuda())

        occupancy = (num_p_in_vx_tv.cpu().numpy() > 0).astype(float)
        occupancy = torch.tensor(occupancy, dtype=torch.float32).to(cfg.device).view(-1, 1)  # shape [N, 1]
        
        indices_torch = torch.tensor(indices_tv.cpu().numpy(), dtype=torch.int32).to(cfg.device)

        batch_indices = torch.full((indices_torch.shape[0], 1), batch_idx, dtype=torch.int32).to(cfg.device)
        indices_combined = torch.cat([batch_indices, indices_torch], dim=1)
        all_voxels.append(occupancy)
        all_indices.append(indices_combined.int())

    features_tc = torch.cat(all_voxels, dim=0)
    indices_tc = torch.cat(all_indices, dim=0)
    
    sparse_tensor = ME.SparseTensor(features=features_tc,
                                    coordinates=indices_tc,
                                    quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)
    del indices_torch, all_voxels, all_indices

    return sparse_tensor , indices_tc



def preprocess(pc):
    from spconv.utils import Point2VoxelGPU3d
    from spconv.pytorch.utils import PointToVoxel

    # Voxel generator
    gen = Point2VoxelGPU3d(
        vsize_xyz=cfg.vsize_xyz,
        coors_range_xyz=cfg.coors_range_xyz,
        num_point_features=cfg.num_point_features,
        max_num_voxels=600000,
        max_num_points_per_voxel=cfg.max_num_points_per_voxel
        )

    batch_size = pc.shape[0]
    all_voxels, all_indices = [], []

    for batch_idx in range(batch_size):
        pc_single = pc[batch_idx]
        pc_single = tv.from_numpy(pc_single.detach().cpu().numpy())

        voxels_tv, indices_tv, num_p_in_vx_tv = gen.point_to_voxel_hash(pc_single.cuda())
        voxels_torch = torch.tensor(voxels_tv.cpu().numpy(), dtype=torch.float32).to(cfg.device)
        indices_torch = torch.tensor(indices_tv.cpu().numpy(), dtype=torch.int32).to(cfg.device)
        ## sub-voxel feature
        indices_torch_trans = indices_torch[:, [2, 1, 0]] 
        voxel_centers = (indices_torch_trans.float() * torch.tensor([0.05, 0.05, 0.05]).to(cfg.device)) + torch.tensor([-3.0, -3.0, -1.0]).to(cfg.device)
        # tensor_to_ply(voxel_centers[0].view(-1, 3), "voxel_centers.ply")
        t_values = voxels_torch[:, :, 3] 
        voxels_torch = voxels_torch[:, :, :3]
        relative_pose = torch.where(voxels_torch == 0, torch.tensor(0.0).to(voxels_torch.device), (voxels_torch - voxel_centers.unsqueeze(1)) / cfg.voxel_size.to(cfg.device))
        relative_pose = torch.cat([relative_pose, t_values.unsqueeze(-1)], dim=2)
        voxels_flatten = relative_pose.view(-1, 4 * cfg.max_num_points_per_voxel)

        valid = num_p_in_vx_tv.cpu().numpy() > 0
        indices_torch = indices_torch[valid]
        ## not using abs -> only half of lidar remain     
        
        mask_0 = (t_values == 0).any(dim=1)
        mask_1 = (t_values == 1).any(dim=1)

        indices_0 = indices_torch[mask_0].to(cfg.device)
        indices_1 = indices_torch[mask_1].to(cfg.device)
        
        t0 = torch.zeros((indices_0.shape[0], 1), dtype=torch.int32).to(cfg.device)
        t1 = torch.ones((indices_1.shape[0], 1), dtype=torch.int32).to(cfg.device)
        batch_indices_0 = torch.full((indices_0.shape[0], 1), batch_idx, dtype=torch.int32).to(cfg.device)
        batch_indices_1 = torch.full((indices_1.shape[0], 1), batch_idx, dtype=torch.int32).to(cfg.device)
        indices_combined_0 = torch.cat([batch_indices_0, indices_0, t0], dim=1)
        indices_combined_1 = torch.cat([batch_indices_1, indices_1, t1], dim=1)
        indices_combined = torch.cat([indices_combined_0, indices_combined_1], dim=0)  # [N_total, 4]
        voxels_flatten = torch.cat([voxels_flatten[mask_0], voxels_flatten[mask_1]], dim=0)  
        del indices_combined_0, indices_combined_1,batch_indices_0, batch_indices_1, t0, t1, indices_0, indices_1
        # batch_indices = torch.full((indices_torch.shape[0], 1), batch_idx, dtype=torch.int32).to(self.device)
        # indices_combined = torch.cat([batch_indices, indices_torch], dim=1)


        all_voxels.append(voxels_flatten) # N X (.num_point_features X .max_num_points_per_voxel)
        all_indices.append(indices_combined.int()) # N x (batch, D, W, H, t)
        
    features_tc = torch.cat(all_voxels, dim=0)
    indices_tc = torch.cat(all_indices, dim=0)
    sparse_tensor = ME.SparseTensor(features=features_tc,
                                    coordinates=indices_tc,
                                    quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE)
    
    
    del voxels_torch, indices_torch, relative_pose, voxel_centers, all_voxels, all_indices
    return sparse_tensor


def transform_point_cloud(point_cloud, pos, quat):
    """
    Transform point cloud to world frame using position and quaternion.
    """
    quat = quat / np.linalg.norm(quat)
    r = R.from_quat(quat)
    rotation_matrix = r.as_matrix()

    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = pos.flatten()


    ones = torch.ones((point_cloud.shape[0], 1), dtype=torch.float32)
    pc_homo = torch.cat([point_cloud, ones], dim=1)
    transformation_matrix_torch = torch.from_numpy(transformation_matrix).float()
    transformed_pc_homo = torch.matmul(transformation_matrix_torch, pc_homo.T).T

    transformed_pc = transformed_pc_homo[:, :3]
    del point_cloud, ones, pc_homo, transformation_matrix_torch, transformed_pc_homo
    return transformed_pc


def pad_or_trim_cloud(pc, target_size=3000):
    n = pc.size(0)
    if n < target_size:
        padding = torch.zeros((target_size - n, 3))
        pc = torch.cat([pc, padding], dim=0)
    elif n > target_size:
        pc = pc[:target_size, :]
    return pc