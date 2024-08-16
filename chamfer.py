import torch
import open3d as o3d
import numpy as np
from chamfer_distance import ChamferDistance as chamfer_dist

def read_ply(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points, dtype=np.float32)
    return torch.tensor(points)

def compute_chamfer_loss(file_path1, file_path2):
    # PLY 파일에서 포인트 클라우드 읽기
    preds = read_ply(file_path1).unsqueeze(0)  # (N, 3) -> (1, N, 3) for batch processing
    preds = torch.nan_to_num(preds, nan=0.0)
    gts_orgin = read_ply(file_path2).unsqueeze(0)
    gts_orgin = torch.nan_to_num(gts_orgin, nan=0.0)

    # GPU로 이동
    preds = preds.cuda()
    gts_orgin = gts_orgin.cuda()

    # Chamfer Distance 계산
    chd = chamfer_dist()
    dist1, dist2, idx1, idx2 = chd(preds, gts_orgin)
    loss2 = torch.mean(dist1) + torch.mean(dist2)
    print(loss2)

    return loss2

if __name__ == "__main__":
    file_path1 = "pts.ply"
    file_path2 = "output.ply"

    loss = compute_chamfer_loss(file_path1, file_path2)
    print(f"Chamfer Loss: {loss.item()}")
