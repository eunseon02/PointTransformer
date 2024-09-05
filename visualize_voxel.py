import open3d as o3d
import numpy as np

# .ply 파일 경로 설정
ply_file_path1 = "voxels_torch.ply"  # 첫 번째 .ply 파일 경로
ply_file_path2 = "voxel_centers.ply"   # 두 번째 .ply 파일 경로

# 첫 번째 .ply 파일 읽기
pcd1 = o3d.io.read_point_cloud(ply_file_path1)
points1 = np.asarray(pcd1.points)  # 포인트 클라우드를 numpy 배열로 변환

# 두 번째 .ply 파일 읽기
pcd2 = o3d.io.read_point_cloud(ply_file_path2)
points2 = np.asarray(pcd2.points)  # 포인트 클라우드를 numpy 배열로 변환

# 복셀 크기 및 범위 설정
voxel_size = 0.05
x_min, y_min, z_min = -3, -3, -1
x_max, y_max, z_max = 3, 3, 1.5

# 첫 번째 파일 범위 내 포인트 필터링
mask1 = (
    (points1[:, 0] >= x_min) & (points1[:, 0] <= x_max) &
    (points1[:, 1] >= y_min) & (points1[:, 1] <= y_max) &
    (points1[:, 2] >= z_min) & (points1[:, 2] <= z_max)
)
points_in_range1 = points1[mask1]

# 두 번째 파일 범위 내 포인트 필터링
mask2 = (
    (points2[:, 0] >= x_min) & (points2[:, 0] <= x_max) &
    (points2[:, 1] >= y_min) & (points2[:, 1] <= y_max) &
    (points2[:, 2] >= z_min) & (points2[:, 2] <= z_max)
)
points_in_range2 = points2[mask2]

import open3d as o3d
import numpy as np

# 복셀 그리드를 선으로 그리기 위한 함수
def create_voxel_lines(voxel_grid):
    lines = []
    points = []
    for voxel in voxel_grid.get_voxels():
        voxel_min_bound = voxel.grid_index * voxel_grid.voxel_size + voxel_grid.origin
        voxel_max_bound = voxel_min_bound + voxel_grid.voxel_size * np.ones(3)

        # 8개의 꼭짓점
        p0 = voxel_min_bound
        p1 = [voxel_max_bound[0], voxel_min_bound[1], voxel_min_bound[2]]
        p2 = [voxel_max_bound[0], voxel_max_bound[1], voxel_min_bound[2]]
        p3 = [voxel_min_bound[0], voxel_max_bound[1], voxel_min_bound[2]]
        p4 = [voxel_min_bound[0], voxel_min_bound[1], voxel_max_bound[2]]
        p5 = [voxel_max_bound[0], voxel_min_bound[1], voxel_max_bound[2]]
        p6 = voxel_max_bound
        p7 = [voxel_min_bound[0], voxel_max_bound[1], voxel_max_bound[2]]

        # 꼭짓점을 리스트에 추가
        idx_base = len(points)
        points.extend([p0, p1, p2, p3, p4, p5, p6, p7])

        # 12개의 모서리를 선으로 연결
        lines.extend([
            [idx_base + 0, idx_base + 1], [idx_base + 1, idx_base + 2],
            [idx_base + 2, idx_base + 3], [idx_base + 3, idx_base + 0],
            [idx_base + 4, idx_base + 5], [idx_base + 5, idx_base + 6],
            [idx_base + 6, idx_base + 7], [idx_base + 7, idx_base + 4],
            [idx_base + 0, idx_base + 4], [idx_base + 1, idx_base + 5],
            [idx_base + 2, idx_base + 6], [idx_base + 3, idx_base + 7]
        ])

    return points, lines

# 첫 번째 파일 복셀 그리드 생성
voxel_grid1 = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
    pcd1, voxel_size=voxel_size, min_bound=[x_min, y_min, z_min], max_bound=[x_max, y_max, z_max]
)

# 두 번째 파일 복셀 그리드 생성
voxel_grid2 = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
    pcd2, voxel_size=voxel_size, min_bound=[x_min, y_min, z_min], max_bound=[x_max, y_max, z_max]
)

# 복셀 그리드를 선으로 변환
points1, lines1 = create_voxel_lines(voxel_grid1)
points2, lines2 = create_voxel_lines(voxel_grid2)

# 첫 번째 파일의 선셋 생성
line_set1 = o3d.geometry.LineSet()
line_set1.points = o3d.utility.Vector3dVector(points1)
line_set1.lines = o3d.utility.Vector2iVector(lines1)

# 두 번째 파일의 선셋 생성
line_set2 = o3d.geometry.LineSet()
line_set2.points = o3d.utility.Vector3dVector(points2)
line_set2.lines = o3d.utility.Vector2iVector(lines2)

# 첫 번째 파일 선택된 범위 내에서의 포인트 생성 (파란색)
pcd_in_range1 = o3d.geometry.PointCloud()
pcd_in_range1.points = o3d.utility.Vector3dVector(points_in_range1)
pcd_in_range1.paint_uniform_color([0.0, 0.0, 1.0])  # 파란색으로 설정

# 두 번째 파일 선택된 범위 내에서의 포인트 생성 (빨간색)
pcd_in_range2 = o3d.geometry.PointCloud()
pcd_in_range2.points = o3d.utility.Vector3dVector(points_in_range2)
pcd_in_range2.paint_uniform_color([1.0, 0.0, 0.0])  # 빨간색으로 설정

# 시각화: 두 개의 포인트 클라우드와 선으로 된 복셀 그리드를 함께 그리기
o3d.visualization.draw([pcd_in_range1, line_set1, pcd_in_range2, line_set2])
