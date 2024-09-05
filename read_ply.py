import open3d as o3d
import numpy as np
from sklearn.neighbors import KDTree

# 두 개의 PLY 파일을 읽습니다.
pcd1 = o3d.io.read_point_cloud("voxels_flatten3.ply")
pcd2 = o3d.io.read_point_cloud("voxels_flatten2.ply")

# 각각의 포인트를 numpy array로 변환
points1 = np.asarray(pcd1.points)
points2 = np.asarray(pcd2.points)

# KDTree를 사용해 가까운 이웃 탐색 (임계값을 정해서 겹치는 부분 계산)
tree = KDTree(points1)
distances, indices = tree.query(points2, k=1)
threshold = 0.05  # 포인트가 겹친다고 간주할 임계값

# 각 포인트에 대해 색상 지정
colors1 = np.array([[1, 0, 0] for _ in range(len(points1))])  # 첫 번째 파일 (빨간색)

colors2 = np.array([[0, 0, 1] for _ in range(len(points2))])  # 두 번째 파일 (파란색)

# 겹치는 포인트는 초록색으로 표시
overlap_indices = np.where(distances < threshold)[0]
colors2[overlap_indices] = [0, 1, 0]  # 겹치는 부분은 초록색

# 색상 적용
pcd1.colors = o3d.utility.Vector3dVector(colors1)
pcd2.colors = o3d.utility.Vector3dVector(colors2)

# 두 포인트 클라우드를 합쳐서 시각화
o3d.visualization.draw_geometries([pcd1, pcd2])
