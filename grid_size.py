import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# PLY 파일을 로드합니다
ply_file_path = "gts_voxel.ply"  # PLY 파일의 경로를 입력하세요
pcd = o3d.io.read_point_cloud(ply_file_path)

# 포인트 클라우드의 좌표를 numpy 배열로 변환합니다
points = np.asarray(pcd.points)

# 3D 플롯을 생성합니다
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 포인트 클라우드를 3D로 시각화합니다
ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c='blue', marker='o')

# 축 레이블 설정
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

# 그래프를 보여줍니다
plt.show()
