import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# PLY 파일 로드
ply_file_path = 'voxels_flatten3.ply'  # 실제 PLY 파일 경로로 변경하세요
pcd = o3d.io.read_point_cloud(ply_file_path)

# 포인트 클라우드 데이터를 numpy 배열로 변환
points = np.asarray(pcd.points)

# 시각화
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap='viridis', s=1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
