import open3d as o3d

# 첫 번째 PLY 파일 읽기
ply_file_1 = "occupancy.ply"  # 첫 번째 PLY 파일 경로
pcd1 = o3d.io.read_point_cloud(ply_file_1)

# 두 번째 PLY 파일 읽기
ply_file_2 = "occupancy3.ply"  # 두 번째 PLY 파일 경로
pcd2 = o3d.io.read_point_cloud(ply_file_2)

# 첫 번째 포인트 클라우드 색상 변경 (빨간색)
pcd1.paint_uniform_color([1, 0, 0])

# 두 번째 포인트 클라우드 색상 변경 (파란색)
pcd2.paint_uniform_color([0, 0, 1])

# 두 개의 포인트 클라우드를 함께 시각화
o3d.visualization.draw_geometries([pcd1, pcd2])
