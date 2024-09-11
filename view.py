import open3d as o3d

# PLY 파일 경로 설정
ply_file_path = "filtered_pc.ply"

# PLY 파일 읽기
pcd = o3d.io.read_point_cloud(ply_file_path)

# 포인트 클라우드 정보 출력
print(pcd)
print(f"Number of points: {len(pcd.points)}")

# 포인트 클라우드 시각화
o3d.visualization.draw_geometries([pcd])
