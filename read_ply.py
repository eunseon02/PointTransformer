import open3d as o3d
import numpy as np

def read_ply(file_path):
    # Open3D를 사용하여 PLY 파일을 읽음
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd

def display_ply_files(file_path1, file_path2):
    # 두 개의 PLY 파일을 읽음
    pcd1 = read_ply(file_path1)
    pcd2 = read_ply(file_path2)

    # 첫 번째 점 구름에 색상 적용 (예: 빨간색)
    pcd1.paint_uniform_color([1, 0, 0])  # 빨간색

    # 두 번째 점 구름에 색상 적용 (예: 파란색)
    pcd2.paint_uniform_color([0, 0, 1])  # 파란색

    # 두 개의 점 구름을 리스트로 묶어 시각화
    o3d.visualization.draw_geometries([pcd1, pcd2])

if __name__ == "__main__":
    # PLY 파일 경로를 지정
    file_path1 = "indices.ply"
    file_path2 = "pts.ply"

    # 두 개의 PLY 파일을 읽어 시각화
    display_ply_files(file_path1, file_path2)
