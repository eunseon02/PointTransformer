import open3d as o3d
import numpy as np

def read_ply(file_path):
    # Open3D를 사용하여 PLY 파일을 읽음
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd

def find_common_points(pcd1, pcd2, threshold=0.01):
    # KD-Tree를 이용하여 두 점 구름 간의 공통 점을 찾음
    pcd_tree = o3d.geometry.KDTreeFlann(pcd2)
    common_indices_pcd1 = []
    common_indices_pcd2 = []

    for i, point in enumerate(np.asarray(pcd1.points)):
        [_, idx, _] = pcd_tree.search_radius_vector_3d(point, threshold)
        if len(idx) > 0:
            common_indices_pcd1.append(i)
            common_indices_pcd2.extend(idx)
    
    return common_indices_pcd1, common_indices_pcd2

def display_ply_files(file_path1, file_path2):
    # 두 개의 PLY 파일을 읽음
    pcd1 = read_ply(file_path1)
    pcd2 = read_ply(file_path2)

    # 공통점 찾기
    common_indices_pcd1, common_indices_pcd2 = find_common_points(pcd1, pcd2)

    # 모든 점을 각각 빨강(PCD1), 파랑(PCD2)으로 설정
    pcd1.paint_uniform_color([1, 0, 0])  # 빨간색
    pcd2.paint_uniform_color([0, 0, 1])  # 파란색

    # 겹치는 포인트는 초록색으로 설정
    colors_pcd1 = np.asarray(pcd1.colors)
    colors_pcd2 = np.asarray(pcd2.colors)

    for i in common_indices_pcd1:
        colors_pcd1[i] = [0, 1, 0]  # 초록색

    for i in common_indices_pcd2:
        colors_pcd2[i] = [0, 1, 0]  # 초록색

    pcd1.colors = o3d.utility.Vector3dVector(colors_pcd1)
    pcd2.colors = o3d.utility.Vector3dVector(colors_pcd2)

    # 두 개의 점 구름을 리스트로 묶어 시각화
    o3d.visualization.draw_geometries([pcd1, pcd2])

if __name__ == "__main__":
    # PLY 파일 경로를 지정
    file_path1 = "12.ply"
    file_path2 = "/root/PointTransformer/dataset/train/batch_1/pts_0018_gt.ply "

    # 두 개의 PLY 파일을 읽어 시각화
    display_ply_files(file_path1, file_path2)
