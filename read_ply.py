import open3d as o3d
import numpy as np

def read_ply(file_path):
    # Open3D를 사용하여 PLY 파일을 읽음
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd

def find_overlapping_points(pcd1, pcd2, threshold=1e-6):
    # 두 점 구름의 포인트를 numpy 배열로 변환
    points1 = np.asarray(pcd1.points)
    points2 = np.asarray(pcd2.points)

    # KDTree를 사용하여 점 간의 거리 계산
    tree1 = o3d.geometry.KDTreeFlann(pcd1)
    overlap_mask = np.zeros(points1.shape[0], dtype=bool)

    for i, point in enumerate(points2):
        [_, idx, _] = tree1.search_knn_vector_3d(point, 1)
        if np.linalg.norm(points1[idx[0]] - point) < threshold:
            overlap_mask[idx[0]] = True

    return overlap_mask

def display_ply_files(file_path1, file_path2):
    # 두 개의 PLY 파일을 읽음
    pcd1 = read_ply(file_path1)
    pcd2 = read_ply(file_path2)

    # 겹치는 포인트 찾기
    overlap_mask1 = find_overlapping_points(pcd1, pcd2)
    overlap_mask2 = find_overlapping_points(pcd2, pcd1)

    # 첫 번째 점 구름에 색상 적용
    colors1 = np.tile([1, 0, 0], (len(pcd1.points), 1))  # 빨간색
    colors1[overlap_mask1] = [0, 1, 0]  # 겹치는 부분은 녹색으로 표시

    # 두 번째 점 구름에 색상 적용
    colors2 = np.tile([0, 0, 1], (len(pcd2.points), 1))  # 파란색
    colors2[overlap_mask2] = [0, 1, 0]  # 겹치는 부분은 녹색으로 표시

    # 색상 적용
    pcd1.colors = o3d.utility.Vector3dVector(colors1)
    pcd2.colors = o3d.utility.Vector3dVector(colors2)

    # 빨간색과 파란색 점 개수 계산
    num_red_points = np.sum(~overlap_mask1)
    num_blue_points = np.sum(~overlap_mask2)
    num_green_points = np.sum(overlap_mask1)  # 녹색 점 (겹치는 점) 개수

    print(f"Number of red points (unique to file 1): {num_red_points}")
    print(f"Number of blue points (unique to file 2): {num_blue_points}")
    print(f"Number of green points (overlapping points): {num_green_points}")

    # 두 개의 점 구름을 리스트로 묶어 시각화
    o3d.visualization.draw_geometries([pcd1, pcd2])

if __name__ == "__main__":
    # PLY 파일 경로를 지정
    file_path1 = "output.ply"
    file_path2 = "points.ply"

    # 두 개의 PLY 파일을 읽어 시각화
    display_ply_files(file_path1, file_path2)
