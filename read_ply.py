import open3d as o3d

def display_ply_files(file_path1, file_path2):
    # Attempt to read the first PLY file
    try:
        pcd1 = o3d.io.read_point_cloud(file_path1)
        if pcd1.is_empty():
            print(f"Error: The point cloud in {file_path1} is empty or the file could not be read.")
            return
        else:
            print(f"Successfully read {file_path1}.")
    except Exception as e:
        print(f"Failed to read {file_path1}: {e}")
        return

    # Attempt to read the second PLY file
    try:
        pcd2 = o3d.io.read_point_cloud(file_path2)
        if pcd2.is_empty():
            print(f"Error: The point cloud in {file_path2} is empty or the file could not be read.")
            return
        else:
            print(f"Successfully read {file_path2}.")
    except Exception as e:
        print(f"Failed to read {file_path2}: {e}")
        return

    # Display both point clouds together
    o3d.visualization.draw_geometries([pcd1, pcd2], window_name="Two PLY Files Viewer")

if __name__ == "__main__":
    # Replace with your actual PLY file paths
    file_path1 = "transformed_pred9.ply"
    file_path2 = "pts_9.ply"
    
    display_ply_files(file_path1, file_path2)
