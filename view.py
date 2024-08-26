import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_ply(filename):
    # Read the PLY file using Open3D
    pcd = o3d.io.read_point_cloud(filename)
    
    # Convert the point cloud to a numpy array
    points = np.asarray(pcd.points)
    
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the point cloud data
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Show plot
    plt.show()

# Example usage
plot_ply("occupancy_grid.ply")
