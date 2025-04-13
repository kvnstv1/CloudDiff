import tkinter as tk
from tkinter import filedialog
import numpy as np
import open3d as o3d


#================================================================================================================================

def selectFile():
    """
    A method to select a file using fileDialog
    """
    root = tk.Tk()
    root.withdraw()
    source_path = filedialog.askopenfilename(title="Select source file")
    target_path = filedialog.askopenfilename(title="Select target file")
    
    if not (source_path and target_path):
        print("Missing files!")
        return
    
    return source_path, target_path

#================================================================================================================================

def processFiles(sourceFile, targetFile):
    """
    A method to load files and returns point clouds
    """
    source = o3d.io.read_point_cloud(sourceFile)
    target = o3d.io.read_point_cloud(targetFile)
    source_down = source.voxel_down_sample(voxel_size=0.8)
    target_down = target.voxel_down_sample(voxel_size=0.8)
    return source_down, target_down

#================================================================================================================================

def visualize(source, target):
    """
    A method to visualize the source and target file on the same frame
    """
    source_pcd = o3d.geometry.PointCloud()
    target_pcd = o3d.geometry.PointCloud()

    source_points = np.asarray(source.points)
    target_points = np.asarray(target.points)

    source_pcd.points = o3d.utility.Vector3dVector(source_points)
    target_pcd.points = o3d.utility.Vector3dVector(target_points)

    source_pcd.paint_uniform_color([1, 0, 0])  # Red for source
    target_pcd.paint_uniform_color([0, 1, 0])  # Green for target

    print(f"Source point cloud: {len(source_pcd.points)} points")
    print(f"Target point cloud: {len(target_pcd.points)} points")

    print("Visualizing point clouds...")
    o3d.visualization.draw_geometries([source_pcd, target_pcd])

#================================================================================================================================

def main():
    sourcePath, targetPath = selectFile()
    source, target = processFiles(sourcePath, targetPath)
    visualize( source, target)

#================================================================================================================================


if __name__ == "__main__":
    main()