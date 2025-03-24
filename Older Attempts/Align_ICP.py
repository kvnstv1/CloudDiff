import tkinter as tk
from tkinter import filedialog
import numpy as np
import open3d as o3d

# File selection
file_path_source = None
file_path_target = None

def select_files():
    global file_path_source, file_path_target
    root = tk.Tk()
    root.withdraw()
    file_path_source = filedialog.askopenfilename(title="Select source file")
    file_path_target = filedialog.askopenfilename(title="Select target file")

select_files()

if file_path_source and file_path_target:
    print(f"The source file is {file_path_source}")
    print(f"The target file is {file_path_target}")
else:
    print("No files selected.")
    exit()

# Load point clouds
pcd1 = o3d.io.read_point_cloud(file_path_source)
pcd2 = o3d.io.read_point_cloud(file_path_target)

points1 = np.asarray(pcd1.points)
points2 = np.asarray(pcd2.points)

# Compute centroids
centroid1 = np.mean(points1, axis=0)
centroid2 = np.mean(points2, axis=0)

# Translate to common origin
translated_points1 = points1 - centroid1
translated_points2 = points2 - centroid2

# Compute scale factor based on standard deviation
scale_factor = np.std(translated_points2) / np.std(translated_points1)

# Scale the source point cloud
scaled_points1 = translated_points1 * scale_factor

# Convert scaled points back to Open3D point cloud
scaled_pcd1 = o3d.geometry.PointCloud()
scaled_pcd1.points = o3d.utility.Vector3dVector(scaled_points1)

# Perform ICP alignment
threshold = 0.05  # Maximum correspondence distance (adjust if needed)
icp_result = o3d.pipelines.registration.registration_icp(
    scaled_pcd1, pcd2, threshold,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
)

# Apply the ICP transformation to the source point cloud
aligned_pcd1 = scaled_pcd1.transform(icp_result.transformation)

# Translate back to match the target centroid
aligned_points1 = np.asarray(aligned_pcd1.points) + centroid2
aligned_pcd1.points = o3d.utility.Vector3dVector(aligned_points1)

# Visualize the aligned point clouds
o3d.visualization.draw_geometries([pcd2, aligned_pcd1])
