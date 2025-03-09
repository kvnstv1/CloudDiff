import tkinter as tk
from tkinter import filedialog
import numpy as np
import open3d as o3d
import pycpd  # Install with: pip install pycpd

# Downsampling function (fix)
def downsample_point_cloud(pcd, num_points=5000):
    downsampled_pcd = pcd.farthest_point_down_sample(num_points)
    return np.asarray(downsampled_pcd.points)

# File selection
def select_files():
    root = tk.Tk()
    root.withdraw()
    file_path_source = filedialog.askopenfilename(title="Select source file")
    file_path_target = filedialog.askopenfilename(title="Select target file")
    return file_path_source, file_path_target

file_path_source, file_path_target = select_files()
if not file_path_source or not file_path_target:
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

# Compute scale factor based on bounding box size ratio
bb1 = np.ptp(translated_points1, axis=0)  # Bounding box size
bb2 = np.ptp(translated_points2, axis=0)
scale_factor = np.mean(bb2 / bb1)  # Compute mean scale ratio

# Scale the source point cloud
scaled_points1 = translated_points1 * scale_factor

# Downsample for CPD
source_sampled = downsample_point_cloud(pcd1, num_points=5000)
target_sampled = downsample_point_cloud(pcd2, num_points=5000)

# Run CPD for initial global alignment (fix input)
cpd = pycpd.RigidRegistration(X=source_sampled, Y=target_sampled)
transformation_matrix = cpd.register()[0]  # Get transformation matrix

# Apply CPD transformation to full source cloud
pcd1.transform(transformation_matrix)

# Run ICP for final fine-tuning
threshold = 0.05
icp_result = o3d.pipelines.registration.registration_icp(
    pcd1, pcd2, threshold,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
)

# Apply ICP transformation
pcd1.transform(icp_result.transformation)

# Visualize the aligned point clouds
o3d.visualization.draw_geometries([pcd2, pcd1])
