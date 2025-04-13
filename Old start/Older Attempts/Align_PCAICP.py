import tkinter as tk
from tkinter import filedialog
import numpy as np
import open3d as o3d

# Function to compute PCA for rough alignment
def compute_pca_alignment(source, target):
    source_mean = np.mean(source, axis=0)
    target_mean = np.mean(target, axis=0)

    # Center the data
    source_centered = source - source_mean
    target_centered = target - target_mean

    # Compute covariance matrices (3x3 instead of NxN)
    cov_source = np.cov(source_centered.T)
    cov_target = np.cov(target_centered.T)

    # Compute eigenvectors (principal axes)
    _, V_source = np.linalg.eigh(cov_source)
    _, V_target = np.linalg.eigh(cov_target)

    # Compute rotation matrix to align principal axes
    R = V_target.T @ V_source.T
    return R


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

# Compute scale factor based on bounding box size ratio
bb1 = np.ptp(translated_points1, axis=0)  # Bounding box size (point-to-point range)
bb2 = np.ptp(translated_points2, axis=0)
scale_factor = np.mean(bb2 / bb1)  # Compute mean scale ratio

# Scale the source point cloud
scaled_points1 = translated_points1 * scale_factor

# Compute PCA-based rough alignment
R_pca = compute_pca_alignment(scaled_points1, translated_points2)
aligned_points1 = scaled_points1 @ R_pca.T  # Apply rotation

# Convert aligned points back to Open3D point cloud
aligned_pcd1 = o3d.geometry.PointCloud()
aligned_pcd1.points = o3d.utility.Vector3dVector(aligned_points1)

# Run ICP after initial alignment
threshold = 0.05  # Adjust as needed
icp_result = o3d.pipelines.registration.registration_icp(
    aligned_pcd1, pcd2, threshold,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
)

# Apply the ICP transformation
aligned_pcd1.transform(icp_result.transformation)

# Translate back to match the target centroid
aligned_points1 = np.asarray(aligned_pcd1.points) + centroid2
aligned_pcd1.points = o3d.utility.Vector3dVector(aligned_points1)

# Visualize the aligned point clouds
o3d.visualization.draw_geometries([pcd2, aligned_pcd1])
