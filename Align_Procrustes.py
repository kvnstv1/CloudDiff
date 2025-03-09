import tkinter as tk
from tkinter import filedialog
import numpy as np
import open3d as o3d

# Function to compute the optimal rotation matrix using SVD-based Procrustes alignment
def compute_optimal_rotation(source, target):
    H = source.T @ target
    U, S, Vt = np.linalg.svd(H)
    R_opt = Vt.T @ U.T
    
    # Ensure a proper rotation (no reflection)
    if np.linalg.det(R_opt) < 0:
        Vt[-1, :] *= -1
        R_opt = Vt.T @ U.T
    
    return R_opt

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

# Ensure both point clouds have the same number of points
min_size = min(scaled_points1.shape[0], translated_points2.shape[0])
scaled_points1 = scaled_points1[:min_size]
translated_points2 = translated_points2[:min_size]

# Compute the optimal rotation matrix using SVD
R_opt = compute_optimal_rotation(scaled_points1, translated_points2)

# Apply rotation
aligned_points1 = scaled_points1 @ R_opt

# Translate back to the target centroid
final_aligned_points1 = aligned_points1 + centroid2

# Ensure correct alignment by checking centroids
centroid_aligned = np.mean(final_aligned_points1, axis=0)
if np.linalg.norm(centroid_aligned - centroid2) > 1e-6:  # Tolerance for centroid alignment
    print("Centroids are not aligned. Adjusting translation.")
    final_aligned_points1 -= (centroid_aligned - centroid2)

# Visualize the aligned point clouds
aligned_pcd1 = o3d.geometry.PointCloud()
aligned_pcd1.points = o3d.utility.Vector3dVector(final_aligned_points1)

o3d.visualization.draw_geometries([pcd2, aligned_pcd1])
