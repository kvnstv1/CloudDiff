import tkinter as tk
from tkinter import filedialog
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R
import open3d as o3d

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

####################################################################################

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

# Perform PCA on scaled points
pca1 = PCA(n_components=3)
pca2 = PCA(n_components=3)

pca1.fit(scaled_points1)
pca2.fit(translated_points2)

axes1 = pca1.components_
axes2 = pca2.components_

# Ensure consistent orientation of PCA axes
for i in range(3):
    if np.dot(axes1[i], axes2[i]) < 0:  # If the axes point in opposite directions
        axes1[i] *= -1  # Flip the eigenvector

# Compute rotation matrix using SVD
U, _, Vt = np.linalg.svd(axes2 @ axes1.T)
R_opt = Vt.T @ U.T

# Ensure a proper rotation (no reflection)
if np.linalg.det(R_opt) < 0:
    Vt[-1, :] *= -1
    R_opt = Vt.T @ U.T

# Apply rotation and translation
aligned_points1 = scaled_points1 @ R_opt.T
final_aligned_points1 = aligned_points1 + centroid2

# Convert to Open3D point cloud
aligned_pcd1 = o3d.geometry.PointCloud()
aligned_pcd1.points = o3d.utility.Vector3dVector(final_aligned_points1)

# Visualize
o3d.visualization.draw_geometries([pcd2, aligned_pcd1])
