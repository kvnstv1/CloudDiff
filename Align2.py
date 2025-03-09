import tkinter as tk
from tkinter import filedialog
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R
import open3d as o3d

file_path_source = None
file_path_target = None

def select_files():
    global file_path_source, file_path_target
    root = tk.Tk()
    root.title("Select a file!")
    root.geometry("400x400")
    root.withdraw()
    file_path_source = filedialog.askopenfilename(title="Select source file")
    file_path_target = filedialog.askopenfilename(title="Select target file")

select_files()

if file_path_source and file_path_target:
    print(f"The source file is {file_path_source}")
    print(f"The target file is {file_path_target}")
else:
    print("No files selected.")

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

scale_factor = np.linalg.norm(translated_points2, axis=None) / np.linalg.norm(translated_points1, axis=None)

# Scale the source point cloud
scaled_points1 = translated_points1 * scale_factor

# Perform PCA on scaled points
pca1 = PCA(n_components=3)
pca2 = PCA(n_components=3)

pca1.fit(scaled_points1)
pca2.fit(translated_points2)

axes1 = pca1.components_
axes2 = pca2.components_

# Align principal axes
rotation = R.align_vectors(axes1, axes2)[0]

# Check if rotation is more than 180 degrees
rotation_angles = rotation.as_euler('xyz', degrees=True)
for i, angle in enumerate(rotation_angles):
    if abs(angle) > 180:
        rotation_angles[i] = -angle  # Adjust angle to be within -180 to 180 degrees

# Convert adjusted angles back to a rotation matrix
adjusted_rotation = R.from_euler('xyz', rotation_angles, degrees=True)
adjusted_rotation_matrix = adjusted_rotation.as_matrix()

# Apply adjusted rotation and translation
aligned_points1 = np.dot(scaled_points1, adjusted_rotation_matrix)
final_aligned_points1 = aligned_points1 + centroid2

# If necessary, invert the Z-axis to correct orientation
final_aligned_points1[:, 2] *= -1  # Invert Z-axis if needed

# Ensure correct alignment by checking centroids
centroid_aligned = np.mean(final_aligned_points1, axis=0)
if np.linalg.norm(centroid_aligned - centroid2) > 1e-6:  # Tolerance for centroid alignment
    print("Centroids are not aligned. Adjust translation.")
    final_aligned_points1 -= centroid_aligned + centroid2

# Visualize
aligned_pcd1 = o3d.geometry.PointCloud()
aligned_pcd1.points = o3d.utility.Vector3dVector(final_aligned_points1)
aligned_pcd1.paint_uniform_color([1, 0, 0])  # Paint red
pcd2.paint_uniform_color([0, 1, 0])  # Paint green

o3d.visualization.draw_geometries([pcd2, aligned_pcd1])