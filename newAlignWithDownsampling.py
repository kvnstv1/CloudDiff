import tkinter as tk
from tkinter import filedialog
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R
from scipy.spatial import KDTree
import open3d as o3d
from scipy.linalg import det

file_path_source = None
file_path_target = None

def chamfer_distance(points1, points2):
    tree = KDTree(points2)
    dist_p1 = tree.query(points1)[0]
    tree = KDTree(points1)
    dist_p2 = tree.query(points2)[0]
    return np.mean(dist_p1) + np.mean(dist_p2)

def select_files():
    global file_path_source, file_path_target
    root = tk.Tk()
    root.title("Select files!")
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

# Downsample point clouds to 50,000 points using farthest point sampling
pcd1_downsampled = pcd1.farthest_point_down_sample(50000)
pcd2_downsampled = pcd2.farthest_point_down_sample(50000)

points1 = np.asarray(pcd1_downsampled.points)
points2 = np.asarray(pcd2_downsampled.points)

# Compute centroids
centroid1 = np.mean(points1, axis=0)
centroid2 = np.mean(points2, axis=0)

# Translate to common origin
translated_points1 = points1 - centroid1
translated_points2 = points2 - centroid2

# Scale the source point cloud
scale_factor = np.std(translated_points2) / np.std(translated_points1)
scaled_points1 = translated_points1 * scale_factor

# Perform PCA on scaled points
pca1 = PCA(n_components=3)
pca2 = PCA(n_components=3)

pca1.fit(scaled_points1)
pca2.fit(translated_points2)

axes1 = pca1.components_
axes2 = pca2.components_

# Ensure right-handed coordinate systems
if det(axes1) < 0:
    axes1[2, :] *= -1
if det(axes2) < 0:
    axes2[2, :] *= -1

print("Axes 1")
print(str(axes1))
print("Axes 1 det:", det(axes1))

print("Axes 2")
print(str(axes2))
print("Axes 2 det:", det(axes2))

# Align principal axes
rotation = R.align_vectors(axes1, axes2)[0]

print("R")
print(rotation.as_matrix())
print("R det:", det(rotation.as_matrix()))

# Apply rotation and translation
aligned_points1 = np.dot(scaled_points1, rotation.as_matrix())
final_aligned_points1 = aligned_points1 + centroid2

# Apply ICP for finer alignment
pcd1_aligned = o3d.geometry.PointCloud()
pcd1_aligned.points = o3d.utility.Vector3dVector(final_aligned_points1)

# ICP registration
reg_p2p = o3d.pipelines.registration.registration_icp(
    pcd1_aligned, pcd2_downsampled, 0.2, np.eye(4),
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
)

# Apply ICP transformation
final_aligned_points1_icp = np.dot(np.vstack((final_aligned_points1.T, np.ones(final_aligned_points1.shape[0]))).T, reg_p2p.transformation).T[:3].T

# Visualize
pcd1_aligned_icp = o3d.geometry.PointCloud()
pcd1_aligned_icp.points = o3d.utility.Vector3dVector(final_aligned_points1_icp)
pcd1_aligned_icp.paint_uniform_color([1, 0, 0])  # Paint red
pcd2_downsampled.paint_uniform_color([0, 1, 0])  # Paint green

# Save aligned point clouds
o3d.io.write_point_cloud("aligned_pcd1.ply", pcd1_aligned_icp)

# Visualize both point clouds
o3d.visualization.draw_geometries([pcd2_downsampled, pcd1_aligned_icp])
