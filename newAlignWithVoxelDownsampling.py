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

def detect_mirroring(axes1, axes2):
    # Check if any axis is flipped (dot product is negative)
    flipped_axes = [np.dot(axes1[i], axes2[i]) < 0 for i in range(3)]
    return flipped_axes

select_files()

if file_path_source and file_path_target:
    print(f"The source file is {file_path_source}")
    print(f"The target file is {file_path_target}")
else:
    print("No files selected.")

pcd1 = o3d.io.read_point_cloud(file_path_source)
pcd2 = o3d.io.read_point_cloud(file_path_target)

# Downsample point clouds using voxel downsampling
voxel_size = 0.01  # Adjust this based on your point cloud density
pcd1_downsampled = o3d.geometry.PointCloud.voxel_down_sample(pcd1, voxel_size)
pcd2_downsampled = o3d.geometry.PointCloud.voxel_down_sample(pcd2, voxel_size)

points1 = np.asarray(pcd1_downsampled.points)
points2 = np.asarray(pcd2_downsampled.points)

# Compute centroids
centroid1 = np.mean(points1, axis=0)
centroid2 = np.mean(points2, axis=0)

# Translate to common origin
translated_points1 = points1 - centroid1

# Check for mirroring before scaling
pca1 = PCA(n_components=3)
pca2 = PCA(n_components=3)

pca1.fit(translated_points1)
pca2.fit(points2 - centroid2)

axes1 = pca1.components_
axes2 = pca2.components_

# Ensure right-handed coordinate systems
if det(axes1) < 0:
    axes1[2, :] *= -1
if det(axes2) < 0:
    axes2[2, :] *= -1

flipped_axes = detect_mirroring(axes1, axes2)

# Apply reflection transformation if needed
reflection_matrix = np.eye(3)
for i, flipped in enumerate(flipped_axes):
    if flipped:
        reflection_matrix[i, i] = -1

# Reflect points1 before scaling and translating
points1_reflected = np.dot(translated_points1, reflection_matrix)

# Scale the source point cloud
scale_factor = np.std(points2 - centroid2) / np.std(points1_reflected)
scaled_points1 = points1_reflected * scale_factor

# Perform PCA on scaled points
pca1.fit(scaled_points1)
axes1_reflected = pca1.components_

# Align principal axes
try:
    rotation = R.align_vectors(axes1_reflected, axes2)[0]
except UserWarning:
    print("Warning: Optimal rotation not uniquely defined. Proceeding with caution.")

# Apply rotation and translation
aligned_points1 = np.dot(scaled_points1, rotation.as_matrix()) + centroid2

# Apply ICP for finer alignment
pcd1_aligned = o3d.geometry.PointCloud()
pcd1_aligned.points = o3d.utility.Vector3dVector(aligned_points1)

# ICP registration with adjusted parameters
reg_p2p = o3d.pipelines.registration.registration_icp(
    pcd1_aligned, pcd2_downsampled, 0.05, np.eye(4),
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=5000)
)

# Apply ICP transformation
final_aligned_points1_icp = np.dot(np.vstack((aligned_points1.T, np.ones(aligned_points1.shape[0]))).T, reg_p2p.transformation).T[:3].T

# Visualize
pcd1_aligned_icp = o3d.geometry.PointCloud()
pcd1_aligned_icp.points = o3d.utility.Vector3dVector(final_aligned_points1_icp)
pcd1_aligned_icp.paint_uniform_color([1, 0, 0])  # Paint red
pcd2_downsampled.paint_uniform_color([0, 1, 0])  # Paint green

# Save aligned point clouds
o3d.io.write_point_cloud("aligned_pcd1.ply", pcd1_aligned_icp)

# Visualize both point clouds
o3d.visualization.draw_geometries([pcd2_downsampled, pcd1_aligned_icp])
