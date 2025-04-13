import tkinter as tk
from tkinter import filedialog
import numpy as np
from sklearn.decomposition import PCA
import open3d as o3d
import scipy.spatial.transform as transform

####################################################################################
# FUNCTIONS:
def compute_rotation_matrix(axes1, axes2):
    """
    Compute the optimal rotation matrix to align two sets of principal axes.
    Uses cross products and Rodrigues' rotation formula instead of SVD.
    
    Parameters:
        axes1 (numpy.ndarray): Principal axes of source cloud (3x3 matrix)
        axes2 (numpy.ndarray): Principal axes of target cloud (3x3 matrix)

    Returns:
        R_opt (numpy.ndarray): 3x3 optimal rotation matrix
    """
    return axes2 @ axes1.T

# Manual PCA function
def manual_pca(data, n_components):
    centered_data = data - np.mean(data, axis=0)
    cov_matrix = np.cov(centered_data.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    selected_eigenvectors = eigenvectors[:, :n_components]
    return selected_eigenvectors

####################################################################################
# STEP 1: READ FILES
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
# Step 2: Convert to python data structures

pcd1 = o3d.io.read_point_cloud(file_path_source)
pcd2 = o3d.io.read_point_cloud(file_path_target)

points1 = np.asarray(pcd1.points)
points2 = np.asarray(pcd2.points)

####################################################################################
# Step 3: Centre the point clouds

centroid1 = np.mean(points1, axis=0)
centroid2 = np.mean(points2, axis=0)
shifted_points1 = points1 - centroid1
shifted_points2 = points2 - centroid2

####################################################################################
# Step 4: Scale the Point Clouds

bb1 = np.ptp(points1, axis=0)  # Bounding box range
bb2 = np.ptp(points2, axis=0)
scale_factor = np.mean(bb2 / bb1)  # Use the mean scale of all axes
scaled_points1 = shifted_points1 * scale_factor

####################################################################################
# Step 5: PCA time

axes1 = manual_pca(scaled_points1, n_components=3)
axes2 = manual_pca(shifted_points2, n_components=3)
rotation = transform.Rotation.align_vectors(axes1,axes2)[0]
rotation_matrix = rotation.as_matrix()



####################################################################################
# Step 7: Apply rotation

aligned_points1 = np.dot(scaled_points1, rotation_matrix)
final_aligned_points1 = aligned_points1 + centroid2

####################################################################################
# Step 9: Visualize

origin = np.array([[0, 0, 0]])  # Single point at the origin
origin_pcd = o3d.geometry.PointCloud()
origin_pcd.points = o3d.utility.Vector3dVector(origin)
origin_pcd.paint_uniform_color([1, 0, 0])  # Red color

aligned_pcd1 = o3d.geometry.PointCloud()
aligned_pcd1.points = o3d.utility.Vector3dVector(final_aligned_points1)
aligned_pcd1.paint_uniform_color([0, 1, 0])  # Green color

centred_pcd2 = o3d.geometry.PointCloud()
centred_pcd2.points = o3d.utility.Vector3dVector(shifted_points2)
centred_pcd2.paint_uniform_color([0, 0, 1])  # Blue color

# Visualize
o3d.visualization.draw_geometries([centred_pcd2, aligned_pcd1, origin_pcd])
