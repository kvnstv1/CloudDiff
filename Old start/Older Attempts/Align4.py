import tkinter as tk
from tkinter import filedialog
import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d

# Manual PCA function
def manual_pca(data, n_components):
    # Center the data
    centered_data = data - np.mean(data, axis=0)
    
    # Compute the covariance matrix
    cov_matrix = np.cov(centered_data.T)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Sort eigenvectors by eigenvalues in descending order
    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
     # Enforce a consistent direction for eigenvectors
    for i in range(n_components):
        if np.sum(eigenvectors[:, i]) < 0:
            eigenvectors[:, i] *= -1  # Flip sign to maintain consistency
    
    return eigenvectors[:, :n_components]

#Procrustes Alignment
def compute_optimal_rotation(source, target):
    H = source.T @ target
    U, S, Vt = np.linalg.svd(H)
    R_opt = Vt.T @ U.T
    
    # Ensure a proper rotation (no reflection)
    if np.linalg.det(R_opt) < 0:
        Vt[-1, :] *= -1
        R_opt = Vt.T @ U.T
    
    return R_opt


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

scale_factor = np.std(translated_points2) / np.std(translated_points1)

# Scale the source point cloud
scaled_points1 = translated_points1 * scale_factor

# Function to flip points along an axis
def flip_points(points, axis):
    if axis == 'x':
        return points * np.array([1, -1, -1])
    elif axis == 'y':
        return points * np.array([-1, 1, -1])
    elif axis == 'z':
        return points * np.array([-1, -1, 1])

# Function to compute rotation angles for flipped points
def compute_rotation_angles(points, target_points):
    axes1 = manual_pca(points, n_components=3)
    axes2 = manual_pca(target_points, n_components=3)
    rotation = R.align_vectors(axes1, axes2)[0]
    return rotation.as_euler('xyz', degrees=True)

# Check flipping along each axis
min_angles = compute_rotation_angles(scaled_points1, translated_points2)
min_points = scaled_points1

for axis in ['x', 'y', 'z']:
    flipped_points = flip_points(scaled_points1, axis)
    angles = compute_rotation_angles(flipped_points, translated_points2)
    
    # Check if the rotation angles are smaller
    if sum(abs(angle) for angle in angles) < sum(abs(angle) for angle in min_angles):
        min_angles = angles
        min_points = flipped_points

# Use min_points for further processing
aligned_points1 = min_points

# Perform manual PCA on aligned points
axes1 = manual_pca(aligned_points1, n_components=3)
axes2 = manual_pca(translated_points2, n_components=3)

# Align principal axes
rotation = R.align_vectors(axes1, axes2)[0]

# Check if rotation is more than 180 degrees
rotation_angles = rotation.as_euler('xyz', degrees=True)
for i, angle in enumerate(rotation_angles):
    if abs(angle) > 180:
        rotation_angles[i] = (angle+180)%360-180  # Adjust angle to be within -180 to 180 degrees

# Convert adjusted angles back to a rotation matrix
adjusted_rotation = R.from_euler('xyz', rotation_angles, degrees=True)
adjusted_rotation_matrix = adjusted_rotation.as_matrix()

# Apply adjusted rotation and translation
aligned_points1 = np.dot(aligned_points1, adjusted_rotation_matrix)
final_aligned_points1 = aligned_points1 + centroid2

# Ensure correct alignment by checking centroids
centroid_aligned = np.mean(final_aligned_points1, axis=0)
if np.linalg.norm(centroid_aligned - centroid2) > 1e-6:  # Tolerance for centroid alignment
    print("Centroids are not aligned. Adjust translation.")
    final_aligned_points1 -= centroid_aligned - centroid2

# Visualize
aligned_pcd1 = o3d.geometry.PointCloud()
aligned_pcd1.points = o3d.utility.Vector3dVector(final_aligned_points1)
#aligned_pcd1.paint_uniform_color([1, 0, 0])  # Paint red
#pcd2.paint_uniform_color([0, 1, 0])  # Paint green

o3d.visualization.draw_geometries([pcd2, aligned_pcd1])
