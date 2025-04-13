import tkinter as tk
from tkinter import filedialog
import open3d as o3d
import numpy as np
from scipy.spatial import KDTree
from scipy.linalg import det, svd
#from pclpy import pcl
from itertools import permutations, product

def kabsch_rotation(A, B):
    """
    Computes optimal rotation matrix using Kabsch algorithm.
    
    Args:
        A: Source points as columns (3xN array)
        B: Target points as columns (3xN array)
        
    Returns:
        3x3 rotation matrix
    """
    # Center the points
    A_centered = A - np.mean(A, axis=1, keepdims=True)
    B_centered = B - np.mean(B, axis=1, keepdims=True)

    # Compute covariance matrix
    H = A_centered @ B_centered.T

    # Singular Value Decomposition
    U, S, Vt = svd(H)

    # Compute rotation matrix
    R = Vt.T @ U.T

    # Ensure right-handed coordinate system
    if det(R) < 0:
        Vt[2,:] *= -1  # Flip z-axis if determinant is negative
        R = Vt.T @ U.T

    return R

# 1.  Read files
def selectFiles():
    """
    Prompts the user to select source and target files using a file dialog.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Prompt the user to select the source file
    source_file_path = filedialog.askopenfilename(title="Select Source File")
    if not source_file_path:
        print("No source file selected.")
        return None, None

    # Prompt the user to select the target file
    target_file_path = filedialog.askopenfilename(title="Select Target File")
    if not target_file_path:
        print("No target file selected.")
        return None, None

    return source_file_path, target_file_path

# 2. Load and Preprocess Point Clouds
def load_and_preprocess(file_path, downsample_factor=50000):
    pcd = o3d.io.read_point_cloud(file_path)
    pcd = pcd.farthest_point_down_sample(downsample_factor)
    points = np.asarray(pcd.points)
    return pcd, points

# 3. Center and Scale Clouds
def center_and_scale(source, target):
    source_centroid = np.mean(source, axis=0)
    target_centroid = np.mean(target, axis=0)
    
    source_centered = source - source_centroid
    target_centered = target - target_centroid
    
    scale_factor = np.std(target_centered) / np.std(source_centered)
    source_scaled = source_centered * scale_factor
    
    return source_scaled, target_centered, source_centroid, target_centroid

# 4. PCA-Based Coarse Alignment
def handle_pca_ambiguity(eigenvectors):
    min_error = float('inf')
    best_axes = None
    for perm in permutations([0,1,2]):
        for signs in [[1,1,1], [1,1,-1], [1,-1,1], [-1,1,1]]:
            modified = eigenvectors[:, perm] * signs
            if abs(det(modified) - 1) < min_error:
                min_error = abs(det(modified) - 1)
                best_axes = modified
    return best_axes

def pca_alignment(source, target):
    source_axes = handle_pca_ambiguity(np.linalg.eig(np.cov(source.T))[1][:, :3])
    target_axes = handle_pca_ambiguity(np.linalg.eig(np.cov(target.T))[1][:, :3])
    R = kabsch_rotation(source_axes.T, target_axes.T)
    return source @ R

# 5. Mirror Handling
def generate_mirrors(points):
    mirrors = []
    for perm in permutations([0,1,2]):
        for signs in product([1,-1], repeat=3):
            if np.prod(signs) > 0:  # Maintain right-handedness
                transform = np.eye(3)[perm,:] * signs
                mirrors.append(points @ transform.T)
    return mirrors

# 6. ICP Refinement
def refine_icp(source, target, threshold=0.1):
    # Convert numpy arrays to Open3D point clouds
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source)
    
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target)
    
    # Estimate normals with consistent orientation
    search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    source_pcd.estimate_normals(search_param=search_param)
    target_pcd.estimate_normals(search_param=search_param)
    
    # Orient normals consistently
    source_pcd.orient_normals_consistent_tangent_plane(k=20)
    target_pcd.orient_normals_consistent_tangent_plane(k=20)
    
    # Run ICP
    reg = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
    )
    return reg.transformation

# 7. Metrics Calculation
def chamfer_distance(source, target):
    tree = KDTree(target)
    dists = tree.query(source)[0]
    return np.mean(dists) + np.mean(KDTree(source).query(target)[0])

def hausdorff_distance(source, target):
    tree = KDTree(target)
    return np.max(tree.query(source)[0])

# 8. Full Alignment Pipeline
def align_and_compare(source_path, target_path):
    # Load and preprocess
    _, source = load_and_preprocess(source_path)
    _, target = load_and_preprocess(target_path)
    
    # Center and scale
    source_scaled, target_centered, s_cent, t_cent = center_and_scale(source, target)
    
    # Coarse PCA alignment
    aligned = pca_alignment(source_scaled, target_centered)
    
    # Handle mirrors
    mirrors = generate_mirrors(aligned)
    best_mirror = min(mirrors, key=lambda x: chamfer_distance(x, target_centered))
    
    # ICP refinement
    transform = refine_icp(best_mirror, target_centered)
    final_aligned = (best_mirror @ transform[:3,:3].T) + transform[:3,3]
    
    # Apply original centroids
    final_aligned += t_cent - s_cent

    # Visualize
    visualize_results(source, target, final_aligned)
    
    # Calculate metrics
    cd = chamfer_distance(final_aligned, target)
    hd = hausdorff_distance(final_aligned, target)
    
    return final_aligned, cd, hd

# 9. Visualization
def visualize_results(source, target, aligned):
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source)
    source_pcd.paint_uniform_color([1,0,0])
    
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target)
    target_pcd.paint_uniform_color([0,1,0])
    
    aligned_pcd = o3d.geometry.PointCloud()
    aligned_pcd.points = o3d.utility.Vector3dVector(aligned)
    aligned_pcd.paint_uniform_color([0,0,1])
    
    o3d.visualization.draw_geometries([source_pcd, target_pcd, aligned_pcd])

def main():
    source, target = selectFiles()
    source_aligned, chamfer, hausdorff = align_and_compare(source, target)
    print(f"Chamfer Distance: {chamfer:.4f}")
    print(f"Hausdorff Distance: {hausdorff:.4f}")

if __name__ == "__main__":
    main()
