import tkinter as tk
from tkinter import filedialog
import numpy as np
from scipy.spatial import KDTree
import open3d as o3d
from scipy.linalg import det, svd
from itertools import permutations, product

# --------------------- Global Variables ---------------------
file_path_source = None
file_path_target = None

# --------------------- Robust PCA Ambiguity Resolution ---------------------
def handle_pca_ambiguity(eigenvectors):
    """Resolves PCA eigenvector ambiguities through full permutation and sign combinations"""
    min_error = float('inf')
    best_axes = None
    
    # Full 24 valid PCA configurations (6 permutations × 4 sign combinations)
    for perm in permutations([0,1,2]):
        # Only valid sign combinations that preserve right-handed systems
        for sign_matrix in [[1,1,1], [1,1,-1], [1,-1,1], [-1,1,1]]:
            modified = eigenvectors[:, perm] * sign_matrix
            current_error = abs(det(modified) - 1)  # Proper rotation matrices have det=1
            if current_error < min_error:
                min_error = current_error
                best_axes = modified
    return best_axes

# --------------------- PCA Calculation ---------------------
def pca_manual(data, n_components):
    """Performs PCA with improved eigenvector handling"""
    mean = np.mean(data, axis=0)
    centered_data = data - mean
    covariance_matrix = np.cov(centered_data.T)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    return handle_pca_ambiguity(eigenvectors[:, :n_components])

# --------------------- Optimal Rotation Calculation ---------------------
def kabsch_rotation(A, B):
    """Kabsch algorithm with enhanced numerical stability"""
    # Center the points
    A_centered = A - np.mean(A, axis=0)
    B_centered = B - np.mean(B, axis=0)
    
    # Compute covariance matrix
    H = A_centered.T @ B_centered
    
    # SVD with full_matrices=False for better numerical stability
    U, S, Vt = svd(H, full_matrices=False)
    
    # Correct rotation matrix
    R = Vt.T @ U.T
    
    # Ensure right-handed system
    if det(R) < 0:
        Vt[-1,:] *= -1
        R = Vt.T @ U.T
    
    return R

# --------------------- Enhanced ICP with Multi-Resolution ---------------------
def enhanced_icp(source, target, initial_threshold):
    """Multi-resolution ICP with adaptive parameters"""
    # Create multi-resolution versions
    voxel_sizes = [0.1, 0.05, 0.02]
    sources = [source.voxel_down_sample(v) for v in voxel_sizes]
    targets = [target.voxel_down_sample(v) for v in voxel_sizes]
    
    transformation = np.eye(4)
    
    # Coarse-to-fine alignment
    for i in reversed(range(len(voxel_sizes))):
        src = sources[i]
        tgt = targets[i]
        
        # Estimate normals with adaptive search
        radius = voxel_sizes[i] * 2
        src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
        tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
        
        # Dynamic threshold based on voxel size
        threshold = max(initial_threshold, voxel_sizes[i] * 1.5)
        
        reg = o3d.pipelines.registration.registration_icp(
            src, tgt, threshold, transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-8,
                relative_rmse=1e-8,
                max_iteration=1000)
        )
        transformation = reg.transformation
    
    return transformation

# --------------------- Comprehensive Mirror Handling ---------------------
def generate_mirror_transforms(points):
    """Generates all valid mirror transformations considering rotational symmetries"""
    versions = []
    
    # Generate all permutation matrices
    for perm in permutations([0,1,2]):
        # Generate all sign combinations for each permutation
        for signs in product([1,-1], repeat=3):
            # Skip invalid left-handed systems
            if np.prod(signs) < 0:
                continue
                
            transform = np.eye(3)[perm,:] * signs
            versions.append(points @ transform.T)
    
    return versions

# --------------------- Chamfer Distance Calculation ---------------------
def chamfer_distance(points1, points2):
    """Calculates Chamfer distance between two point clouds"""
    tree = KDTree(points2)
    dist_p1 = tree.query(points1)[0]
    tree = KDTree(points1)
    dist_p2 = tree.query(points2)[0]
    return np.mean(dist_p1) + np.mean(dist_p2)

# --------------------- Alignment Validation Tools ---------------------
def alignment_quality_report(source, target, transform):
    """Quantitative alignment assessment"""
    aligned = source.transform(transform)
    distances = aligned.compute_point_cloud_distance(target)
    print(f"Alignment Quality Report:")
    print(f"  Mean distance: {np.mean(distances):.4f}")
    print(f"  Median distance: {np.median(distances):.4f}")
    print(f"  95th percentile: {np.percentile(distances, 95):.4f}")

# --------------------- Core Pipeline ---------------------
def main():
    # File selection
    root = tk.Tk()
    root.withdraw()
    source_path = filedialog.askopenfilename(title="Select source file")
    target_path = filedialog.askopenfilename(title="Select target file")
    
    if not (source_path and target_path):
        print("Missing files!")
        return

    # Data loading and preprocessing
    source = o3d.io.read_point_cloud(source_path).voxel_down_sample(0.01)
    target = o3d.io.read_point_cloud(target_path).voxel_down_sample(0.01)
    
    # Centering and scaling
    source_pts = np.asarray(source.points)
    target_pts = np.asarray(target.points)
    
    source_centered = source_pts - np.mean(source_pts, axis=0)
    target_centered = target_pts - np.mean(target_pts, axis=0)
    
    scale_factor = np.linalg.norm(target_centered) / np.linalg.norm(source_centered)
    source_scaled = source_centered * scale_factor

    # PCA Alignment with full ambiguity resolution
    source_axes = pca_manual(source_scaled, 3)
    target_axes = pca_manual(target_centered, 3)
    R = kabsch_rotation(source_axes.T, target_axes.T)
    aligned_source = source_scaled @ R

    # Mirror handling with validation
    mirrored_versions = generate_mirror_transforms(aligned_source)
    best_version = min(mirrored_versions, 
                      key=lambda x: chamfer_distance(x, target_centered))
    
    # ICP refinement
    source_aligned_pcd = o3d.geometry.PointCloud()
    source_aligned_pcd.points = o3d.utility.Vector3dVector(best_version)
    
    initial_threshold = np.percentile(np.linalg.norm(target_centered - best_version, axis=1), 95)
    final_transform = enhanced_icp(source_aligned_pcd, target, initial_threshold)
    
    # Apply final transformation
    source_aligned_pcd.transform(final_transform)
    
    # Results validation
    alignment_quality_report(source_aligned_pcd, target, final_transform)
    
    # Visualize and save
    source_aligned_pcd.paint_uniform_color([1,0,0])
    target.paint_uniform_color([0,1,0])
    o3d.visualization.draw_geometries([source_aligned_pcd, target])
    
    combined = source_aligned_pcd + target
    o3d.io.write_point_cloud("aligned_result.ply", combined)

if __name__ == "__main__":
    main()
