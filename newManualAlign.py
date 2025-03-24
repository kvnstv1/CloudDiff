import tkinter as tk
from tkinter import filedialog
import numpy as np
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

def pca_manual(data, n_components):
    #This part already works fine, so great...
    mean = np.mean(data, axis=0)
    centered_data = data - mean

    #PCA TIME
    #Compute the covariance matrix
    covariance_matrix = np.cov(centered_data.T)
    print(f"\nThe convariance matrix is \n {covariance_matrix}")
    #Compute the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    #Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    print(f"\nSorted eigenvectors are \n{eigenvectors}")
    #Selecting componenrts
    principal_components = eigenvectors[:, :n_components]
    print(f"\nThe principal components selected are: \n{principal_components}")
    
    return principal_components

def align_vectors_manual(axes, axes_target):
    rotation_matrix = np.eye(3)
    
    for i in range(3):
        #The cross product gives the rotation axis
        cross_product = np.cross(axes[i], axes_target[i])
        rotation_axis = cross_product / np.linalg.norm(cross_product)
        print(f"\nThe calculated rotation axis is \n{rotation_axis}")
        #The dot product gives the cos() of the rotation angle
        dot_product = np.dot(axes[i], axes_target[i])
        #Compute  rotation angle
        rotation_angle = np.arccos(dot_product)
        print(f"\nThe calculated rotation angle is \n{rotation_angle}")
        #Rodrigues' formula
        # See: https://people.eecs.berkeley.edu/~ug/slide/pipeline/assignments/as5/rotation.html 
        K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                      [rotation_axis[2], 0, -rotation_axis[0]],
                      [-rotation_axis[1], rotation_axis[0], 0]])
        rotation = np.eye(3) + np.sin(rotation_angle) * K + (1 - np.cos(rotation_angle)) * np.dot(K, K)
        # Apply rotation matrix
        rotation_matrix = np.dot(rotation, rotation_matrix)
        print(f"\nThe final rotation matrix is \n{rotation_matrix}")
    
    return rotation_matrix

select_files()

if file_path_source and file_path_target:
    print(f"The source file is {file_path_source}\n")
    print(f"The target file is {file_path_target}\n")
else:
    print("No files selected.")

pcd1 = o3d.io.read_point_cloud(file_path_source)
pcd2 = o3d.io.read_point_cloud(file_path_target)

#Downsample point clouds to 50,000 points using farthest point sampling
pcd1_downsampled = pcd1.farthest_point_down_sample(50000)
pcd2_downsampled = pcd2.farthest_point_down_sample(50000)

points1 = np.asarray(pcd1_downsampled.points)
points2 = np.asarray(pcd2_downsampled.points)

#Compute centroids
centroid1 = np.mean(points1, axis=0)
centroid2 = np.mean(points2, axis=0)

#Translate to common origin
translated_points1 = points1 - centroid1
translated_points2 = points2 - centroid2

#Scale the source point cloud
scale_factor = np.std(translated_points2) / np.std(translated_points1)
scaled_points1 = translated_points1 * scale_factor

#Perform manual PCA on scaled points
axes1 = pca_manual(scaled_points1, 3)
axes2 = pca_manual(translated_points2, 3)

#Ensure right-handed coordinate systems
if det(axes1) < 0:
    axes1[2, :] *= -1
if det(axes2) < 0:
    axes2[2, :] *= -1

print("Axes 1\n")
print(str(axes1))
print("\nAxes 1 det:", det(axes1))

print("Axes 2\n")
print(str(axes2))
print("\nAxes 2 det:", det(axes2))

#ALIGNMENT!!!!!
rotation_matrix = align_vectors_manual(axes1, axes2)

#Second verification after the stuff in the other method really
print("R\n")
print(rotation_matrix)
print("\nR det:", det(rotation_matrix))

#Pre ICP step of applying rotation and translation
aligned_points1 = np.dot(scaled_points1, rotation_matrix)
final_aligned_points1 = aligned_points1 + centroid2

#Apply ICP for finer alignment
pcd1_aligned = o3d.geometry.PointCloud()
pcd1_aligned.points = o3d.utility.Vector3dVector(final_aligned_points1)

#ICP registration using o3d
reg_p2p = o3d.pipelines.registration.registration_icp(
    pcd1_aligned, pcd2_downsampled, 0.2, np.eye(4),
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=4000) #Increasing max iterations does not make things better.
)

#Apply ICP transformation
final_aligned_points1_icp = np.dot(np.vstack((final_aligned_points1.T, np.ones(final_aligned_points1.shape[0]))).T, reg_p2p.transformation).T[:3].T

#Visualize
pcd1_aligned_icp = o3d.geometry.PointCloud()
pcd1_aligned_icp.points = o3d.utility.Vector3dVector(final_aligned_points1_icp)
pcd1_aligned_icp.paint_uniform_color([1, 0, 0])  # Paint red
pcd2_downsampled.paint_uniform_color([0, 1, 0])  # Paint green

#Write output file so that we don't need to repeat experiments each time.
#Combine the two point clouds into one
combined_pcd = o3d.geometry.PointCloud()
combined_pcd.points = o3d.utility.Vector3dVector(
    np.vstack((
        np.asarray(pcd2_downsampled.points),
        np.asarray(pcd1_aligned_icp.points)
    ))
)

#Assign colors to
colors = np.vstack((
    np.tile([0, 1, 0], (len(pcd2_downsampled.points), 1)),  # Green for source PCD
    np.tile([1, 0, 0], (len(pcd1_aligned_icp.points), 1))   # Red for aligned PCD
))
combined_pcd.colors = o3d.utility.Vector3dVector(colors)

#Save the combined point cloud to a PLY file
o3d.io.write_point_cloud("combined_pcd.ply", combined_pcd)


#Save aligned point clouds
#o3d.io.write_point_cloud("aligned_pcd1.ply", pcd1_aligned_icp)

#Visualize both point clouds
o3d.visualization.draw_geometries([pcd2_downsampled, pcd1_aligned_icp])
