import tkinter as tk
from tkinter import filedialog
import numpy as np
from scipy.spatial import KDTree
import open3d as o3d
from scipy.linalg import det


file_path_source = None
file_path_target = None

def mirror_axes(points):
    """Now returns 9 versions including original at center 
    Order of operations checked using an online source.
    @see: https://gamedev.stackexchange.com/questions/149062/how-to-mirror-reflect-flip-a-4d-transformation-matrix
    """
    mirrored_versions = []
    axes = [1, -1]
    
    # Generate all 8 mirrored/axis-flipped versions
    for x in axes:
        for y in axes:
            for z in axes:
                mirror_matrix = np.array([
                    [x, 0, 0],
                    [0, y, 0],
                    [0, 0, z]
                ])
                mirrored = points@mirror_matrix
                mirrored_versions.append(mirrored)
    
    # Insert original copy at position 4 (grid center)
    mirrored_versions.insert(4, points.copy())  # Index 4 is center in 3x3 grid
    return mirrored_versions


def visualize_mirror_grid(mirrored_versions, optimal_mirror, target_points, centroid_target):
    """
    Visualizes 9 versions (8 mirrors + original) in 3x3 grid with targets
    """
    # Calculate grid spacing
    max_dim = np.max(np.ptp(target_points, axis=0)) * 3
    
    # 3x3 grid positions (9 cells)
    grid_positions = [
        (-max_dim, max_dim, 0), (0, max_dim, 0), (max_dim, max_dim, 0),
        (-max_dim, 0, 0),       (0, 0, 0),       (max_dim, 0, 0),
        (-max_dim, -max_dim, 0),(0, -max_dim, 0),(max_dim, -max_dim, 0)
    ]

    geometries = []
    
    # Create coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=max_dim/3)

    for i, mirrored in enumerate(mirrored_versions):
        # Position in grid
        pos = grid_positions[i]
        
        # Create target instance (green)
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(target_points + centroid_target + pos)
        target_pcd.paint_uniform_color([0, 0.5, 0])  # Darker green
        
        # Create source version
        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(mirrored + centroid_target + pos)
        
        # Highlight if optimal
        if np.array_equal(mirrored, optimal_mirror):
            source_pcd.paint_uniform_color([1, 1, 0])  # Bright yellow
            bbox = source_pcd.get_axis_aligned_bounding_box()
            bbox.color = (1, 0, 0)  # Red box
            geometries.append(bbox)
        else:
            source_pcd.paint_uniform_color([0.8, 0.8, 1])  # Light blue
        
        geometries.extend([target_pcd, source_pcd, coord_frame.translate(pos)])

    o3d.visualization.draw_geometries(
        geometries,
        window_name="3x3 Mirror Grid",
        zoom=0.5,
        front=[-0.5, -0.5, 1],
        lookat=centroid_target,
        up=[0, 0, 1]
    )


def visualize_points(points1, points2):
    """
    Visualize the points given together for debugging purposes.
    """
    # Create point clouds for visualization
    pcd1_aligned = o3d.geometry.PointCloud()
    pcd1_aligned.points = o3d.utility.Vector3dVector(points1)
    pcd1_aligned.paint_uniform_color([1, 0, 0])  # Paint red

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points2)
    pcd2.paint_uniform_color([0, 1, 0])  # Paint green

    # Visualize both point clouds
    o3d.visualization.draw_geometries([pcd2, pcd1_aligned])


import numpy as np

def mirror_axes(points):
    """
    A method that returns all versions of an object 
    mirrored along all its axes using matrix multiplication.
    """
    mirrored_versions = []
    axes = [1, -1]
    
    # Create all possible transformation matrices
    for x in axes:
        for y in axes:
            for z in axes:
                # Create transformation matrix for mirroring
                transform_matrix = np.diag([x, y, z])
                
                # Apply transformation using matrix multiplication
                mirrored = np.transpose(np.matmul(transform_matrix, np.transpose(points)))
                mirrored_versions.append(mirrored)
    
    return mirrored_versions



def chamfer_distance(points1, points2):
    """
    A method that calculates the Chamfer distance between 2 objects in 3d space.
    The Chamfer distance is the distance from each point int he source to its nearest neighbour
    in the target, and from each point in the target to its nearest point in the source

    @see: https://medium.com/@sim30217/chamfer-distance-4207955e8612
    """
    tree = KDTree(points2)
    dist_p1 = tree.query(points1)[0]
    tree = KDTree(points1)
    dist_p2 = tree.query(points2)[0]
    return np.mean(dist_p1) + np.mean(dist_p2)

# def find_optimal_mirror(points1, points2):
#     """
#     A method that finds the optimal mirror image from a set of mirrored images of the 
#     source file that has been centred and roughly aligned.
#     """
#     mirrored_versions = mirror_axes(points1)
#     distances = [chamfer_distance(mirrored, points2) for mirrored in mirrored_versions]
#     min_distance = min(distances)
#     optimal_mirror_index = distances.index(min_distance)
#     optimal_mirror = mirrored_versions[optimal_mirror_index]
#     visualize_points(points2, optimal_mirror)
#     return min_distance, optimal_mirror

def find_optimal_mirror(points1, points2, centroid):
    """
    Returns optimal mirror with visual highlighting and proper positioning
    """
    mirrored_versions = mirror_axes(points1)
    distances = [chamfer_distance(mirrored, points2) for mirrored in mirrored_versions]
    min_distance = min(distances)
    optimal_idx = distances.index(min_distance)
    optimal_mirror = mirrored_versions[optimal_idx]
    
    # Create highlighted version (yellow) for visualization
    optimal_highlight = o3d.geometry.PointCloud()
    optimal_highlight.points = o3d.utility.Vector3dVector(optimal_mirror + centroid)
    optimal_highlight.paint_uniform_color([1, 0.7, 0])  # Yellow
    
    # Create original target (green)
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(points2 + centroid)
    target_pcd.paint_uniform_color([0, 1, 0])
    
    o3d.visualization.draw_geometries([target_pcd, optimal_highlight],
                                     window_name="Optimal Mirror Preview",
                                     zoom=0.8,
                                     front=[0, -1, 0.5],
                                     lookat=centroid,
                                     up=[0, 0, 1])
    return min_distance, optimal_mirror



def select_files():
    """
    A method to select files using a simple button and a filechooser
    using tkinter.
    """
    global file_path_source, file_path_target
    root = tk.Tk()
    root.title("Select files!")
    root.geometry("400x400")
    root.withdraw()
    file_path_source = filedialog.askopenfilename(title="Select source file")
    file_path_target = filedialog.askopenfilename(title="Select target file")

def pca_manual(data, n_components):
    """
    A method to do Principal Component Analysis on an object 'data' 
    to find "n_components" number of components.
    Returns a list of the principal components.
    """
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
    """
    A method to align axes by calculating a rotation matrix that transforms 
    the source to the target.
    NOT BEING USED AT THE MOMENT.
    """
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
        #See: https://people.eecs.berkeley.edu/~ug/slide/pipeline/assignments/as5/rotation.html 
        K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                      [rotation_axis[2], 0, -rotation_axis[0]],
                      [-rotation_axis[1], rotation_axis[0], 0]])
        rotation = np.eye(3) + np.sin(rotation_angle) * K + (1 - np.cos(rotation_angle)) * np.dot(K, K)
        # Apply rotation matrix
        rotation_matrix = np.dot(rotation, rotation_matrix)
        print(f"\nThe final rotation matrix is \n{rotation_matrix}")
    
    return rotation_matrix

def transpose_method(axes, axes_target):
    """
    A simpler method to calculate the rotation matrix by assuming that
    the coordinate system remains consistent. 
    """
    print("This should be rotation matrix stuff")
    print("Axes:")
    print(str(axes))
    print("Target:")
    print(str(axes_target))
    T = np.transpose(axes_target)
    print("Transpose:")
    print(str(T))

    #matmul does matrix multiplication. Otherwise it does 
    #element-wise multiplication.
    P = np.matmul(axes,T)
    print("Product:")
    print(str(P))

    print("Identity:")
    print(str(np.matmul(P,np.transpose(P))))

    return P


def main():

    """
    Calls the m,ethods in the right order.
    """
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
    #rotation_matrix = align_vectors_manual(axes1, axes2)
    rotation_matrix = transpose_method(axes1, axes2)

    #Second verification after the stuff in the other method really
    print("R\n")
    print(rotation_matrix)
    print("\nR det:", det(rotation_matrix))

    #Pre ICP step of applying rotation and translation
    aligned_points1 = np.dot(scaled_points1, rotation_matrix)
    final_aligned_points1 = aligned_points1

    # Find the optimal mirrored version
    min_distance, optimal_mirror = find_optimal_mirror(final_aligned_points1, translated_points2, centroid2)
    print(f"The optimal mirror is \n{optimal_mirror}")

    mirrored_versions = mirror_axes(final_aligned_points1)

   # Visualize grid
    visualize_mirror_grid(
    mirrored_versions,
    optimal_mirror,
    translated_points2,
    centroid2
    )

    optimal_mirror = optimal_mirror + centroid2

    #Testing purposes only
    #optimal_mirror = final_aligned_points1

    # We're using the optimal mirrored version for finer alignment 
    pcd1_aligned = o3d.geometry.PointCloud()
    pcd1_aligned.points = o3d.utility.Vector3dVector(optimal_mirror)

    #ICP registration using o3d
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd1_aligned, pcd2_downsampled, 0.2, np.eye(4),         # Changing 0.2 to 0.5 doesn't help much either
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000) #Increasing max iterations does not make things better.
    )
    print(f"The registration attempt is \n{reg_p2p} ")

    #Apply ICP transformation
    final_aligned_points1_icp = np.dot(np.vstack((optimal_mirror.T, np.ones(optimal_mirror.shape[0]))).T, reg_p2p.transformation).T[:3].T

    #Visualize
    visualize_points(np.asarray(pcd2_downsampled.points), final_aligned_points1_icp)

    #Write output file so that we don't need to repeat experiments each time.
    #Combine the two point clouds into one
    # combined_pcd = o3d.geometry.PointCloud()
    # combined_pcd.points = o3d.utility.Vector3dVector(
    #     np.vstack((
    #         np.asarray(pcd2_downsampled.points),
    #         np.asarray(pcd1_aligned_icp.points)
    #     ))
    # )

    #Assign colors to
    # colors = np.vstack((
    #     np.tile([0, 1, 0], (len(pcd2_downsampled.points), 1)),  # Green for source PCD
    #     np.tile([1, 0, 0], (len(pcd1_aligned_icp.points), 1))   # Red for aligned PCD
    # ))
    # combined_pcd.colors = o3d.utility.Vector3dVector(colors)

    #Save the combined point cloud to a PLY file
    #o3d.io.write_point_cloud("combined_pcd.ply", combined_pcd)


    #Save aligned point clouds
    #o3d.io.write_point_cloud("aligned_pcd1.ply", pcd1_aligned_icp)


if __name__ == "__main__":
    main()