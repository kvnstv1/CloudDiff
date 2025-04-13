import open3d as o3d
import numpy as np
import tkinter as tk
from tkinter import filedialog, simpledialog

def sample_random_points(input_file, output_file, num_points=50000):
    # Load the point cloud
    print(f"Loading point cloud from {input_file}...")
    pcd = o3d.io.read_point_cloud(input_file)
    
    # Get the total number of points
    total_points = len(np.asarray(pcd.points))
    print(f"Loaded {total_points} points")
    
    # Check if we have enough points
    if total_points <= num_points:
        print(f"Input cloud has fewer than {num_points} points. Using all points.")
        o3d.io.write_point_cloud(output_file, pcd)
        print(f"Wrote all {total_points} points to {output_file}")
        return
    
    # Sample random indices
    indices = np.random.choice(total_points, num_points, replace=False)
    
    # Create a new point cloud with the sampled points
    sampled_pcd = o3d.geometry.PointCloud()
    points = np.asarray(pcd.points)[indices]
    sampled_pcd.points = o3d.utility.Vector3dVector(points)
    
    # Copy colors if they exist
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)[indices]
        sampled_pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Copy normals if they exist
    if pcd.has_normals():
        normals = np.asarray(pcd.normals)[indices]
        sampled_pcd.normals = o3d.utility.Vector3dVector(normals)
    
    # Write the sampled point cloud to a PLY file
    o3d.io.write_point_cloud(output_file, sampled_pcd, write_ascii=True)
    print(f"Successfully wrote {num_points} randomly sampled points to {output_file}")

def main():
    # Create and hide the Tkinter root window
    root = tk.Tk()
    root.withdraw()
    
    # Ask user for input file using file dialog
    input_filename = filedialog.askopenfilename(
        title="Select input PLY file",
        filetypes=[("PLY files", "*.ply"), ("All files", "*.*")]
    )
    
    if not input_filename:
        print("No input file selected. Exiting.")
        return
    
    # Ask user for output file using file dialog
    output_filename = filedialog.asksaveasfilename(
        title="Save sampled PLY file as",
        defaultextension=".ply",
        filetypes=[("PLY files", "*.ply"), ("All files", "*.*")]
    )
    
    if not output_filename:
        print("No output file specified. Exiting.")
        return
    
    # Ask user for number of points using a dialog
    num_points_str = simpledialog.askstring(
        "Number of Points", 
        "Number of points to sample:",
        initialvalue="50000"
    )
    
    num_points = 50000  # default
    
    if num_points_str:
        try:
            num_points = int(num_points_str)
            if num_points <= 0:
                print("Number of points must be positive. Using default of 50000.")
                num_points = 50000
        except ValueError:
            print("Invalid number, using default of 50000")
    
    # Sample the points
    sample_random_points(input_filename, output_filename, num_points)

if __name__ == "__main__":
    main()
