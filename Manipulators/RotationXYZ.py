import tkinter as tk
from tkinter import filedialog
import open3d as o3d
import numpy as np
import os

def rotate_ply_file(file_path, rotation_axis, rotation_angle):
    # Read the PLY file
    pcd = o3d.io.read_point_cloud(file_path)
    
    # Define the rotation matrix
    rotation_angle = np.deg2rad(rotation_angle)  # Convert to radians
    if rotation_axis == 'x':
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(rotation_angle), -np.sin(rotation_angle)],
            [0, np.sin(rotation_angle), np.cos(rotation_angle)]
        ])
    elif rotation_axis == 'y':
        rotation_matrix = np.array([
            [np.cos(rotation_angle), 0, np.sin(rotation_angle)],
            [0, 1, 0],
            [-np.sin(rotation_angle), 0, np.cos(rotation_angle)]
        ])
    elif rotation_axis == 'z':
        rotation_matrix = np.array([
            [np.cos(rotation_angle), -np.sin(rotation_angle), 0],
            [np.sin(rotation_angle), np.cos(rotation_angle), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("Invalid rotation axis. Must be 'x', 'y', or 'z'.")
    
    # Apply the rotation
    points = np.asarray(pcd.points)
    rotated_points = np.dot(points, rotation_matrix)
    pcd.points = o3d.utility.Vector3dVector(rotated_points)
    
    # Write the rotated point cloud to a new PLY file
    file_name, file_extension = os.path.splitext(file_path)
    new_file_path = f"{file_name}_rotated{file_extension}"
    o3d.io.write_point_cloud(new_file_path, pcd)

def select_file():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(title="Select a PLY File", filetypes=[("PLY files", "*.ply"), ("All files", "*.*")])
    
    if file_path:
        print(f"Selected file: {file_path}")
        # Specify the rotation axis and angle (90 degrees around the z-axis)
        rotation_axis = 'x'
        rotation_angle = 90
        rotate_ply_file(file_path, rotation_axis, rotation_angle)
    else:
        print("No file selected.")

# Run the program
select_file()
