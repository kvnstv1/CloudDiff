import tkinter as tk
from tkinter import filedialog
import open3d as o3d
import os
import numpy as np

def mirror_ply_file(file_path, mirror_axis):
    # Read the PLY file
    pcd = o3d.io.read_point_cloud(file_path)
    
    # Mirror the point cloud
    points = np.asarray(pcd.points)
    if mirror_axis == 'x':
        points[:, 0] *= -1
    elif mirror_axis == 'y':
        points[:, 1] *= -1
    elif mirror_axis == 'z':
        points[:, 2] *= -1
    else:
        raise ValueError("Invalid mirror axis. Must be 'x', 'y', or 'z'.")
    
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Write the mirrored point cloud to a new PLY file
    file_name, file_extension = os.path.splitext(file_path)
    new_file_path = f"{file_name}_mirrored{file_extension}"
    o3d.io.write_point_cloud(new_file_path, pcd)

def select_file():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(title="Select a PLY File", filetypes=[("PLY files", "*.ply"), ("All files", "*.*")])
    
    if file_path:
        print(f"Selected file: {file_path}")
        # Specify the mirror axis (e.g., 'x', 'y', or 'z')
        mirror_axis = 'y'  # Change this to 'y' or 'z' as needed
        mirror_ply_file(file_path, mirror_axis)
    else:
        print("No file selected.")

# Run the program
select_file()
