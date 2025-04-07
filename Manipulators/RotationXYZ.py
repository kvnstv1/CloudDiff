import tkinter as tk
from tkinter import filedialog
import open3d as o3d
import numpy as np
import os


def rotate_ply_file(file_path, rotation_axis, rotation_angle):

    pcd = o3d.io.read_point_cloud(file_path)

    
    
    # Define the rotation matrix
    rotation_angle = np.deg2rad(rotation_angle) 
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
    

    pcd.rotate(rotation_matrix, center=(0, 0, 0))  # Rotate around the origin
    
    # #Rotation time!
    # points = np.asarray(pcd.points)
    # rotated_points = np.matmul(points, np.transpose(rotation_matrix))
    # pcd.points = o3d.utility.Vector3dVector(rotated_points)
    
    #Write to file
    file_name, file_extension = os.path.splitext(file_path)
    new_file_path = f"{file_name}_rotated{file_extension}"
    o3d.io.write_point_cloud(new_file_path, pcd, write_ascii=True)

def select_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select a PLY File", filetypes=[("PLY files", "*.ply"), ("All files", "*.*")])
    
    if file_path:
        print(f"Selected file: {file_path}")
        #CHange these
        rotation_axis = 'y'
        rotation_angle = -40
        rotate_ply_file(file_path, rotation_axis, rotation_angle)
    else:
        print("No file selected.")


select_file()
