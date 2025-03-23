import tkinter as tk
from tkinter import filedialog
import open3d as o3d
import os
import numpy as np

def translate_ply_file(file_path, translation_vector):
    # Read the PLY file
    pcd = o3d.io.read_point_cloud(file_path)
    
    # Translate the point cloud
    translation_vector = np.array(translation_vector)
    pcd.translate(translation_vector)
    
    # Write the translated point cloud to a new PLY file
    file_name, file_extension = os.path.splitext(file_path)
    new_file_path = f"{file_name}_translated{file_extension}"
    o3d.io.write_point_cloud(new_file_path, pcd)

def select_file():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(title="Select a PLY File", filetypes=[("PLY files", "*.ply"), ("All files", "*.*")])
    
    if file_path:
        print(f"Selected file: {file_path}")
        # Specify the translation vector (300 units along the x-axis)
        translation_vector = [300, 500, -250]
        translate_ply_file(file_path, translation_vector)
    else:
        print("No file selected.")

# Run the program
select_file()
