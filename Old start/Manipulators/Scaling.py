import open3d as o3d
import numpy as np
import tkinter as tk
from tkinter import filedialog, simpledialog

def scale(input_file, output_file, scaleFactor):
    print(f"Loading point cloud from {input_file}...")
    pcd = o3d.io.read_point_cloud(input_file)
    points = np.asarray(pcd.points)
    points = scaleFactor * points
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(output_file, pcd, write_ascii=True)
        
    return
    

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

    scaleFactor = simpledialog.askstring(
        "Scale factor",
        "Enter the scale factor:",
        initialvalue="1"
    )

    try:
        scaleFactor = float(scaleFactor)
    except (ValueError, TypeError):
        print("Invalid scale factor.")
        return
    
    if not output_filename:
        print("No output file specified. Exiting.")
        return
    
    scale(input_filename, output_filename, scaleFactor)

if __name__ == "__main__":
    main()
