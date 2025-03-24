import tkinter as tk
from tkinter import filedialog
import open3d as o3d

def select_file():
    file_path = filedialog.askopenfilename(title="Select a PLY file", filetypes=[("PLY files", "*.ply"), ("All files", "*.*")])
    if file_path:
        visualize_ply(file_path)

def visualize_ply(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    o3d.visualization.draw_geometries([pcd])

root = tk.Tk()
root.title("PLY File Visualizer")

open_button = tk.Button(root, text="Open PLY File", command=select_file)
open_button.pack()

root.mainloop()
