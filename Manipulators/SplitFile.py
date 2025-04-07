import tkinter as tk
from tkinter import filedialog
from plyfile import PlyData, PlyElement
import numpy as np

def split_ply_file(file_path):
    # Read the PLY file
    plydata = PlyData.read(file_path)
    original_file_name = file_path.split('/')[-1].split('.')[0]

    # Extract vertex data
    vertices = plydata['vertex']
    vertex_data = vertices.data

    # Split into even and odd rows
    even_vertices = vertex_data[::2]
    odd_vertices = vertex_data[1::2]

    # Create new PlyData objects for even and odd files
    even_element = PlyElement.describe(np.array(even_vertices.tolist(), dtype=vertex_data.dtype), 'vertex')
    odd_element = PlyElement.describe(np.array(odd_vertices.tolist(), dtype=vertex_data.dtype), 'vertex')

    # Preserve headers and update vertex count
    even_plydata = PlyData([even_element], text=True)
    odd_plydata = PlyData([odd_element], text=True)

    # Write to new files
    even_file_name = f"{original_file_name}_even.ply"
    odd_file_name = f"{original_file_name}_odd.ply"

    with open(even_file_name, 'w') as even_file:
        even_plydata.write(even_file)

    with open(odd_file_name, 'w') as odd_file:
        odd_plydata.write(odd_file)

    print(f"Files created: {even_file_name}, {odd_file_name}")

def select_and_process_file():
    # File chooser dialog
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(title="Select a PLY File", filetypes=[("PLY files", "*.ply"), ("All files", "*.*")])
    
    if file_path:
        split_ply_file(file_path)

# Run the program
select_and_process_file()
