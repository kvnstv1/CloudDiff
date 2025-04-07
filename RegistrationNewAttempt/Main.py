import numpy as np
from scipy.spatial import KDTree
import open3d as o3d
from scipy.linalg import det
import tkinter as tk
from tkinter import filedialog



def selectFiles():
    """
    Prompts the user to select source and target files using a file dialog.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Prompt the user to select the source file
    source_file_path = filedialog.askopenfilename(title="Select Source File")
    if not source_file_path:
        print("No source file selected.")
        return None, None

    # Prompt the user to select the target file
    target_file_path = filedialog.askopenfilename(title="Select Target File")
    if not target_file_path:
        print("No target file selected.")
        return None, None

    return source_file_path, target_file_path





def main():

    sourceFile, targetFile = selectFiles()

    


if __name__ == "__main__":
    main()

