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

def readPlyFile(filename):
    """
    Reads a ply file.
    Returns a ndarray that is ready to be operated on
    """

    pcd = o3d.io.read_point_cloud(filename)
    pcd = pcd.farthest_point_down_sample(50000)
    data = np.asarray(pcd.points)
    data = np.transpose(data)

    return data

def writePlyFile(data):
    """
    Writes a ply file with a given filename and data.
    """
    filename = filedialog.asksaveasfilename(
        title="Save sampled PLY file as",
        defaultextension=".ply",
        filetypes=[("PLY files", "*.ply"), ("All files", "*.*")]
    )
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data.T)
    o3d.io.write_point_cloud(filename, pcd, write_ascii=True)
    print(f"Wrote data to {filename} successfully.")

def writeCombinedData(source, target):
    """
    This method combines two ply files into one file.
    The source cloud is coloured green, and the target is coloured red. 
    """
    sourcePCD = o3d.geometry.PointCloud()
    targetPCD = o3d.geometry.PointCloud()
    sourcePCD.points = o3d.utility.Vector3dVector(np.transpose(source))
    targetPCD.points = o3d.utility.Vector3dVector(np.transpose(target))
    sourcePCD.paint_uniform_color([1,0,0])
    targetPCD.paint_uniform_color([0,1,0])
    combined = sourcePCD + targetPCD
    filename = filedialog.asksaveasfilename(
        title="Let's save the combined data now",
        defaultextension = ".ply",
        filetypes = [("PLY files", "*.ply"), ("All files", "*.*")]
    )
    o3d.io.write_point_cloud(filename, combined, write_ascii=True)

def centreAndScale(source, target):
    """
    Centres and scales the source and target
    Assumes source and target are arrays of shape(3,n)
    """
    sourceCentroid = np.mean(source, axis=1)[:, np.newaxis]
    targetCentroid = np.mean(target, axis=1)[:, np.newaxis]

    source = source - sourceCentroid
    target = target - targetCentroid

    #Note: Using Euclidean norms
    sourceDist = np.mean(np.linalg.norm(source, axis=0))
    targetDist = np.mean(np.linalg.norm(target, axis=0))

    scale = targetDist/sourceDist
    source = source * scale

    return source, target


def main():

    sourceFile, targetFile = selectFiles()
    source = readPlyFile(sourceFile)
    target = readPlyFile(targetFile)
    source, target = centreAndScale(source,target)
    writeCombinedData(source, target)



if __name__ == "__main__":
    main()

