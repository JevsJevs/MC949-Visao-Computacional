"""Utility functions to image loading and processing
"""

import cv2
import sys
import numpy as np
from pathlib import Path

try:
# Works in scripts
    current_file = Path(__file__).resolve()
except NameError:
    # Fallback for interactive sessions like Jupyter
    current_file = Path(sys.argv[0]).resolve() if sys.argv[0] else Path.cwd()

current_dir = current_file.parent

BASE_DATA_PATH = current_dir.parent.parent.parent / "data"


def load_images(datasetDirFromData: str) -> dict[str, np.ndarray]:
    """
    Load all images from a raw panorama folder into a dictionary.

    Args:
        datasetDirFromData (str): Considering the data folder as base, define the dataset path from it.
            Ex: Dataset at path (data/T1/Dataset) -> datasetDirFromData = "T1/Dataset"

    Returns:
        Dict[str, np.ndarray]: A dictionary mapping filename (without extension)
                             to the loaded OpenCV image.
    """
    path = BASE_DATA_PATH / Path(datasetDirFromData)
    images: dict[str, np.ndarray] = {}
    
    for file in sorted(path.glob("*")):  # iterate over files
        if file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            img = cv2.imread(str(file))
            if img is not None:
                # Use filename without extension as key
                key = file.stem  
                images[key] = img
    
    return images


def save_image(img: np.ndarray, filename : str, path: str = "interim", img_format : str = ".jpg") -> bool:
    """
    Save an OpenCV image to disk.

    Args:
        img (np.ndarray): The image to save.
        filename (str): Name of the output file (without extension).
        path (str, optional): Subfolder under BASE_DATA_PATH to save the image. Default is "interim".
        format (str, optional): Image format/extension (e.g., 'jpg', 'png'). Default is 'jpg'.

    Returns:
        bool: True if the image was saved successfully, False otherwise.
    """
    # Ensure format does not have a leading dot
    img_format = img_format.lstrip(".")
    output_path = BASE_DATA_PATH / Path(path)
    output_path.mkdir(parents=True, exist_ok=True)
    finalDir = output_path / f"{filename}.{img_format}"
    
    success = cv2.imwrite(finalDir, img)
    return success

def Draw_points(image, pts, repro):
    if repro == False:
        image = cv2.drawKeypoints(image, pts, image, color=(0, 255, 0), flags=0)
    else:
        for p in pts:
            image = cv2.circle(image, tuple(p), 2, (0, 0, 255), -1)
    return image

def img_downscale(img, downscale):
	downscale = int(downscale/2)
	i = 1
	while(i <= downscale):
		img = cv2.pyrDown(img)
		i = i + 1
	return img

def to_ply(path, point_cloud, colors, densify):
    out_points = point_cloud.reshape(-1, 3) * 200
    out_colors = colors.reshape(-1, 3)
    print(out_colors.shape, out_points.shape)
    verts = np.hstack([out_points, out_colors])

    # cleaning point cloud
    mean = np.mean(verts[:, :3], axis=0)
    temp = verts[:, :3] - mean
    dist = np.sqrt(temp[:, 0] ** 2 + temp[:, 1] ** 2 + temp[:, 2] ** 2)
    #print(dist.shape, np.mean(dist))
    indx = np.where(dist < np.mean(dist) + 300)
    verts = verts[indx]
    #print( verts.shape)
    ply_header = '''ply
		format ascii 1.0
		element vertex %(vert_num)d
		property float x
		property float y
		property float z
		property uchar blue
		property uchar green
		property uchar red
		end_header
		'''
    if not densify:
        with open('../../data/T2/results/' + 'sparse.ply', 'w') as f:
            f.write(ply_header % dict(vert_num=len(verts)))
            np.savetxt(f, verts, '%f %f %f %d %d %d')
    else:
        with open('../../data/T2/results/' + 'dense.ply', 'w') as f:
            f.write(ply_header % dict(vert_num=len(verts)))
            np.savetxt(f, verts, '%f %f %f %d %d %d')