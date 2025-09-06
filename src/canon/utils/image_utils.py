"""Utility functions to image loading and processing
"""

import cv2
import sys
from pathlib import Path

try:
# Works in scripts
    current_file = Path(__file__).resolve()
except NameError:
    # Fallback for interactive sessions like Jupyter
    current_file = Path(sys.argv[0]).resolve() if sys.argv[0] else Path.cwd()

current_dir = current_file.parent

BASE_DATA_PATH = current_dir.parent.parent.parent / "data"


def load_images(datasetDirFromData: str) -> dict[str, cv2.Mat]:
    """
    Load all images from a raw panorama folder into a dictionary.

    Args:
        datasetDirFromData (str): Considering the data folder as base, define the dataset path from it.
            Ex: Dataset at path (data/T1/Dataset) -> datasetDirFromData = "T1/Dataset"

    Returns:
        Dict[str, cv2.Mat]: A dictionary mapping filename (without extension)
                             to the loaded OpenCV image.
    """
    path = BASE_DATA_PATH / Path(datasetDirFromData)
    images: dict[str, cv2.Mat] = {}
    
    for file in sorted(path.glob("*")):  # iterate over files
        if file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            img = cv2.imread(str(file))
            if img is not None:
                # Use filename without extension as key
                key = file.stem  
                images[key] = img
    
    return images


def save_image(img: cv2.Mat, filename : str, path: str = "interim", img_format : str = ".jpg") -> bool:
    """
    Save an OpenCV image to disk.

    Args:
        img (cv2.Mat): The image to save.
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