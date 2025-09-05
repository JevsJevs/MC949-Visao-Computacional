"""Utility functions to image loading and processing
"""

import cv2
import os
import sys
from pathlib import Path
from src.process import feature_extraction


try:
# Works in scripts
    current_file = Path(__file__).resolve()
except NameError:
    # Fallback for interactive sessions like Jupyter
    current_file = Path(sys.argv[0]).resolve() if sys.argv[0] else Path.cwd()

current_dir = current_file.parent

BASE_DATA_PATH = current_dir.parent.parent.parent.parent / "data" / "T2"


def load_raw_images(object: str) -> dict[str, cv2.Mat]:
    """
    Load all images from a raw panorama folder into a dictionary.

    Args:
        panorama (str): Name of the panorama folder under "raw".

    Returns:
        Dict[str, cv2.Mat]: A dictionary mapping filename (without extension)
                             to the loaded OpenCV image.
    """
    path = BASE_DATA_PATH / "interim" / object
    images: dict[str, cv2.Mat] = {}
    
    for file in sorted(path.glob("*")):  # iterate over files
        if file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            img = cv2.imread(str(file))
            if img is not None:
                # Use filename without extension as key
                key = file.stem  
                images[key] = img
    
    return images
