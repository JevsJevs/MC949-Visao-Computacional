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

BASE_DATA_PATH = current_dir.parent.parent.parent / "data"


def load_raw_images(panorama: str) -> dict[str, cv2.Mat]:
    """
    Load all images from a raw panorama folder into a dictionary.

    Args:
        panorama (str): Name of the panorama folder under "raw".

    Returns:
        Dict[str, cv2.Mat]: A dictionary mapping filename (without extension)
                             to the loaded OpenCV image.
    """
    path = BASE_DATA_PATH / "raw" / panorama
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
    output_path = BASE_DATA_PATH / path / f"{filename}.{img_format}"
    
    return cv2.imwrite(output_path, img)


def draw_pairing_lines(img1: cv2.Mat, img2: cv2.Mat, alg: str, filename: str,
                       path: str = "interim", img_format: str = ".jpg"):

    algs = {
        "SIFT": feature_extraction.SIFT,
        "ORB": feature_extraction.ORB,
        "AKAZE": feature_extraction.AKAZE
    }

    func = algs.get(alg.upper())

    kp1, des1 = func(img1)
    kp2, des2 = func(img2)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Aplica Lowe ratio test
    good_matches = david_loew_ratio_test(matches)

    # Desenha as linhas
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=2)

    # --- Marcar início e fim ---
    h1, w1 = img1.shape[:2]

    for m in good_matches:
        pt1 = tuple(map(int, kp1[m.queryIdx].pt))  # ponto na img1
        pt2 = tuple(map(int, kp2[m.trainIdx].pt))  # ponto na img2
        pt2 = (int(pt2[0] + w1), int(pt2[1]))      # compensar deslocamento da concatenação

        # "X" no ponto inicial (img1)
        cv2.drawMarker(img_matches, pt1, (0, 255, 0), markerType=cv2.MARKER_TILTED_CROSS,
                       markerSize=12, thickness=2, line_type=cv2.LINE_AA)

        # "O" no ponto final (img2)
        cv2.circle(img_matches, pt2, 6, (0, 0, 255), 2, lineType=cv2.LINE_AA)

    save_image(img_matches, filename, path, img_format)

    return img_matches


def david_loew_ratio_test(matches: cv2.DMatch, ratio=0.75):
    """
    Receives the matches returned by knnMatch and applies the ratio test.
    
    Args:
        matches (list[list[cv2.DMatch]]): list of lists of matches returned by knnMatch.
        ratio (float): Lowe's ratio test factor (default=0.75).
    
    Returns:
        list[cv2.DMatch]: list of good (accepted) matches.
    """
    good_matches = []
    for m, n in matches:  
        if m.distance < ratio * n.distance:
            good_matches.append(m)
    return good_matches



    
    
    