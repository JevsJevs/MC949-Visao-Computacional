"""Feature extraction methods, corresponding to step 2 of our project"""

import cv2
import numpy as np
from typing import List, Tuple, Optional

def SIFT(img: cv2.Mat) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
    """
    Run SIFT on an image and return keypoints and descriptors.

    Args:
        img: Input image as a NumPy array (BGR).

    Returns:
        A tuple of (keypoints, descriptors). Descriptors can be None if no keypoints found.
    """
    sift = cv2.SIFT_create()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # SIFT works on grayscale

    # Detect keypoints and descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors