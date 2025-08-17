"""Feature extraction methods, corresponding to step 2 of our project"""

import cv2
import numpy as np
from typing import List, Tuple, Optional

def SIFT(
    img: cv2.Mat,
    nfeatures: int = 0,
    nOctaveLayers: int = 3,
    contrastThreshold: float = 0.04,
    edgeThreshold: int = 10,
    sigma: float = 1.6
) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
    """
    Run SIFT on an image and return keypoints and descriptors.

    Args:
        img: Input image as a NumPy array (BGR).
        nfeatures: Maximum number of keypoints to retain (0 = all).
        nOctaveLayers: Number of layers per octave in the Gaussian pyramid.
        contrastThreshold: Higher value filters out more low-contrast keypoints.
        edgeThreshold: Threshold to filter edge-like keypoints (lower = stricter).
        sigma: Gaussian blur applied in the first octave.

    Returns:
        A tuple of (keypoints, descriptors).
        - keypoints: List of detected keypoints.
        - descriptors: NumPy array of shape (N, 128) with SIFT descriptors,
          or None if no keypoints found.
    """
    sift = cv2.SIFT_create(
        nfeatures=nfeatures,
        nOctaveLayers=nOctaveLayers,
        contrastThreshold=contrastThreshold,
        edgeThreshold=edgeThreshold,
        sigma=sigma,
    )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors


def ORB(
    img: cv2.Mat,
    nfeatures: int = 500,
    scaleFactor: float = 1.2,
    nlevels: int = 8,
    edgeThreshold: int = 31,
    firstLevel: int = 0,
    WTA_K: int = 2,
    scoreType: int = cv2.ORB_HARRIS_SCORE,
    patchSize: int = 31,
    fastThreshold: int = 20
) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
    """
    Run ORB on an image and return keypoints and descriptors.

    Args:
        img: Input image as a NumPy array (BGR).
        nfeatures: Maximum number of keypoints to retain.
        scaleFactor: Pyramid decimation ratio (e.g., 1.2 means each level is 20% smaller).
        nlevels: Number of pyramid levels.
        edgeThreshold: Minimum distance from the image border to consider a keypoint.
        firstLevel: Pyramid level to start.
        WTA_K: Number of points used in BRIEF descriptor (2, 3, or 4).
        scoreType: Keypoint ranking method (cv2.ORB_HARRIS_SCORE or cv2.ORB_FAST_SCORE).
        patchSize: Size of the patch used by the descriptor.
        fastThreshold: Threshold for the FAST corner detector (lower = more keypoints).

    Returns:
        A tuple of (keypoints, descriptors).
        - keypoints: List of detected keypoints.
        - descriptors: NumPy array of shape (N, 32) with binary ORB descriptors,
          or None if no keypoints found.
    """
    orb = cv2.ORB_create(
        nfeatures=nfeatures,
        scaleFactor=scaleFactor,
        nlevels=nlevels,
        edgeThreshold=edgeThreshold,
        firstLevel=firstLevel,
        WTA_K=WTA_K,
        scoreType=scoreType,
        patchSize=patchSize,
        fastThreshold=fastThreshold,
    )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors