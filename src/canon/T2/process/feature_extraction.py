"""Feature extraction methods for T2 - 3D reconstruction project
This module extends T1 feature extraction for 3D reconstruction context.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from pathlib import Path
from tqdm import tqdm

# Import from T1 for reuse
from canon.T1.process import feature_extraction as t1_features


def extract_features_from_image_set(
    images: Dict[str, np.ndarray],
    detector_type: str = "SIFT",
    **detector_params
) -> Dict[str, Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]]:
    """
    Extract features from a set of images for 3D reconstruction.
    
    Args:
        images: Dictionary mapping image names to np.ndarray images
        detector_type: Type of detector ("SIFT", "ORB", "AKAZE")
        **detector_params: Parameters for the detector
    
    Returns:
        Dictionary mapping image names to (keypoints, descriptors) tuples
    """
    detector_functions = {
        "SIFT": t1_features.SIFT,
        "ORB": t1_features.ORB,
        "AKAZE": t1_features.AKAZE
    }
    
    if detector_type not in detector_functions:
        raise ValueError(f"Unsupported detector type: {detector_type}")
    
    detector_func = detector_functions[detector_type]
    features = {}
    
    print(f"Extracting {detector_type} features from {len(images)} images...")
    
    # Using tqdm for progress bar when processing many images
    for img_name, img in tqdm(images.items(), desc=f"Extracting {detector_type}", unit="image"):
        keypoints, descriptors = detector_func(img, **detector_params)
        features[img_name] = (keypoints, descriptors)
        
        # Only print details for small datasets to avoid spam
        if len(images) <= 10:
            print(f"  {img_name}: {len(keypoints) if keypoints else 0} keypoints")
    
    return features


def compare_detectors(
    images: Dict[str, np.ndarray],
    detectors: List[str] = ["SIFT", "ORB", "AKAZE"],
    sample_images: Optional[List[str]] = None
) -> Dict[str, Dict[str, Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]]]:
    """
    Compare different feature detectors on a sample of images.
    
    Args:
        images: Dictionary of images
        detectors: List of detector types to compare
        sample_images: List of image names to use for comparison (if None, uses first 3)
    
    Returns:
        Nested dictionary: detector_type -> image_name -> (keypoints, descriptors)
    """
    if sample_images is None:
        sample_images = list(images.keys())[:3]
    
    sample_imgs = {name: images[name] for name in sample_images if name in images}
    
    comparison_results = {}
    
    for detector in detectors:
        print(f"\n=== Testing {detector} ===")
        try:
            features = extract_features_from_image_set(sample_imgs, detector)
            comparison_results[detector] = features
        except Exception as e:
            print(f"Error with {detector}: {e}")
            comparison_results[detector] = {}
    
    # Print comparison summary
    print("\n=== Feature Detection Comparison ===")
    print(f"{'Image':<15} {'SIFT':<8} {'ORB':<8} {'AKAZE':<8}")
    print("-" * 45)
    
    for img_name in sample_images:
        row = f"{img_name:<15}"
        for detector in detectors:
            if detector in comparison_results and img_name in comparison_results[detector]:
                kp_count = len(comparison_results[detector][img_name][0] or [])
                row += f"{kp_count:<8}"
            else:
                row += f"{'ERROR':<8}"
        print(row)
    
    return comparison_results


def extract_features_with_metadata(
    images: Dict[str, np.ndarray],
    detector_type: str = "SIFT",
    save_visualizations: bool = True,
    output_path: str = "T2/interim",
    **detector_params
) -> Dict[str, Dict]:
    """
    Extract features with additional metadata for 3D reconstruction pipeline.
    
    Args:
        images: Dictionary of images
        detector_type: Detector type
        save_visualizations: Whether to save keypoint visualizations
        output_path: Path to save visualizations
        **detector_params: Detector parameters
    
    Returns:
        Dictionary with features and metadata for each image
    """
    from canon.utils import image_utils
    
    features_with_metadata = {}
    features = extract_features_from_image_set(images, detector_type, **detector_params)
    
    # Add progress bar for metadata processing, especially when saving visualizations
    desc = f"Processing {detector_type} metadata"
    if save_visualizations:
        desc += " + saving visualizations"
    
    for img_name, (keypoints, descriptors) in tqdm(features.items(), desc=desc, unit="image"):
        img = images[img_name]
        
        # Create metadata
        metadata = {
            'keypoints': keypoints,
            'descriptors': descriptors,
            'num_keypoints': len(keypoints) if keypoints else 0,
            'image_shape': img.shape,
            'detector_type': detector_type,
            'detector_params': detector_params
        }
        
        # Save visualization if requested
        if save_visualizations and keypoints:
            img_with_kp = cv2.drawKeypoints(
                img, keypoints, None,
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                color=(0, 255, 0)
            )
            
            filename = f"{detector_type}_{img_name}_keypoints"
            success = image_utils.save_image(img_with_kp, filename, output_path)
            if success:
                metadata['visualization_saved'] = f"{output_path}/{filename}.jpg"
        
        features_with_metadata[img_name] = metadata
    
    return features_with_metadata


# Convenience functions for specific detectors optimized for 3D reconstruction
def extract_sift_for_3d(images: Dict[str, np.ndarray], **kwargs) -> Dict[str, Dict]:
    """Extract SIFT features optimized for 3D reconstruction."""
    default_params = {
        'nfeatures': 2000,  # More features for 3D
        'contrastThreshold': 0.03,  # Lower threshold for more features
        'edgeThreshold': 15
    }
    default_params.update(kwargs)
    
    return extract_features_with_metadata(images, "SIFT", **default_params)


def extract_orb_for_3d(images: Dict[str, np.ndarray], **kwargs) -> Dict[str, Dict]:
    """Extract ORB features optimized for 3D reconstruction."""
    default_params = {
        'nfeatures': 3000,  # More features for 3D
        'scaleFactor': 1.2,
        'fastThreshold': 15  # Lower threshold for more features
    }
    default_params.update(kwargs)
    
    return extract_features_with_metadata(images, "ORB", **default_params)


def extract_akaze_for_3d(images: Dict[str, np.ndarray], **kwargs) -> Dict[str, Dict]:
    """Extract AKAZE features optimized for 3D reconstruction."""
    default_params = {
        'threshold': 0.0005,  # Lower threshold for more features
        'nOctaves': 5,
        'nOctaveLayers': 4
    }
    default_params.update(kwargs)
    
    return extract_features_with_metadata(images, "AKAZE", **default_params)
