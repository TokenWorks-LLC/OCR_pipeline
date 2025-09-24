"""
Image preprocessing functions for handling rotated, faded, and inverted pages.
"""
import logging
from typing import Tuple, Optional

import cv2
import numpy as np
from skimage import filters
from skimage.exposure import adjust_gamma
from skimage.morphology import rectangle

from config import PREPROCESSING

logger = logging.getLogger(__name__)

def _ensure_odd(n: int) -> int:
    n = int(n)
    return n if n % 2 == 1 else n + 1

def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)



def apply_gamma(img: np.ndarray, gamma: float = None) -> np.ndarray:
    """Apply gamma correction to brighten dark images."""
    if gamma is None:
        gamma = PREPROCESSING['gamma_correction']

    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    # Convert to float and apply gamma correction
    img_float = img.astype(np.float32) / 255.0
    corrected = adjust_gamma(_clip01(img_float), gamma)
    out = (_clip01(corrected) * 255.0).astype(np.uint8)
    return out



def clahe_rgb(img: np.ndarray, clip_limit: float = None, grid_size: Tuple[int, int] = None) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to RGB image."""
    if clip_limit is None:
        clip_limit = PREPROCESSING['clahe_clip_limit']
    if grid_size is None:
        grid_size = PREPROCESSING['clahe_grid_size']
    
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    l_channel = clahe.apply(l_channel)
    
    # Merge channels and convert back to BGR
    lab = cv2.merge([l_channel, a_channel, b_channel])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def deskew_small(img: np.ndarray, angle_threshold: float = None) -> np.ndarray:
    """Deskew image by correcting small rotation angles."""
    if angle_threshold is None:
        angle_threshold = PREPROCESSING['deskew_angle_threshold']
    
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply binary threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return img
        
        # Get the largest contour (assuming it's the page)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get minimum area rectangle
        rect = cv2.minAreaRect(largest_contour)
        angle = rect[2]
        
        # Normalize angle
        if angle < -45:
            angle = 90 + angle
        elif angle > 45:
            angle = angle - 90
        
        # Only correct small angles
        if abs(angle) < angle_threshold and abs(angle) > 0.5:
            # Get rotation matrix
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Apply rotation
            rotated = cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            logger.debug(f"Deskewed image by {angle:.2f} degrees")
            return rotated
        
        return img
        
    except Exception as e:
        logger.warning(f"Deskewing failed: {e}")
        return img


def binarize_sauvola(img: np.ndarray, window_size: int = None, k: float = None) -> np.ndarray:
    """Apply Sauvola binarization for better text detection in faded images."""
    if window_size is None:
        window_size = PREPROCESSING['binarize_window_size']
    if k is None:
        k = PREPROCESSING['binarize_k']
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    try:
        # Apply Sauvola threshold
        threshold = filters.threshold_sauvola(gray, window_size=window_size, k=k)
        binary = gray > threshold
        
        # Convert back to uint8
        result = (binary * 255).astype(np.uint8)
        
        # Convert back to 3-channel if input was 3-channel
        if len(img.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        return result
        
    except Exception as e:
        logger.warning(f"Sauvola binarization failed: {e}")
        # Fallback to simple adaptive threshold
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, window_size, 2
        )
        if len(img.shape) == 3:
            adaptive = cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)
        return adaptive


def invert_if_better(img: np.ndarray, confidence_func) -> np.ndarray:
    """Invert image if the inverted version has better OCR confidence."""
    try:
        # Get confidence for normal image
        normal_conf = confidence_func(img)
        
        # Invert image
        inverted = 255 - img
        inverted_conf = confidence_func(inverted)
        
        if inverted_conf > normal_conf * 1.2:  # 20% improvement threshold
            logger.debug(f"Using inverted image (conf: {inverted_conf:.3f} vs {normal_conf:.3f})")
            return inverted
        
        return img
        
    except Exception as e:
        logger.warning(f"Inversion check failed: {e}")
        return img


def preprocess_faded(img: np.ndarray, apply_dilation: bool = True) -> np.ndarray:
    """Apply preprocessing pipeline for faded/low-contrast images."""
    try:
        # Step 1: Gamma correction
        result = apply_gamma(img)
        
        # Step 2: CLAHE
        result = clahe_rgb(result)
        
        # Step 3: Adaptive binarization
        result = binarize_sauvola(result)
        
        # Step 4: Optional mild dilation to connect broken characters
        if apply_dilation:
            kernel_size = PREPROCESSING['dilation_kernel_size']
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
            
            if len(result.shape) == 3:
                gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
                dilated = cv2.dilate(gray, kernel, iterations=1)
                result = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
            else:
                result = cv2.dilate(result, kernel, iterations=1)
        
        return result
        
    except Exception as e:
        logger.warning(f"Faded preprocessing failed: {e}")
        return img


def preprocess_pipeline(img: np.ndarray, confidence_func) -> np.ndarray:
    """
    Main preprocessing pipeline that applies all corrections.
    
    Args:
        img: Input image
        confidence_func: Function to evaluate OCR confidence for inversion check
    
    Returns:
        Preprocessed image
    """
    logger.debug("Starting preprocessing pipeline")
    
    # Step 1: Deskew small angles
    result = deskew_small(img)
    
    # Step 2: Apply faded image enhancements
    result = preprocess_faded(result)
    
    # Step 3: Check if inversion improves OCR
    result = invert_if_better(result, confidence_func)
    
    logger.debug("Preprocessing pipeline complete")
    return result
