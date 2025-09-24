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
    gx, gy = max(1, int(grid_size[0])), max(1, int(grid_size[1]))

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

     # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(gx, gy))
    l2 = clahe.apply(l)

    # Merge channels and convert back to BGR
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)



def deskew_small(img: np.ndarray, angle_threshold: float = None) -> np.ndarray:
    """Deskew image by correcting small rotation angles."""
    if angle_threshold is None:
        angle_threshold = PREPROCESSING['deskew_angle_threshold']
    
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        h, w = gray.shape[:2]
        
        # NEW: Projection variance sweep across ±threshold angles
        steps = max(7, int(angle_threshold * 4) | 1)  # odd num steps
        angles = np.linspace(-float(angle_threshold), float(angle_threshold), steps)
        
        best_angle, best_var = 0.0, -1.0
        for ang in angles:
            M = cv2.getRotationMatrix2D((w // 2, h // 2), ang, 1.0)
            rot = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            proj = np.sum(rot, axis=1).astype(np.float32)
            var = float(np.var(proj))
            if var > best_var:
                best_var, best_angle = var, ang
        
        # Only correct if meaningful (>=0.2° skew)
        if abs(best_angle) >= 0.2:
            M = cv2.getRotationMatrix2D((w // 2, h // 2), best_angle, 1.0)
            rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            logger.debug(f"Deskewed image by {best_angle:.2f} degrees")
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
    
    ws = _ensure_odd(window_size)
    
    # Convert to grayscale if needed
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    
    try:
        # Apply Sauvola threshold
        threshold = filters.threshold_sauvola(gray, window_size=ws, k=float(k))
        binary = (gray > threshold).astype(np.uint8) * 255
        
        # Convert back to 3-channel if input was 3-channel
        if img.ndim == 3:
            binary = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        return binary
        
    except Exception as e:
        logger.warning(f"Sauvola binarization failed: {e}")
        # Fallback to simple adaptive threshold
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, ws, 2
        )
        if img.ndim == 3:
            adaptive = cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)
        return adaptive



def invert_if_better(img: np.ndarray, confidence_func, min_gain_ratio: float = 1.1, min_gain_abs: float = 0.02) -> np.ndarray:
    """Invert image if the inverted version has better OCR confidence."""
    try:
        # Get confidence for normal image
        normal_conf = float(confidence_func(img))
        
        # Invert image
        inverted = (255 - img).astype(np.uint8)
        inverted_conf = float(confidence_func(inverted))
        
        # NEW: require both ratio and absolute gain
        if (inverted_conf >= normal_conf * min_gain_ratio) and ((inverted_conf - normal_conf) >= min_gain_abs):
            logger.debug(f"Using inverted image (conf: {inverted_conf:.3f} vs {normal_conf:.3f})")
            return inverted
        
        return img
        
    except Exception as e:
        logger.warning(f"Inversion check failed: {e}")
        return img


def preprocess_faded(img: np.ndarray, apply_dilation: bool = None) -> np.ndarray:
    """Apply preprocessing pipeline for faded/low-contrast images."""
    try:
        if apply_dilation is None:
            apply_dilation = bool(PREPROCESSING.get('apply_dilation', False))
        
        # Step 1: Gamma correction
        result = apply_gamma(img)
        
        # Step 2: CLAHE
        result = clahe_rgb(result)
        
        # Step 3: Adaptive binarization
        result = binarize_sauvola(result)
        
        # Step 4: Optional mild dilation to connect broken characters
        if apply_dilation:
            kernel_size = int(PREPROCESSING['dilation_kernel_size'])
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, kernel_size), max(1, kernel_size)))
            
            if result.ndim == 3:
                gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
                dilated = cv2.dilate(gray, kernel, iterations=1)
                result = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
            else:
                result = cv2.dilate(result, kernel, iterations=1)
        
        return result
        
    except Exception as e:
        logger.warning(f"Faded preprocessing failed: {e}")
        return img

def preprocess_pipeline(img: np.ndarray, confidence_func) -> dict:
    """
    Main preprocessing pipeline that applies all corrections.

    Returns:
        dict with:
          - 'pretty': enhanced BGR for OCR
          - 'binary': binarized for layout ops
          - 'angle': applied deskew angle (float)
    """
    logger.debug("Starting preprocessing pipeline")
    # 1) Deskew (works on original image)
    deskewed = deskew_small(img)
    # 2) Build a pretty RGB for OCR (gamma + CLAHE), no binarization here
    pretty = clahe_rgb(apply_gamma(deskewed))
    # 3) Binary for layout ops (from pretty, more stable)
    binary = binarize_sauvola(pretty)

    # 4) Consider inversion only for OCR image, not binary layout
    pretty_inv_decided = invert_if_better(pretty, confidence_func)

    logger.debug("Preprocessing pipeline complete")
    return {
        "pretty": pretty_inv_decided,
        "binary": binary,
        "angle": 0.0  # Optional: record angle if you store it inside deskew_small
    }
