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

# Import config with fallback
try:
    from .config import PREPROCESSING
except ImportError:
    try:
        from config import PREPROCESSING
    except ImportError:
        # Fallback configuration
        PREPROCESSING = {
            'gamma_correction': 0.9,
            'clahe_clip_limit': 3.0,
            'clahe_grid_size': (8, 8),
            'binarize_window_size': 31,
            'binarize_k': 0.2,
            'dilation_kernel_size': 1,
            'apply_dilation': False,
            'deskew_angle_threshold': 3.0
        }

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


def apply_bilateral_filter(img: np.ndarray, d: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
    """
    Apply bilateral filter for edge-preserving denoising.
    
    Args:
        img: Input image
        d: Diameter of each pixel neighborhood
        sigma_color: Filter sigma in the color space
        sigma_space: Filter sigma in the coordinate space
    """
    try:
        if img.ndim == 3:
            # Apply bilateral filter to each channel
            filtered = cv2.bilateralFilter(img, d, sigma_color, sigma_space)
        else:
            # Grayscale image
            filtered = cv2.bilateralFilter(img, d, sigma_color, sigma_space)
        
        logger.debug(f"Applied bilateral filter (d={d}, σ_color={sigma_color}, σ_space={sigma_space})")
        return filtered
        
    except Exception as e:
        logger.warning(f"Bilateral filtering failed: {e}")
        return img


def filter_short_lines(text_lines: list, min_length: int = 3, min_alpha_chars: int = 2) -> list:
    """
    Filter out very short text lines that are likely noise.
    
    Args:
        text_lines: List of text line strings
        min_length: Minimum character length
        min_alpha_chars: Minimum alphabetic characters
        
    Returns:
        Filtered list of text lines
    """
    filtered_lines = []
    
    for line in text_lines:
        if not line or not isinstance(line, str):
            continue
        
        line_clean = line.strip()
        
        # Check minimum length
        if len(line_clean) < min_length:
            continue
        
        # Check minimum alphabetic characters
        alpha_count = sum(1 for c in line_clean if c.isalpha())
        if alpha_count < min_alpha_chars:
            continue
        
        # Check if line is not just punctuation or numbers
        non_punct_chars = sum(1 for c in line_clean if c.isalnum())
        if non_punct_chars < min_alpha_chars:
            continue
        
        filtered_lines.append(line)
    
    logger.debug(f"Filtered {len(text_lines)} → {len(filtered_lines)} text lines")
    return filtered_lines


def sauvola_binarization_quality(img: np.ndarray, config: dict = None) -> np.ndarray:
    """
    High-quality Sauvola binarization for text detection assistance.
    
    Args:
        img: Input image
        config: Configuration dict with sauvola_k parameter
        
    Returns:
        Binarized image for detection assistance
    """
    config = config or {}
    k = config.get('sauvola_k', 0.3)  # Default quality k value
    window_size = config.get('sauvola_window_size', 25)
    
    try:
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img.copy()
        
        # Apply Sauvola binarization
        from skimage.filters import threshold_sauvola
        threshold = threshold_sauvola(gray, window_size=window_size, k=k)
        binary = (gray > threshold).astype(np.uint8) * 255
        
        logger.debug(f"Sauvola binarization: window_size={window_size}, k={k}")
        return binary
        
    except Exception as e:
        logger.warning(f"Sauvola binarization failed: {e}")
        # Fallback to adaptive threshold
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 
            window_size if window_size % 2 == 1 else window_size + 1, 10
        )
        return adaptive


def crop_with_padding(image: np.ndarray, bbox: Tuple[int, int, int, int], 
                     pad_percent: float = 0.025) -> np.ndarray:
    """
    Crop image region with percentage-based padding.
    
    Args:
        image: Input image
        bbox: Bounding box (x1, y1, x2, y2)
        pad_percent: Padding as percentage of bbox dimensions
        
    Returns:
        Cropped and padded image region
    """
    try:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        
        # Calculate padding
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        pad_x = int(bbox_w * pad_percent)
        pad_y = int(bbox_h * pad_percent)
        
        # Apply padding and clamp to image bounds
        x1_pad = max(0, x1 - pad_x)
        y1_pad = max(0, y1 - pad_y)
        x2_pad = min(w, x2 + pad_x)
        y2_pad = min(h, y2 + pad_y)
        
        # Crop the region
        cropped = image[y1_pad:y2_pad, x1_pad:x2_pad]
        
        return cropped
        
    except Exception as e:
        logger.warning(f"Crop with padding failed: {e}")
        return image[y1:y2, x1:x2] if bbox else image


def normalize_to_recognizer_height(image: np.ndarray, target_height: int = 48, 
                                  preserve_aspect: bool = True) -> np.ndarray:
    """
    Normalize image height for recognizer input with aspect ratio preservation.
    
    Args:
        image: Input image
        target_height: Target height in pixels (typical: 32, 48, 64)
        preserve_aspect: Whether to preserve aspect ratio with padding
        
    Returns:
        Normalized image
    """
    try:
        h, w = image.shape[:2]
        
        if h == target_height:
            return image
        
        if preserve_aspect:
            # Calculate scale factor based on height
            scale = target_height / h
            new_w = int(w * scale)
            
            # Resize maintaining aspect ratio
            resized = cv2.resize(image, (new_w, target_height), interpolation=cv2.INTER_CUBIC)
            
            # If we need consistent width, we could add horizontal padding here
            return resized
        else:
            # Direct resize (may distort aspect ratio)
            resized = cv2.resize(image, (w, target_height), interpolation=cv2.INTER_CUBIC)
            return resized
            
    except Exception as e:
        logger.warning(f"Height normalization failed: {e}")
        return image


def quality_preprocessing_pipeline(img: np.ndarray, config: dict = None) -> dict:
    """
    Quality-focused preprocessing pipeline for academic PDFs.
    
    Args:
        img: Input image
        config: Configuration dict with preprocessing parameters
        
    Returns:
        Dict with processed images and metadata
    """
    config = config or {}
    preprocessing_config = config.get('preprocessing', {})
    
    logger.debug("Starting quality preprocessing pipeline")
    
    try:
        result = img.copy()
        
        # Step 1: Enhanced bilateral filtering for noise reduction
        if preprocessing_config.get('bilateral_filter', True):
            d = preprocessing_config.get('bilateral_d', 7)
            sigma_color = preprocessing_config.get('bilateral_sigma_color', 50)
            sigma_space = preprocessing_config.get('bilateral_sigma_space', 50)
            
            result = cv2.bilateralFilter(result, d, sigma_color, sigma_space)
            logger.debug(f"Applied bilateral filter: d={d}, σ_color={sigma_color}, σ_space={sigma_space}")
        
        # Step 2: Advanced CLAHE with higher quality settings
        clip_limit = preprocessing_config.get('clahe_clip_limit', 3.5)
        grid_size = preprocessing_config.get('clahe_grid_size', (8, 8))
        
        if len(result.shape) == 3:
            # Convert to LAB for better CLAHE results
            lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
            l_enhanced = clahe.apply(l_channel)
            
            # Merge back and convert to RGB
            lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
            result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
        else:
            # Grayscale CLAHE
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
            result = clahe.apply(result)
        
        logger.debug(f"Applied CLAHE: clip_limit={clip_limit}, grid_size={grid_size}")
        
        # Step 3: Generate high-quality Sauvola binary for detection assistance
        binary_for_detection = sauvola_binarization_quality(result, preprocessing_config)
        
        # Step 4: Optional gamma correction for very dark images
        if preprocessing_config.get('gamma_correction', False):
            gamma = preprocessing_config.get('gamma', 1.2)
            result_float = result.astype(np.float32) / 255.0
            result_gamma = np.power(result_float, 1.0/gamma)
            result = (result_gamma * 255.0).astype(np.uint8)
            logger.debug(f"Applied gamma correction: γ={gamma}")
        
        return {
            'enhanced': result,  # Main enhanced image for OCR
            'binary': binary_for_detection,  # Binary for detection assistance
            'original': img,  # Keep original for reference
            'metadata': {
                'bilateral_filter': preprocessing_config.get('bilateral_filter', True),
                'clahe_clip_limit': clip_limit,
                'sauvola_k': preprocessing_config.get('sauvola_k', 0.3),
                'gamma_applied': preprocessing_config.get('gamma_correction', False)
            }
        }
        
    except Exception as e:
        logger.error(f"Quality preprocessing pipeline failed: {e}")
        return {
            'enhanced': img,
            'binary': img,
            'original': img,
            'metadata': {'error': str(e)}
        }


def enhanced_clahe_preprocessing(img: np.ndarray, config: dict = None) -> np.ndarray:
    """
    Enhanced preprocessing with CLAHE, bilateral filtering, and contrast adjustment.
    
    Args:
        img: Input image
        config: Optional configuration dict
        
    Returns:
        Preprocessed image
    """
    config = config or {}
    
    try:
        result = img.copy()
        
        # Step 1: Mild bilateral filtering for denoising
        if config.get('bilateral_filter', True):
            d = config.get('bilateral_d', 5)  # Smaller for performance
            sigma_color = config.get('bilateral_sigma_color', 50)
            sigma_space = config.get('bilateral_sigma_space', 50)
            result = apply_bilateral_filter(result, d, sigma_color, sigma_space)
        
        # Step 2: Gamma correction for dark images
        if config.get('gamma_correction', True):
            gamma = config.get('gamma', 1.2)
            result = apply_gamma(result, gamma)
        
        # Step 3: CLAHE for local contrast enhancement
        if config.get('clahe', True):
            clip_limit = config.get('clahe_clip_limit', 3.0)
            grid_size = config.get('clahe_grid_size', (8, 8))
            result = clahe_rgb(result, clip_limit, grid_size)
        
        return result
        
    except Exception as e:
        logger.warning(f"Enhanced preprocessing failed: {e}")
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
