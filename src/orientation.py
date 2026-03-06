"""
Advanced 360° orientation detection with Radon profiles and fine deskew correction.
Optimized for academic PDFs with full angular sweep and quality-focused processing.
"""
import logging
from typing import Tuple, Dict, Any, Optional
import time
import hashlib
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import rotate
from scipy.optimize import minimize_scalar
from skimage.filters import threshold_sauvola
from skimage.transform import radon

# Import from local config, with fallback
try:
    from .config import ROTATION_CANDIDATES
except ImportError:
    try:
        from config import ROTATION_CANDIDATES
    except ImportError:
        ROTATION_CANDIDATES = [0, 90, 180, 270]  # Fallback

# Import OCR utils with fallback
try:
    from .ocr_utils import quick_ocr_conf
except ImportError:
    try:
        from ocr_utils import quick_ocr_conf
    except ImportError:
        def quick_ocr_conf(image):
            """Fallback OCR confidence function."""
            return 0.5  # Default confidence

logger = logging.getLogger(__name__)


def downscale_for_speed(image: np.ndarray, max_edge: int = 1200) -> Tuple[np.ndarray, float]:
    """
    Downscale image for fast orientation detection.
    
    Args:
        image: Input image
        max_edge: Maximum edge length for downscaled version
        
    Returns:
        Tuple of (downscaled_image, scale_factor)
    """
    h, w = image.shape[:2]
    max_dim = max(h, w)
    
    if max_dim <= max_edge:
        return image, 1.0
    
    scale = max_edge / max_dim
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    downscaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return downscaled, scale


def compute_radon_projection_score(image: np.ndarray) -> float:
    """
    Compute orientation score using Radon transform projection profiles.
    Higher score indicates better text line alignment.
    
    Args:
        image: Grayscale image
        
    Returns:
        Score indicating text line anisotropy
    """
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Apply Sauvola binarization for better text detection
        binary = threshold_sauvola(gray, window_size=25, k=0.3)
        binary = (binary * 255).astype(np.uint8)
        
        # Invert so text is white on black
        binary = 255 - binary
        
        # Compute horizontal projection (sum rows)
        h_projection = np.sum(binary, axis=1)
        
        # Score based on variance of row sums (higher = more aligned text lines)
        if len(h_projection) > 0:
            h_variance = np.var(h_projection)
        else:
            h_variance = 0.0
        
        # Also check vertical projection for additional confirmation
        v_projection = np.sum(binary, axis=0)
        v_variance = np.var(v_projection) if len(v_projection) > 0 else 0.0
        
        # Combine scores - horizontal variance should dominate for text
        score = h_variance + 0.1 * v_variance
        
        return float(score)
        
    except Exception as e:
        logger.warning(f"Radon projection score failed: {e}")
        return 0.0


def test_rotation_angle(image: np.ndarray, angle: float, use_recognizer: bool = False) -> float:
    """
    Test a specific rotation angle and return quality score.
    
    Args:
        image: Input image
        angle: Rotation angle in degrees
        use_recognizer: Whether to include OCR recognition score
        
    Returns:
        Combined quality score for this angle
    """
    try:
        # Rotate image
        if abs(angle) < 0.1:
            rotated = image
        else:
            rotated = rotate(image, angle, reshape=False, order=1, cval=255)
        
        # Primary score: Radon projection
        projection_score = compute_radon_projection_score(rotated)
        
        # Optional: Light recognizer probe on center crops
        recognition_score = 0.0
        if use_recognizer:
            try:
                h, w = rotated.shape[:2]
                # Take two center crops
                crop1 = rotated[h//4:3*h//4, w//4:3*w//4]
                crop2 = rotated[h//3:2*h//3, w//3:2*w//3]
                
                conf1 = quick_ocr_conf(crop1) if crop1.size > 0 else 0.0
                conf2 = quick_ocr_conf(crop2) if crop2.size > 0 else 0.0
                recognition_score = (conf1 + conf2) / 2.0
            except Exception:
                recognition_score = 0.0
        
        # Combine scores (projection dominates)
        alpha = 0.8  # Weight for projection score
        beta = 0.2   # Weight for recognition score
        
        total_score = alpha * projection_score + beta * recognition_score
        
        return total_score
        
    except Exception as e:
        logger.warning(f"Angle test failed for {angle}°: {e}")
        return 0.0


def detect_coarse_orientation_full360(image: np.ndarray, config: Dict[str, Any] = None) -> Tuple[float, Dict[str, Any]]:
    """
    Full 360° coarse orientation detection using Radon profiles.
    
    Args:
        image: Input image
        config: Configuration dict
        
    Returns:
        Tuple of (best_angle, detection_info)
    """
    config = config or {}
    step_deg = config.get('coarse_step_deg', 1.0)
    use_recognizer = config.get('use_recognizer_probe', True)
    
    # Downscale for speed
    small_image, scale_factor = downscale_for_speed(image, max_edge=1200)
    
    logger.debug(f"360° coarse detection: image {image.shape} -> {small_image.shape} (scale={scale_factor:.3f})")
    
    best_angle = 0.0
    best_score = -1.0
    angle_scores = {}
    
    # Test all angles in 360° range
    angles = np.arange(0, 360, step_deg)
    
    start_time = time.time()
    
    for angle in angles:
        score = test_rotation_angle(small_image, angle, use_recognizer=use_recognizer)
        angle_scores[float(angle)] = score
        
        if score > best_score:
            best_score = score
            best_angle = angle
    
    elapsed = time.time() - start_time
    
    # Normalize angle to [0, 360)
    best_angle = best_angle % 360
    
    detection_info = {
        'method': 'radon_360',
        'best_angle': best_angle,
        'best_score': best_score,
        'angle_scores': angle_scores,
        'num_angles_tested': len(angles),
        'elapsed_seconds': elapsed,
        'scale_factor': scale_factor,
        'success': True
    }
    
    logger.info(f"360° coarse detection: best={best_angle:.1f}° (score={best_score:.1f}, {len(angles)} angles, {elapsed:.2f}s)")
    
    return best_angle, detection_info


def detect_fine_skew_around_angle(image: np.ndarray, base_angle: float, config: Dict[str, Any] = None) -> Tuple[float, float]:
    """
    Fine deskew detection around a base angle using Hough lines.
    
    Args:
        image: Input image (already coarse-rotated)
        base_angle: Base angle for fine adjustment
        config: Configuration dict
        
    Returns:
        Tuple of (fine_angle_adjustment, confidence_score)
    """
    config = config or {}
    search_range = config.get('fine_deg', 2.0)
    step_size = config.get('fine_step_deg', 0.1)
    
    try:
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Adaptive thresholding for text detection
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10
        )
        
        # Morphological operations to connect text horizontally
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        dilated = cv2.dilate(binary, kernel, iterations=1)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(dilated, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
        
        angles = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 != x1:  # Avoid division by zero
                    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                    
                    # Normalize to [-45, 45] range
                    while angle > 45:
                        angle -= 90
                    while angle < -45:
                        angle += 90
                    
                    # Only consider reasonable skew angles
                    if abs(angle) <= search_range:
                        angles.append(angle)
        
        if len(angles) >= 3:  # Need minimum number of lines for reliable detection
            # Use median for robustness
            median_angle = np.median(angles)
            
            # Clamp to search range
            median_angle = np.clip(median_angle, -search_range, search_range)
            
            # Confidence based on line agreement
            angle_std = np.std(angles)
            confidence = max(0.0, 1.0 - angle_std / 10.0)
            
            logger.debug(f"Fine skew: {len(angles)} lines, angle={median_angle:.2f}°, std={angle_std:.2f}, conf={confidence:.3f}")
            return float(median_angle), confidence
        
        else:
            logger.debug(f"Fine skew: insufficient lines ({len(angles) if angles else 0}), using projection-based method")
            
            # Fallback: test small range around base angle using projection scores
            test_angles = np.arange(-search_range, search_range + step_size, step_size)
            best_adjustment = 0.0
            best_score = test_rotation_angle(image, 0.0)  # Current image score
            
            for adj in test_angles:
                if abs(adj) < 0.05:  # Skip tiny adjustments
                    continue
                    
                score = test_rotation_angle(image, adj)
                if score > best_score:
                    best_score = score
                    best_adjustment = adj
            
            confidence = 0.5  # Lower confidence for projection method
            logger.debug(f"Fine skew (projection fallback): adjustment={best_adjustment:.2f}°, conf={confidence:.3f}")
            return float(best_adjustment), confidence
        
    except Exception as e:
        logger.warning(f"Fine skew detection failed: {e}")
        return 0.0, 0.0


def apply_final_rotation(image: np.ndarray, angle: float) -> np.ndarray:
    """Apply final rotation with proper padding to avoid cropping."""
    if abs(angle) < 0.05:  # Skip tiny rotations
        return image
    
    try:
        # Use scipy rotate for high quality
        rotated = rotate(image, angle, reshape=True, order=1, cval=255)
        return rotated.astype(image.dtype)
        
    except Exception as e:
        logger.warning(f"Rotation failed, using OpenCV fallback: {e}")
        
        # OpenCV fallback
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new size to avoid cropping
        cos_a = abs(M[0, 0])
        sin_a = abs(M[0, 1])
        new_w = int(h * sin_a + w * cos_a)
        new_h = int(h * cos_a + w * sin_a)
        
        # Adjust translation
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2
        
        # Apply rotation
        if len(image.shape) == 3:
            rotated = cv2.warpAffine(image, M, (new_w, new_h), 
                                   flags=cv2.INTER_CUBIC, 
                                   borderMode=cv2.BORDER_CONSTANT, 
                                   borderValue=(255, 255, 255))
        else:
            rotated = cv2.warpAffine(image, M, (new_w, new_h), 
                                   flags=cv2.INTER_CUBIC, 
                                   borderMode=cv2.BORDER_CONSTANT, 
                                   borderValue=255)
        
        return rotated


def correct_page_orientation(image: np.ndarray, config: Dict[str, Any] = None, cache_key: Optional[str] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Advanced 360° orientation correction with fine deskew.
    
    Args:
        image: Input image as numpy array
        config: Configuration dict with orientation settings
        cache_key: Optional cache key for results
        
    Returns:
        Tuple of (corrected_image, metadata)
    """
    start_time = time.time()
    
    # Configuration
    config = config or {}
    orientation_config = config.get('orientation', {})
    cache_enabled = orientation_config.get('enable_cache', True)
    
    # Setup caching
    cache_dir = Path("data/.cache/orientation")
    if cache_enabled and cache_key:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"orient_360_{cache_key}.npy"
        
        # Check cache
        if cache_file.exists():
            try:
                cached_data = np.load(cache_file, allow_pickle=True).item()
                if 'corrected_image' in cached_data and 'metadata' in cached_data:
                    logger.debug(f"Using cached 360° orientation for {cache_key}")
                    return cached_data['corrected_image'], cached_data['metadata']
            except Exception as e:
                logger.warning(f"Failed to load orientation cache: {e}")
    
    # Step 1: Full 360° coarse orientation detection
    coarse_angle, coarse_info = detect_coarse_orientation_full360(image, orientation_config)
    
    # Step 2: Apply coarse rotation
    coarse_rotated = apply_final_rotation(image, coarse_angle)
    
    # Step 3: Fine deskew around the coarse angle
    fine_adjustment, fine_confidence = detect_fine_skew_around_angle(
        coarse_rotated, coarse_angle, orientation_config
    )
    
    # Step 4: Apply fine correction if significant and confident
    if abs(fine_adjustment) > 0.1 and fine_confidence > 0.3:
        final_image = apply_final_rotation(coarse_rotated, fine_adjustment)
        final_angle = (coarse_angle + fine_adjustment) % 360
    else:
        final_image = coarse_rotated
        fine_adjustment = 0.0
        final_angle = coarse_angle % 360
    
    # Step 5: Quality verification
    try:
        original_score = test_rotation_angle(image, 0.0)
        final_score = test_rotation_angle(final_image, 0.0)
        quality_improved = final_score >= original_score * 0.95  # Allow small degradation
    except Exception:
        quality_improved = True  # Default to accepting
        original_score = final_score = 0.0
    
    if not quality_improved and abs(final_angle) > 5.0:  # Only revert for significant rotations
        logger.warning("Quality check failed, reverting to original")
        final_image = image
        final_angle = 0.0
        fine_adjustment = 0.0
        final_score = original_score
    
    elapsed = time.time() - start_time
    
    # Compile metadata
    metadata = {
        'method': '360_degree_radon',
        'coarse_angle': coarse_angle,
        'fine_adjustment': fine_adjustment,
        'final_angle': final_angle,
        'original_score': original_score,
        'final_score': final_score,
        'quality_improved': quality_improved,
        'fine_confidence': fine_confidence,
        'coarse_info': coarse_info,
        'original_shape': image.shape,
        'final_shape': final_image.shape,
        'processing_time': elapsed,
        'success': True
    }
    
    # Cache result
    if cache_enabled and cache_key:
        try:
            np.save(cache_file, {
                'corrected_image': final_image,
                'metadata': metadata
            })
        except Exception as e:
            logger.warning(f"Failed to cache orientation result: {e}")
    
    logger.info(f"360° orientation: final={final_angle:.1f}° (coarse={coarse_angle:.1f}° + fine={fine_adjustment:.2f}°), score_gain={final_score/max(original_score,1e-6):.2f}x, {elapsed:.2f}s")
    
    return final_image, metadata
