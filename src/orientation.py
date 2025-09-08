"""
Robust orientation detection using Tesseract OSD and fallback methods.
"""
import logging
from typing import Tuple, Dict, Any
import subprocess
import tempfile
import os

import cv2
import numpy as np
import pytesseract

from config import ROTATION_CANDIDATES
from ocr_utils import quick_ocr_conf

logger = logging.getLogger(__name__)


def detect_orientation_tesseract_osd(img: np.ndarray) -> Dict[str, Any]:
    """
    Use Tesseract OSD (Orientation and Script Detection) to detect orientation.
    
    Args:
        img: Input image as numpy array
        
    Returns:
        Dict with orientation info or empty dict if failed
    """
    try:
        # Save image to temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            cv2.imwrite(tmp.name, img)
            tmp_path = tmp.name
        
        try:
            # Run Tesseract OSD
            osd_result = pytesseract.image_to_osd(tmp_path, output_type=pytesseract.Output.DICT)
            
            rotation = osd_result.get('rotate', 0)
            confidence = osd_result.get('orientation_conf', 0)
            
            logger.debug(f"Tesseract OSD: rotate={rotation}°, confidence={confidence}")
            
            return {
                'method': 'tesseract_osd',
                'rotation': rotation,
                'confidence': confidence,
                'success': True
            }
            
        except pytesseract.TesseractError as e:
            logger.warning(f"Tesseract OSD failed: {e}")
            return {'success': False, 'error': str(e)}
            
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except:
                pass
                
    except Exception as e:
        logger.warning(f"OSD detection failed: {e}")
        return {'success': False, 'error': str(e)}


def detect_orientation_ocr_score(img: np.ndarray, candidates: list = None) -> Dict[str, Any]:
    """
    Fallback orientation detection using OCR confidence scores.
    
    Args:
        img: Input image
        candidates: List of rotation angles to try
        
    Returns:
        Dict with best orientation info
    """
    if candidates is None:
        candidates = ROTATION_CANDIDATES
    
    best_angle = 0
    best_conf = 0.0
    scores = {}
    
    for angle in candidates:
        # Rotate image
        if angle == 0:
            rotated = img
        elif angle == 90:
            rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            rotated = cv2.rotate(img, cv2.ROTATE_180)
        elif angle == 270:
            rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            # Custom angle rotation
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img, matrix, (w, h))
        
        # Quick OCR confidence check
        conf = quick_ocr_conf(rotated)
        scores[angle] = conf
        
        if conf > best_conf:
            best_conf = conf
            best_angle = angle
    
    logger.debug(f"OCR score orientation: angles={scores}, best={best_angle}° (conf={best_conf:.3f})")
    
    return {
        'method': 'ocr_score',
        'rotation': best_angle,
        'confidence': best_conf,
        'scores': scores,
        'success': True
    }


def detect_orientation(img: np.ndarray) -> Tuple[int, Dict[str, Any]]:
    """
    Detect image orientation with Tesseract OSD fallback to OCR scoring.
    
    Args:
        img: Input image
        
    Returns:
        Tuple of (rotation_angle, detection_info)
    """
    # Try Tesseract OSD first
    osd_result = detect_orientation_tesseract_osd(img)
    
    if osd_result.get('success') and osd_result.get('confidence', 0) > 3.0:
        # OSD succeeded with good confidence
        rotation = osd_result['rotation']
        # Convert to our standard angles
        if rotation not in ROTATION_CANDIDATES:
            rotation = min(ROTATION_CANDIDATES, key=lambda x: abs(x - rotation))
        
        logger.info(f"Using Tesseract OSD: {rotation}° (confidence={osd_result.get('confidence')})")
        return rotation, osd_result
    
    # Fallback to OCR scoring
    logger.info("Tesseract OSD failed or low confidence, using OCR score method")
    score_result = detect_orientation_ocr_score(img)
    
    return score_result['rotation'], score_result


def apply_rotation(img: np.ndarray, angle: int) -> np.ndarray:
    """Apply rotation to image."""
    if angle == 0:
        return img
    elif angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        # Custom angle rotation
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(img, matrix, (w, h))
