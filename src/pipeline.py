"""
Main image processing pipeline that coordinates preprocessing, OCR, and extraction.
"""
import logging
from typing import List, Tuple, Optional

import cv2
import numpy as np

from config import ROTATION_CANDIDATES
from preprocess import preprocess_pipeline
from ocr_utils import quick_ocr_conf, ocr_ensemble
from lang_and_extract import extract_translations, detect_lang
from llama_correction import correct_ocr_lines

logger = logging.getLogger(__name__)


def try_rotations(img: np.ndarray, candidates: List[int] = None) -> Tuple[np.ndarray, int]:
    """
    Try different rotation angles and return the best orientation based on OCR confidence.
    
    Args:
        img: Input image
        candidates: List of rotation angles to try (default: from config)
    
    Returns:
        Tuple of (rotated_image, best_angle)
    """
    if candidates is None:
        candidates = ROTATION_CANDIDATES
    
    best_img = img.copy()
    best_angle = 0
    best_conf = 0.0
    
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
            rotated = cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
        
        # Get OCR confidence for this orientation
        try:
            conf = quick_ocr_conf(rotated)
            logger.debug(f"Rotation {angle}°: confidence = {conf:.3f}")
            
            if conf > best_conf:
                best_conf = conf
                best_angle = angle
                best_img = rotated.copy()
                
        except Exception as e:
            logger.warning(f"Failed to evaluate rotation {angle}°: {e}")
    
    if best_angle != 0:
        logger.info(f"Best orientation: {best_angle}° (confidence: {best_conf:.3f})")
    
    return best_img, best_angle


def preprocess_image_pipeline(img: np.ndarray) -> np.ndarray:
    """
    Complete preprocessing pipeline: rotation → deskew → fade rescue → invert-if-better.
    
    Args:
        img: Input image
    
    Returns:
        Preprocessed image
    """
    logger.info("Starting image preprocessing pipeline")
    
    # Step 1: Find best rotation
    img, rotation_angle = try_rotations(img)
    
    # Step 2: Apply preprocessing (deskew, fade rescue, inversion check)
    img = preprocess_pipeline(img, quick_ocr_conf)
    
    logger.info("Image preprocessing pipeline complete")
    return img


def process_image(img: np.ndarray, extract_all_text: bool = True) -> List[dict]:
    """
    Complete pipeline: preprocess image, run OCR ensemble, extract text.
    
    Args:
        img: Input image (BGR format from cv2.imread)
        extract_all_text: If True, extract all OCR'd text; if False, only extract translation patterns
    
    Returns:
        List of text dictionaries
    """
    logger.info("Starting image processing pipeline")
    
    try:
        # Step 1: Preprocess image
        preprocessed_img = preprocess_image_pipeline(img)
        
        # Step 2: Run OCR ensemble
        ocr_lines = ocr_ensemble(preprocessed_img)
        logger.info(f"OCR found {len(ocr_lines)} text lines")
        
        # Step 2.5: Apply LLM correction if available
        # Try to detect dominant language for better correction
        all_text = ' '.join([line.text for line in ocr_lines])
        language_hint = detect_lang(all_text) if len(all_text.strip()) > 20 else None
        if language_hint and language_hint not in ['tr', 'de', 'fr', 'en', 'it']:
            language_hint = None  # Only use hint if it's a target language
        
        corrected_lines, correction_stats = correct_ocr_lines(ocr_lines, language_hint)
        if correction_stats.get('available', False):
            logger.info(f"LLM correction applied (lang={language_hint}): {correction_stats.get('lines_changed', 0)}/{correction_stats.get('lines_processed', 0)} lines changed")
            ocr_lines = corrected_lines
        else:
            logger.debug("LLM correction not available or disabled")
        
        if not ocr_lines:
            logger.warning("No text detected by OCR")
            return []
        
        # Step 3: Extract text (either all text or just translations)
        if extract_all_text:
            # Extract all OCR'd text
            text_entries = []
            for i, line in enumerate(ocr_lines):
                try:
                    # Convert numpy types to Python types for JSON serialization
                    bbox = getattr(line, 'bbox', (0, 0, 0, 0))
                    if hasattr(bbox, '__len__') and len(bbox) >= 4:
                        bbox = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                    else:
                        bbox = (0, 0, 0, 0)
                    
                    text_entries.append({
                        'lang': 'unknown',  # We don't know the language of general text
                        'text': str(getattr(line, 'text', '')),
                        'confidence': float(getattr(line, 'conf', 0.0)),  # Convert to Python float
                        'bbox': bbox,
                        'engine': str(getattr(line, 'engine', 'unknown')),
                        'line_number': i + 1
                    })
                except Exception as e:
                    logger.warning(f"Error processing OCR line {i+1}: {e}. Line object: {line}")
                    # Add a fallback entry
                    text_entries.append({
                        'lang': 'unknown',
                        'text': 'Error extracting text',
                        'confidence': 0.0,
                        'bbox': (0, 0, 0, 0),
                        'engine': 'error',
                        'line_number': i + 1
                    })
            logger.info(f"Extracted {len(text_entries)} text lines")
            return text_entries
        else:
            # Extract only translation patterns (old behavior)
            translations = extract_translations(ocr_lines)
            logger.info(f"Extracted {len(translations)} translations")
            return translations
        
        return translations
        
    except Exception as e:
        logger.error(f"Image processing pipeline failed: {e}")
        return []


def process_image_with_metadata(img: np.ndarray, page_id: str = None) -> dict:
    """
    Process image and return results with metadata.
    
    Args:
        img: Input image
        page_id: Optional page identifier
    
    Returns:
        Dictionary with processing results and metadata
    """
    logger.info(f"Processing image: {page_id or 'unnamed'}")
    
    # Process image
    translations = process_image(img)
    
    # Compile metadata
    metadata = {
        'page_id': page_id,
        'image_shape': img.shape,
        'total_translations': len(translations),
        'languages_found': list(set(t['lang'] for t in translations)),
        'engines_used': list(set(t['engine'] for t in translations)),
        'avg_confidence': np.mean([t['conf'] for t in translations]) if translations else 0.0,
    }
    
    return {
        'translations': translations,
        'metadata': metadata
    }


def batch_process_images(images: List[np.ndarray], page_ids: List[str] = None) -> List[dict]:
    """
    Process multiple images in batch.
    
    Args:
        images: List of input images
        page_ids: Optional list of page identifiers
    
    Returns:
        List of processing results for each image
    """
    if page_ids is None:
        page_ids = [f"page_{i:03d}" for i in range(len(images))]
    
    if len(images) != len(page_ids):
        raise ValueError("Number of images must match number of page IDs")
    
    results = []
    
    for i, (img, page_id) in enumerate(zip(images, page_ids)):
        logger.info(f"Processing batch item {i+1}/{len(images)}: {page_id}")
        
        try:
            result = process_image_with_metadata(img, page_id)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to process {page_id}: {e}")
            results.append({
                'translations': [],
                'metadata': {
                    'page_id': page_id,
                    'error': str(e)
                }
            })
    
    return results
