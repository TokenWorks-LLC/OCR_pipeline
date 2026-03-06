"""
OCR utilities for running PaddleOCR and Tesseract engines and merging results.
"""
import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Callable
import warnings

import cv2
import numpy as np

# Suppress PaddleOCR warnings
warnings.filterwarnings('ignore')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False

from config import (
    TESSERACT_CONFIG, TESSERACT_LANGUAGES, 
    QUICK_OCR_MAX_WORDS, QUICK_OCR_CONFIDENCE_THRESHOLD,
    CONFIDENCE_THRESHOLD, NMS_IOU_THRESHOLD, TESSERACT_CMD, TESSDATA_PREFIX,
    OCR_ENGINE, get_ocr_engine_config
)

# Optional Turkish corrections
try:
    from turkish_corrections import apply_turkish_corrections
    TURKISH_CORRECTIONS_AVAILABLE = True
except ImportError:
    TURKISH_CORRECTIONS_AVAILABLE = False
    def apply_turkish_corrections(text: str) -> str:
        """Fallback function when turkish_corrections is not available."""
        return text

# Configure Tesseract paths after import
if TESSERACT_AVAILABLE:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    os.environ['TESSDATA_PREFIX'] = TESSDATA_PREFIX

logger = logging.getLogger(__name__)


@dataclass
class Line:
    """Represents a line of text detected by OCR."""
    text: str
    conf: float
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    engine: str


# Global PaddleOCR instance (lazy initialization)
_paddle_ocr = None


def get_paddle() -> Optional[PaddleOCR]:
    """Get PaddleOCR instance with lazy initialization."""
    global _paddle_ocr
    
    if not PADDLE_AVAILABLE:
        return None
    
    if _paddle_ocr is None:
        # Set up device with modern Paddle API
        try:
            import paddle
            device = "gpu" if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0 else "cpu"
            paddle.device.set_device(device)
        except ImportError:
            device = "cpu"
        
        # Try different language configurations with modern parameters
        configs_to_try = [
            {'use_textline_orientation': True, 'lang': 'latin'},
            {'use_textline_orientation': True, 'lang': 'en'},
            {'use_textline_orientation': False, 'lang': 'latin'},  # Without angle classification
            {'lang': 'latin'},  # Minimal config
        ]
        
        for i, config in enumerate(configs_to_try):
            try:
                _paddle_ocr = PaddleOCR(**config)
                logger.info(f"PaddleOCR initialized successfully with config {i+1}: {config} on {device}")
                break
            except Exception as e:
                logger.debug(f"PaddleOCR config {i+1} failed: {e}")
                if i == len(configs_to_try) - 1:  # Last attempt
                    logger.error(f"All PaddleOCR configurations failed. Last error: {e}")
                    return None
    
    return _paddle_ocr


def bbox_from_points(points: List[List[float]]) -> Tuple[int, int, int, int]:
    """Convert PaddleOCR point format to (x, y, w, h) bbox."""
    points = np.array(points)
    x_min, y_min = points.min(axis=0).astype(int)
    x_max, y_max = points.max(axis=0).astype(int)
    return (x_min, y_min, x_max - x_min, y_max - y_min)


def ocr_paddle_lines(img: np.ndarray) -> List[Line]:
    """Run PaddleOCR and return detected lines."""
    paddle = get_paddle()
    if paddle is None:
        logger.warning("PaddleOCR not available")
        return []
    
    try:
        # Use the correct method name for newer PaddleOCR versions
        results = paddle.ocr(img)  # Remove cls parameter - it's handled by use_textline_orientation in config
        lines = []
        
        if results and len(results) > 0:
            result = results[0]  # Get first page result
            
            # Handle new PaddleOCR 3.x format (OCRResult object with dict interface)
            if hasattr(result, 'keys') and 'rec_texts' in result:
                texts = result['rec_texts']
                scores = result['rec_scores']  
                polys = result.get('rec_polys', [])
                
                for i, (text, score) in enumerate(zip(texts, scores)):
                    if text.strip() and score >= CONFIDENCE_THRESHOLD:
                        # Get bounding box from polys if available
                        if i < len(polys) and len(polys[i]) > 0:
                            poly = polys[i]
                            bbox = bbox_from_points(poly)
                        else:
                            # Fallback bbox
                            bbox = (0, 0, 100, 20)
                        
                        # Apply Turkish corrections
                        corrected_text = apply_turkish_corrections(text.strip())
                        lines.append(Line(
                            text=corrected_text,
                            conf=float(score),
                            bbox=bbox,
                            engine="paddle"
                        ))
            
            # Handle legacy PaddleOCR 2.x format (fallback)
            elif isinstance(result, list):
                for detection in result:
                    if len(detection) >= 2:
                        bbox_points, (text, conf) = detection[0], detection[1]
                        
                        if text.strip() and conf >= CONFIDENCE_THRESHOLD:
                            bbox = bbox_from_points(bbox_points)
                            # Apply Turkish corrections
                            corrected_text = apply_turkish_corrections(text.strip())
                            lines.append(Line(
                                text=corrected_text,
                                conf=float(conf),
                                bbox=bbox,
                                engine="paddle"
                            ))
        
        logger.debug(f"PaddleOCR found {len(lines)} lines")
        return lines
        
    except Exception as e:
        logger.error(f"PaddleOCR failed: {e}")
        return []


def ocr_tesseract_lines(img: np.ndarray, langs: str = None) -> List[Line]:
    """Run Tesseract OCR and return detected lines."""
    if not TESSERACT_AVAILABLE:
        logger.warning("Tesseract not available")
        return []
    
    if langs is None:
        langs = TESSERACT_LANGUAGES
    
    try:
        # Get detailed data from Tesseract
        data = pytesseract.image_to_data(
            img, 
            lang=langs, 
            config=TESSERACT_CONFIG,
            output_type=pytesseract.Output.DICT
        )
        
        lines = []
        current_line_text = []
        current_line_conf = []
        current_line_bbox = None
        current_line_num = -1
        
        # Group words into lines
        for i in range(len(data['text'])):
            word_text = data['text'][i].strip()
            word_conf = float(data['conf'][i]) / 100.0  # Convert to 0-1 range
            line_num = data['line_num'][i]
            
            if not word_text or word_conf < 0:
                continue
            
            # Start new line
            if line_num != current_line_num:
                # Finish previous line
                if current_line_text and current_line_bbox:
                    full_text = ' '.join(current_line_text)
                    avg_conf = np.mean(current_line_conf)
                    
                    if avg_conf >= CONFIDENCE_THRESHOLD:
                        # Apply Turkish corrections
                        corrected_text = apply_turkish_corrections(full_text)
                        lines.append(Line(
                            text=corrected_text,
                            conf=avg_conf,
                            bbox=current_line_bbox,
                            engine="tesseract"
                        ))
                
                # Start new line
                current_line_text = [word_text]
                current_line_conf = [word_conf]
                current_line_bbox = (
                    data['left'][i], 
                    data['top'][i],
                    data['width'][i], 
                    data['height'][i]
                )
                current_line_num = line_num
            else:
                # Add to current line
                current_line_text.append(word_text)
                current_line_conf.append(word_conf)
                
                # Expand bounding box
                if current_line_bbox:
                    x1 = min(current_line_bbox[0], data['left'][i])
                    y1 = min(current_line_bbox[1], data['top'][i])
                    x2 = max(current_line_bbox[0] + current_line_bbox[2], data['left'][i] + data['width'][i])
                    y2 = max(current_line_bbox[1] + current_line_bbox[3], data['top'][i] + data['height'][i])
                    current_line_bbox = (x1, y1, x2 - x1, y2 - y1)
        
        # Handle last line
        if current_line_text and current_line_bbox:
            full_text = ' '.join(current_line_text)
            avg_conf = np.mean(current_line_conf)
            
            if avg_conf >= CONFIDENCE_THRESHOLD:
                # Apply Turkish corrections  
                corrected_text = apply_turkish_corrections(full_text)
                lines.append(Line(
                    text=corrected_text,
                    conf=avg_conf,
                    bbox=current_line_bbox,
                    engine="tesseract"
                ))
        
        logger.debug(f"Tesseract found {len(lines)} lines")
        return lines
        
    except Exception as e:
        logger.error(f"Tesseract failed: {e}")
        return []


def calculate_iou(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
    """Calculate Intersection over Union (IoU) of two bounding boxes."""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Calculate intersection
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    
    intersection = (xi2 - xi1) * (yi2 - yi1)
    union = w1 * h1 + w2 * h2 - intersection
    
    return intersection / union if union > 0 else 0.0


def merge_lines_nms(lines: List[Line], iou_thresh: float = None) -> List[Line]:
    """Merge overlapping lines using Non-Maximum Suppression."""
    if iou_thresh is None:
        iou_thresh = NMS_IOU_THRESHOLD
    
    if not lines:
        return []
    
    # Sort by confidence (descending)
    lines = sorted(lines, key=lambda l: l.conf, reverse=True)
    merged = []
    
    while lines:
        # Take the highest confidence line
        current = lines.pop(0)
        merged.append(current)
        
        # Remove overlapping lines with lower confidence
        remaining = []
        for line in lines:
            iou = calculate_iou(current.bbox, line.bbox)
            if iou < iou_thresh:
                remaining.append(line)
            else:
                logger.debug(f"Merged overlapping line (IoU: {iou:.3f})")
        
        lines = remaining
    
    return merged


def quick_ocr_conf(img: np.ndarray, max_words: int = None) -> float:
    """Get a quick OCR confidence score for orientation/inversion decisions."""
    if max_words is None:
        max_words = QUICK_OCR_MAX_WORDS
    
    try:
        # Try Tesseract first (faster)
        if TESSERACT_AVAILABLE:
            try:
                data = pytesseract.image_to_data(
                    img,
                    config='--oem 1 --psm 6',
                    output_type=pytesseract.Output.DICT
                )
                
                confidences = [
                    float(conf) / 100.0 for conf in data['conf']
                    if conf != -1 and data['text'][i].strip()
                    for i in range(len(data['conf']))
                ][:max_words]
                
                if confidences:
                    return np.mean(confidences)
            except Exception as e:
                logger.debug(f"Quick Tesseract failed: {e}")
        
        # Fallback to PaddleOCR
        paddle = get_paddle()
        if paddle:
            results = paddle.ocr(img)  # Remove cls=False parameter
            if results and results[0]:
                confidences = [
                    detection[1][1] for detection in results[0]
                    if len(detection) >= 2 and detection[1][0].strip()
                ][:max_words]
                
                if confidences:
                    return np.mean(confidences)
        
        return 0.0
        
    except Exception as e:
        logger.debug(f"Quick OCR failed: {e}")
        return 0.0


def ocr_ensemble(img: np.ndarray) -> List[Line]:
    """Run both OCR engines and merge results."""
    all_lines = []
    
    # Run PaddleOCR
    paddle_lines = ocr_paddle_lines(img)
    all_lines.extend(paddle_lines)
    
    # Run Tesseract
    tesseract_lines = ocr_tesseract_lines(img)
    all_lines.extend(tesseract_lines)
    
    # Merge overlapping detections
    merged_lines = merge_lines_nms(all_lines)
    
    logger.info(f"OCR ensemble: {len(paddle_lines)} paddle + {len(tesseract_lines)} tesseract → {len(merged_lines)} merged")
    
    return merged_lines


# Enhanced OCR functions supporting multiple engines
def ocr_with_engine(img: np.ndarray, engine: str = None, profile: str = None, device: str = None) -> List[Line]:
    """
    Run OCR using specified engine with fallback support.
    
    Args:
        img: Input image
        engine: OCR engine name (defaults to config)
        profile: Performance profile (defaults to config)  
        device: Device to use (defaults to config)
        
    Returns:
        List of detected text lines
    """
    engine = engine or OCR_ENGINE['engine']
    profile = profile or OCR_ENGINE['profile']
    device = device or OCR_ENGINE['device']
    
    # Get engine configuration
    engine_config = get_ocr_engine_config(engine, profile)
    engine_config['device'] = device
    
    # Try primary engine
    try:
        lines = _run_single_engine(img, engine, engine_config)
        if lines:
            logger.info(f"OCR successful with {engine}: {len(lines)} lines detected")
            return lines
        else:
            logger.warning(f"No lines detected with {engine}, trying fallback")
    except Exception as e:
        logger.error(f"OCR failed with {engine}: {e}")
    
    # Try fallback engines
    fallback_engines = OCR_ENGINE.get('fallback_engines', ['paddle'])
    for fallback_engine in fallback_engines:
        if fallback_engine != engine:  # Don't retry the same engine
            try:
                logger.info(f"Trying fallback engine: {fallback_engine}")
                fallback_config = get_ocr_engine_config(fallback_engine, profile)
                fallback_config['device'] = device
                lines = _run_single_engine(img, fallback_engine, fallback_config)
                if lines:
                    logger.info(f"Fallback successful with {fallback_engine}: {len(lines)} lines detected")
                    return lines
            except Exception as e:
                logger.error(f"Fallback engine {fallback_engine} failed: {e}")
    
    # Final fallback to original ensemble if all else fails
    logger.warning("All configured engines failed, using original ensemble")
    return ocr_ensemble(img)


def _run_single_engine(img: np.ndarray, engine: str, config: dict) -> List[Line]:
    """
    Run OCR using a single engine.
    
    Args:
        img: Input image
        engine: Engine name
        config: Engine configuration
        
    Returns:
        List of detected text lines
    """
    if engine == 'paddle':
        return ocr_paddle_lines(img)
    elif engine == 'tesseract':
        return ocr_tesseract_lines(img)
    elif engine in ['doctr', 'mmocr', 'kraken']:
        return _run_new_engine(img, engine, config)
    else:
        raise ValueError(f"Unknown OCR engine: {engine}")


def _run_new_engine(img: np.ndarray, engine: str, config: dict) -> List[Line]:
    """
    Run OCR using new engine implementations (docTR, MMOCR, Kraken).
    
    Args:
        img: Input image (OpenCV format)
        engine: Engine name
        config: Engine configuration
        
    Returns:
        List of detected text lines
    """
    try:
        from engines import create_engine
        
        # Create engine instance
        engine_instance = create_engine(engine, config.get('profile', 'balanced'), config.get('device', 'auto'))
        
        # Convert OpenCV image to PIL Image
        from PIL import Image
        if len(img.shape) == 3:
            # BGR to RGB conversion for OpenCV images
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(img_rgb)
        else:
            pil_image = Image.fromarray(img)
        
        # Run inference
        spans = engine_instance.infer_page(pil_image)
        
        # Convert spans to Line objects
        lines = []
        for span in spans:
            # Convert bbox from [x1, y1, x2, y2] to (x, y, w, h)
            x1, y1, x2, y2 = span['bbox']
            bbox = (x1, y1, x2 - x1, y2 - y1)
            
            line = Line(
                text=span['text'],
                conf=span['conf'],
                bbox=bbox,
                engine=engine
            )
            lines.append(line)
        
        logger.debug(f"{engine} extracted {len(lines)} lines")
        return lines
        
    except ImportError as e:
        logger.error(f"Engine {engine} not available: {e}")
        return []
    except Exception as e:
        logger.error(f"Engine {engine} failed: {e}")
        return []


def get_engine_info() -> dict:
    """
    Get information about available OCR engines.
    
    Returns:
        Dictionary with engine availability and versions
    """
    try:
        from engines import check_engine_dependencies
        return check_engine_dependencies()
    except ImportError:
        # Fallback to basic info
        info = {}
        
        # Check PaddleOCR
        if PADDLE_AVAILABLE:
            try:
                from paddleocr import PaddleOCR
                info['paddle'] = {'available': True, 'version': 'unknown'}
            except:
                info['paddle'] = {'available': False, 'error': 'import failed'}
        else:
            info['paddle'] = {'available': False, 'error': 'not installed'}
            
        # Check Tesseract
        if TESSERACT_AVAILABLE:
            try:
                import pytesseract
                version = pytesseract.get_tesseract_version()
                info['tesseract'] = {'available': True, 'version': str(version)}
            except:
                info['tesseract'] = {'available': True, 'version': 'unknown'}
        else:
            info['tesseract'] = {'available': False, 'error': 'not installed'}
        
        return info
