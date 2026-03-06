"""
OCR Engine Interface for Multiple OCR Backend Support

This module defines the base interface that all OCR engines must implement
to ensure consistent output format and interoperability within the pipeline.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from PIL import Image
import numpy as np


class OcrEngine(ABC):
    """
    Abstract base class for OCR engines.
    
    All OCR engine implementations must inherit from this class and implement
    the infer_page method to return normalized text spans with consistent schema.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the OCR engine with configuration.
        
        Args:
            config: Engine-specific configuration dictionary
        """
        self.config = config or {}
        self.device = self._detect_device()
        
    def _detect_device(self) -> str:
        """Detect whether GPU is available and should be used."""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"
    
    @abstractmethod
    def infer_page(self, image: Union[Image.Image, np.ndarray]) -> List[Dict[str, Any]]:
        """
        Run OCR inference on a single page image.
        
        Args:
            image: Input image as PIL Image or numpy array
            
        Returns:
            List of text spans, each with the following normalized schema:
            {
                "text": str,                    # Recognized text content
                "bbox": [x1, y1, x2, y2],     # Bounding box in page coordinates  
                "conf": float,                 # Confidence score (0.0 - 1.0)
                "level": str,                  # "word" or "line" 
                "page_width": int,             # Original page width for bbox normalization
                "page_height": int             # Original page height for bbox normalization
            }
            
        Notes:
            - Bounding boxes should be in absolute page coordinates (pixels)
            - Confidence scores should be normalized to [0.0, 1.0] range
            - Text should be cleaned (no excessive whitespace, normalized)
            - Level indicates granularity: "word" for word-level, "line" for line-level
        """
        raise NotImplementedError("Subclasses must implement infer_page method")
    
    def preprocess_image(self, image: Union[Image.Image, np.ndarray]) -> Union[Image.Image, np.ndarray]:
        """
        Optional preprocessing of input image before OCR inference.
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image in same format as input
        """
        return image
    
    def postprocess_spans(self, spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Optional postprocessing of OCR spans after inference.
        
        Args:
            spans: Raw OCR spans from inference
            
        Returns:
            Processed spans with same schema
        """
        return spans
    
    def get_engine_info(self) -> Dict[str, Any]:
        """
        Get information about the engine for logging and debugging.
        
        Returns:
            Dictionary with engine metadata:
            {
                "name": str,           # Engine name (e.g., "doctr", "mmocr")
                "version": str,        # Engine version
                "device": str,         # Device being used ("cpu" or "cuda")
                "models": Dict,        # Model information
                "config": Dict         # Current configuration
            }
        """
        return {
            "name": self.__class__.__name__.replace("Engine", "").lower(),
            "version": "unknown",
            "device": self.device,
            "models": {},
            "config": self.config
        }


class MockOcrEngine(OcrEngine):
    """
    Mock OCR engine for testing and development.
    
    Returns dummy text spans for any input image.
    """
    
    def infer_page(self, image: Union[Image.Image, np.ndarray]) -> List[Dict[str, Any]]:
        """Return mock OCR results for testing."""
        # Get image dimensions
        if isinstance(image, Image.Image):
            width, height = image.size
        else:
            height, width = image.shape[:2]
            
        # Return mock text spans
        return [
            {
                "text": "Mock OCR Result Line 1",
                "bbox": [50, 50, 300, 80],
                "conf": 0.95,
                "level": "line",
                "page_width": width,
                "page_height": height
            },
            {
                "text": "Mock OCR Result Line 2", 
                "bbox": [50, 100, 280, 130],
                "conf": 0.87,
                "level": "line",
                "page_width": width,
                "page_height": height
            }
        ]
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Return mock engine information."""
        return {
            "name": "mock",
            "version": "1.0.0",
            "device": self.device,
            "models": {"detector": "mock_detector", "recognizer": "mock_recognizer"},
            "config": self.config
        }


def normalize_bbox(bbox: List[float], page_width: int, page_height: int) -> List[int]:
    """
    Normalize bounding box coordinates to absolute pixel values.
    
    Args:
        bbox: Bounding box as [x1, y1, x2, y2] (may be normalized or absolute)
        page_width: Page width in pixels
        page_height: Page height in pixels
        
    Returns:
        Bounding box as [x1, y1, x2, y2] in absolute pixel coordinates
    """
    x1, y1, x2, y2 = bbox
    
    # If coordinates appear to be normalized (0-1 range), convert to absolute
    if all(0 <= coord <= 1 for coord in bbox):
        x1 = int(x1 * page_width)
        y1 = int(y1 * page_height)
        x2 = int(x2 * page_width)
        y2 = int(y2 * page_height)
    else:
        # Ensure integer coordinates and clamp to page bounds
        x1 = max(0, min(int(x1), page_width))
        y1 = max(0, min(int(y1), page_height))
        x2 = max(x1, min(int(x2), page_width))
        y2 = max(y1, min(int(y2), page_height))
        
    return [x1, y1, x2, y2]


def normalize_confidence(conf: float) -> float:
    """
    Normalize confidence score to [0.0, 1.0] range.
    
    Args:
        conf: Raw confidence score
        
    Returns:
        Normalized confidence in [0.0, 1.0] range
    """
    return max(0.0, min(1.0, float(conf)))


def clean_text(text: str) -> str:
    """
    Clean and normalize OCR text output.
    
    Args:
        text: Raw OCR text
        
    Returns:
        Cleaned text with normalized whitespace
    """
    if not text:
        return ""
    
    # Remove excessive whitespace and normalize
    text = " ".join(text.split())
    
    # Remove non-printable characters except newlines and tabs
    text = "".join(char for char in text if char.isprintable() or char in "\n\t")
    
    return text.strip()