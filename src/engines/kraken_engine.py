"""
Kraken OCR Engine Implementation

This module provides a Kraken backend for the OCR pipeline, implementing
the OcrEngine interface for specialized text recognition, particularly
for historical documents and non-Latin scripts.
"""

import logging
from typing import List, Dict, Any, Union
from PIL import Image
import numpy as np

from ..ocr_engine import OcrEngine, normalize_bbox, normalize_confidence, clean_text

logger = logging.getLogger(__name__)


class KrakenEngine(OcrEngine):
    """
    Kraken OCR engine implementation for specialized text recognition.
    
    Particularly suited for historical documents, manuscripts, and
    non-Latin scripts including Arabic, Persian, and ancient scripts.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Kraken engine with configuration.
        
        Args:
            config: Configuration dictionary with options:
                - model: Model name or path (default: auto-detected)
                - segmentation_model: Segmentation model for layout analysis
                - bidi: Enable bidirectional text support (default: True)
                - reading_order: Enable reading order detection (default: True)
                - device: Device to use ('auto', 'cpu', 'cuda')
                - line_level: Return line-level results (default: True)
        """
        super().__init__(config)
        
        # Default configuration
        self.model_name = self.config.get('model', None)  # Auto-detect if None
        self.segmentation_model = self.config.get('segmentation_model', 'blla.mlmodel')
        self.bidi = self.config.get('bidi', True)
        self.reading_order = self.config.get('reading_order', True)
        self.line_level = self.config.get('line_level', True)
        
        # Override device if specified in config
        if 'device' in self.config:
            if self.config['device'] == 'auto':
                self.device = self._detect_device()
            else:
                self.device = self.config['device']
        
        # Initialize models
        self.segmentation_model_obj = None
        self.recognition_model_obj = None
        self._init_models()
        
    def _init_models(self):
        """Initialize Kraken segmentation and recognition models."""
        try:
            import kraken
            from kraken import blla, rpred
            from kraken.lib import models
            
            logger.info(f"Initializing Kraken models: seg={self.segmentation_model}, rec={self.model_name}, device={self.device}")
            
            # Load segmentation model for layout analysis
            try:
                self.segmentation_model_obj = models.load_any(self.segmentation_model)
                if hasattr(self.segmentation_model_obj, 'to') and self.device == "cuda":
                    self.segmentation_model_obj = self.segmentation_model_obj.to('cuda')
                logger.info(f"Loaded segmentation model: {self.segmentation_model}")
            except Exception as e:
                logger.warning(f"Failed to load segmentation model {self.segmentation_model}: {e}")
                self.segmentation_model_obj = None
            
            # Load recognition model
            if self.model_name:
                try:
                    self.recognition_model_obj = models.load_any(self.model_name)
                    if hasattr(self.recognition_model_obj, 'to') and self.device == "cuda":
                        self.recognition_model_obj = self.recognition_model_obj.to('cuda')
                    logger.info(f"Loaded recognition model: {self.model_name}")
                except Exception as e:
                    logger.warning(f"Failed to load recognition model {self.model_name}: {e}")
                    self.recognition_model_obj = None
            
            logger.info("Kraken models initialized successfully")
            
        except ImportError:
            raise ImportError(
                "Kraken is not installed. Install it with: pip install kraken"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Kraken models: {e}")
    
    def infer_page(self, image: Union[Image.Image, np.ndarray]) -> List[Dict[str, Any]]:
        """
        Run Kraken OCR inference on a single page.
        
        Args:
            image: Input image as PIL Image or numpy array
            
        Returns:
            List of normalized text spans with bbox, confidence, and text
        """
        try:
            from kraken import blla, rpred
            from kraken.lib import segmentation
            
            # Convert numpy array to PIL Image if needed
            if isinstance(image, np.ndarray):
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = Image.fromarray(image)
                elif len(image.shape) == 3 and image.shape[2] == 4:
                    image = Image.fromarray(image).convert('RGB')
                elif len(image.shape) == 2:
                    image = Image.fromarray(image).convert('RGB')
                else:
                    raise ValueError(f"Unsupported image array shape: {image.shape}")
            
            # Ensure PIL Image is in RGB mode
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            page_width, page_height = image.size
            spans = []
            
            # Step 1: Perform layout analysis (line segmentation)
            if self.segmentation_model_obj:
                try:
                    # Use loaded segmentation model
                    baseline_seg = blla.segment(image, model=self.segmentation_model_obj)
                except Exception as e:
                    logger.warning(f"Segmentation with model failed: {e}, using default")
                    baseline_seg = segmentation.segment(image, text_direction='horizontal-lr')
            else:
                # Use default segmentation
                baseline_seg = segmentation.segment(image, text_direction='horizontal-lr')
            
            # Step 2: Perform text recognition on segmented lines
            if self.recognition_model_obj and baseline_seg['lines']:
                try:
                    # Use loaded recognition model
                    pred_it = rpred.rpred(
                        network=self.recognition_model_obj,
                        im=image,
                        bounds=baseline_seg,
                        bidi_reordering=self.bidi
                    )
                    
                    # Extract results
                    for line_result in pred_it:
                        if line_result.prediction and line_result.prediction.strip():
                            # Extract bounding box from baseline
                            bbox = self._extract_line_bbox(line_result, page_width, page_height)
                            
                            spans.append({
                                "text": clean_text(line_result.prediction),
                                "bbox": bbox,
                                "conf": normalize_confidence(line_result.avg_conf if hasattr(line_result, 'avg_conf') else 0.5),
                                "level": "line",
                                "page_width": page_width,
                                "page_height": page_height
                            })
                            
                except Exception as e:
                    logger.error(f"Kraken recognition failed: {e}")
            
            else:
                # Fallback: just return segmented regions without recognition
                for line in baseline_seg.get('lines', []):
                    bbox = self._extract_segmentation_bbox(line, page_width, page_height)
                    spans.append({
                        "text": "[segmented line]",  # Placeholder text
                        "bbox": bbox,
                        "conf": 0.5,  # Default confidence
                        "level": "line",
                        "page_width": page_width,
                        "page_height": page_height
                    })
            
            logger.debug(f"Kraken extracted {len(spans)} text spans")
            return spans
            
        except Exception as e:
            logger.error(f"Kraken inference failed: {e}")
            return []
    
    def _extract_line_bbox(self, line_result, page_width: int, page_height: int) -> List[int]:
        """
        Extract bounding box from Kraken line result.
        
        Args:
            line_result: Kraken recognition result for a line
            page_width: Page width in pixels
            page_height: Page height in pixels
            
        Returns:
            Normalized bounding box as [x1, y1, x2, y2]
        """
        try:
            # Extract bbox from line result
            if hasattr(line_result, 'bbox'):
                x1, y1, x2, y2 = line_result.bbox
                return normalize_bbox([x1, y1, x2, y2], page_width, page_height)
            elif hasattr(line_result, 'baseline'):
                # Extract from baseline coordinates
                baseline = line_result.baseline
                if baseline:
                    x_coords = [point[0] for point in baseline]
                    y_coords = [point[1] for point in baseline]
                    x1, x2 = min(x_coords), max(x_coords)
                    y1, y2 = min(y_coords), max(y_coords)
                    # Add some padding for line height
                    y1 = max(0, y1 - 10)
                    y2 = min(page_height, y2 + 10)
                    return normalize_bbox([x1, y1, x2, y2], page_width, page_height)
            
            # Fallback
            return [0, 0, page_width, page_height]
            
        except Exception as e:
            logger.warning(f"Failed to extract line bbox: {e}")
            return [0, 0, page_width, page_height]
    
    def _extract_segmentation_bbox(self, line_info, page_width: int, page_height: int) -> List[int]:
        """
        Extract bounding box from segmentation result.
        
        Args:
            line_info: Line information from segmentation
            page_width: Page width in pixels
            page_height: Page height in pixels
            
        Returns:
            Normalized bounding box as [x1, y1, x2, y2]
        """
        try:
            # Extract coordinates from line segmentation
            if isinstance(line_info, dict):
                # Try different possible keys
                for key in ['bbox', 'boundary', 'polygon', 'coords']:
                    if key in line_info:
                        coords = line_info[key]
                        if coords:
                            x_coords = [point[0] for point in coords]
                            y_coords = [point[1] for point in coords]
                            x1, x2 = min(x_coords), max(x_coords)
                            y1, y2 = min(y_coords), max(y_coords)
                            return normalize_bbox([x1, y1, x2, y2], page_width, page_height)
            
            # Fallback
            return [0, 0, page_width, page_height]
            
        except Exception as e:
            logger.warning(f"Failed to extract segmentation bbox: {e}")
            return [0, 0, page_width, page_height]
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get Kraken engine information."""
        try:
            import kraken
            kraken_version = kraken.__version__
        except:
            kraken_version = "unknown"
        
        return {
            "name": "kraken",
            "version": kraken_version,
            "device": self.device,
            "models": {
                "segmentation": self.segmentation_model,
                "recognition": self.model_name,
                "bidi_support": self.bidi
            },
            "config": self.config
        }


class KrakenEngineFactory:
    """Factory for creating Kraken engine instances with different profiles."""
    
    @staticmethod
    def create_general_engine(device: str = "auto") -> KrakenEngine:
        """Create a general-purpose Kraken engine."""
        config = {
            'model': None,  # Auto-detect
            'segmentation_model': 'blla.mlmodel',
            'bidi': True,
            'reading_order': True,
            'device': device,
            'line_level': True
        }
        return KrakenEngine(config)
    
    @staticmethod
    def create_akkadian_engine(device: str = "auto") -> KrakenEngine:
        """Create a Kraken engine optimized for Akkadian cuneiform."""
        config = {
            'model': None,  # Should be set to specific Akkadian model when available
            'segmentation_model': 'blla.mlmodel',
            'bidi': False,  # Akkadian is typically left-to-right
            'reading_order': True,
            'device': device,
            'line_level': True
        }
        return KrakenEngine(config)
    
    @staticmethod
    def create_arabic_engine(device: str = "auto") -> KrakenEngine:
        """Create a Kraken engine optimized for Arabic scripts."""
        config = {
            'model': None,  # Should be set to specific Arabic model
            'segmentation_model': 'blla.mlmodel',
            'bidi': True,  # Arabic is right-to-left with BiDi support
            'reading_order': True,
            'device': device,
            'line_level': True
        }
        return KrakenEngine(config)