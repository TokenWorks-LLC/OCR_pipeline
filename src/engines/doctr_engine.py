"""
DocTR OCR Engine Implementation

This module provides a docTR backend for the OCR pipeline, implementing
the OcrEngine interface for seamless integration.
"""

import logging
from typing import List, Dict, Any, Union
from PIL import Image
import numpy as np

from ..ocr_engine import OcrEngine, normalize_bbox, normalize_confidence, clean_text

logger = logging.getLogger(__name__)


class DoctrEngine(OcrEngine):
    """
    DocTR OCR engine implementation using Mindee's docTR library.
    
    Supports both detection and recognition with pre-trained models.
    Uses DBNet for detection and SAR/CRNN for recognition by default.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize DocTR engine with configuration.
        
        Args:
            config: Configuration dictionary with options:
                - det_arch: Detection architecture (default: 'db_resnet50')
                - reco_arch: Recognition architecture (default: 'sar')
                - pretrained: Use pretrained models (default: True)
                - assume_straight_pages: Assume straight text pages (default: True)
                - export_as_straight_boxes: Export as straight boxes (default: True)
                - device: Device to use ('auto', 'cpu', 'cuda')
        """
        super().__init__(config)
        
        # Default configuration
        self.det_arch = self.config.get('det_arch', 'db_resnet50')
        self.reco_arch = self.config.get('reco_arch', 'sar')
        self.pretrained = self.config.get('pretrained', True)
        self.assume_straight_pages = self.config.get('assume_straight_pages', True)
        self.export_as_straight_boxes = self.config.get('export_as_straight_boxes', True)
        
        # Override device if specified in config
        if 'device' in self.config:
            if self.config['device'] == 'auto':
                self.device = self._detect_device()
            else:
                self.device = self.config['device']
        
        # Initialize model
        self.model = None
        self._init_model()
        
    def _init_model(self):
        """Initialize the docTR OCR model."""
        try:
            from doctr.models import ocr_predictor
            
            logger.info(f"Initializing docTR model: det={self.det_arch}, reco={self.reco_arch}, device={self.device}")
            
            self.model = ocr_predictor(
                det_arch=self.det_arch,
                reco_arch=self.reco_arch,
                pretrained=self.pretrained,
                assume_straight_pages=self.assume_straight_pages,
                export_as_straight_boxes=self.export_as_straight_boxes
            )
            
            # Move to device if CUDA is available and requested
            if self.device == "cuda":
                try:
                    self.model = self.model.cuda()
                    logger.info("DocTR model moved to CUDA")
                except Exception as e:
                    logger.warning(f"Failed to move docTR model to CUDA: {e}, using CPU")
                    self.device = "cpu"
            
            logger.info("DocTR model initialized successfully")
            
        except ImportError:
            raise ImportError(
                "docTR is not installed. Install it with: pip install python-doctr"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize docTR model: {e}")
    
    def infer_page(self, image: Union[Image.Image, np.ndarray]) -> List[Dict[str, Any]]:
        """
        Run docTR OCR inference on a single page.
        
        Args:
            image: Input image as PIL Image or numpy array
            
        Returns:
            List of normalized text spans with bbox, confidence, and text
        """
        if self.model is None:
            raise RuntimeError("DocTR model not initialized")
        
        try:
            # Convert numpy array to PIL Image if needed
            if isinstance(image, np.ndarray):
                if len(image.shape) == 3 and image.shape[2] == 3:
                    # RGB array
                    image = Image.fromarray(image)
                elif len(image.shape) == 3 and image.shape[2] == 4:
                    # RGBA array
                    image = Image.fromarray(image).convert('RGB')
                elif len(image.shape) == 2:
                    # Grayscale array
                    image = Image.fromarray(image).convert('RGB')
                else:
                    raise ValueError(f"Unsupported image array shape: {image.shape}")
            
            # Ensure PIL Image is in RGB mode
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            page_width, page_height = image.size
            
            # Run docTR inference
            from doctr.io import DocumentFile
            
            # Create DocumentFile from PIL Image
            doc = DocumentFile.from_images([image])
            result = self.model(doc)
            
            # Extract text spans from docTR result
            spans = []
            
            if result.pages and len(result.pages) > 0:
                page = result.pages[0]
                
                # Iterate through page structure: page -> blocks -> lines -> words
                for block in page.blocks:
                    for line in block.lines:
                        if self.config.get('word_level', False):
                            # Word-level extraction
                            for word in line.words:
                                bbox = self._extract_bbox(word.geometry, page_width, page_height)
                                spans.append({
                                    "text": clean_text(word.value),
                                    "bbox": bbox,
                                    "conf": normalize_confidence(word.confidence),
                                    "level": "word",
                                    "page_width": page_width,
                                    "page_height": page_height
                                })
                        else:
                            # Line-level extraction (default)
                            line_text = " ".join([word.value for word in line.words])
                            line_conf = np.mean([word.confidence for word in line.words]) if line.words else 0.0
                            bbox = self._extract_bbox(line.geometry, page_width, page_height)
                            
                            spans.append({
                                "text": clean_text(line_text),
                                "bbox": bbox,
                                "conf": normalize_confidence(line_conf),
                                "level": "line",
                                "page_width": page_width,
                                "page_height": page_height
                            })
            
            logger.debug(f"DocTR extracted {len(spans)} text spans")
            return spans
            
        except Exception as e:
            logger.error(f"DocTR inference failed: {e}")
            return []
    
    def _extract_bbox(self, geometry, page_width: int, page_height: int) -> List[int]:
        """
        Extract bounding box from docTR geometry.
        
        Args:
            geometry: DocTR geometry object
            page_width: Page width in pixels
            page_height: Page height in pixels
            
        Returns:
            Normalized bounding box as [x1, y1, x2, y2]
        """
        try:
            # DocTR geometry is typically (N, 2) array with corner points
            # Or it might have a bounding_box property
            if hasattr(geometry, 'bounding_box'):
                # Use bounding box if available
                bbox = geometry.bounding_box
                x1, y1, x2, y2 = bbox
            else:
                # Extract from polygon points
                points = np.array(geometry)
                x1, y1 = points.min(axis=0)
                x2, y2 = points.max(axis=0)
            
            # DocTR coordinates are typically normalized (0-1)
            return normalize_bbox([x1, y1, x2, y2], page_width, page_height)
            
        except Exception as e:
            logger.warning(f"Failed to extract bbox from geometry: {e}")
            # Return fallback bbox
            return [0, 0, page_width, page_height]
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get DocTR engine information."""
        try:
            import doctr
            doctr_version = doctr.__version__
        except:
            doctr_version = "unknown"
        
        return {
            "name": "doctr",
            "version": doctr_version,
            "device": self.device,
            "models": {
                "detector": self.det_arch,
                "recognizer": self.reco_arch,
                "pretrained": self.pretrained
            },
            "config": self.config
        }


class DoctrEngineFactory:
    """Factory for creating DocTR engine instances with different profiles."""
    
    @staticmethod
    def create_fast_engine(device: str = "auto") -> DoctrEngine:
        """Create a fast DocTR engine configuration."""
        config = {
            'det_arch': 'db_resnet50',
            'reco_arch': 'crnn_vgg16_bn',  # Faster than SAR
            'pretrained': True,
            'device': device,
            'assume_straight_pages': True,
            'export_as_straight_boxes': True,
            'word_level': False
        }
        return DoctrEngine(config)
    
    @staticmethod
    def create_quality_engine(device: str = "auto") -> DoctrEngine:
        """Create a high-quality DocTR engine configuration."""
        config = {
            'det_arch': 'db_resnet50', 
            'reco_arch': 'sar',  # Higher quality than CRNN
            'pretrained': True,
            'device': device,
            'assume_straight_pages': False,  # Handle rotated text
            'export_as_straight_boxes': False,  # Preserve rotation info
            'word_level': False
        }
        return DoctrEngine(config)
    
    @staticmethod
    def create_word_level_engine(device: str = "auto") -> DoctrEngine:
        """Create a word-level DocTR engine configuration."""
        config = {
            'det_arch': 'db_resnet50',
            'reco_arch': 'sar',
            'pretrained': True,
            'device': device,
            'assume_straight_pages': True,
            'export_as_straight_boxes': True,
            'word_level': True  # Extract individual words
        }
        return DoctrEngine(config)