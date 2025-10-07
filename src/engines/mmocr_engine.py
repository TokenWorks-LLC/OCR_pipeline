"""
MMOCR OCR Engine Implementation

This module provides an MMOCR backend for the OCR pipeline, implementing
the OcrEngine interface for seamless integration with OpenMMLab's MMOCR.
"""

import logging
from typing import List, Dict, Any, Union
from PIL import Image
import numpy as np

from ..ocr_engine import OcrEngine, normalize_bbox, normalize_confidence, clean_text

logger = logging.getLogger(__name__)


class MmocrEngine(OcrEngine):
    """
    MMOCR OCR engine implementation using OpenMMLab's MMOCR library.
    
    Supports state-of-the-art detection and recognition models including
    DBNet++, ABINet, and PARSeq.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize MMOCR engine with configuration.
        
        Args:
            config: Configuration dictionary with options:
                - det_config: Detection model config path or model name
                - det_weights: Detection model weights path
                - rec_config: Recognition model config path or model name  
                - rec_weights: Recognition model weights path
                - recognizer: Recognition model type ('abinet', 'parseq', 'crnn')
                - device: Device to use ('auto', 'cpu', 'cuda')
                - batch_mode: Enable batch processing (default: False)
        """
        super().__init__(config)
        
        # Default configuration - use high-quality models
        self.recognizer = self.config.get('recognizer', 'abinet')  # 'abinet' for quality, 'parseq' for speed
        self.batch_mode = self.config.get('batch_mode', False)
        
        # Model configurations
        self.det_config = self.config.get('det_config', 'dbnetpp_resnet50-oclip_fpnc_1200e_icdar2015')
        self.rec_config = self._get_rec_config()
        
        # Override device if specified in config
        if 'device' in self.config:
            if self.config['device'] == 'auto':
                self.device = self._detect_device()
            else:
                self.device = self.config['device']
        
        # Initialize models
        self.detector = None
        self.recognizer_model = None
        self._init_models()
        
    def _get_rec_config(self) -> str:
        """Get recognition model config based on recognizer type."""
        recognizer_configs = {
            'abinet': 'abinet_20e_st-an_mj',  # High quality
            'parseq': 'parseq_20e_st-an_mj',  # Fast
            'crnn': 'crnn_academic_dataset',   # Basic
            'sar': 'sar_resnet31_parallel-decoder_5e_st-an_mj'  # Alternative
        }
        
        return self.config.get('rec_config', recognizer_configs.get(self.recognizer, recognizer_configs['abinet']))
    
    def _init_models(self):
        """Initialize MMOCR detection and recognition models."""
        try:
            from mmocr.apis import MMOCRInferencer
            
            logger.info(f"Initializing MMOCR models: det={self.det_config}, rec={self.rec_config}, device={self.device}")
            
            # Initialize MMOCR inferencer with detection and recognition models
            self.inferencer = MMOCRInferencer(
                det=self.det_config,
                rec=self.rec_config,
                device=self.device
            )
            
            logger.info("MMOCR models initialized successfully")
            
        except ImportError:
            raise ImportError(
                "MMOCR is not installed. Install it with: "
                "pip install mmengine mmcv mmdet && pip install mmocr"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize MMOCR models: {e}")
    
    def infer_page(self, image: Union[Image.Image, np.ndarray]) -> List[Dict[str, Any]]:
        """
        Run MMOCR inference on a single page.
        
        Args:
            image: Input image as PIL Image or numpy array
            
        Returns:
            List of normalized text spans with bbox, confidence, and text
        """
        if self.inferencer is None:
            raise RuntimeError("MMOCR models not initialized")
        
        try:
            # Convert PIL Image to numpy array if needed
            if isinstance(image, Image.Image):
                page_width, page_height = image.size
                image_array = np.array(image)
            else:
                if len(image.shape) == 3:
                    page_height, page_width = image.shape[:2]
                else:
                    page_height, page_width = image.shape
                image_array = image
                
            # Ensure image is in RGB format
            if len(image_array.shape) == 3 and image_array.shape[2] == 4:
                # Convert RGBA to RGB
                image_array = image_array[:, :, :3]
            elif len(image_array.shape) == 2:
                # Convert grayscale to RGB
                image_array = np.stack([image_array] * 3, axis=2)
            
            # Run MMOCR inference
            result = self.inferencer(image_array, return_vis=False, save_pred=False)
            
            # Extract text spans from MMOCR result
            spans = []
            
            if result and 'predictions' in result and len(result['predictions']) > 0:
                predictions = result['predictions'][0]  # First (and only) image
                
                # Extract detection and recognition results
                if 'det_polygons' in predictions and 'rec_texts' in predictions:
                    det_polygons = predictions['det_polygons']
                    rec_texts = predictions['rec_texts']
                    rec_scores = predictions.get('rec_scores', [1.0] * len(rec_texts))
                    
                    for i, (polygon, text, score) in enumerate(zip(det_polygons, rec_texts, rec_scores)):
                        if text.strip():  # Only include non-empty text
                            bbox = self._polygon_to_bbox(polygon, page_width, page_height)
                            
                            spans.append({
                                "text": clean_text(text),
                                "bbox": bbox,
                                "conf": normalize_confidence(score),
                                "level": "line",  # MMOCR typically returns line-level results
                                "page_width": page_width,
                                "page_height": page_height
                            })
            
            logger.debug(f"MMOCR extracted {len(spans)} text spans")
            return spans
            
        except Exception as e:
            logger.error(f"MMOCR inference failed: {e}")
            return []
    
    def _polygon_to_bbox(self, polygon: List[List[float]], page_width: int, page_height: int) -> List[int]:
        """
        Convert polygon coordinates to bounding box.
        
        Args:
            polygon: Polygon points as list of [x, y] coordinates
            page_width: Page width in pixels
            page_height: Page height in pixels
            
        Returns:
            Bounding box as [x1, y1, x2, y2]
        """
        try:
            # Extract x and y coordinates
            if isinstance(polygon[0], (list, tuple)):
                # Format: [[x1, y1], [x2, y2], ...]
                x_coords = [point[0] for point in polygon]
                y_coords = [point[1] for point in polygon]
            else:
                # Format: [x1, y1, x2, y2, ...] (flattened)
                x_coords = polygon[::2]
                y_coords = polygon[1::2]
            
            # Calculate bounding box
            x1, x2 = min(x_coords), max(x_coords)
            y1, y2 = min(y_coords), max(y_coords)
            
            return normalize_bbox([x1, y1, x2, y2], page_width, page_height)
            
        except Exception as e:
            logger.warning(f"Failed to convert polygon to bbox: {e}")
            # Return fallback bbox
            return [0, 0, page_width, page_height]
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get MMOCR engine information."""
        try:
            import mmocr
            mmocr_version = mmocr.__version__
        except:
            mmocr_version = "unknown"
        
        try:
            import mmcv
            mmcv_version = mmcv.__version__
        except:
            mmcv_version = "unknown"
            
        try:
            import mmdet
            mmdet_version = mmdet.__version__
        except:
            mmdet_version = "unknown"
        
        return {
            "name": "mmocr",
            "version": mmocr_version,
            "device": self.device,
            "models": {
                "detector": self.det_config,
                "recognizer": self.rec_config,
                "recognizer_type": self.recognizer
            },
            "dependencies": {
                "mmcv": mmcv_version,
                "mmdet": mmdet_version
            },
            "config": self.config
        }


class MmocrEngineFactory:
    """Factory for creating MMOCR engine instances with different profiles."""
    
    @staticmethod
    def create_fast_engine(device: str = "auto") -> MmocrEngine:
        """Create a fast MMOCR engine configuration."""
        config = {
            'recognizer': 'parseq',  # Fast recognition model
            'det_config': 'dbnet_resnet18_fpnc_1200e_icdar2015',  # Lighter detection model
            'device': device,
            'batch_mode': False
        }
        return MmocrEngine(config)
    
    @staticmethod
    def create_quality_engine(device: str = "auto") -> MmocrEngine:
        """Create a high-quality MMOCR engine configuration."""
        config = {
            'recognizer': 'abinet',  # High-quality recognition model
            'det_config': 'dbnetpp_resnet50-oclip_fpnc_1200e_icdar2015',  # Best detection model
            'device': device,
            'batch_mode': False
        }
        return MmocrEngine(config)
    
    @staticmethod
    def create_research_engine(device: str = "auto") -> MmocrEngine:
        """Create a research-grade MMOCR engine with latest models."""
        config = {
            'recognizer': 'parseq',  # State-of-the-art permuted autoregressive model
            'det_config': 'dbnetpp_resnet50-oclip_fpnc_1200e_icdar2015',  # DBNet++ with oCLIP
            'device': device,
            'batch_mode': True  # Enable batch processing for efficiency
        }
        return MmocrEngine(config)


# Alternative interface for direct model specification
class CustomMmocrEngine(MmocrEngine):
    """
    Custom MMOCR engine that allows direct specification of model configs and weights.
    """
    
    def __init__(self, det_config: str, det_weights: str, rec_config: str, rec_weights: str, device: str = "auto"):
        """
        Initialize with custom model paths.
        
        Args:
            det_config: Path to detection model config
            det_weights: Path to detection model weights
            rec_config: Path to recognition model config  
            rec_weights: Path to recognition model weights
            device: Device to use
        """
        config = {
            'det_config': det_config,
            'det_weights': det_weights,
            'rec_config': rec_config,
            'rec_weights': rec_weights,
            'device': device
        }
        super().__init__(config)