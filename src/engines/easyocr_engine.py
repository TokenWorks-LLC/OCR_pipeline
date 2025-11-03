"""
EasyOCR Engine Implementation

This module provides an EasyOCR backend for the OCR pipeline.
EasyOCR supports 80+ languages and is easy to use.
"""

import logging
from typing import List, Dict, Any, Union, Optional
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class EasyOCREngine:
    """
    EasyOCR engine wrapper for multi-language OCR support.
    
    Supports 80+ languages with simple initialization.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize EasyOCR engine with configuration.
        
        Args:
            config: Configuration dictionary with options:
                - lang_list: List of language codes (default: ['en'])
                - gpu: Enable GPU (default: True if CUDA available)
                - model_storage_directory: Model cache directory
                - download_enabled: Allow model downloads (default: True)
        """
        self.config = config or {}
        
        # Default configuration - check if CUDA is actually available
        self._gpu_available = self._check_gpu()
        
        self.lang_list = self.config.get('lang_list', ['en'])
        self.gpu = self.config.get('gpu', self._gpu_available)
        self.model_storage_directory = self.config.get('model_storage_directory', None)
        self.download_enabled = self.config.get('download_enabled', True)
        
        # Initialize reader
        self.reader = None
        self._init_reader()
    
    def _check_gpu(self) -> bool:
        """Check if GPU is available for EasyOCR."""
        try:
            import torch
            has_cuda = torch.cuda.is_available()
            if has_cuda:
                logger.info(f"CUDA is available: {torch.cuda.get_device_name(0)}")
            else:
                logger.info("CUDA is not available, will use CPU")
            return has_cuda
        except ImportError:
            logger.warning("PyTorch not available, cannot check GPU")
            return False
        
    def _init_reader(self):
        """Initialize the EasyOCR Reader."""
        try:
            import easyocr
            
            logger.info(f"Initializing EasyOCR: languages={self.lang_list}, gpu={self.gpu}")
            
            self.reader = easyocr.Reader(
                self.lang_list,
                gpu=self.gpu,
                model_storage_directory=self.model_storage_directory,
                download_enabled=self.download_enabled
            )
            
            logger.info("EasyOCR reader initialized successfully")
            
        except ImportError:
            raise ImportError(
                "EasyOCR is not installed. Install it with: pip install easyocr"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize EasyOCR reader: {e}")
    
    def ocr(self, image: Union[Image.Image, np.ndarray]) -> List[List]:
        """
        Run EasyOCR on an image.
        
        Args:
            image: Input image as PIL Image or numpy array
            
        Returns:
            List of [bbox, text, confidence] for each detection
        """
        if self.reader is None:
            raise RuntimeError("EasyOCR reader not initialized")
        
        try:
            # Convert PIL Image to numpy if needed
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            # Run OCR
            results = self.reader.readtext(image)
            
            # EasyOCR returns: [[bbox, text, conf], ...]
            # We need to convert to our format
            formatted_results = []
            for bbox, text, conf in results:
                formatted_results.append({
                    'bbox': bbox,
                    'text': text,
                    'confidence': conf
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"EasyOCR inference failed: {e}")
            return []
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get information about the engine."""
        return {
            "name": "easyocr",
            "version": "1.7.2",
            "languages": self.lang_list,
            "device": "cuda" if self.gpu else "cpu"
        }


class EasyOCREngineFactory:
    """Factory for creating EasyOCR engine instances."""
    
    @staticmethod
    def create(config: Dict[str, Any] = None) -> EasyOCREngine:
        """Create an EasyOCR engine instance."""
        return EasyOCREngine(config)
