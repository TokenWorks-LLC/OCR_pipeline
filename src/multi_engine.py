"""
Multi-Engine OCR Orchestration

Handles running multiple OCR engines and combining their results.
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class MultiEngineOrchestrator:
    """
    Orchestrator for running multiple OCR engines and combining results.
    """
    
    def __init__(self, ensemble_config: Dict[str, Any] = None):
        """
        Initialize multi-engine orchestrator.
        
        Args:
            ensemble_config: Configuration for ensemble system
        """
        self.config = ensemble_config or {}
        self.engines = {}
        self.ensemble = None
        
        # Initialize ensemble voter
        from ensemble import OCREnsemble
        self.ensemble = OCREnsemble(self.config.get('voting', {}))
    
    def add_engine(self, name: str, engine: Any):
        """Add an OCR engine to the orchestrator."""
        self.engines[name] = engine
        logger.info(f"Added OCR engine: {name}")
    
    def run_all_engines(self, image: np.ndarray) -> Dict[str, List[Dict]]:
        """
        Run all registered engines on an image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dict mapping engine name to list of detections
        """
        results = {}
        
        for engine_name, engine in self.engines.items():
            try:
                logger.debug(f"Running {engine_name}...")
                
                if engine_name == 'paddle':
                    # PaddleOCR
                    ocr_result = engine.ocr(image)
                    boxes = self._parse_paddle_result(ocr_result)
                    results[engine_name] = boxes
                    
                elif engine_name == 'easyocr':
                    # EasyOCR
                    boxes = engine.ocr(image)
                    results[engine_name] = boxes
                    
                elif engine_name == 'doctr':
                    # DocTR
                    boxes = engine.infer_page(image)
                    results[engine_name] = boxes
                    
                elif engine_name == 'tesseract':
                    # Tesseract (via ocr_utils)
                    from ocr_utils import ocr_tesseract_lines
                    lines = ocr_tesseract_lines(image)
                    boxes = [{
                        'bbox': line.bbox,
                        'text': line.text,
                        'confidence': line.conf
                    } for line in lines]
                    results[engine_name] = boxes
                
                logger.info(f"{engine_name}: found {len(results.get(engine_name, []))} text boxes")
                
            except Exception as e:
                logger.error(f"Engine {engine_name} failed: {e}")
                results[engine_name] = []
        
        return results
    
    def run_with_ensemble(self, image: np.ndarray) -> List[Dict]:
        """
        Run all engines and combine results using ensemble voting.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of combined detection dicts
        """
        # Run all engines
        engine_results = self.run_all_engines(image)
        
        # Combine with ensemble
        if len(engine_results) == 0:
            return []
        elif len(engine_results) == 1:
            # Only one engine, return its results directly
            return list(engine_results.values())[0]
        else:
            # Multiple engines, use ensemble
            combined = self.ensemble.combine_results(engine_results)
            
            # Log statistics
            stats = self.ensemble.get_statistics(engine_results)
            logger.info(f"Ensemble: {stats['total_detections']} detections from {stats['num_engines']} engines -> {len(combined)} combined")
            
            return combined
    
    def _parse_paddle_result(self, result: List) -> List[Dict]:
        """Parse PaddleOCR result into standard format."""
        boxes = []
        
        if result and len(result) > 0:
            page_result = result[0]
            
            if hasattr(page_result, 'keys'):
                # New PaddleOCR format
                rec_texts = page_result.get('rec_texts', [])
                rec_scores = page_result.get('rec_scores', [])
                rec_polys = page_result.get('rec_polys', [])
                rec_boxes = page_result.get('rec_boxes', [])
                
                bboxes = rec_boxes if rec_boxes is not None and len(rec_boxes) > 0 else rec_polys
                
                for i in range(len(rec_texts)):
                    text = rec_texts[i] if i < len(rec_texts) else ''
                    score = rec_scores[i] if i < len(rec_scores) else 0.0
                    bbox = bboxes[i] if i < len(bboxes) else None
                    
                    if bbox is not None:
                        boxes.append({
                            'bbox': bbox,
                            'text': text,
                            'confidence': score
                        })
            else:
                # Old PaddleOCR format
                if hasattr(page_result, '__iter__'):
                    for item in page_result:
                        if isinstance(item, (list, tuple)) and len(item) >= 2:
                            boxes.append({
                                'bbox': item[0],
                                'text': item[1][0] if isinstance(item[1], tuple) else '',
                                'confidence': item[1][1] if isinstance(item[1], tuple) else 0.0
                            })
        
        return boxes
