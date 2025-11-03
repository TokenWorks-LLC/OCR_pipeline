#!/usr/bin/env python3
"""
Orchestrator adapter for backward compatibility with ocr_pipeline.py.

This module provides a simplified interface to the orchestrator while
maintaining the existing CLI commands and behavior.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from orchestrator import PipelineOrchestrator, PipelineConfig, PageTask
from cache_store import CacheStore
from pdf_utils import extract_page_count

logger = logging.getLogger(__name__)


@dataclass
class SimplifiedOCRResult:
    """Simplified OCR result compatible with legacy interface."""
    text: str
    confidence: float
    boxes: int
    processing_time: float
    cache_hits: Dict[str, bool]


class OrchestratorAdapter:
    """
    Adapter to route ocr_pipeline.py commands through the orchestrator.
    
    Maintains backward compatibility while using the new orchestrator-based
    architecture with deterministic caching.
    """
    
    def __init__(self, profile_path: Optional[str] = None, invalidate: Optional[str] = None):
        """
        Initialize orchestrator adapter.
        
        Args:
            profile_path: Path to profile JSON (optional)
            invalidate: Cache stage to invalidate (render, ocr, fusion, llm, all)
        """
        self.profile_path = profile_path
        
        # Create default config (can be overridden by profile)
        self.config = PipelineConfig(
            max_page_workers=4,
            max_gpu_workers=1,
            cache_enabled=True,
            cache_dir="cache/pipeline",
            cache_max_size_gb=10.0,
            keep_page_order=True,
            queue_size=100,
            torch_deterministic=True,
            blas_num_threads=1
        )
        
        # Initialize cache
        self.cache = CacheStore(
            cache_dir=self.config.cache_dir,
            max_size_gb=self.config.cache_max_size_gb,
            enabled=self.config.cache_enabled
        )
        
        # Handle cache invalidation if requested
        if invalidate:
            count = self.cache.invalidate(invalidate)
            logger.info(f"Invalidated {count} cache entries for stage: {invalidate}")
        
        # Initialize orchestrator
        self.orchestrator = PipelineOrchestrator(
            config=self.config,
            cache_store=self.cache
        )
        
        logger.info("Orchestrator adapter initialized")
    
    def process_single_page(
        self,
        pdf_path: Path,
        page_num: int,
        ocr_engine,
        ocr_config: Dict,
        llm_corrector=None
    ) -> SimplifiedOCRResult:
        """
        Process a single page (backward compatible interface).
        
        Args:
            pdf_path: Path to PDF file
            page_num: Page number (1-indexed)
            ocr_engine: OCR engine instance
            ocr_config: OCR configuration dict
            llm_corrector: Optional LLM corrector
            
        Returns:
            SimplifiedOCRResult with text and metadata
        """
        logger.info(f"Processing {pdf_path.name}, page {page_num}")
        
        # Process through orchestrator
        tasks = self.orchestrator.process_pdf_parallel(
            pdf_path=str(pdf_path),
            page_range=(page_num, page_num),
            ocr_engine=ocr_engine,
            ocr_config=ocr_config,
            llm_corrector=llm_corrector
        )
        
        if not tasks or len(tasks) == 0:
            raise RuntimeError(f"Failed to process page {page_num}")
        
        task = tasks[0]
        
        # Convert to simplified result
        result = SimplifiedOCRResult(
            text=task.final_text,
            confidence=self._calculate_confidence(task),
            boxes=len(task.detection_boxes) if task.detection_boxes else 0,
            processing_time=sum(task.stage_timings.values()),
            cache_hits=task.cache_hits
        )
        
        return result
    
    def process_batch(
        self,
        pdf_path: Path,
        page_range: Optional[Tuple[int, int]],
        ocr_engine,
        ocr_config: Dict,
        llm_corrector=None
    ) -> List[SimplifiedOCRResult]:
        """
        Process multiple pages (backward compatible interface).
        
        Args:
            pdf_path: Path to PDF file
            page_range: (start, end) tuple or None for all pages
            ocr_engine: OCR engine instance
            ocr_config: OCR configuration dict
            llm_corrector: Optional LLM corrector
            
        Returns:
            List of SimplifiedOCRResult
        """
        # Determine page range
        if page_range is None:
            try:
                total_pages = extract_page_count(str(pdf_path))
                page_range = (1, total_pages)
            except Exception as e:
                logger.error(f"Failed to get page count: {e}")
                raise
        
        logger.info(f"Processing {pdf_path.name}, pages {page_range[0]}-{page_range[1]}")
        
        # Process through orchestrator
        tasks = self.orchestrator.process_pdf_parallel(
            pdf_path=str(pdf_path),
            page_range=page_range,
            ocr_engine=ocr_engine,
            ocr_config=ocr_config,
            llm_corrector=llm_corrector
        )
        
        # Convert to simplified results
        results = []
        for task in tasks:
            result = SimplifiedOCRResult(
                text=task.final_text,
                confidence=self._calculate_confidence(task),
                boxes=len(task.detection_boxes) if task.detection_boxes else 0,
                processing_time=sum(task.stage_timings.values()),
                cache_hits=task.cache_hits
            )
            results.append(result)
        
        return results
    
    def _calculate_confidence(self, task: PageTask) -> float:
        """
        Calculate overall confidence from recognition results.
        
        Args:
            task: PageTask with recognition results
            
        Returns:
            Average confidence score
        """
        if not task.recognition_results:
            return 0.0
        
        confidences = []
        for result in task.recognition_results:
            if isinstance(result, dict) and 'confidence' in result:
                confidences.append(result['confidence'])
            elif isinstance(result, dict) and 'conf' in result:
                confidences.append(result['conf'])
        
        if not confidences:
            return 0.0
        
        return sum(confidences) / len(confidences)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        return self.orchestrator.get_statistics()
    
    def cleanup(self):
        """Cleanup resources."""
        self.orchestrator.cleanup()


def create_orchestrator_adapter(
    profile_path: Optional[str] = None,
    invalidate: Optional[str] = None
) -> OrchestratorAdapter:
    """
    Factory function to create orchestrator adapter.
    
    Args:
        profile_path: Path to profile JSON
        invalidate: Cache stage to invalidate
        
    Returns:
        OrchestratorAdapter instance
    """
    return OrchestratorAdapter(
        profile_path=profile_path,
        invalidate=invalidate
    )
