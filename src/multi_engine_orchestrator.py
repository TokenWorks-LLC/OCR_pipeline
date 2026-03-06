#!/usr/bin/env python3
"""
Multi-engine OCR orchestrator with ROVER fusion.

Coordinates multiple OCR engines (PaddleOCR, docTR, MMOCR, Kraken) and fuses
their outputs using ROVER algorithm for improved accuracy.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from cache_store import CacheStore
from rover_fusion import ROVERFusion, Hypothesis
from engines import create_engine, get_available_engines, ENGINE_AVAILABILITY

logger = logging.getLogger(__name__)


@dataclass
class EngineConfig:
    """Configuration for a single OCR engine."""
    name: str
    enabled: bool = True
    timeout: float = 30.0
    quality_mode: str = 'balanced'  # fast, balanced, quality
    min_confidence: float = 0.0
    max_lines: Optional[int] = None


@dataclass
class EngineResult:
    """Result from a single OCR engine."""
    engine_name: str
    text: str
    confidence: float
    processing_time: float
    success: bool
    error: Optional[str] = None
    word_boxes: Optional[List] = None


class MultiEngineOrchestrator:
    """
    Orchestrates multiple OCR engines with ROVER fusion.
    
    Features:
    - Runs multiple engines in parallel with timeouts
    - Caches per-engine results for reproducibility
    - Fuses outputs using ROVER algorithm
    - Graceful fail-soft: continues with N-1 engines on failures
    - Structured logging for engine performance
    """
    
    def __init__(
        self,
        engine_configs: List[EngineConfig],
        cache_dir: str = "cache/pipeline",
        enable_cache: bool = True,
        fusion_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize orchestrator.
        
        Args:
            engine_configs: List of engine configurations
            cache_dir: Directory for caching results
            enable_cache: Whether to enable caching
            fusion_weights: Optional weights for ROVER fusion {engine_name: weight}
        """
        self.engine_configs = {cfg.name: cfg for cfg in engine_configs if cfg.enabled}
        self.cache = CacheStore(cache_dir=cache_dir, enabled=enable_cache)
        
        # Set up ROVER fusion
        if fusion_weights is None:
            fusion_weights = {name: 1.0 for name in self.engine_configs.keys()}
        
        self.rover = ROVERFusion(weights=fusion_weights)
        
        # Statistics
        self.stats = {
            'engine_runs': {name: 0 for name in self.engine_configs.keys()},
            'engine_successes': {name: 0 for name in self.engine_configs.keys()},
            'engine_failures': {name: 0 for name in self.engine_configs.keys()},
            'engine_timeouts': {name: 0 for name in self.engine_configs.keys()},
            'cache_hits': 0,
            'cache_misses': 0,
            'fusion_runs': 0
        }
        
        logger.info(f"MultiEngineOrchestrator initialized with engines: {list(self.engine_configs.keys())}")
        logger.info(f"Fusion weights: {fusion_weights}")
    
    def _run_single_engine(
        self,
        engine_name: str,
        image: np.ndarray,
        render_hash: str,
        languages: List[str]
    ) -> EngineResult:
        """
        Run OCR with a single engine.
        
        Args:
            engine_name: Name of engine
            image: Image as numpy array
            render_hash: Hash of rendered image for caching
            languages: List of language codes
            
        Returns:
            EngineResult with text and metadata
        """
        config = self.engine_configs[engine_name]
        self.stats['engine_runs'][engine_name] += 1
        
        start_time = time.time()
        
        try:
            # Check cache first
            engine_version = "1.0"  # TODO: Get actual version from engine
            cached = self.cache.get_ocr_result(
                render_hash=render_hash,
                engine=engine_name,
                engine_version=engine_version,
                languages=languages
            )
            
            if cached:
                self.stats['cache_hits'] += 1
                logger.debug(f"Cache hit for {engine_name}")
                return EngineResult(
                    engine_name=engine_name,
                    text=cached.get('text', ''),
                    confidence=cached.get('confidence', 0.0),
                    processing_time=time.time() - start_time,
                    success=True,
                    word_boxes=cached.get('word_boxes', [])
                )
            
            self.stats['cache_misses'] += 1
            
            # Create engine instance
            engine = create_engine(
                engine_name=engine_name,
                profile=config.quality_mode
            )
            
            if engine is None:
                raise RuntimeError(f"Failed to create engine {engine_name}")
            
            # Run OCR with timeout
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(engine.run_ocr, image, languages)
                
                try:
                    result = future.result(timeout=config.timeout)
                except TimeoutError:
                    self.stats['engine_timeouts'][engine_name] += 1
                    logger.warning(f"Timeout for {engine_name} after {config.timeout}s")
                    return EngineResult(
                        engine_name=engine_name,
                        text="",
                        confidence=0.0,
                        processing_time=config.timeout,
                        success=False,
                        error="timeout"
                    )
            
            # Extract text and confidence
            text = result.get('text', '')
            confidence = result.get('confidence', 0.0)
            word_boxes = result.get('word_boxes', [])
            
            # Apply filters
            if config.min_confidence and confidence < config.min_confidence:
                logger.debug(f"{engine_name} below min confidence: {confidence:.3f}")
            
            if config.max_lines:
                lines = text.split('\n')
                if len(lines) > config.max_lines:
                    logger.debug(f"{engine_name} exceeded max lines: {len(lines)}")
                    text = '\n'.join(lines[:config.max_lines])
            
            # Cache the result
            cache_data = {
                'text': text,
                'confidence': confidence,
                'word_boxes': word_boxes,
                'engine': engine_name,
                'engine_version': engine_version
            }
            
            self.cache.put_ocr_result(
                render_hash=render_hash,
                engine=engine_name,
                engine_version=engine_version,
                languages=languages,
                result=cache_data
            )
            
            self.stats['engine_successes'][engine_name] += 1
            
            return EngineResult(
                engine_name=engine_name,
                text=text,
                confidence=confidence,
                processing_time=time.time() - start_time,
                success=True,
                word_boxes=word_boxes
            )
            
        except Exception as e:
            self.stats['engine_failures'][engine_name] += 1
            logger.error(f"Error in {engine_name}: {e}", exc_info=True)
            return EngineResult(
                engine_name=engine_name,
                text="",
                confidence=0.0,
                processing_time=time.time() - start_time,
                success=False,
                error=str(e)
            )
    
    def run_engines_parallel(
        self,
        image: np.ndarray,
        render_hash: str,
        languages: List[str]
    ) -> Dict[str, EngineResult]:
        """
        Run all configured engines in parallel.
        
        Args:
            image: Image as numpy array
            render_hash: Hash of rendered image
            languages: List of language codes
            
        Returns:
            Dict mapping engine_name -> EngineResult
        """
        results = {}
        
        # Run engines in parallel
        with ThreadPoolExecutor(max_workers=len(self.engine_configs)) as executor:
            future_to_engine = {
                executor.submit(
                    self._run_single_engine,
                    engine_name,
                    image,
                    render_hash,
                    languages
                ): engine_name
                for engine_name in self.engine_configs.keys()
            }
            
            for future in as_completed(future_to_engine):
                engine_name = future_to_engine[future]
                try:
                    result = future.result()
                    results[engine_name] = result
                    
                    if result.success:
                        logger.debug(
                            f"{engine_name}: {len(result.text)} chars, "
                            f"conf={result.confidence:.3f}, time={result.processing_time:.2f}s"
                        )
                    else:
                        logger.warning(
                            f"{engine_name} failed: {result.error}"
                        )
                except Exception as e:
                    logger.error(f"Unexpected error collecting {engine_name} result: {e}")
                    results[engine_name] = EngineResult(
                        engine_name=engine_name,
                        text="",
                        confidence=0.0,
                        processing_time=0.0,
                        success=False,
                        error=str(e)
                    )
        
        return results
    
    def fuse_results(
        self,
        engine_results: Dict[str, EngineResult]
    ) -> Tuple[str, float, Dict]:
        """
        Fuse multiple engine results using ROVER.
        
        Args:
            engine_results: Dict of engine_name -> EngineResult
            
        Returns:
            (fused_text, fused_confidence, provenance_dict)
        """
        self.stats['fusion_runs'] += 1
        
        # Filter successful results
        successful = {
            name: result for name, result in engine_results.items()
            if result.success and result.text
        }
        
        if not successful:
            logger.warning("No successful engine results to fuse")
            return "", 0.0, {}
        
        if len(successful) == 1:
            # Only one engine succeeded
            engine_name, result = list(successful.items())[0]
            logger.info(f"Only {engine_name} succeeded, using its output directly")
            return result.text, result.confidence, {
                'method': 'single',
                'engine': engine_name
            }
        
        # Create hypotheses for ROVER
        hypotheses = []
        for engine_name, result in successful.items():
            hyp = Hypothesis(
                text=result.text,
                confidence=result.confidence,
                engine=engine_name
            )
            hypotheses.append(hyp)
        
        # Fuse using ROVER
        fused_text, fused_conf, provenance_list = self.rover.fuse(hypotheses)
        
        # Build provenance dict
        provenance = {
            'method': 'rover',
            'engines': list(successful.keys()),
            'per_char_engines': provenance_list,
            'char_count': len(fused_text)
        }
        
        logger.info(
            f"ROVER fusion: {len(successful)} engines -> "
            f"{len(fused_text)} chars, conf={fused_conf:.3f}"
        )
        
        # Cache fusion result
        engine_hashes = sorted([
            f"{name}_{result.confidence:.4f}"
            for name, result in successful.items()
        ])
        
        self.cache.put_fusion_result(
            engine_hashes=engine_hashes,
            result={
                'text': fused_text,
                'confidence': fused_conf,
                'provenance': provenance,
                'engines': list(successful.keys())
            }
        )
        
        return fused_text, fused_conf, provenance
    
    def process_image(
        self,
        image: np.ndarray,
        render_hash: str,
        languages: List[str] = None
    ) -> Tuple[str, float, Dict]:
        """
        Process image with all engines and fuse results.
        
        Args:
            image: Image as numpy array
            render_hash: Hash of rendered image
            languages: List of language codes (default: ['en'])
            
        Returns:
            (fused_text, fused_confidence, metadata)
        """
        if languages is None:
            languages = ['en']
        
        # Run all engines
        start_time = time.time()
        engine_results = self.run_engines_parallel(image, render_hash, languages)
        
        # Fuse results
        fused_text, fused_conf, provenance = self.fuse_results(engine_results)
        
        total_time = time.time() - start_time
        
        # Build metadata
        metadata = {
            'engines': {
                name: {
                    'success': result.success,
                    'confidence': result.confidence,
                    'processing_time': result.processing_time,
                    'text_length': len(result.text) if result.success else 0,
                    'error': result.error
                }
                for name, result in engine_results.items()
            },
            'fusion': {
                'text_length': len(fused_text),
                'confidence': fused_conf,
                'provenance': provenance
            },
            'timing': {
                'total': total_time,
                'fusion_only': total_time - max(
                    (r.processing_time for r in engine_results.values()),
                    default=0.0
                )
            }
        }
        
        return fused_text, fused_conf, metadata
    
    def get_statistics(self) -> Dict:
        """Get orchestrator statistics."""
        stats = dict(self.stats)
        
        # Calculate success rates
        stats['success_rates'] = {}
        for engine_name in self.engine_configs.keys():
            runs = stats['engine_runs'][engine_name]
            if runs > 0:
                successes = stats['engine_successes'][engine_name]
                stats['success_rates'][engine_name] = successes / runs
        
        # Cache hit rate
        total_cache_ops = stats['cache_hits'] + stats['cache_misses']
        if total_cache_ops > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / total_cache_ops
        else:
            stats['cache_hit_rate'] = 0.0
        
        return stats
    
    def log_statistics(self):
        """Log current statistics."""
        stats = self.get_statistics()
        
        logger.info("=== Multi-Engine Orchestrator Statistics ===")
        logger.info(f"Total fusion runs: {stats['fusion_runs']}")
        logger.info(f"Cache hit rate: {stats['cache_hit_rate']:.1%}")
        
        for engine_name in sorted(self.engine_configs.keys()):
            runs = stats['engine_runs'][engine_name]
            successes = stats['engine_successes'][engine_name]
            failures = stats['engine_failures'][engine_name]
            timeouts = stats['engine_timeouts'][engine_name]
            success_rate = stats['success_rates'].get(engine_name, 0.0)
            
            logger.info(
                f"{engine_name:12s}: {runs:3d} runs, "
                f"{successes:3d} success, {failures:3d} fail, {timeouts:3d} timeout "
                f"({success_rate:.1%})"
            )
