"""
Deterministic cache layer for OCR pipeline artifacts.
Ensures byte-for-byte reproducibility with SHA-based verification.
"""

import hashlib
import json
import logging
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
import numpy as np
from PIL import Image
import io

logger = logging.getLogger(__name__)


class CacheStore:
    """
    Deterministic cache for pipeline artifacts.
    
    Keys are based on: (pdf_sha1, page_num, dpi, config_hash)
    Stores: rendered images, orientation angles, detection boxes
    Verifies round-trip equality via SHA256 over pixel bytes.
    """
    
    def __init__(self, cache_dir: str = "cache/pipeline", max_size_gb: float = 10.0, enabled: bool = True):
        """
        Initialize cache store.
        
        Args:
            cache_dir: Directory for cache storage
            max_size_gb: Maximum cache size in GB
            enabled: Whether caching is enabled
        """
        self.cache_dir = Path(cache_dir)
        self.max_size_gb = max_size_gb
        self.enabled = enabled
        
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._stats = {
                'hits': 0,
                'misses': 0,
                'stores': 0,
                'evictions': 0,
                'parity_failures': 0
            }
        
        logger.info(f"Cache store initialized: {cache_dir}, enabled={enabled}")
    
    def _compute_pdf_hash(self, pdf_path: str) -> str:
        """Compute SHA1 hash of PDF file."""
        sha1 = hashlib.sha1()
        with open(pdf_path, 'rb') as f:
            while chunk := f.read(8192):
                sha1.update(chunk)
        return sha1.hexdigest()
    
    def _compute_config_hash(self, config: Dict[str, Any]) -> str:
        """Compute hash of configuration dict."""
        # Sort keys for deterministic hash
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def _compute_image_hash(self, image: np.ndarray) -> str:
        """Compute SHA256 hash of image pixel data."""
        return hashlib.sha256(image.tobytes()).hexdigest()
    
    def _make_key(self, pdf_hash: str, page_num: int, artifact_type: str, config_hash: str = "") -> str:
        """
        Generate cache key.
        
        Args:
            pdf_hash: SHA1 hash of PDF file
            page_num: Page number (1-indexed)
            artifact_type: Type of artifact (render, orientation, detection)
            config_hash: Hash of relevant config
        
        Returns:
            Cache key string
        """
        parts = [pdf_hash[:16], f"p{page_num}", artifact_type]
        if config_hash:
            parts.append(config_hash[:8])
        return "_".join(parts)
    
    def _get_cache_path(self, key: str, extension: str = "pkl") -> Path:
        """Get filesystem path for cache key."""
        # Use first 2 chars of key for sharding
        shard = key[:2]
        shard_dir = self.cache_dir / shard
        shard_dir.mkdir(exist_ok=True)
        return shard_dir / f"{key}.{extension}"
    
    def get_rendered_image(
        self,
        pdf_path: str,
        page_num: int,
        dpi: int,
        verify: bool = True
    ) -> Optional[np.ndarray]:
        """
        Get cached rendered image.
        
        Args:
            pdf_path: Path to PDF file
            page_num: Page number
            dpi: Rendering DPI
            verify: Whether to verify image hash
        
        Returns:
            Rendered image as numpy array, or None if not cached
        """
        if not self.enabled:
            return None
        
        try:
            pdf_hash = self._compute_pdf_hash(pdf_path)
            config_hash = self._compute_config_hash({'dpi': dpi})
            key = self._make_key(pdf_hash, page_num, "render", config_hash)
            cache_path = self._get_cache_path(key, "npz")
            
            if not cache_path.exists():
                self._stats['misses'] += 1
                return None
            
            # Load image
            data = np.load(cache_path)
            image = data['image']
            stored_hash = str(data.get('hash', ''))
            
            # Verify integrity if requested
            if verify and stored_hash:
                computed_hash = self._compute_image_hash(image)
                if computed_hash != stored_hash:
                    logger.warning(f"Cache parity failure for {key}: hash mismatch")
                    self._stats['parity_failures'] += 1
                    cache_path.unlink()
                    return None
            
            self._stats['hits'] += 1
            logger.debug(f"Cache hit: render page {page_num}")
            return image
            
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
            return None
    
    def put_rendered_image(
        self,
        pdf_path: str,
        page_num: int,
        dpi: int,
        image: np.ndarray
    ) -> bool:
        """
        Store rendered image in cache.
        
        Args:
            pdf_path: Path to PDF file
            page_num: Page number
            dpi: Rendering DPI
            image: Rendered image
        
        Returns:
            True if stored successfully
        """
        if not self.enabled:
            return False
        
        try:
            pdf_hash = self._compute_pdf_hash(pdf_path)
            config_hash = self._compute_config_hash({'dpi': dpi})
            key = self._make_key(pdf_hash, page_num, "render", config_hash)
            cache_path = self._get_cache_path(key, "npz")
            
            # Compute hash for verification
            image_hash = self._compute_image_hash(image)
            
            # Store with hash
            np.savez_compressed(
                cache_path,
                image=image,
                hash=image_hash,
                dpi=dpi,
                page_num=page_num
            )
            
            self._stats['stores'] += 1
            logger.debug(f"Cache store: render page {page_num}")
            return True
            
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
            return False
    
    def get_orientation(
        self,
        pdf_path: str,
        page_num: int,
        image_hash: str
    ) -> Optional[Tuple[float, np.ndarray]]:
        """
        Get cached orientation result.
        
        Args:
            pdf_path: Path to PDF
            page_num: Page number
            image_hash: Hash of input image
        
        Returns:
            (angle, deskewed_image) or None
        """
        if not self.enabled:
            return None
        
        try:
            pdf_hash = self._compute_pdf_hash(pdf_path)
            key = self._make_key(pdf_hash, page_num, "orient", image_hash[:8])
            cache_path = self._get_cache_path(key, "npz")
            
            if not cache_path.exists():
                self._stats['misses'] += 1
                return None
            
            data = np.load(cache_path)
            angle = float(data['angle'])
            deskewed = data['deskewed_image']
            
            self._stats['hits'] += 1
            logger.debug(f"Cache hit: orientation page {page_num}")
            return (angle, deskewed)
            
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
            return None
    
    def put_orientation(
        self,
        pdf_path: str,
        page_num: int,
        image_hash: str,
        angle: float,
        deskewed_image: np.ndarray
    ) -> bool:
        """Store orientation result in cache."""
        if not self.enabled:
            return False
        
        try:
            pdf_hash = self._compute_pdf_hash(pdf_path)
            key = self._make_key(pdf_hash, page_num, "orient", image_hash[:8])
            cache_path = self._get_cache_path(key, "npz")
            
            np.savez_compressed(
                cache_path,
                angle=angle,
                deskewed_image=deskewed_image,
                page_num=page_num
            )
            
            self._stats['stores'] += 1
            logger.debug(f"Cache store: orientation page {page_num}")
            return True
            
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
            return False
    
    def get_detection_boxes(
        self,
        pdf_path: str,
        page_num: int,
        image_hash: str,
        config: Dict[str, Any]
    ) -> Optional[list]:
        """
        Get cached detection boxes.
        
        Args:
            pdf_path: Path to PDF
            page_num: Page number
            image_hash: Hash of input image
            config: Detection config (model, threshold, etc.)
        
        Returns:
            List of detection boxes or None
        """
        if not self.enabled:
            return None
        
        try:
            pdf_hash = self._compute_pdf_hash(pdf_path)
            config_hash = self._compute_config_hash(config)
            key = self._make_key(pdf_hash, page_num, "detect", f"{image_hash[:8]}_{config_hash[:8]}")
            cache_path = self._get_cache_path(key, "pkl")
            
            if not cache_path.exists():
                self._stats['misses'] += 1
                return None
            
            with open(cache_path, 'rb') as f:
                boxes = pickle.load(f)
            
            self._stats['hits'] += 1
            logger.debug(f"Cache hit: detection page {page_num}")
            return boxes
            
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
            return None
    
    def put_detection_boxes(
        self,
        pdf_path: str,
        page_num: int,
        image_hash: str,
        config: Dict[str, Any],
        boxes: list
    ) -> bool:
        """Store detection boxes in cache."""
        if not self.enabled:
            return False
        
        try:
            pdf_hash = self._compute_pdf_hash(pdf_path)
            config_hash = self._compute_config_hash(config)
            key = self._make_key(pdf_hash, page_num, "detect", f"{image_hash[:8]}_{config_hash[:8]}")
            cache_path = self._get_cache_path(key, "pkl")
            
            with open(cache_path, 'wb') as f:
                pickle.dump(boxes, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            self._stats['stores'] += 1
            logger.debug(f"Cache store: detection page {page_num}")
            return True
            
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
            return False
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        if not self.enabled:
            return {}
        
        stats = self._stats.copy()
        total = stats['hits'] + stats['misses']
        if total > 0:
            stats['hit_rate'] = stats['hits'] / total
        else:
            stats['hit_rate'] = 0.0
        return stats
    
    def get_size_mb(self) -> float:
        """Get current cache size in MB."""
        if not self.enabled:
            return 0.0
        
        total_size = sum(f.stat().st_size for f in self.cache_dir.rglob('*') if f.is_file())
        return total_size / (1024 * 1024)
    
    def evict_lru(self, target_mb: Optional[float] = None) -> int:
        """
        Evict least-recently-used cache entries.
        
        Args:
            target_mb: Target size in MB (defaults to max_size_gb)
        
        Returns:
            Number of entries evicted
        """
        if not self.enabled:
            return 0
        
        if target_mb is None:
            target_mb = self.max_size_gb * 1024
        
        # Get all cache files with access times
        files = [(f, f.stat().st_atime, f.stat().st_size) 
                for f in self.cache_dir.rglob('*') if f.is_file()]
        
        # Sort by access time (oldest first)
        files.sort(key=lambda x: x[1])
        
        current_size_mb = sum(size for _, _, size in files) / (1024 * 1024)
        evicted = 0
        
        for file_path, _, size in files:
            if current_size_mb <= target_mb:
                break
            
            try:
                file_path.unlink()
                current_size_mb -= size / (1024 * 1024)
                evicted += 1
            except Exception as e:
                logger.warning(f"Failed to evict {file_path}: {e}")
        
        if evicted > 0:
            self._stats['evictions'] += evicted
            logger.info(f"Evicted {evicted} cache entries, size: {current_size_mb:.1f} MB")
        
        return evicted
    
    def get_ocr_result(
        self,
        render_hash: str,
        engine: str,
        engine_version: str,
        languages: List[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached OCR result for a specific engine.
        
        Args:
            render_hash: Hash of the rendered image
            engine: Engine name (paddle, doctr, etc.)
            engine_version: Engine version string
            languages: List of languages
        
        Returns:
            OCR result dict or None
        """
        if not self.enabled:
            return None
        
        try:
            # Create content-addressed key: sha1(render_hash|engine|version|langs)
            langs_str = ','.join(sorted(languages))
            key_input = f"{render_hash}|{engine}|{engine_version}|{langs_str}"
            key = hashlib.sha1(key_input.encode()).hexdigest()[:24]
            cache_path = self._get_cache_path(f"ocr_{key}", "json")
            
            if not cache_path.exists():
                self._stats['misses'] += 1
                return None
            
            with open(cache_path, 'r', encoding='utf-8') as f:
                result = json.load(f)
            
            self._stats['hits'] += 1
            logger.debug(f"Cache hit: OCR {engine}")
            return result
            
        except Exception as e:
            logger.warning(f"Cache read error (OCR): {e}")
            return None
    
    def put_ocr_result(
        self,
        render_hash: str,
        engine: str,
        engine_version: str,
        languages: List[str],
        result: Dict[str, Any]
    ) -> bool:
        """Store OCR result in cache."""
        if not self.enabled:
            return False
        
        try:
            langs_str = ','.join(sorted(languages))
            key_input = f"{render_hash}|{engine}|{engine_version}|{langs_str}"
            key = hashlib.sha1(key_input.encode()).hexdigest()[:24]
            cache_path = self._get_cache_path(f"ocr_{key}", "json")
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            self._stats['stores'] += 1
            logger.debug(f"Cache store: OCR {engine}")
            return True
            
        except Exception as e:
            logger.warning(f"Cache write error (OCR): {e}")
            return False
    
    def get_fusion_result(
        self,
        engine_hashes: List[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached fusion result.
        
        Args:
            engine_hashes: List of OCR result hashes (sorted)
        
        Returns:
            Fusion result dict or None
        """
        if not self.enabled:
            return None
        
        try:
            # Create content-addressed key: sha1(sorted(engine_hashes))
            sorted_hashes = ','.join(sorted(engine_hashes))
            key = hashlib.sha1(sorted_hashes.encode()).hexdigest()[:24]
            cache_path = self._get_cache_path(f"fusion_{key}", "json")
            
            if not cache_path.exists():
                self._stats['misses'] += 1
                return None
            
            with open(cache_path, 'r', encoding='utf-8') as f:
                result = json.load(f)
            
            self._stats['hits'] += 1
            logger.debug("Cache hit: fusion")
            return result
            
        except Exception as e:
            logger.warning(f"Cache read error (fusion): {e}")
            return None
    
    def put_fusion_result(
        self,
        engine_hashes: List[str],
        result: Dict[str, Any]
    ) -> bool:
        """Store fusion result in cache."""
        if not self.enabled:
            return False
        
        try:
            sorted_hashes = ','.join(sorted(engine_hashes))
            key = hashlib.sha1(sorted_hashes.encode()).hexdigest()[:24]
            cache_path = self._get_cache_path(f"fusion_{key}", "json")
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            self._stats['stores'] += 1
            logger.debug("Cache store: fusion")
            return True
            
        except Exception as e:
            logger.warning(f"Cache write error (fusion): {e}")
            return False
    
    def get_llm_result(
        self,
        model: str,
        prompt_template_version: str,
        normalized_text_batch: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached LLM correction result.
        
        Args:
            model: LLM model identifier
            prompt_template_version: Version of prompt template
            normalized_text_batch: Normalized input text batch
        
        Returns:
            LLM result dict or None
        """
        if not self.enabled:
            return None
        
        try:
            # Create content-addressed key: sha1(model|template_version|text)
            key_input = f"{model}|{prompt_template_version}|{normalized_text_batch}"
            key = hashlib.sha1(key_input.encode()).hexdigest()[:24]
            cache_path = self._get_cache_path(f"llm_{key}", "json")
            
            if not cache_path.exists():
                self._stats['misses'] += 1
                return None
            
            with open(cache_path, 'r', encoding='utf-8') as f:
                result = json.load(f)
            
            self._stats['hits'] += 1
            logger.debug("Cache hit: LLM")
            return result
            
        except Exception as e:
            logger.warning(f"Cache read error (LLM): {e}")
            return None
    
    def put_llm_result(
        self,
        model: str,
        prompt_template_version: str,
        normalized_text_batch: str,
        result: Dict[str, Any]
    ) -> bool:
        """Store LLM correction result in cache."""
        if not self.enabled:
            return False
        
        try:
            key_input = f"{model}|{prompt_template_version}|{normalized_text_batch}"
            key = hashlib.sha1(key_input.encode()).hexdigest()[:24]
            cache_path = self._get_cache_path(f"llm_{key}", "json")
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            self._stats['stores'] += 1
            logger.debug("Cache store: LLM")
            return True
            
        except Exception as e:
            logger.warning(f"Cache write error (LLM): {e}")
            return False
    
    def invalidate(self, stage: str = 'all') -> int:
        """
        Invalidate cache for specific stage(s).
        
        Args:
            stage: Stage to invalidate ('render', 'ocr', 'fusion', 'llm', 'all')
        
        Returns:
            Number of entries invalidated
        """
        if not self.enabled:
            return 0
        
        count = 0
        
        if stage == 'all':
            return self.clear()
        
        # Map stage names to file prefixes
        stage_prefixes = {
            'render': ['render_'],
            'orientation': ['orient_'],
            'detection': ['detect_'],
            'ocr': ['ocr_'],
            'fusion': ['fusion_'],
            'llm': ['llm_']
        }
        
        if stage not in stage_prefixes:
            logger.warning(f"Unknown stage '{stage}'. Valid: {list(stage_prefixes.keys()) + ['all']}")
            return 0
        
        prefixes = stage_prefixes[stage]
        
        for f in self.cache_dir.rglob('*'):
            if f.is_file():
                # Check if filename starts with any of the stage prefixes
                if any(f.name.startswith(prefix) for prefix in prefixes):
                    try:
                        f.unlink()
                        count += 1
                    except Exception as e:
                        logger.warning(f"Failed to delete {f}: {e}")
        
        logger.info(f"Invalidated {count} cache entries for stage '{stage}'")
        return count
    
    def clear(self) -> int:
        """Clear all cache entries. Returns number cleared."""
        if not self.enabled:
            return 0
        
        count = 0
        for f in self.cache_dir.rglob('*'):
            if f.is_file():
                try:
                    f.unlink()
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete {f}: {e}")
        
        logger.info(f"Cleared {count} cache entries")
        return count
