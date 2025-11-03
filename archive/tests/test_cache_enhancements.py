#!/usr/bin/env python3
"""
Unit tests for cache store enhancements (4-stage caching).

Tests the new cache methods for OCR, fusion, and LLM stages.
"""

import json
import tempfile
import unittest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from cache_store import CacheStore


class TestCacheStoreEnhancements(unittest.TestCase):
    """Test enhanced cache store with 4-stage caching."""
    
    def setUp(self):
        """Set up test cache store with temp directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = CacheStore(
            cache_dir=self.temp_dir,
            max_size_gb=1.0,
            enabled=True
        )
    
    def tearDown(self):
        """Clean up temp directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_ocr_cache_roundtrip(self):
        """Test OCR result caching."""
        # Prepare test data
        render_hash = "abc123def456"
        engine = "paddle"
        version = "2.7.3"
        languages = ["en", "de"]
        
        result = {
            "text": "Sample OCR text",
            "lines": [
                {"text": "Line 1", "bbox": [10, 20, 100, 40], "conf": 0.95},
                {"text": "Line 2", "bbox": [10, 50, 100, 70], "conf": 0.88}
            ],
            "confidence": 0.915
        }
        
        # Store
        success = self.cache.put_ocr_result(
            render_hash=render_hash,
            engine=engine,
            engine_version=version,
            languages=languages,
            result=result
        )
        self.assertTrue(success)
        
        # Retrieve
        retrieved = self.cache.get_ocr_result(
            render_hash=render_hash,
            engine=engine,
            engine_version=version,
            languages=languages
        )
        
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["text"], result["text"])
        self.assertEqual(len(retrieved["lines"]), 2)
        self.assertEqual(retrieved["confidence"], result["confidence"])
    
    def test_ocr_cache_miss(self):
        """Test OCR cache miss for non-existent entry."""
        result = self.cache.get_ocr_result(
            render_hash="nonexistent",
            engine="paddle",
            engine_version="1.0",
            languages=["en"]
        )
        self.assertIsNone(result)
    
    def test_ocr_cache_language_sensitivity(self):
        """Test that OCR cache is sensitive to language changes."""
        render_hash = "test123"
        engine = "doctr"
        version = "0.6.0"
        
        result_en = {"text": "English text", "lang": "en"}
        result_de = {"text": "German text", "lang": "de"}
        
        # Store English
        self.cache.put_ocr_result(
            render_hash=render_hash,
            engine=engine,
            engine_version=version,
            languages=["en"],
            result=result_en
        )
        
        # Store German
        self.cache.put_ocr_result(
            render_hash=render_hash,
            engine=engine,
            engine_version=version,
            languages=["de"],
            result=result_de
        )
        
        # Retrieve should get correct ones
        retrieved_en = self.cache.get_ocr_result(
            render_hash=render_hash,
            engine=engine,
            engine_version=version,
            languages=["en"]
        )
        
        retrieved_de = self.cache.get_ocr_result(
            render_hash=render_hash,
            engine=engine,
            engine_version=version,
            languages=["de"]
        )
        
        self.assertEqual(retrieved_en["text"], "English text")
        self.assertEqual(retrieved_de["text"], "German text")
    
    def test_fusion_cache_roundtrip(self):
        """Test fusion result caching."""
        engine_hashes = ["hash1_paddle", "hash2_doctr", "hash3_mmocr", "hash4_kraken"]
        
        fusion_result = {
            "text": "Fused consensus text",
            "provenance": {
                "char_0_5": "paddle",
                "char_6_10": "doctr",
                "char_11_15": "rover_consensus"
            },
            "confidence": 0.92,
            "engines_used": ["paddle", "doctr", "mmocr", "kraken"]
        }
        
        # Store
        success = self.cache.put_fusion_result(
            engine_hashes=engine_hashes,
            result=fusion_result
        )
        self.assertTrue(success)
        
        # Retrieve
        retrieved = self.cache.get_fusion_result(engine_hashes=engine_hashes)
        
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["text"], fusion_result["text"])
        self.assertEqual(retrieved["confidence"], 0.92)
        self.assertEqual(len(retrieved["engines_used"]), 4)
    
    def test_fusion_cache_order_independence(self):
        """Test that fusion cache is order-independent (sorted internally)."""
        hashes_ordered = ["hash1", "hash2", "hash3"]
        hashes_reversed = ["hash3", "hash2", "hash1"]
        
        result = {"text": "Order should not matter"}
        
        # Store with one order
        self.cache.put_fusion_result(
            engine_hashes=hashes_ordered,
            result=result
        )
        
        # Retrieve with different order - should get same result
        retrieved = self.cache.get_fusion_result(engine_hashes=hashes_reversed)
        
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["text"], "Order should not matter")
    
    def test_llm_cache_roundtrip(self):
        """Test LLM correction result caching."""
        model = "qwen2.5:7b-instruct"
        template_version = "v1.2"
        text_batch = "This is a test text with som OCR erors."
        
        llm_result = {
            "corrected_text": "This is a test text with some OCR errors.",
            "edit_distance": 2,
            "edit_ratio": 0.05,
            "changes": [
                {"original": "som", "corrected": "some", "position": 25},
                {"original": "erors", "corrected": "errors", "position": 38}
            ]
        }
        
        # Store
        success = self.cache.put_llm_result(
            model=model,
            prompt_template_version=template_version,
            normalized_text_batch=text_batch,
            result=llm_result
        )
        self.assertTrue(success)
        
        # Retrieve
        retrieved = self.cache.get_llm_result(
            model=model,
            prompt_template_version=template_version,
            normalized_text_batch=text_batch
        )
        
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["corrected_text"], llm_result["corrected_text"])
        self.assertEqual(retrieved["edit_distance"], 2)
        self.assertEqual(len(retrieved["changes"]), 2)
    
    def test_llm_cache_model_sensitivity(self):
        """Test that LLM cache is sensitive to model changes."""
        text = "Test text"
        template = "v1"
        
        result_qwen = {"corrected_text": "Qwen corrected", "model": "qwen"}
        result_mistral = {"corrected_text": "Mistral corrected", "model": "mistral"}
        
        # Store with qwen
        self.cache.put_llm_result("qwen2.5:7b", template, text, result_qwen)
        
        # Store with mistral
        self.cache.put_llm_result("mistral:7b", template, text, result_mistral)
        
        # Retrieve should get correct ones
        retrieved_qwen = self.cache.get_llm_result("qwen2.5:7b", template, text)
        retrieved_mistral = self.cache.get_llm_result("mistral:7b", template, text)
        
        self.assertEqual(retrieved_qwen["corrected_text"], "Qwen corrected")
        self.assertEqual(retrieved_mistral["corrected_text"], "Mistral corrected")
    
    def test_invalidate_stage_ocr(self):
        """Test selective invalidation of OCR cache."""
        # Store OCR result
        self.cache.put_ocr_result("hash1", "paddle", "2.7", ["en"], {"text": "OCR"})
        
        # Store fusion result
        self.cache.put_fusion_result(["hash1"], {"text": "Fusion"})
        
        # Invalidate OCR only
        count = self.cache.invalidate("ocr")
        self.assertGreater(count, 0)
        
        # OCR should be gone
        self.assertIsNone(self.cache.get_ocr_result("hash1", "paddle", "2.7", ["en"]))
        
        # Fusion should still be there
        self.assertIsNotNone(self.cache.get_fusion_result(["hash1"]))
    
    def test_invalidate_stage_fusion(self):
        """Test selective invalidation of fusion cache."""
        # Store both
        self.cache.put_ocr_result("hash1", "paddle", "2.7", ["en"], {"text": "OCR"})
        self.cache.put_fusion_result(["hash1"], {"text": "Fusion"})
        
        # Invalidate fusion only
        count = self.cache.invalidate("fusion")
        self.assertGreater(count, 0)
        
        # Fusion should be gone
        self.assertIsNone(self.cache.get_fusion_result(["hash1"]))
        
        # OCR should still be there
        self.assertIsNotNone(self.cache.get_ocr_result("hash1", "paddle", "2.7", ["en"]))
    
    def test_invalidate_all(self):
        """Test invalidation of all cache stages."""
        # Store everything
        self.cache.put_ocr_result("hash1", "paddle", "2.7", ["en"], {"text": "OCR"})
        self.cache.put_fusion_result(["hash1"], {"text": "Fusion"})
        self.cache.put_llm_result("qwen", "v1", "text", {"text": "LLM"})
        
        # Invalidate all
        count = self.cache.invalidate("all")
        self.assertGreater(count, 0)
        
        # Everything should be gone
        self.assertIsNone(self.cache.get_ocr_result("hash1", "paddle", "2.7", ["en"]))
        self.assertIsNone(self.cache.get_fusion_result(["hash1"]))
        self.assertIsNone(self.cache.get_llm_result("qwen", "v1", "text"))
    
    def test_cache_stats(self):
        """Test cache statistics tracking."""
        # Initial stats
        stats = self.cache.get_stats()
        initial_hits = stats.get('hits', 0)
        initial_misses = stats.get('misses', 0)
        
        # Trigger a miss
        self.cache.get_ocr_result("nonexistent", "engine", "1.0", ["en"])
        
        # Store and retrieve (hit)
        self.cache.put_ocr_result("test", "engine", "1.0", ["en"], {"text": "test"})
        self.cache.get_ocr_result("test", "engine", "1.0", ["en"])
        
        # Check stats updated
        stats = self.cache.get_stats()
        self.assertGreater(stats['misses'], initial_misses)
        self.assertGreater(stats['hits'], initial_hits)


if __name__ == '__main__':
    unittest.main()
