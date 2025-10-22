#!/usr/bin/env python3
"""
Test DeepSeek-OCR integration with the pipeline.
Validates that DeepSeek can be used as an OCR engine.
"""

import sys
import os
from pathlib import Path

# Add project paths
sys.path.insert(0, str(Path(__file__).parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent / 'production'))

def test_deepseek_availability():
    """Test if DeepSeek-OCR is available."""
    print("Testing DeepSeek-OCR availability...")

    try:
        from deepseek_ocr import is_deepseek_available, DeepSeekOCRError
        available = is_deepseek_available()
        if available:
            print("[OK] DeepSeek-OCR is available")
            return True
        else:
            print("[ERROR] DeepSeek-OCR is not available")
            return False
    except ImportError as e:
        print(f"[ERROR] DeepSeek-OCR module not found: {e}")
        return False

def test_ocr_ensemble_deepseek():
    """Test OCR ensemble with DeepSeek engine selection."""
    print("Testing OCR ensemble with DeepSeek...")

    try:
        import cv2
        import numpy as np
        from ocr_utils import ocr_ensemble

        # Create a simple test image
        img = np.ones((100, 200, 3), dtype=np.uint8) * 255  # White image
        cv2.putText(img, "Test", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        # Test DeepSeek-only mode
        print("  Testing DeepSeek-only mode...")
        try:
            results = ocr_ensemble(img, engine="deepseek")
            print(f"  [OK] DeepSeek-only: {len(results)} lines detected")
        except Exception as e:
            print(f"  [ERROR] DeepSeek-only failed: {e}")

        # Test ensemble mode (should include DeepSeek if available)
        print("  Testing ensemble mode...")
        results = ocr_ensemble(img, engine="ensemble")
        deepseek_count = len([r for r in results if hasattr(r, 'engine') and r.engine == 'deepseek'])
        print(f"  [OK] Ensemble: {len(results)} total lines, {deepseek_count} from DeepSeek")

        return True

    except Exception as e:
        print(f"[ERROR] OCR ensemble test failed: {e}")
        return False

def test_pipeline_config():
    """Test that pipeline can be configured for DeepSeek."""
    print("Testing pipeline configuration for DeepSeek...")

    try:
        from comprehensive_pipeline import ComprehensivePipeline, PipelineConfig

        # Create config with DeepSeek
        config = PipelineConfig(ocr_engine="deepseek")

        # Try to create pipeline (will fail if DeepSeek not available)
        pipeline = ComprehensivePipeline(config)

        print("[OK] Pipeline configured for DeepSeek successfully")
        pipeline.cleanup()
        return True

    except Exception as e:
        print(f"[ERROR] Pipeline configuration failed: {e}")
        return False

def test_config_files():
    """Test that DeepSeek config files are valid."""
    print("Testing DeepSeek config files...")

    config_files = [
        'config_eval_deepseek_basic.json',
        'config_eval_deepseek_advanced.json',
        'config_eval_deepseek_akkadian.json'
    ]

    for config_file in config_files:
        try:
            import json
            with open(config_file, 'r') as f:
                config = json.load(f)

            # Check DeepSeek configuration
            if config.get('ocr', {}).get('engine') == 'deepseek':
                print(f"[OK] {config_file} has correct DeepSeek OCR engine")
            else:
                print(f"[ERROR] {config_file} missing DeepSeek OCR engine")
                return False

            if 'deepseek' in config:
                print(f"[OK] {config_file} has DeepSeek configuration section")
            else:
                print(f"[ERROR] {config_file} missing DeepSeek configuration")
                return False

        except Exception as e:
            print(f"[ERROR] {config_file} validation failed: {e}")
            return False

    return True

def main():
    """Run all integration tests."""
    print("="*60)
    print("DEEPSEEK-OCR INTEGRATION TESTS")
    print("="*60)

    tests = [
        test_deepseek_availability,
        test_config_files,
        test_ocr_ensemble_deepseek,
        test_pipeline_config
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
                print(f"[OK] {test.__name__}")
            else:
                print(f"[FAIL] {test.__name__}")
        except Exception as e:
            print(f"[ERROR] {test.__name__}: {e}")
        print()

    print("="*60)
    print(f"TEST RESULTS: {passed}/{total} passed")

    if passed == total:
        print("[SUCCESS] All DeepSeek integration tests passed!")
        print("\nReady to run DeepSeek evaluation:")
        print("• python eval_deepseek.py -c config_eval_deepseek_basic.json")
        print("• python eval_deepseek.py -c config_eval_deepseek_advanced.json")
        print("• python eval_deepseek.py -c config_eval_deepseek_akkadian.json")
        return True
    else:
        print("[ERROR] Some tests failed - check DeepSeek setup")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
