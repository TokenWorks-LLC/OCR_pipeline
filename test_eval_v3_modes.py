#!/usr/bin/env python3
"""
Test script for proper eval v3 modes - verifies the corrected implementation
"""

import sys
import os
import json
from pathlib import Path

# Add project paths
sys.path.insert(0, str(Path(__file__).parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent / 'production'))

def test_config_loading():
    """Test that the updated configs load correctly."""
    print("Testing config loading...")

    configs_to_test = [
        'config_eval_basic_v3.json',
        'config_eval_advanced_v3.json',
        'config_eval_akkadian_v3.json'
    ]

    for config_file in configs_to_test:
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)

            print(f"[OK] {config_file} loaded successfully")

            # Check that advanced and akkadian configs have LLM V3 enabled
            if 'advanced' in config_file or 'akkadian' in config_file:
                llm_config = config.get('llm', {})
                if llm_config.get('enable_llm_v3', False):
                    print(f"   [OK] LLM V3 enabled in {config_file}")
                else:
                    print(f"   [ERROR] LLM V3 not enabled in {config_file}")

            # Check for llm_v3 section
            if 'llm_v3' in config:
                print(f"   [OK] llm_v3 section found in {config_file}")
            else:
                print(f"   [ERROR] llm_v3 section missing in {config_file}")

        except Exception as e:
            print(f"[ERROR] Failed to load {config_file}: {e}")
            return False

    return True

def test_pipeline_initialization():
    """Test that pipeline initializes correctly with LLM V3."""
    print("Testing pipeline initialization...")

    try:
        from comprehensive_pipeline import ComprehensivePipeline, PipelineConfig

        # Test advanced_v3 config
        with open('config_eval_advanced_v3.json', 'r') as f:
            config_data = json.load(f)

        pipeline_config = PipelineConfig(
            enable_llm_v3=True,
            enable_llm_correction=True,
            llm_provider=config_data['llm']['provider'],
            llm_model=config_data['llm']['model_id'],
            llm_base_url=config_data['llm']['base_url'],
            llm_timeout=config_data['llm']['timeout'],
            max_concurrent_corrections=config_data['llm']['max_workers'],
            enable_reading_order=config_data['ocr']['enable_reading_order'],
            dpi=config_data['ocr']['dpi']
        )

        pipeline = ComprehensivePipeline(pipeline_config)

        # Check that LLM V3 corrector is initialized
        if hasattr(pipeline, 'llm_v3_corrector') and pipeline.llm_v3_corrector:
            print("[OK] LLM V3 corrector initialized successfully")
        else:
            print("[ERROR] LLM V3 corrector not initialized")
            return False

        # Check that legacy LLM corrector is not initialized (should be None)
        if pipeline.llm_corrector is None:
            print("[OK] Legacy LLM corrector correctly disabled")
        else:
            print("[ERROR] Legacy LLM corrector should be disabled when V3 is enabled")
            return False

        pipeline.cleanup()
        return True

    except Exception as e:
        print(f"[ERROR] Pipeline initialization failed: {e}")
        return False

def test_akkadian_mode():
    """Test that akkadian mode works with LLM V3."""
    print("Testing akkadian mode...")

    try:
        from comprehensive_pipeline import ComprehensivePipeline, PipelineConfig

        # Test akkadian_v3 config
        with open('config_eval_akkadian_v3.json', 'r') as f:
            config_data = json.load(f)

        pipeline_config = PipelineConfig(
            enable_llm_v3=True,
            enable_llm_correction=True,
            enable_akkadian_extraction=True,
            llm_provider=config_data['llm']['provider'],
            llm_model=config_data['llm']['model_id'],
            llm_base_url=config_data['llm']['base_url'],
            llm_timeout=config_data['llm']['timeout'],
            max_concurrent_corrections=config_data['llm']['max_workers'],
            enable_reading_order=config_data['ocr']['enable_reading_order'],
            dpi=config_data['ocr']['dpi']
        )

        pipeline = ComprehensivePipeline(pipeline_config)

        # Check that both LLM V3 and Akkadian extractor are initialized
        if pipeline.llm_v3_corrector:
            print("[OK] LLM V3 corrector initialized for akkadian mode")
        else:
            print("[ERROR] LLM V3 corrector not initialized for akkadian mode")
            return False

        if pipeline.akkadian_extractor:
            print("[OK] Akkadian extractor initialized for akkadian mode")
        else:
            print("[ERROR] Akkadian extractor not initialized for akkadian mode")
            return False

        pipeline.cleanup()
        return True

    except Exception as e:
        print(f"[ERROR] Akkadian mode test failed: {e}")
        return False

def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("EVAL V3 MODES INTEGRATION TESTS")
    print("="*60)

    tests = [
        test_config_loading,
        test_pipeline_initialization,
        test_akkadian_mode
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
                print(f"[OK] {test.__name__} PASSED")
            else:
                print(f"[ERROR] {test.__name__} FAILED")
        except Exception as e:
            print(f"[ERROR] {test.__name__} ERROR: {e}")
        print()

    print("="*60)
    print(f"TEST RESULTS: {passed}/{total} passed")

    if passed == total:
        print("SUCCESS: ALL TESTS PASSED!")
        print("\nReady for testing:")
        print("• python run_evaluation.py -c config_eval_basic_v3.json")
        print("• python run_evaluation.py -c config_eval_advanced_v3.json")
        print("• python run_evaluation.py -c config_eval_akkadian_v3.json")
        return True
    else:
        print("[ERROR] SOME TESTS FAILED")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
