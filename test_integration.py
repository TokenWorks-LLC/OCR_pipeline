#!/usr/bin/env python3
"""
Integration test for multi-engine OCR implementation.
Tests engine availability, factory pattern, and pipeline integration.
"""
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_engine_availability():
    """Test engine availability detection."""
    print("🔍 Testing Engine Availability...")
    
    try:
        from engines import ENGINE_AVAILABILITY
        
        print(f"  PaddleOCR: {'✅ Available' if ENGINE_AVAILABILITY['paddle'] else '❌ Not Available'}")
        print(f"  docTR: {'✅ Available' if ENGINE_AVAILABILITY['doctr'] else '❌ Not Available'}")
        print(f"  MMOCR: {'✅ Available' if ENGINE_AVAILABILITY['mmocr'] else '❌ Not Available'}")
        print(f"  Kraken: {'✅ Available' if ENGINE_AVAILABILITY['kraken'] else '❌ Not Available'}")
        
        return True
    except ImportError as e:
        print(f"❌ Engine availability test failed: {e}")
        return False

def test_factory_pattern():
    """Test engine factory pattern."""
    print("\n🏭 Testing Factory Pattern...")
    
    try:
        from engines import create_engine
        
        # Test with mock engine (should always work)
        mock_engine = create_engine('mock')
        if mock_engine:
            print("  ✅ Mock engine creation successful")
        else:
            print("  ❌ Mock engine creation failed")
            return False
            
        # Test with unavailable engine (should return None gracefully)
        invalid_engine = create_engine('nonexistent')
        if invalid_engine is None:
            print("  ✅ Invalid engine handled gracefully")
        else:
            print("  ❌ Invalid engine should return None")
            return False
            
        return True
    except Exception as e:
        print(f"❌ Factory pattern test failed: {e}")
        return False

def test_config_system():
    """Test configuration system."""
    print("\n⚙️ Testing Configuration System...")
    
    try:
        from config import get_ocr_engine_config, OCR_PROFILES
        
        # Test default config
        default_config = get_ocr_engine_config()
        print(f"  ✅ Default config: {default_config['engine']}")
        
        # Test profile configs
        for profile_name in OCR_PROFILES:
            profile_config = get_ocr_engine_config(profile_name)
            print(f"  ✅ {profile_name.capitalize()} profile: {profile_config['engine']}")
            
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_ocr_utils_integration():
    """Test OCR utils integration."""
    print("\n🔧 Testing OCR Utils Integration...")
    
    try:
        from ocr_utils import ocr_with_engine
        
        # Test function exists and is callable
        if callable(ocr_with_engine):
            print("  ✅ ocr_with_engine function available")
        else:
            print("  ❌ ocr_with_engine function not callable")
            return False
            
        return True
    except ImportError as e:
        print(f"❌ OCR utils integration test failed: {e}")
        return False

def test_pipeline_integration():
    """Test pipeline integration."""
    print("\n🔄 Testing Pipeline Integration...")
    
    try:
        from pipeline import process_image
        import inspect
        
        # Check if process_image accepts engine parameters
        sig = inspect.signature(process_image)
        params = list(sig.parameters.keys())
        
        if 'engine_name' in params:
            print("  ✅ Pipeline supports engine_name parameter")
        else:
            print("  ❌ Pipeline missing engine_name parameter")
            return False
            
        if 'engine_config' in params:
            print("  ✅ Pipeline supports engine_config parameter")
        else:
            print("  ❌ Pipeline missing engine_config parameter")
            return False
            
        return True
    except Exception as e:
        print(f"❌ Pipeline integration test failed: {e}")
        return False

def main():
    """Run all integration tests."""
    print("🧪 Multi-Engine OCR Integration Tests\n")
    print("=" * 50)
    
    tests = [
        test_engine_availability,
        test_factory_pattern,
        test_config_system,
        test_ocr_utils_integration,
        test_pipeline_integration,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All integration tests passed! Multi-engine implementation ready.")
        return 0
    else:
        print("⚠️ Some tests failed. Check implementation before proceeding.")
        return 1

if __name__ == '__main__':
    sys.exit(main())