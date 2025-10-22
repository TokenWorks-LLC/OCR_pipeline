#!/usr/bin/env python3
"""
Test DeepSeek basic mode on sample image
"""
import sys
import os
import time
import cv2

# Add src to path
sys.path.insert(0, 'src')

from deepseek_ocr import ocr_deepseek_lines, DeepSeekOCRError

def test_deepseek_basic():
    print("=== TESTING DEEPSEEK BASIC MODE ===")

    # Check if test image exists
    test_image_path = 'data/input/test.png'
    if not os.path.exists(test_image_path):
        print(f"ERROR: Test image not found at {test_image_path}")
        return False

    print("Loading test image...")
    img = cv2.imread(test_image_path)
    if img is None:
        print("ERROR: Failed to load test image")
        return False

    print(f"Image loaded: {img.shape}")

    print("Testing DeepSeek-OCR (timeout: 300s)...")
    start_time = time.time()

    try:
        lines = ocr_deepseek_lines(img, timeout=300)
        duration = time.time() - start_time

        print(".1f"        print(f"Lines detected: {len(lines)}")

        if lines:
            print("Sample output:")
            for i, line in enumerate(lines[:3]):
                print(f"  {i+1}: {line.text[:50]}...")

        return True

    except DeepSeekOCRError as e:
        duration = time.time() - start_time
        print(".1f"        print(f"Error details: {e}")
        return False
    except Exception as e:
        duration = time.time() - start_time
        print(".1f"        print(f"Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_deepseek_basic()
    sys.exit(0 if success else 1)
