#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick test of Akkadian detection with trained LM."""

import sys
import os
import logging
from pathlib import Path

# Set UTF-8 output
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Enable debug logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Set environment variable to load Akkadian LM
os.environ['AKKADIAN_LM_PATH'] = 'models/akkadian_char_lm.json'

from lang_and_akkadian import LanguageDetector

# Test samples
test_samples = [
    ("Akkadian 1", "a-na-kam i-ma-at Sa-ar-la Sa-bu-a-i-û ha-ra-na-am"),
    ("Akkadian 2", "KÙ.BABBAR ša-pí-il-tum i-ma-at"),
    ("Akkadian 3", "šu-ku-ul-tum a-na DINGIR"),
    ("Akkadian short", "a-na-ku"),
    ("Turkish", "Yüzey araştırmasına Prof.Dr. Kutlu Emre katılmıştır"),
    ("German", "Die Untersuchung der archäologischen Funde"),
    ("English", "The archaeological research project was completed"),
]

print("=== Akkadian Detection Test (with char LM) ===\n")

detector = LanguageDetector(akkadian_lm_path='models/akkadian_char_lm.json')

for label, text in test_samples:
    # Test Akkadian detection
    is_akk, akk_conf = detector.is_akkadian_transliteration(text)
    
    status = "[AKKADIAN]" if is_akk else "[NOT AKK ]"
    print(f"\n{label:15} | {status} | conf={akk_conf:.2f}")
    print(f"  Text: {text[:60]}")

print("\n=== Summary ===")
print("Detection working - Akkadian samples should show [AKKADIAN] status")
print("Threshold: confidence > 0.3 for positive detection")

