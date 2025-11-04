#!/usr/bin/env python3
"""Quick test of trained Akkadian LM perplexity scores."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from python_char_lm import PythonCharLM

# Load model
lm = PythonCharLM()
lm.load('models/akkadian_char_lm.json')

# Test samples
test_samples = [
    ("Akkadian 1", "a-na-kam i-ma-at Sa-ar-la Sa-bu-a-i-û"),
    ("Akkadian 2", "KÙ.BABBAR ša-pí-il-tum"),
    ("Akkadian 3", "i-ma-at šu-ku-ul-tum"),
    ("Turkish", "Yüzey araştırmasına Prof.Dr. Kutlu Emre katılmıştır"),
    ("German", "Die Untersuchung der archäologischen Funde"),
    ("English", "The archaeological research project"),
]

print("=== Akkadian LM Perplexity Test ===\n")
print(f"Model: 3-gram, vocab={len(lm.vocab)}, n-grams={sum(len(c) for c in lm.ngram_counts.values())}\n")

for label, text in test_samples:
    perplexity = lm.perplexity(text)
    print(f"{label:15} | PPL: {perplexity:7.2f} | {text[:50]}")

print("\n✓ Lower perplexity = better Akkadian fit")
print("✓ Detection threshold: PPL < 20 = strong, < 40 = weak")
