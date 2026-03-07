"""Spot-check Akkadian detections from client CSV."""
import csv
import re

csv_path = "reports/gold_full_20251008_1213/client_translations.csv"

# Read CSV
rows = list(csv.DictReader(open(csv_path, encoding='utf-8-sig')))

print(f"Total pairs in CSV: {len(rows)}")
print(f"\n{'='*80}")
print("SPOT-CHECK: First 15 Akkadian Detections")
print(f"{'='*80}\n")

# Akkadian markers for manual verification
MARKERS = {"DUMU", "LUGAL", "KÙ.BABBAR", "KUBABBAR", "URU", "É", "KUR", "KIIB", "DAM.QAR"}
DIACRITICS = set('šṣṭḫāēīūáíúà')

false_positives = []

for i, row in enumerate(rows[:15], 1):
    akk_text = row['akkadian_text']
    pdf = row['pdf_name']
    page = row['page']
    
    # Check for markers and diacritics
    has_marker = any(m in akk_text for m in MARKERS)
    has_diacritic = any(c in akk_text for c in DIACRITICS)
    has_syllabic = bool(re.search(r'\b[a-z]{1,4}-[a-z]{1,4}', akk_text, re.I))
    
    # Simple heuristic: if no markers AND no diacritics AND has Turkish/German words
    turkish_words = {"ve", "için", "ile", "bu", "bir", "da", "de"}
    german_words = {"und", "der", "die", "das", "mit", "für"}
    has_turkish = any(word in akk_text.lower() for word in turkish_words)
    has_german = any(word in akk_text.lower() for word in german_words)
    
    is_likely_fp = (not has_marker) and (not has_diacritic) and (has_turkish or has_german)
    
    status = "❌ FP?" if is_likely_fp else "✅ OK"
    
    print(f"{i}. {status} | {pdf} p.{page}")
    print(f"   Markers: {has_marker} | Diacritics: {has_diacritic} | Syllabic: {has_syllabic}")
    print(f"   Text: {akk_text[:100]}...")
    print()
    
    if is_likely_fp:
        false_positives.append((i, pdf, page, akk_text[:60]))

print(f"{'='*80}")
print(f"SPOT-CHECK RESULTS (n=15)")
print(f"{'='*80}")
print(f"False Positives: {len(false_positives)}/15 = {100*len(false_positives)/15:.1f}%")
print(f"Target: ≤10% (1-2 FPs acceptable)")
print()

if false_positives:
    print("⚠️  Potential False Positives:")
    for idx, pdf, page, text in false_positives:
        print(f"  {idx}. {pdf} p.{page}: {text}...")
else:
    print("✅ No obvious false positives detected!")

print(f"\n{'='*80}")
if len(false_positives) <= 2:
    print("✅ GO: False positive rate ≤10%")
else:
    print("⚠️  NO-GO: False positive rate >10% - needs tuning")
