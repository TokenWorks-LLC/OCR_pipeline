"""
Debug scoring for Akkadian texts to find optimal threshold.
"""

import sys
sys.path.insert(0, 'src')

from lang_and_akkadian import is_akkadian_transliteration

# Test texts with expected outcomes
test_cases = [
    # TRUE AKKADIAN
    ("A-du-da DUMU ru-ba-im", True, "AKT 4 - syllabic + determinative"),
    ("sa-ru-pá-am i-sé-er", True, "AKT 4b - syllabic + diacritics"),
    ("DUMU ru-ba-im KÙ.BABBAR", True, "Determinatives + syllabic"),
    ("sé-pá-am lu-ú-ší-ib", True, "Pure syllabic with diacritics"),
    ("i-na É.GAL-lim", True, "Syllabic + logogram"),
    ("a-na A-šur-i-mi-ti", True, "Long personal name"),
    
    # FALSE POSITIVES (Turkish/German)
    ("Yüzey araştırmasına Prof. Dr. Tahsin Özgüç'ün başkanlığında", False, "Turkish prose"),
    ("EIN NEUES ALTASSYRISCHES RECHTSDOKUMENT AUS KANES", False, "German capitals"),
    ("Bu tablo üzerinde birçok satır bulunmaktadır", False, "Turkish academic"),
    ("a-qul şeklinde yazılmıştır", False, "Turkish with hyphen"),
    ("Prof. Dr. ve Doç. Dr. ile birlikte", False, "Turkish academic titles"),
]

# Test with threshold sweep
config_base = {
    "require_diacritic_or_marker": True
}

print("=" * 80)
print("SCORING ANALYSIS")
print("=" * 80)

print("\n### INDIVIDUAL SCORES ###\n")
scores = []
for text, expected, label in test_cases:
    cfg = {**config_base, "threshold": 0.0}  # Just get score
    is_akk, score = is_akkadian_transliteration(text, config=cfg)
    scores.append((score, expected, label, text[:60]))
    expect_str = "AKKA" if expected else "NOT-"
    print(f"[{expect_str}] {score:.3f} | {label:30s} | {text[:60]}")

# Find optimal threshold
print("\n\n### THRESHOLD SWEEP ###\n")
print(f"{'Threshold':<12} {'TP':<6} {'FP':<6} {'FN':<6} {'TN':<6} {'Precision':<12} {'Recall':<12}")
print("-" * 80)

for threshold in [0.20, 0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
    tp = fp = fn = tn = 0
    for score, expected_akk, _, _ in scores:
        detected = score >= threshold
        if detected and expected_akk:
            tp += 1
        elif detected and not expected_akk:
            fp += 1
        elif not detected and expected_akk:
            fn += 1
        else:
            tn += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"{threshold:<12.2f} {tp:<6} {fp:<6} {fn:<6} {tn:<6} {precision:<12.2f} {recall:<12.2f} (F1={f1:.2f})")

print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)

# Find best threshold with FP=0
best_threshold = None
best_recall = 0
for threshold in [round(x*0.05, 2) for x in range(1, 20)]:  # 0.05 to 0.95
    tp = fp = fn = 0
    for score, expected_akk, _, _ in scores:
        detected = score >= threshold
        if detected and expected_akk:
            tp += 1
        elif detected and not expected_akk:
            fp += 1
        elif not detected and expected_akk:
            fn += 1
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    if fp == 0 and recall > best_recall:
        best_recall = recall
        best_threshold = threshold

print(f"Best threshold with 0 FP: {best_threshold:.2f} (Recall={best_recall*100:.1f}%)")
print(f"Current threshold (0.65): Catches {100*1/6:.1f}% of true Akkadian")

# Show what we're missing at current threshold
print("\n### MISSED AKKADIAN AT THRESHOLD=0.65 ###\n")
for score, expected, label, text in scores:
    if expected and score < 0.65:
        print(f"Score {score:.3f} | {label:30s} | {text}")
