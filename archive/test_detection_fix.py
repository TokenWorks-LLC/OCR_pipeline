"""
Quick test to verify Akkadian detection config is working correctly.
Tests both false positives (Turkish/German) and true positives (Akkadian).
"""

import sys
sys.path.insert(0, 'src')

from lang_and_akkadian import is_akkadian_transliteration

# Test cases
print("=" * 60)
print("AKKADIAN DETECTION FIX VALIDATION")
print("=" * 60)

# FALSE POSITIVES (should NOT be detected as Akkadian with strict config)
false_positives = [
    ("Turkish prose", "Yüzey araştırmasına Prof. Dr. Tahsin Özgüç'ün başkanlığında"),
    ("German prose", "EIN NEUES ALTASSYRISCHES RECHTSDOKUMENT AUS KANES"),
    ("Turkish academic", "Bu tablo üzerinde birçok satır bulunmaktadır"),
    ("Hyphenated Turkish", "a-qul şeklinde yazılmıştır"),
]

# TRUE POSITIVES (should be detected as Akkadian with strict config)
true_positives = [
    ("AKT 4 text", "A-du-da DUMU ru-ba-im"),
    ("AKT 4b text", "sa-ru-pá-am i-sé-er"),
    ("With determinative", "DUMU ru-ba-im KÙ.BABBAR"),
    ("With diacritics", "sé-pá-am lu-ú-ší-ib"),
]

# Test with strict config (threshold 0.20, require diacritic/marker)
strict_config = {
    "threshold": 0.20,
    "require_diacritic_or_marker": True
}

print("\n### FALSE POSITIVE TESTS (should all be False) ###\n")
fp_count = 0
for label, text in false_positives:
    is_akk, score = is_akkadian_transliteration(text, config=strict_config)
    result = "❌ FAIL (FP)" if is_akk else "✅ PASS"
    print(f"{result} | {label:20s} | score={score:.3f} | {text[:50]}")
    if is_akk:
        fp_count += 1

print(f"\nFalse Positive Rate: {fp_count}/{len(false_positives)} = {100*fp_count/len(false_positives):.1f}%")

print("\n### TRUE POSITIVE TESTS (should all be True) ###\n")
tp_count = 0
for label, text in true_positives:
    is_akk, score = is_akkadian_transliteration(text, config=strict_config)
    result = "✅ PASS" if is_akk else "❌ FAIL (FN)"
    print(f"{result} | {label:20s} | score={score:.3f} | {text[:50]}")
    if is_akk:
        tp_count += 1

print(f"\nTrue Positive Rate: {tp_count}/{len(true_positives)} = {100*tp_count/len(true_positives):.1f}%")

# Test with old config (threshold 0.20, no diacritic requirement)
print("\n" + "=" * 60)
print("COMPARISON: OLD CONFIG (threshold=0.20, no requirements)")
print("=" * 60)

old_config = {
    "threshold": 0.20,
    "require_diacritic_or_marker": False
}

print("\n### FALSE POSITIVE TESTS WITH OLD CONFIG ###\n")
old_fp_count = 0
for label, text in false_positives:
    is_akk, score = is_akkadian_transliteration(text, config=old_config)
    result = "❌ FP" if is_akk else "✅ OK"
    print(f"{result} | {label:20s} | score={score:.3f}")
    if is_akk:
        old_fp_count += 1

print(f"\nOld Config FP Rate: {old_fp_count}/{len(false_positives)} = {100*old_fp_count/len(false_positives):.1f}%")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Strict Config (0.65 threshold): {fp_count} false positives, {tp_count} true positives")
print(f"Old Config (0.20 threshold): {old_fp_count} false positives")
print(f"\nReduction in FP: {old_fp_count - fp_count} ({100*(old_fp_count-fp_count)/max(old_fp_count,1):.1f}% improvement)")

# GO/NO-GO decision
if fp_count == 0 and tp_count >= len(true_positives) * 0.75:
    print("\n✅ GO: Detection fix successful!")
else:
    print(f"\n⚠️  NO-GO: FP={fp_count}, TP={tp_count}/{len(true_positives)}")
