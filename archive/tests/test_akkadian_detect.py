"""
Unit tests for Akkadian detection with config-aware gate.

Tests the fix for false positives: require_diacritic_or_marker gate + negative lexicon.
Validates 100% precision and 100% recall on canonical test cases.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lang_and_akkadian import is_akkadian_transliteration


class TestAkkadianDetection:
    """Test Akkadian detection with validated config."""
    
    def test_false_positives_eliminated(self):
        """Turkish/German prose should NOT be detected as Akkadian with strict config."""
        
        strict_config = {
            "threshold": 0.20,
            "require_diacritic_or_marker": True
        }
        
        false_positive_cases = [
            # Turkish academic prose
            "Yüzey araştırmasına Prof. Dr. Tahsin Özgüç'ün başkanlığında",
            # German title
            "EIN NEUES ALTASSYRISCHES RECHTSDOKUMENT AUS KANES",
            # Turkish scholarly text
            "Bu tablo üzerinde birçok satır bulunmaktadır",
            # Turkish with hyphenated word
            "a-qul şeklinde yazılmıştır",
        ]
        
        for text in false_positive_cases:
            is_akk, score = is_akkadian_transliteration(text, config=strict_config)
            assert not is_akk, f"False positive detected: '{text[:50]}...' (score={score:.3f})"
            assert score < 0.20, f"Score too high for non-Akkadian: {score:.3f}"
    
    def test_true_positives_preserved(self):
        """Real Akkadian transliterations should be detected with strict config."""
        
        strict_config = {
            "threshold": 0.20,
            "require_diacritic_or_marker": True
        }
        
        true_positive_cases = [
            # AKT 4 - syllabic with determinative
            "A-du-da DUMU ru-ba-im",
            # AKT 4b - syllabic with diacritics
            "sa-ru-pá-am i-sé-er",
            # Determinatives and syllabic
            "DUMU ru-ba-im KÙ.BABBAR",
            # Pure syllabic with diacritics
            "sé-pá-am lu-ú-ší-ib",
        ]
        
        for text in true_positive_cases:
            is_akk, score = is_akkadian_transliteration(text, config=strict_config)
            assert is_akk, f"False negative: '{text}' (score={score:.3f})"
            assert score >= 0.20, f"Score too low for Akkadian: {score:.3f}"
    
    def test_config_honored_threshold(self):
        """Detection must honor the threshold from config."""
        
        test_text = "a-na É.GAL-lim"  # Akkadian with marker and syllabic
        
        # With low threshold (0.20)
        low_config = {"threshold": 0.20, "require_diacritic_or_marker": True}
        is_akk_low, score_low = is_akkadian_transliteration(test_text, config=low_config)
        
        # With high threshold (0.80)
        high_config = {"threshold": 0.80, "require_diacritic_or_marker": True}
        is_akk_high, score_high = is_akkadian_transliteration(test_text, config=high_config)
        
        # Score should be the same
        assert score_low == score_high, "Score should not change with threshold"
        
        # Detection should change based on threshold
        if 0.20 <= score_low < 0.80:
            assert is_akk_low and not is_akk_high, "Threshold not being honored"
    
    def test_config_honored_gate(self):
        """Detection must honor require_diacritic_or_marker gate."""
        
        # The gate prevents syllabic pattern bonus without diacritics/markers
        # However, the LM can still contribute independently
        # So we test that the syllabic BONUS is removed, not that score becomes zero
        
        test_text = "a-na-ku-ma"  # Syllabic pattern, no diacritics/markers
        
        # With gate disabled
        no_gate_config = {"threshold": 0.20, "require_diacritic_or_marker": False}
        is_akk_no_gate, score_no_gate = is_akkadian_transliteration(test_text, config=no_gate_config)
        
        # With gate enabled
        gate_config = {"threshold": 0.20, "require_diacritic_or_marker": True}
        is_akk_gate, score_gate = is_akkadian_transliteration(test_text, config=gate_config)
        
        # Without gate: should detect (pure syllabic pattern gets +0.45)
        assert is_akk_no_gate, f"Should detect syllabic pattern without gate (score={score_no_gate:.3f})"
        
        # With gate: syllabic bonus (+0.45) should be removed
        # This should reduce score by at least 0.45
        assert score_gate < score_no_gate, "Gate should reduce score"
        assert (score_no_gate - score_gate) >= 0.40, \
            f"Gate should remove ~0.45 syllabic bonus (reduction={score_no_gate-score_gate:.2f})"
        
        # Most importantly: real Turkish/German prose should score 0.0 with gate
        # (This is tested in test_false_positives_eliminated)
    
    def test_negative_lexicon_penalty(self):
        """Common Turkish/German/English words should reduce score."""
        
        config = {"threshold": 0.20, "require_diacritic_or_marker": False}
        
        # Pure Akkadian syllabic pattern
        pure_text = "a-na-ku"
        is_akk_pure, score_pure = is_akkadian_transliteration(pure_text, config=config)
        
        # Same pattern with Turkish function words
        with_turkish = "a-na-ku ve bu için"
        is_akk_turk, score_turk = is_akkadian_transliteration(with_turkish, config=config)
        
        # Score should be lower when Turkish words are present
        assert score_turk < score_pure, "Negative lexicon penalty not applied"
    
    def test_precision_recall_100_percent(self):
        """
        Full test suite: must achieve 100% precision and 100% recall.
        This is the canonical acceptance test from DETECTION_FIX_SUMMARY.md.
        """
        
        strict_config = {
            "threshold": 0.20,
            "require_diacritic_or_marker": True
        }
        
        # All test cases from the fix validation
        test_cases = [
            # (text, expected_is_akkadian)
            ("Yüzey araştırmasına Prof. Dr. Tahsin Özgüç'ün başkanlığında", False),
            ("EIN NEUES ALTASSYRISCHES RECHTSDOKUMENT AUS KANES", False),
            ("Bu tablo üzerinde birçok satır bulunmaktadır", False),
            ("a-qul şeklinde yazılmıştır", False),
            ("A-du-da DUMU ru-ba-im", True),
            ("sa-ru-pá-am i-sé-er", True),
            ("DUMU ru-ba-im KÙ.BABBAR", True),
            ("sé-pá-am lu-ú-ší-ib", True),
        ]
        
        tp = fp = tn = fn = 0
        
        for text, expected_akk in test_cases:
            is_akk, score = is_akkadian_transliteration(text, config=strict_config)
            
            if is_akk and expected_akk:
                tp += 1
            elif is_akk and not expected_akk:
                fp += 1
                print(f"FALSE POSITIVE: '{text[:50]}...' (score={score:.3f})")
            elif not is_akk and expected_akk:
                fn += 1
                print(f"FALSE NEGATIVE: '{text[:50]}...' (score={score:.3f})")
            else:
                tn += 1
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Assert 100% precision and recall
        assert precision == 1.0, f"Precision = {precision:.2%} (expected 100%)"
        assert recall == 1.0, f"Recall = {recall:.2%} (expected 100%)"
        assert fp == 0, f"False positives = {fp} (expected 0)"
        assert fn == 0, f"False negatives = {fn} (expected 0)"
        
        print(f"\n✅ PASS: Precision={precision:.0%}, Recall={recall:.0%}, FP={fp}, FN={fn}")


if __name__ == "__main__":
    # Run tests
    import pytest
    pytest.main([__file__, "-v", "-s"])
