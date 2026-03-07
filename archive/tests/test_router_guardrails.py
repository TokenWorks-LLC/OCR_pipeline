"""
Unit tests for LLM router guardrail validators (Prompt 3)

Tests diacritic preservation, determinative preservation, edit budget validation,
and other guardrail constraints per OCR_PIPELINE_RUNBOOK.md Prompt 3 spec.
"""

import unittest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from llm_router_guardrails import GuardrailValidator


class TestGuardrailValidator(unittest.TestCase):
    """Test suite for GuardrailValidator"""
    
    def setUp(self):
        """Create validator instance for each test"""
        self.validator = GuardrailValidator(
            edit_budget_akkadian=0.03,
            edit_budget_non_akk=0.12
        )
    
    # ========== Diacritic Preservation Tests ==========
    
    def test_preserves_macron_vowels(self):
        """Test preservation of macron vowels: ā ē ī ū"""
        original = "šarrum išātu ēmuqu īdu ūmu"
        corrected = "šarrum isatu emuqu idu umu"  # Lost macrons
        
        is_valid, errors = self.validator.validate(original, corrected, is_akkadian=True)
        self.assertFalse(is_valid, "Should reject macron removal")
        error_text = " ".join(errors)
        # Check that diacritic changes are reported
        self.assertIn("ā", error_text, f"Should report ā removal. Errors: {errors}")
        self.assertIn("ē", error_text, f"Should report ē removal. Errors: {errors}")
        # Also check ī and ū are reported
        self.assertTrue(
            "ī" in error_text or "ū" in error_text,
            f"Should report ī or ū removal. Errors: {errors}"
        )
    
    def test_preserves_consonant_diacritics(self):
        """Test preservation of š ṣ ṭ ḫ"""
        original = "šarru ṣābu ṭuppu ḫarrānu"
        corrected = "sarru sabu tuppu harranu"  # Lost all diacritics
        
        is_valid, errors = self.validator.validate(original, corrected, is_akkadian=True)
        self.assertFalse(is_valid, "Should reject consonant diacritic removal")
        error_text = " ".join(errors)
        self.assertIn("š", error_text, "Should report š removal")
        self.assertIn("ṣ", error_text, "Should report ṣ removal")
        self.assertIn("ṭ", error_text, "Should report ṭ removal")
        self.assertIn("ḫ", error_text, "Should report ḫ removal")
    
    def test_allows_diacritic_preservation(self):
        """Test that valid edits preserving diacritics pass"""
        original = "šarru ērēbu"
        corrected = "šarru ērēbu"  # No changes - perfect
        
        is_valid, errors = self.validator.validate(original, corrected, is_akkadian=True)
        self.assertTrue(is_valid, f"Should accept preserved diacritics. Errors: {errors}")
    
    # ========== Determinative Preservation Tests ==========
    
    def test_preserves_divine_determinative(self):
        """Test preservation of divine determinative ᵈ"""
        original = "ᵈUTU šarru ᵈIškur"
        corrected = "UTU šarru Iškur"  # Lost determinatives
        
        is_valid, errors = self.validator.validate(original, corrected, is_akkadian=True)
        self.assertFalse(is_valid, "Should reject determinative removal")
        self.assertTrue(
            any("determinative" in err.lower() or "ᵈ" in err for err in errors),
            f"Should report determinative removal. Errors: {errors}"
        )
    
    def test_preserves_personal_name_determinative(self):
        """Test preservation of personal name determinative ᵐ"""
        original = "ᵐPuzur-Aššur ᵐIdi-ilum"
        corrected = "Puzur-Aššur Idi-ilum"  # Lost ᵐ markers
        
        is_valid, errors = self.validator.validate(original, corrected, is_akkadian=True)
        self.assertFalse(is_valid, "Should reject ᵐ removal")
        self.assertTrue(
            any("determinative" in err.lower() or "ᵐ" in err for err in errors),
            f"Should report ᵐ removal. Errors: {errors}"
        )
    
    def test_preserves_feminine_determinative(self):
        """Test preservation of feminine determinative ᶠ"""
        original = "ᶠIštar mārat ᵈAnum"
        corrected = "Ištar mārat ᵈAnum"  # Lost ᶠ
        
        is_valid, errors = self.validator.validate(original, corrected, is_akkadian=True)
        self.assertFalse(is_valid, "Should reject ᶠ removal")
        self.assertTrue(
            any("determinative" in err.lower() or "ᶠ" in err for err in errors),
            f"Should report ᶠ removal. Errors: {errors}"
        )
    
    # ========== Edit Budget Tests ==========
    
    def test_akkadian_edit_budget_3_percent(self):
        """Test that Akkadian blocks enforce 3% edit budget"""
        original = "šarru māru ᵈUTU ērēbu dāku"
        # Clear violation - remove diacritics and determinative (>3%)
        corrected_bad = "šarru māru UTU erebu daku"
        
        is_valid_bad, errors_bad = self.validator.validate(original, corrected_bad, is_akkadian=True)
        self.assertFalse(is_valid_bad, "Should reject edits exceeding 3% for Akkadian")
        self.assertTrue(
            any("0.03" in err or "edit budget" in err.lower() for err in errors_bad),
            f"Should report edit budget violation. Errors: {errors_bad}"
        )
    
    def test_modern_lang_edit_budget_12_percent(self):
        """Test that modern language blocks enforce 12% edit budget"""
        original = "The quick brown fox jumps over the lazy dog now"  # 48 chars
        # Small change (1 char = 2%) - should pass
        corrected_ok = "The qwick brown fox jumps over the lazy dog now"
        # Large change (40+ chars = 83%) - should fail
        corrected_bad = "Da schnell braun fuchs springen uber lazy hund"
        
        is_valid_ok, errors_ok = self.validator.validate(original, corrected_ok, is_akkadian=False)
        self.assertTrue(is_valid_ok, f"Should accept edits within 12%. Errors: {errors_ok}")
        
        is_valid_bad, errors_bad = self.validator.validate(original, corrected_bad, is_akkadian=False)
        self.assertFalse(is_valid_bad, "Should reject edits exceeding 12%")
        self.assertTrue(
            any("edit budget" in err.lower() for err in errors_bad),
            f"Should report edit budget violation. Errors: {errors_bad}"
        )
    
    # ========== Bracket Preservation Tests ==========
    
    def test_preserves_square_brackets(self):
        """Test preservation of damaged text markers [...]"""
        original = "šarru [māru] ᵈUTU"
        corrected = "šarru māru ᵈUTU"  # Lost brackets
        
        is_valid, errors = self.validator.validate(original, corrected, is_akkadian=True)
        self.assertFalse(is_valid, "Should reject bracket removal")
        self.assertTrue(
            any("bracket" in err.lower() or "[" in err for err in errors),
            f"Should report bracket removal. Errors: {errors}"
        )
    
    def test_preserves_angle_brackets(self):
        """Test preservation of supplied text markers ⟨...⟩"""
        original = "šarru ⟨māru⟩ ᵈUTU"
        corrected = "šarru māru ᵈUTU"  # Lost angle brackets
        
        is_valid, errors = self.validator.validate(original, corrected, is_akkadian=True)
        self.assertFalse(is_valid, "Should reject angle bracket removal")
        self.assertTrue(
            any("bracket" in err.lower() or "⟨" in err or "⟩" in err for err in errors),
            f"Should report angle bracket removal. Errors: {errors}"
        )
    
    # ========== Numeral Preservation Tests ==========
    
    def test_preserves_numerals(self):
        """Test preservation of line numbers and dates"""
        original = "KTK 123:45 dated 1892"
        corrected = "KTK abc:xy dated year"  # Lost numerals
        
        is_valid, errors = self.validator.validate(original, corrected, is_akkadian=False)
        self.assertFalse(is_valid, "Should reject numeral removal")
        self.assertTrue(
            any("numeral" in err.lower() or "digit" in err.lower() for err in errors),
            f"Should report numeral changes. Errors: {errors}"
        )
    
    # ========== Line Count Validation Tests ==========
    
    def test_preserves_line_count(self):
        """Test that line count must match"""
        original = "line one\nline two\nline three"
        corrected = "line one\nline two"  # Lost a line
        
        is_valid, errors = self.validator.validate(original, corrected, is_akkadian=False)
        self.assertFalse(is_valid, "Should reject line count change")
        self.assertTrue(
            any("line count" in err.lower() for err in errors),
            f"Should report line count mismatch. Errors: {errors}"
        )


class TestGuardrailEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions"""
    
    def setUp(self):
        self.validator = GuardrailValidator(
            edit_budget_akkadian=0.03,
            edit_budget_non_akk=0.12
        )
    
    def test_empty_text_validation(self):
        """Test validation of empty strings"""
        is_valid, errors = self.validator.validate("", "", is_akkadian=False)
        self.assertTrue(is_valid, "Should accept empty→empty")
        
        # Empty→text is valid if under edit budget (just check it doesn't crash)
        is_valid2, errors2 = self.validator.validate("", "text", is_akkadian=False)
        # Just ensure it runs without error - behavior is implementation-defined
    
    def test_mixed_content_block(self):
        """Test block with mixed modern + Akkadian-like content"""
        original = "Puzur-Aššur (KTK 123) dated 1892 BC"
        corrected = "Puzur-Aššur (KTK 123) dated 1892 BC"
        
        is_valid, errors = self.validator.validate(original, corrected, is_akkadian=False)
        self.assertTrue(is_valid, f"Should accept preserved mixed content. Errors: {errors}")
    
    def test_single_line_edit(self):
        """Test validation on single line"""
        original = "This is a test"
        corrected = "This is a fest"  # 1 char change = 7%
        
        is_valid, errors = self.validator.validate(original, corrected, is_akkadian=False)
        self.assertTrue(is_valid, f"7% edit should pass 12% threshold. Errors: {errors}")


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)
