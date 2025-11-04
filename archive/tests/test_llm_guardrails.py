"""
Unit Tests for LLM Guardrails

Tests the content-preservation guardrails in the LLM corrector:
- Edit budget enforcement
- Bracket preservation
- Line break preservation
- Vocabulary explosion detection
- JSON schema validation
"""

import unittest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from llm.corrector import LLMCorrector, GuardrailViolation
from llm.json_schemas import (
    CorrectionResponse, EditOperation, CorrectionRequest,
    CorrectionFlags, ContextInfo
)
from pydantic import ValidationError


class TestEditBudgetGuardrail(unittest.TestCase):
    """Test edit budget guardrail."""
    
    def setUp(self):
        self.corrector = LLMCorrector()
    
    def test_edit_budget_within_limit(self):
        """Test that edits within budget pass."""
        original = "Hello world this is a test"
        corrected = "Hello world this is a fest"  # 1 char change
        
        # Should not raise
        self.corrector._validate_edit_budget(original, corrected, max_ratio=0.12)
    
    def test_edit_budget_exceeds_limit(self):
        """Test that edits exceeding budget fail."""
        original = "Hello world"
        corrected = "Completely different text"
        
        with self.assertRaises(GuardrailViolation) as cm:
            self.corrector._validate_edit_budget(original, corrected, max_ratio=0.12)
        
        self.assertIn("edit ratio", str(cm.exception).lower())
    
    def test_edit_budget_transliteration_strict(self):
        """Test strict 3% budget for transliteration."""
        original = "ša-ar-ru šarrum LUGAL"  # 21 chars
        corrected = "ša-ar-ru šarrum LUGAL!"  # 22 chars (1 added)
        
        # 1/21 = 4.7% > 3%
        with self.assertRaises(GuardrailViolation):
            self.corrector._validate_edit_budget(original, corrected, max_ratio=0.03)


class TestBracketPreservation(unittest.TestCase):
    """Test bracket preservation guardrail."""
    
    def setUp(self):
        self.corrector = LLMCorrector()
    
    def test_brackets_preserved(self):
        """Test that preserved brackets pass."""
        original = "Text with [brackets] and (parens) and {braces}"
        corrected = "Text with [brackets] and (parens) and {braces}"
        
        # Should not raise
        self.corrector._validate_bracket_preservation(original, corrected)
    
    def test_missing_square_bracket(self):
        """Test detection of missing square brackets."""
        original = "Text with [brackets]"
        corrected = "Text with brackets"
        
        with self.assertRaises(GuardrailViolation) as cm:
            self.corrector._validate_bracket_preservation(original, corrected)
        
        self.assertIn("bracket", str(cm.exception).lower())
    
    def test_missing_parenthesis(self):
        """Test detection of missing parentheses."""
        original = "Text with (parens)"
        corrected = "Text with parens)"
        
        with self.assertRaises(GuardrailViolation):
            self.corrector._validate_bracket_preservation(original, corrected)
    
    def test_extra_brackets_rejected(self):
        """Test that extra brackets are rejected."""
        original = "Text without brackets"
        corrected = "Text with [brackets]"
        
        with self.assertRaises(GuardrailViolation):
            self.corrector._validate_bracket_preservation(original, corrected)


class TestLineBreakPreservation(unittest.TestCase):
    """Test line break preservation guardrail."""
    
    def setUp(self):
        self.corrector = LLMCorrector()
    
    def test_line_breaks_preserved(self):
        """Test that preserved line breaks pass."""
        original = "Line 1\nLine 2\nLine 3"
        corrected = "Line 1\nLine 2\nLine 3"
        
        # Should not raise
        self.corrector._validate_line_breaks(original, corrected)
    
    def test_missing_line_break(self):
        """Test detection of missing line breaks."""
        original = "Line 1\nLine 2"
        corrected = "Line 1 Line 2"
        
        with self.assertRaises(GuardrailViolation) as cm:
            self.corrector._validate_line_breaks(original, corrected)
        
        self.assertIn("line break", str(cm.exception).lower())
    
    def test_extra_line_break(self):
        """Test detection of extra line breaks."""
        original = "Line 1 Line 2"
        corrected = "Line 1\nLine 2"
        
        with self.assertRaises(GuardrailViolation):
            self.corrector._validate_line_breaks(original, corrected)


class TestVocabularyExplosion(unittest.TestCase):
    """Test vocabulary explosion detection."""
    
    def setUp(self):
        self.corrector = LLMCorrector()
    
    def test_normal_correction(self):
        """Test that normal corrections pass."""
        original = "Die Ubersetzung ist gut"
        corrected = "Die Übersetzung ist gut"
        
        # Should not raise (minimal change)
        self.corrector._validate_vocabulary_explosion(original, corrected)
    
    def test_vocabulary_explosion_detected(self):
        """Test detection of vocabulary explosion."""
        original = "Hello"
        corrected = "Hello world this is a lot of added text"
        
        with self.assertRaises(GuardrailViolation) as cm:
            self.corrector._validate_vocabulary_explosion(original, corrected, max_increase=0.15)
        
        self.assertIn("vocabulary explosion", str(cm.exception).lower())
    
    def test_minor_additions_allowed(self):
        """Test that minor additions are allowed."""
        original = "Hello world test"  # 15 alpha chars
        corrected = "Hello world test!"  # 15 alpha chars (! is non-alpha)
        
        # Should not raise (same alpha count)
        self.corrector._validate_vocabulary_explosion(original, corrected)


class TestJSONSchemaValidation(unittest.TestCase):
    """Test JSON schema validation."""
    
    def test_valid_response(self):
        """Test validation of valid response."""
        valid_data = {
            "span_id": "test_123",
            "mode": "fix_typos_only",
            "lang_detected": "de",
            "corrected_text": "Die Übersetzung",
            "applied_edits": [
                {"pos": 4, "from": "U", "to": "Ü", "type": "subst"}
            ],
            "edit_ratio": 0.02,
            "diacritic_restored": True,
            "confidence": 0.95,
            "notes": "Fixed umlaut"
        }
        
        # Should not raise
        response = CorrectionResponse(**valid_data)
        
        self.assertEqual(response.span_id, "test_123")
        self.assertEqual(response.mode, "fix_typos_only")
        self.assertEqual(response.lang_detected, "de")
        self.assertEqual(response.edit_ratio, 0.02)
    
    def test_missing_required_field(self):
        """Test rejection of response missing required fields."""
        invalid_data = {
            "span_id": "test_123",
            # Missing mode, lang_detected, corrected_text, etc.
        }
        
        with self.assertRaises(ValidationError):
            CorrectionResponse(**invalid_data)
    
    def test_invalid_edit_ratio(self):
        """Test rejection of invalid edit ratio."""
        invalid_data = {
            "span_id": "test_123",
            "mode": "fix_typos_only",
            "lang_detected": "en",
            "corrected_text": "Test",
            "applied_edits": [],
            "edit_ratio": 1.5,  # Invalid: > 1.0
            "diacritic_restored": False,
            "confidence": 0.9,
            "notes": ""
        }
        
        with self.assertRaises(ValidationError):
            CorrectionResponse(**invalid_data)
    
    def test_invalid_mode(self):
        """Test rejection of invalid mode."""
        invalid_data = {
            "span_id": "test_123",
            "mode": "invalid_mode",  # Invalid
            "lang_detected": "en",
            "corrected_text": "Test",
            "applied_edits": [],
            "edit_ratio": 0.1,
            "diacritic_restored": False,
            "confidence": 0.9,
            "notes": ""
        }
        
        with self.assertRaises(ValidationError):
            CorrectionResponse(**invalid_data)
    
    def test_edit_operation_validation(self):
        """Test EditOperation validation."""
        valid_edit = EditOperation(
            pos=5,
            **{"from": "a", "to": "ä", "type": "subst"}
        )
        
        self.assertEqual(valid_edit.pos, 5)
        self.assertEqual(getattr(valid_edit, 'from'), "a")
        self.assertEqual(valid_edit.to, "ä")
        self.assertEqual(valid_edit.type, "subst")


class TestCorrectionRequest(unittest.TestCase):
    """Test CorrectionRequest schema."""
    
    def test_valid_request(self):
        """Test creation of valid request."""
        request = CorrectionRequest(
            schema_version="1.0",
            span_id="test_line",
            language_hint="de",
            original_text="Die Ubersetzung",
            context=ContextInfo(
                prev_line="Vorheriger Text",
                next_line="Nächster Text"
            ),
            flags=CorrectionFlags(
                is_transliteration_suspected=False,
                max_relative_change_chars=0.12,
                mode="fix_typos_only"
            )
        )
        
        self.assertEqual(request.schema_version, "1.0")
        self.assertEqual(request.span_id, "test_line")
        self.assertEqual(request.language_hint, "de")
        self.assertFalse(request.flags.is_transliteration_suspected)
        self.assertEqual(request.flags.max_relative_change_chars, 0.12)
    
    def test_transliteration_request(self):
        """Test creation of transliteration protection request."""
        request = CorrectionRequest(
            schema_version="1.0",
            span_id="akk_line",
            language_hint="unknown",
            original_text="ša-ar-ru šarrum",
            context=ContextInfo(
                prev_line="",
                next_line=""
            ),
            flags=CorrectionFlags(
                is_transliteration_suspected=True,
                max_relative_change_chars=0.03,
                mode="protect_transliteration"
            )
        )
        
        self.assertTrue(request.flags.is_transliteration_suspected)
        self.assertEqual(request.flags.max_relative_change_chars, 0.03)
        self.assertEqual(request.flags.mode, "protect_transliteration")


class TestGuardrailIntegration(unittest.TestCase):
    """Test guardrail integration in corrector."""
    
    def setUp(self):
        self.corrector = LLMCorrector()
    
    def test_all_guardrails_pass(self):
        """Test that valid corrections pass all guardrails."""
        original = "Die Ubersetzung [1] ist wichtig."
        corrected = "Die Übersetzung [1] ist wichtig."
        
        # Should not raise
        self.corrector._apply_guardrails(
            original=original,
            corrected=corrected,
            is_transliteration=False
        )
    
    def test_transliteration_strict_budget(self):
        """Test that transliteration has strict budget."""
        original = "ša-ar-ru šarrum LUGAL"
        corrected = "ša-ar-ru šarrum LUGAL extra text"  # Too many changes
        
        with self.assertRaises(GuardrailViolation):
            self.corrector._apply_guardrails(
                original=original,
                corrected=corrected,
                is_transliteration=True
            )
    
    def test_bracket_violation_caught(self):
        """Test that bracket violations are caught."""
        original = "Text with [brackets]"
        corrected = "Text with brackets"
        
        with self.assertRaises(GuardrailViolation):
            self.corrector._apply_guardrails(
                original=original,
                corrected=corrected,
                is_transliteration=False
            )


def run_tests():
    """Run all tests and return results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestEditBudgetGuardrail))
    suite.addTests(loader.loadTestsFromTestCase(TestBracketPreservation))
    suite.addTests(loader.loadTestsFromTestCase(TestLineBreakPreservation))
    suite.addTests(loader.loadTestsFromTestCase(TestVocabularyExplosion))
    suite.addTests(loader.loadTestsFromTestCase(TestJSONSchemaValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestCorrectionRequest))
    suite.addTests(loader.loadTestsFromTestCase(TestGuardrailIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
