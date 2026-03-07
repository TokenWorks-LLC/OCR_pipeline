#!/usr/bin/env python3
"""
Unit tests for Akkadian span protection during LLM correction.

Verifies that Akkadian transliteration is never altered by LLM typo correction.
"""
import sys
import unittest
from pathlib import Path

# Add tools to path
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from run_page_text import AkkadianSpanProtector


class TestAkkadianSpanProtector(unittest.TestCase):
    """Test Akkadian span protection."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.protector = AkkadianSpanProtector()
    
    def test_find_simple_akkadian_span(self):
        """Test finding simple Akkadian span with diacritics."""
        text = "The word a-na-ku-um is Akkadian for 'I'."
        spans = self.protector.find_akkadian_spans(text)
        
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0][2], "a-na-ku-um")
    
    def test_find_multiple_spans(self):
        """Test finding multiple Akkadian spans."""
        text = "Forms like a-na-ku and šar-ru-um appear in texts."
        spans = self.protector.find_akkadian_spans(text)
        
        self.assertEqual(len(spans), 2)
        span_texts = [s[2] for s in spans]
        self.assertIn("a-na-ku", span_texts)
        self.assertIn("šar-ru-um", span_texts)
    
    def test_find_marker_spans(self):
        """Test finding spans with Akkadian markers."""
        text = "The king is called LUGAL in Akkadian."
        spans = self.protector.find_akkadian_spans(text)
        
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0][2], "LUGAL")
    
    def test_find_determinative_spans(self):
        """Test finding spans with determinatives."""
        text = "The deity ᵈNanna was worshipped."
        spans = self.protector.find_akkadian_spans(text)
        
        # Should find text before determinative
        self.assertGreater(len(spans), 0)
    
    def test_protect_and_unprotect(self):
        """Test protection wrapping and unwrapping."""
        text = "The word a-na-ku is Akkadian."
        protected, original_spans = self.protector.protect_spans(text)
        
        # Check protection
        self.assertIn("<AKK>", protected)
        self.assertIn("</AKK>", protected)
        self.assertEqual(len(original_spans), 1)
        self.assertEqual(original_spans[0], "a-na-ku")
        
        # Check unprotection
        unprotected = self.protector.unprotect_spans(protected)
        self.assertEqual(unprotected, text)
    
    def test_validate_unchanged_spans(self):
        """Test validation of unchanged protected spans."""
        text = "Forms a-na-ku and šar-ru appear."
        protected, original_spans = self.protector.protect_spans(text)
        
        # Simulate unchanged LLM output
        corrected = protected.replace("Forms", "Forms:")  # Only non-Akkadian change
        
        is_valid, error = self.protector.validate_protection(original_spans, corrected)
        self.assertTrue(is_valid, f"Validation should pass: {error}")
    
    def test_validate_detect_altered_span(self):
        """Test validation detects altered protected spans."""
        text = "The word a-na-ku is Akkadian."
        protected, original_spans = self.protector.protect_spans(text)
        
        # Simulate LLM altering Akkadian span
        corrupted = protected.replace("a-na-ku", "anaku")
        
        is_valid, error = self.protector.validate_protection(original_spans, corrupted)
        self.assertFalse(is_valid, "Validation should fail when span is altered")
        self.assertIn("altered", error.lower())
    
    def test_validate_detect_missing_span(self):
        """Test validation detects missing protected spans."""
        text = "Forms a-na-ku and šar-ru appear."
        protected, original_spans = self.protector.protect_spans(text)
        
        # Simulate LLM removing a span
        corrupted = protected.replace("<AKK>a-na-ku</AKK>", "anaku")  # Remove tags
        
        is_valid, error = self.protector.validate_protection(original_spans, corrupted)
        self.assertFalse(is_valid, "Validation should fail when span is removed")
    
    def test_no_akkadian_in_text(self):
        """Test text with no Akkadian content."""
        text = "This is plain English text with no Akkadian."
        spans = self.protector.find_akkadian_spans(text)
        
        self.assertEqual(len(spans), 0)
        
        # Protection should return original text
        protected, original_spans = self.protector.protect_spans(text)
        self.assertEqual(protected, text)
        self.assertEqual(len(original_spans), 0)
    
    def test_complex_akkadian_text(self):
        """Test complex Akkadian transliteration."""
        text = """
        1. a-na LUGAL be-lí-ia
        2. qí-bí-ma um-ma ᵈNanna-ma-an-sum
        3. a-na šarri be-lí-ia lu-ú šul-mu
        """
        
        spans = self.protector.find_akkadian_spans(text)
        
        # Should find multiple Akkadian spans
        self.assertGreater(len(spans), 3)
        
        # Protection and validation
        protected, original_spans = self.protector.protect_spans(text)
        is_valid, error = self.protector.validate_protection(original_spans, protected)
        self.assertTrue(is_valid, f"Complex text validation failed: {error}")
        
        # Unprotect should restore original
        unprotected = self.protector.unprotect_spans(protected)
        # Normalize whitespace for comparison
        self.assertEqual(unprotected.strip(), text.strip())
    
    def test_mixed_content_protection(self):
        """Test protection of mixed Akkadian and modern language."""
        text = "The Akkadian word a-na-ku means 'I' in English, similar to šarru meaning 'king'."
        
        protected, original_spans = self.protector.protect_spans(text)
        
        # Should protect Akkadian but not English
        self.assertIn("<AKK>", protected)
        self.assertIn("a-na-ku", original_spans)
        self.assertIn("šarru", original_spans)
        
        # English words should remain unprotected
        self.assertNotIn("<AKK>English</AKK>", protected)
        self.assertNotIn("<AKK>king</AKK>", protected)
    
    def test_edit_distance_zero_for_identical(self):
        """Test edit distance calculation for identical strings."""
        from run_page_text import SimpleLLMCorrector
        
        s1 = "a-na-ku-um"
        s2 = "a-na-ku-um"
        
        distance = SimpleLLMCorrector._edit_distance(s1, s2)
        self.assertEqual(distance, 0)
    
    def test_edit_distance_simple_substitution(self):
        """Test edit distance for simple substitution."""
        from run_page_text import SimpleLLMCorrector
        
        s1 = "a-na-ku"
        s2 = "a-na-ka"
        
        distance = SimpleLLMCorrector._edit_distance(s1, s2)
        self.assertEqual(distance, 1)  # One substitution: u -> a
    
    def test_protection_with_em_dash(self):
        """Test protection works with em-dash instead of hyphen (OCR error)."""
        text = "The word a—na—ku is corrupted by OCR."
        
        spans = self.protector.find_akkadian_spans(text)
        
        # Should still detect with em-dash
        self.assertGreater(len(spans), 0)
        
        protected, original_spans = self.protector.protect_spans(text)
        self.assertGreater(len(original_spans), 0)


class TestAkkadianDetector(unittest.TestCase):
    """Test page-level Akkadian detection."""
    
    def setUp(self):
        """Set up test fixtures."""
        from run_page_text import AkkadianDetector
        self.detector = AkkadianDetector()
    
    def test_detect_pure_akkadian_page(self):
        """Test detection of pure Akkadian page."""
        text = """
        1. a-na LUGAL be-lí-ia
        2. qí-bí-ma um-ma ᵈNanna-ma-an-sum
        3. a-na šarri be-lí-ia lu-ú šul-mu
        4. ṭup-pa-ka iš-tu KUR aš-šur
        5. i-na pa-ni-ia iš-šá-ak-na-am-ma
        """
        
        has_akkadian, meta = self.detector.detect_page(text)
        
        self.assertTrue(has_akkadian)
        self.assertGreater(meta["qualified_lines"], 0)
        self.assertGreater(meta["ratio"], 0)
    
    def test_detect_mixed_page(self):
        """Test detection of mixed Akkadian and modern text."""
        text = """
        This letter demonstrates typical Old Assyrian formulas:
        
        1. a-na LUGAL be-lí-ia
        2. qí-bí-ma um-ma servant-GN
        
        The formula is standard in business correspondence.
        """
        
        has_akkadian, meta = self.detector.detect_page(text)
        
        # Should detect Akkadian even with modern text
        self.assertTrue(has_akkadian)
        self.assertGreater(meta["qualified_lines"], 0)
    
    def test_no_akkadian_page(self):
        """Test page with no Akkadian content."""
        text = """
        This is a modern academic text discussing
        the trade networks in ancient Anatolia.
        The merchants used various routes.
        """
        
        has_akkadian, meta = self.detector.detect_page(text)
        
        self.assertFalse(has_akkadian)
        self.assertEqual(meta["qualified_lines"], 0)
    
    def test_false_positive_prevention(self):
        """Test that common false positives are filtered out."""
        text = """
        The German phrase "im Namen des Königs" means
        "in the name of the king" and appears frequently.
        This is scholarly prose, not Akkadian transliteration.
        """
        
        has_akkadian, meta = self.detector.detect_page(text)
        
        # Should not detect as Akkadian (no diacritics/markers)
        self.assertFalse(has_akkadian)


class TestCSVOutput(unittest.TestCase):
    """Test CSV output formatting."""
    
    def test_newline_escaping(self):
        """Test that newlines are properly escaped in CSV output."""
        text = "Line 1\nLine 2\nLine 3"
        escaped = text.replace('\n', '\\n')
        
        self.assertEqual(escaped, "Line 1\\nLine 2\\nLine 3")
        self.assertNotIn('\n', escaped)
    
    def test_quote_handling(self):
        """Test that quotes are handled in CSV."""
        text = 'Text with "quoted" content'
        
        # CSV writer should handle this, but we can test manually
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['test', text, 'true'])
        
        result = output.getvalue()
        self.assertIn('quoted', result)


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestAkkadianSpanProtector))
    suite.addTests(loader.loadTestsFromTestCase(TestAkkadianDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestCSVOutput))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_tests())
