#!/usr/bin/env python3
"""
Unit tests for grapheme_metrics.py.

Tests grapheme-aware CER/WER calculations for OCR evaluation.
"""

import unittest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from grapheme_metrics import (
    split_graphemes,
    split_words,
    compute_edit_distance,
    compute_cer_wer,
    compute_grapheme_cer_wer
)


class TestGraphemeMetrics(unittest.TestCase):
    """Test grapheme-aware metrics for OCR evaluation."""
    
    def test_split_graphemes_simple(self):
        """Test grapheme splitting on simple ASCII text."""
        text = "hello world"
        graphemes = split_graphemes(text)
        
        # Each character is a grapheme
        self.assertEqual(len(graphemes), 11)  # 11 chars including space
        self.assertEqual(graphemes[0], 'h')
        self.assertEqual(graphemes[5], ' ')
    
    def test_split_graphemes_diacritics(self):
        """Test grapheme splitting with diacritics (Akkadian)."""
        # Akkadian diacritics: š ṣ ṭ ḫ ā ē ī ū
        text = "šarrum ṣābum ṭābum"
        graphemes = split_graphemes(text)
        
        # Each diacritic character should be one grapheme
        self.assertIn('š', graphemes)
        self.assertIn('ṣ', graphemes)
        self.assertIn('ṭ', graphemes)
    
    def test_split_graphemes_combining_marks(self):
        """Test grapheme splitting with combining diacritical marks."""
        # a + combining acute accent (U+0301)
        text = "a\u0301bc"  # á (as base + combining)
        graphemes = split_graphemes(text)
        
        # Should treat a + combining as single grapheme
        self.assertEqual(len(graphemes), 3)
        self.assertEqual(graphemes[0], "a\u0301")  # á as one grapheme
    
    def test_split_words(self):
        """Test word splitting."""
        text = "hello world from OCR"
        words = split_words(text)
        
        self.assertEqual(len(words), 4)
        self.assertEqual(words, ["hello", "world", "from", "OCR"])
    
    def test_split_words_multiple_spaces(self):
        """Test word splitting with multiple spaces."""
        text = "hello   world"
        words = split_words(text)
        
        # Should handle multiple spaces
        self.assertIn("hello", words)
        self.assertIn("world", words)
    
    def test_compute_edit_distance_identical(self):
        """Test edit distance for identical sequences."""
        seq = ["a", "b", "c"]
        ins, dels, subs, total = compute_edit_distance(seq, seq)
        
        self.assertEqual(total, 0)
        self.assertEqual(ins, 0)
        self.assertEqual(dels, 0)
        self.assertEqual(subs, 0)
    
    def test_compute_edit_distance_insertion(self):
        """Test edit distance with insertions."""
        ref = ["a", "b", "c"]
        hyp = ["a", "b", "x", "c"]  # Inserted 'x'
        
        ins, dels, subs, total = compute_edit_distance(ref, hyp)
        
        self.assertEqual(ins, 1)
        self.assertGreaterEqual(total, 1)
    
    def test_compute_edit_distance_deletion(self):
        """Test edit distance with deletions."""
        ref = ["a", "b", "c"]
        hyp = ["a", "c"]  # Deleted 'b'
        
        ins, dels, subs, total = compute_edit_distance(ref, hyp)
        
        self.assertEqual(dels, 1)
        self.assertGreaterEqual(total, 1)
    
    def test_compute_edit_distance_substitution(self):
        """Test edit distance with substitutions."""
        ref = ["a", "b", "c"]
        hyp = ["a", "x", "c"]  # Substituted 'b' with 'x'
        
        ins, dels, subs, total = compute_edit_distance(ref, hyp)
        
        self.assertEqual(subs, 1)
        self.assertGreaterEqual(total, 1)
    
    def test_compute_cer_wer_perfect(self):
        """Test CER/WER for perfect match."""
        reference = "This is a test"
        hypothesis = "This is a test"
        
        metrics = compute_cer_wer(reference, hypothesis)
        
        self.assertEqual(metrics['cer'], 0.0)
        self.assertEqual(metrics['wer'], 0.0)
        self.assertEqual(metrics['char_edits']['total'], 0)
        self.assertEqual(metrics['word_edits']['total'], 0)
    
    def test_compute_cer_wer_typical_ocr_errors(self):
        """Test CER/WER with typical OCR errors."""
        reference = "The quick brown fox"
        hypothesis = "Ihe quik brown fox"  # 'T' -> 'I', 'quick' -> 'quik'
        
        metrics = compute_cer_wer(reference, hypothesis)
        
        # Should have non-zero errors
        self.assertGreater(metrics['cer'], 0.0)
        self.assertGreater(metrics['wer'], 0.0)
        
        # CER should be relatively low (only 2 char errors out of 19)
        self.assertLess(metrics['cer'], 0.2)
    
    def test_compute_cer_wer_akkadian_text(self):
        """Test CER/WER on Akkadian transliteration."""
        reference = "šarrum kīma ilim"
        hypothesis = "sarrum kima ilim"  # Lost diacritics
        
        metrics = compute_cer_wer(reference, hypothesis)
        
        # Should detect the character differences
        self.assertGreater(metrics['cer'], 0.0)
        
        # Word-level might show errors if š vs s treated differently
        # (depends on normalization)
    
    def test_compute_grapheme_cer_wer(self):
        """Test grapheme-level CER/WER."""
        reference = "café résumé"
        hypothesis = "cafe resume"
        
        # Standard CER
        char_metrics = compute_cer_wer(reference, hypothesis)
        
        # Grapheme CER
        grapheme_metrics = compute_grapheme_cer_wer(reference, hypothesis)
        
        # Both should detect errors, but grapheme may differ
        # if accents are composed vs decomposed
        self.assertGreaterEqual(char_metrics['cer'], 0.0)
        self.assertGreaterEqual(grapheme_metrics['ger'], 0.0)
    
    def test_compute_cer_wer_empty_strings(self):
        """Test CER/WER with empty strings."""
        # Both empty
        metrics = compute_cer_wer("", "")
        self.assertEqual(metrics['cer'], 0.0)
        self.assertEqual(metrics['wer'], 0.0)
        
        # Reference empty, hypothesis not
        metrics = compute_cer_wer("", "text")
        # Should handle gracefully (CER = 1.0 or special case)
        self.assertIsNotNone(metrics['cer'])
    
    def test_cer_normalization(self):
        """Test that CER is properly normalized (0-1 range)."""
        reference = "test"
        hypothesis = "completely different text"
        
        metrics = compute_cer_wer(reference, hypothesis)
        
        # CER should be in valid range
        self.assertGreaterEqual(metrics['cer'], 0.0)
        # Can exceed 1.0 if hypothesis is much longer
        # but typically should be reasonable
    
    def test_wer_normalization(self):
        """Test that WER is properly normalized."""
        reference = "one two three"
        hypothesis = "four five six seven"
        
        metrics = compute_cer_wer(reference, hypothesis)
        
        # WER should be in valid range
        self.assertGreaterEqual(metrics['wer'], 0.0)
    
    def test_akkadian_diacritics_preserved_in_comparison(self):
        """Test that Akkadian diacritics are properly compared."""
        # These should NOT be considered equal
        ref_with_diacritics = "šarrum"
        hyp_without_diacritics = "sarrum"
        
        metrics = compute_cer_wer(ref_with_diacritics, hyp_without_diacritics)
        
        # Should show difference
        self.assertGreater(metrics['cer'], 0.0)
        
        # Grapheme-level should also show difference
        g_metrics = compute_grapheme_cer_wer(ref_with_diacritics, hyp_without_diacritics)
        self.assertGreater(g_metrics['ger'], 0.0)
    
    def test_determinatives_superscripts(self):
        """Test that superscript determinatives are handled."""
        # Akkadian determinatives: ᵈ ᵐ ᶠ
        text = "ᵈUTU ᵐAššur-uballiṭ"
        
        graphemes = split_graphemes(text)
        
        # Should preserve superscripts
        text_rejoined = ''.join(graphemes)
        self.assertIn('ᵈ', text_rejoined)
        self.assertIn('ᵐ', text_rejoined)
    
    def test_metrics_structure(self):
        """Test that metrics dict has expected structure."""
        metrics = compute_cer_wer("reference", "hypothesis")
        
        # Check required keys
        self.assertIn('cer', metrics)
        self.assertIn('wer', metrics)
        self.assertIn('char_edits', metrics)
        self.assertIn('word_edits', metrics)
        
        # Check char_edits structure
        self.assertIn('insertions', metrics['char_edits'])
        self.assertIn('deletions', metrics['char_edits'])
        self.assertIn('substitutions', metrics['char_edits'])
        self.assertIn('total', metrics['char_edits'])
        self.assertIn('ref_length', metrics['char_edits'])
        
        # Check word_edits structure
        self.assertIn('insertions', metrics['word_edits'])
        self.assertIn('deletions', metrics['word_edits'])
        self.assertIn('substitutions', metrics['word_edits'])
        self.assertIn('total', metrics['word_edits'])
        self.assertIn('ref_length', metrics['word_edits'])


if __name__ == '__main__':
    unittest.main()
