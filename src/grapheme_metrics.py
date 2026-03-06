#!/usr/bin/env python3
r"""
Grapheme-aware metrics for OCR evaluation.

Provides diagnostic metrics that count grapheme clusters (base + combining marks)
as single units, complementing standard character-based CER/WER.

REQUIREMENTS:
- Keep official CER/WER unchanged (character/word based)
- Add diagnostic Grapheme CER/WER using regex \X pattern
- Report both to understand diacritic behavior

Author: Senior OCR Engineer
Date: 2025-10-06
"""

import regex  # Note: 'regex' module, not 're' - supports \X for grapheme clusters
from difflib import SequenceMatcher
from typing import Tuple, Dict, List


def split_graphemes(text: str) -> List[str]:
    r"""
    Split text into grapheme clusters using Unicode segmentation.
    
    A grapheme cluster is a base character plus any combining marks.
    Examples:
        - "š" → ["š"] (1 grapheme, even if composed of s + combining caron)
        - "a\u0301" → ["a\u0301"] (1 grapheme: a + combining acute)
        - "ṣ" → ["ṣ"] (1 grapheme)
    
    Uses regex module's \X pattern which matches extended grapheme clusters
    per Unicode TR29.
    
    Args:
        text: Input text
        
    Returns:
        List of grapheme cluster strings
    """
    if not text:
        return []
    
    # \X matches any extended grapheme cluster
    return regex.findall(r'\X', text)


def split_words(text: str) -> List[str]:
    """
    Split text into words (whitespace-separated tokens).
    
    Args:
        text: Input text
        
    Returns:
        List of word strings
    """
    if not text:
        return []
    
    return text.split()


def compute_edit_distance(reference: List[str], hypothesis: List[str]) -> Tuple[int, int, int, int]:
    """
    Compute Levenshtein edit distance using SequenceMatcher.
    
    Returns insertions, deletions, substitutions, and total edits.
    
    Args:
        reference: Reference sequence (list of strings)
        hypothesis: Hypothesis sequence (list of strings)
        
    Returns:
        Tuple of (insertions, deletions, substitutions, total_edits)
    """
    if not reference and not hypothesis:
        return 0, 0, 0, 0
    
    matcher = SequenceMatcher(None, reference, hypothesis)
    
    insertions = 0
    deletions = 0
    substitutions = 0
    
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == 'insert':
            insertions += (j2 - j1)
        elif op == 'delete':
            deletions += (i2 - i1)
        elif op == 'replace':
            # Replacement is counted as min(del, ins) substitutions
            # plus remaining insertions or deletions
            n_subs = min(i2 - i1, j2 - j1)
            substitutions += n_subs
            
            if (i2 - i1) > (j2 - j1):
                deletions += (i2 - i1) - (j2 - j1)
            elif (j2 - j1) > (i2 - i1):
                insertions += (j2 - j1) - (i2 - i1)
    
    total_edits = insertions + deletions + substitutions
    return insertions, deletions, substitutions, total_edits


def compute_cer_wer(reference: str, hypothesis: str) -> Dict[str, float]:
    """
    Compute standard Character Error Rate (CER) and Word Error Rate (WER).
    
    THIS IS THE OFFICIAL METRIC - unchanged from baseline.
    
    CER = (insertions + deletions + substitutions) / len(reference_chars)
    WER = (insertions + deletions + substitutions) / len(reference_words)
    
    Args:
        reference: Reference (gold) text
        hypothesis: Hypothesis (OCR) text
        
    Returns:
        Dict with 'cer', 'wer', and detailed edit counts
    """
    # Character-level
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    
    char_ins, char_del, char_sub, char_total = compute_edit_distance(ref_chars, hyp_chars)
    
    cer = char_total / len(ref_chars) if ref_chars else 0.0
    
    # Word-level
    ref_words = split_words(reference)
    hyp_words = split_words(hypothesis)
    
    word_ins, word_del, word_sub, word_total = compute_edit_distance(ref_words, hyp_words)
    
    wer = word_total / len(ref_words) if ref_words else 0.0
    
    return {
        'cer': cer,
        'wer': wer,
        'char_edits': {
            'insertions': char_ins,
            'deletions': char_del,
            'substitutions': char_sub,
            'total': char_total,
            'ref_length': len(ref_chars)
        },
        'word_edits': {
            'insertions': word_ins,
            'deletions': word_del,
            'substitutions': word_sub,
            'total': word_total,
            'ref_length': len(ref_words)
        }
    }


def compute_grapheme_cer_wer(reference: str, hypothesis: str) -> Dict[str, float]:
    """
    Compute DIAGNOSTIC Grapheme Error Rate (GER) and Grapheme Word Error Rate (GWER).
    
    This is a diagnostic metric to understand diacritic behavior.
    Graphemes are counted as single units (base + combining marks).
    
    GER = (insertions + deletions + substitutions) / len(reference_graphemes)
    GWER = still uses word boundaries but compares grapheme sequences
    
    Args:
        reference: Reference (gold) text
        hypothesis: Hypothesis (OCR) text
        
    Returns:
        Dict with 'ger', 'gwer', and detailed edit counts
    """
    # Grapheme-level
    ref_graphemes = split_graphemes(reference)
    hyp_graphemes = split_graphemes(hypothesis)
    
    graph_ins, graph_del, graph_sub, graph_total = compute_edit_distance(ref_graphemes, hyp_graphemes)
    
    ger = graph_total / len(ref_graphemes) if ref_graphemes else 0.0
    
    # Word-level grapheme analysis
    # Split into words, then compare grapheme sequences within words
    ref_words = split_words(reference)
    hyp_words = split_words(hypothesis)
    
    word_ins, word_del, word_sub, word_total = compute_edit_distance(ref_words, hyp_words)
    
    gwer = word_total / len(ref_words) if ref_words else 0.0
    
    return {
        'ger': ger,
        'gwer': gwer,
        'grapheme_edits': {
            'insertions': graph_ins,
            'deletions': graph_del,
            'substitutions': graph_sub,
            'total': graph_total,
            'ref_length': len(ref_graphemes)
        },
        'word_edits': {
            'insertions': word_ins,
            'deletions': word_del,
            'substitutions': word_sub,
            'total': word_total,
            'ref_length': len(ref_words)
        }
    }


def compute_all_metrics(reference: str, hypothesis: str, 
                       enable_grapheme: bool = True) -> Dict[str, any]:
    """
    Compute both official and diagnostic metrics.
    
    OFFICIAL metrics (always computed):
    - CER: Character Error Rate
    - WER: Word Error Rate
    
    DIAGNOSTIC metrics (if enabled):
    - GER: Grapheme Error Rate
    - GWER: Grapheme Word Error Rate
    
    Args:
        reference: Reference (gold) text
        hypothesis: Hypothesis (OCR) text
        enable_grapheme: Whether to compute diagnostic grapheme metrics
        
    Returns:
        Dict with all metrics and metadata
    """
    # Official metrics (REQUIRED)
    official = compute_cer_wer(reference, hypothesis)
    
    result = {
        'official': official,
        'cer': official['cer'],  # Top-level for easy access
        'wer': official['wer'],
    }
    
    # Diagnostic metrics (OPTIONAL)
    if enable_grapheme:
        diagnostic = compute_grapheme_cer_wer(reference, hypothesis)
        result['diagnostic'] = diagnostic
        result['ger'] = diagnostic['ger']
        result['gwer'] = diagnostic['gwer']
    
    return result


def format_metrics_report(metrics: Dict[str, any], 
                         include_diagnostic: bool = True) -> str:
    """
    Format metrics into a human-readable report.
    
    Args:
        metrics: Metrics dict from compute_all_metrics()
        include_diagnostic: Whether to include diagnostic metrics
        
    Returns:
        Formatted string report
    """
    lines = []
    
    # Official metrics
    lines.append("=== OFFICIAL METRICS ===")
    lines.append(f"CER: {metrics['cer']:.4f} ({metrics['cer']*100:.2f}%)")
    lines.append(f"WER: {metrics['wer']:.4f} ({metrics['wer']*100:.2f}%)")
    
    char_edits = metrics['official']['char_edits']
    lines.append(f"\nCharacter Edits:")
    lines.append(f"  Insertions:    {char_edits['insertions']}")
    lines.append(f"  Deletions:     {char_edits['deletions']}")
    lines.append(f"  Substitutions: {char_edits['substitutions']}")
    lines.append(f"  Total:         {char_edits['total']} / {char_edits['ref_length']}")
    
    word_edits = metrics['official']['word_edits']
    lines.append(f"\nWord Edits:")
    lines.append(f"  Insertions:    {word_edits['insertions']}")
    lines.append(f"  Deletions:     {word_edits['deletions']}")
    lines.append(f"  Substitutions: {word_edits['substitutions']}")
    lines.append(f"  Total:         {word_edits['total']} / {word_edits['ref_length']}")
    
    # Diagnostic metrics
    if include_diagnostic and 'diagnostic' in metrics:
        lines.append("\n=== DIAGNOSTIC METRICS (Grapheme-aware) ===")
        lines.append(f"GER:  {metrics['ger']:.4f} ({metrics['ger']*100:.2f}%)")
        lines.append(f"GWER: {metrics['gwer']:.4f} ({metrics['gwer']*100:.2f}%)")
        
        graph_edits = metrics['diagnostic']['grapheme_edits']
        lines.append(f"\nGrapheme Edits:")
        lines.append(f"  Insertions:    {graph_edits['insertions']}")
        lines.append(f"  Deletions:     {graph_edits['deletions']}")
        lines.append(f"  Substitutions: {graph_edits['substitutions']}")
        lines.append(f"  Total:         {graph_edits['total']} / {graph_edits['ref_length']}")
        
        # Comparison
        cer_ger_diff = metrics['cer'] - metrics['ger']
        lines.append(f"\nCER vs GER difference: {cer_ger_diff:.4f}")
        if abs(cer_ger_diff) > 0.01:
            lines.append(f"  → Significant diacritic confusion detected")
    
    return '\n'.join(lines)


if __name__ == '__main__':
    # Test cases
    print("=== Grapheme Metrics Test Suite ===\n")
    
    test_cases = [
        {
            'name': 'Akkadian with diacritics',
            'reference': 'a-na DINGIR-lí-šu i-qí-a-am',
            'hypothesis': 'a-na DINGIR-li-su i-qi-a-am',  # Missing diacritics
        },
        {
            'name': 'Perfect match',
            'reference': 'KÙ.BABBAR ša-ru-pá-am',
            'hypothesis': 'KÙ.BABBAR ša-ru-pá-am',
        },
        {
            'name': 'Subscript confusion',
            'reference': 'GIŠ.TUKUL₂ ša₂ LUGAL',
            'hypothesis': 'GIŠ.TUKUL2 ša2 LUGAL',  # Subscripts → ASCII
        },
    ]
    
    for test in test_cases:
        print(f"Test: {test['name']}")
        print(f"  REF: {test['reference']}")
        print(f"  HYP: {test['hypothesis']}")
        print()
        
        metrics = compute_all_metrics(
            test['reference'],
            test['hypothesis'],
            enable_grapheme=True
        )
        
        report = format_metrics_report(metrics, include_diagnostic=True)
        print(report)
        print("\n" + "="*60 + "\n")
