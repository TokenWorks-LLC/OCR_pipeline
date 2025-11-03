"""
LLM correction trigger heuristics.
Detects when text should be sent to LLM for correction.
"""

from typing import Dict, Tuple
import unicodedata
import math
from collections import Counter


# Expected diacritics by language
LANGUAGE_DIACRITICS = {
    'de': set('äöüßÄÖÜ'),
    'tr': set('çğıİöşüÇĞÖŞÜ'),
    'fr': set('àâæçéèêëïîôùûüÿœÀÂÆÇÉÈÊËÏÎÔÙÛÜŸŒ'),
    'it': set('àèéìíîòóùúÀÈÉÌÍÎÒÓÙÚ'),
}

# Calibrated confidence thresholds by language
DEFAULT_CONFIDENCE_THRESHOLDS = {
    'en': 0.86,
    'de': 0.85,
    'tr': 0.83,
    'fr': 0.85,
    'it': 0.85,
    'other': 0.88,
}


def has_mojibake(text: str) -> bool:
    """
    Detect mojibake patterns in text.
    
    Args:
        text: Input text
    
    Returns:
        True if mojibake detected
    """
    from .encoding import has_mojibake as encoding_has_mojibake
    return encoding_has_mojibake(text)


def has_diacritic_mismatch(text: str, language: str, min_length: int = 200) -> bool:
    """
    Detect if text is missing expected diacritics for the language.
    
    Turkish/German/French/Italian text should contain language-specific diacritics.
    If text is long enough but lacks expected characters, flag for LLM review.
    
    Args:
        text: Input text
        language: Language code (de, tr, fr, it)
        min_length: Minimum text length to check
    
    Returns:
        True if diacritic mismatch detected
    """
    if len(text) < min_length or language not in LANGUAGE_DIACRITICS:
        return False
    
    expected = LANGUAGE_DIACRITICS[language]
    
    # Check if text contains any expected diacritics
    for char in text:
        if char in expected:
            return False  # Found expected diacritics, no mismatch
    
    # Long text in this language should have diacritics
    return True


def calculate_char_lm_anomaly(text: str, window: int = 3) -> float:
    """
    Calculate character-level language model anomaly score.
    
    Uses simple n-gram frequency to detect unusual character sequences
    that might indicate OCR errors.
    
    Args:
        text: Input text
        window: N-gram window size
    
    Returns:
        Z-score of anomaly (higher = more unusual)
    """
    if len(text) < window * 10:
        return 0.0  # Too short to analyze
    
    # Extract n-grams
    ngrams = []
    for i in range(len(text) - window + 1):
        ngrams.append(text[i:i+window])
    
    if not ngrams:
        return 0.0
    
    # Calculate n-gram frequencies
    freq = Counter(ngrams)
    counts = list(freq.values())
    
    if len(counts) < 2:
        return 0.0
    
    # Calculate mean and std of frequencies
    mean_freq = sum(counts) / len(counts)
    variance = sum((x - mean_freq) ** 2 for x in counts) / len(counts)
    std_freq = math.sqrt(variance)
    
    if std_freq == 0:
        return 0.0
    
    # Find rare n-grams (frequency < mean - std)
    threshold = mean_freq - std_freq
    rare_count = sum(1 for c in counts if c < threshold)
    rare_ratio = rare_count / len(counts)
    
    # Convert to z-score approximation
    # High ratio of rare n-grams = high anomaly
    z_score = rare_ratio * 10  # Scale to make z>2.5 meaningful
    
    return z_score


def get_calibrated_threshold(language: str, thresholds: Dict[str, float] = None) -> float:
    """
    Get calibrated confidence threshold for a language.
    
    Args:
        language: Language code
        thresholds: Optional custom thresholds
    
    Returns:
        Confidence threshold (0.0-1.0)
    """
    if thresholds is None:
        thresholds = DEFAULT_CONFIDENCE_THRESHOLDS
    
    return thresholds.get(language, thresholds.get('other', 0.88))


def should_trigger_llm(
    text: str,
    confidence: float,
    language: str,
    custom_thresholds: Dict[str, float] = None
) -> Tuple[bool, str]:
    """
    Determine if text should be sent to LLM for correction.
    
    Triggers LLM if ANY of these conditions are met:
    1. Low confidence (below calibrated threshold)
    2. Mojibake detected
    3. Diacritic mismatch for language
    4. Character-level anomaly (z > 2.5)
    
    Args:
        text: OCR text
        confidence: OCR confidence score (0.0-1.0)
        language: Language code
        custom_thresholds: Optional custom confidence thresholds
    
    Returns:
        (should_trigger, reason)
    """
    # Check 1: Low confidence
    threshold = get_calibrated_threshold(language, custom_thresholds)
    if confidence < threshold:
        return True, f"low_confidence ({confidence:.3f} < {threshold:.3f})"
    
    # Check 2: Mojibake
    if has_mojibake(text):
        return True, "mojibake_detected"
    
    # Check 3: Diacritic mismatch
    if has_diacritic_mismatch(text, language):
        return True, f"diacritic_mismatch (expected {language} diacritics missing)"
    
    # Check 4: Character-level anomaly
    anomaly_score = calculate_char_lm_anomaly(text)
    if anomaly_score > 2.5:
        return True, f"char_lm_anomaly (z={anomaly_score:.2f})"
    
    # No triggers
    return False, "high_confidence"
