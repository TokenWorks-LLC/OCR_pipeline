"""
Encoding utilities for detecting and repairing mojibake.
Ensures Turkish/German diacritics survive end-to-end.
"""

import re
from typing import Tuple


# Common mojibake patterns from cp1252→utf-8 double-encoding
MOJIBAKE_PATTERNS = [
    'Ã¤', 'Ã¶', 'Ã¼', 'ÃŸ',  # German: ä ö ü ß
    'Ã„', 'Ã–', 'Ãœ',         # German uppercase: Ä Ö Ü
    'Ã§', 'Ä±', 'ÄŸ', 'Å',   # Turkish: ç ı ğ Ş
    'Ã©', 'Ã¨', 'Ãª', 'Ã«',  # French: é è ê ë
    'Ã ', 'Ã¢',              # French: à â
    'Â',                      # Spurious non-breaking space marker
    'â€™', 'â€œ', 'â€',     # Smart quotes
    '�',                      # Replacement character
]

# Expected diacritics for each language
LANGUAGE_DIACRITICS = {
    'de': set('äöüßÄÖÜ'),
    'tr': set('çğıİöşüÇĞÖŞÜ'),
    'fr': set('àâæçéèêëïîôùûüÿœÀÂÆÇÉÈÊËÏÎÔÙÛÜŸŒ'),
    'it': set('àèéìíîòóùúÀÈÉÌÍÎÒÓÙÚ'),
}


def has_mojibake(text: str) -> bool:
    """
    Detect if text contains mojibake patterns.
    
    Args:
        text: Input string
    
    Returns:
        True if mojibake detected
    """
    if not text:
        return False
    
    for pattern in MOJIBAKE_PATTERNS:
        if pattern in text:
            return True
    
    return False


def count_mojibake_markers(text: str) -> int:
    """
    Count number of mojibake markers in text.
    
    Args:
        text: Input string
    
    Returns:
        Count of mojibake patterns found
    """
    count = 0
    for pattern in MOJIBAKE_PATTERNS:
        count += text.count(pattern)
    return count


def repair_mojibake(text: str, threshold: float = 0.8) -> Tuple[str, bool]:
    """
    Attempt to repair mojibake by treating as mis-decoded cp1252→utf-8.
    
    Args:
        text: Potentially corrupted text
        threshold: Reduction ratio required to accept repair (default: 80%)
    
    Returns:
        (repaired_text, was_repaired)
    """
    if not text or not has_mojibake(text):
        return text, False
    
    # Count mojibake markers before
    before_count = count_mojibake_markers(text)
    
    try:
        # Attempt repair: encode as latin-1, decode as utf-8
        # This reverses the cp1252→utf-8 double-encoding
        repaired = text.encode('latin-1').decode('utf-8', errors='ignore')
        
        # Count mojibake markers after
        after_count = count_mojibake_markers(repaired)
        
        # Calculate reduction
        if before_count == 0:
            reduction = 0.0
        else:
            reduction = (before_count - after_count) / before_count
        
        # Accept repair if reduction meets threshold
        if reduction >= threshold:
            return repaired, True
        else:
            # Repair didn't help enough
            return text, False
            
    except (UnicodeDecodeError, UnicodeEncodeError):
        # Repair failed
        return text, False


def has_expected_diacritics(text: str, lang: str, min_length: int = 200) -> bool:
    """
    Check if text has expected diacritics for the language.
    
    Args:
        text: Input text
        lang: Language code (de, tr, fr, it)
        min_length: Minimum text length to check
    
    Returns:
        True if expected diacritics are present
    """
    if len(text) < min_length or lang not in LANGUAGE_DIACRITICS:
        return True  # Don't flag short text or unknown languages
    
    expected = LANGUAGE_DIACRITICS[lang]
    
    # Check if text contains any expected diacritics
    for char in text:
        if char in expected:
            return True
    
    return False


def apply_fallback_fixes(text: str) -> str:
    """
    Apply deterministic fallback fixes for common mojibake.
    Used when LLM is disabled or rejected.
    
    Args:
        text: Input text
    
    Returns:
        Text with common mojibake patterns replaced
    """
    # Common cp1252→utf-8 replacements
    replacements = {
        'Ã¼': 'ü',
        'Ã¶': 'ö',
        'Ã¤': 'ä',
        'ÃŸ': 'ß',
        'Ãœ': 'Ü',
        'Ã–': 'Ö',
        'Ã„': 'Ä',
        'Ã§': 'ç',
        'Ä±': 'ı',
        'ÄŸ': 'ğ',
        'Å': 'Ş',
        'Ã©': 'é',
        'Ã¨': 'è',
        'Ãª': 'ê',
        'Ã«': 'ë',
        'Ã ': 'à',
        'Ã¢': 'â',
        'â€™': "'",
        'â€œ': '"',
        'â€': '"',
        'Â ': ' ',  # Non-breaking space
    }
    
    result = text
    for mojibake, correct in replacements.items():
        result = result.replace(mojibake, correct)
    
    return result
