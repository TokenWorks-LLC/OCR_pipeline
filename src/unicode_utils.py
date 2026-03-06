"""
Unicode normalization utilities for preserving diacritics in cuneiform transliteration.
"""

import unicodedata


def nfc(s: str) -> str:
    """Normalize to NFC form (composed characters)."""
    return unicodedata.normalize("NFC", s)


def space_collapse(s: str) -> str:
    """Collapse multiple whitespace to single spaces."""
    return " ".join(s.split())


def normalize_preserve(s: str) -> str:
    """
    Normalize text while preserving punctuation and diacritics.
    
    This function:
    - Normalizes to NFC (composed Unicode)
    - Collapses whitespace
    - Preserves all punctuation, diacritics, and special characters
    
    Critical for cuneiform transliteration which uses:
    - Colons, hyphens, brackets
    - Combining diacritics (á, š, ṣ, ḫ, etc.)
    - Subscript numbers
    """
    return space_collapse(nfc(s))
