#!/usr/bin/env python3
"""
Transliteration-aware text normalization for Akkadian/cuneiform OCR.

NON-NEGOTIABLES:
- Never strip or fold diacritics
- Use NFC (not NFKD/NFKC)
- Never delete: - : . [ ] ( ) ʾ ʿ or subscripts (₁₂₃₄₅₆₇₈₉₀)
- Never auto-expand/contract hyphenation or merge lines
- Never lowercase transliteration
- Preserve case as printed

Author: Senior OCR Engineer
Date: 2025-10-06
"""

import unicodedata
import re
from typing import Set

# Akkadian/Cuneiform transliteration diacritics
TRANSLIT_DIACRITICS = {
    # Akkadian long vowels and consonants
    'š', 'ṣ', 'ṭ', 'ḫ', 'ā', 'ē', 'ī', 'ū', 
    'â', 'ê', 'î', 'û', 'á', 'é', 'í', 'ú',
    # Additional diacritics
    'ʾ', 'ʿ',  # ayin and aleph
    # Subscript digits for Sumerian
    '₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉',
}

# Common Sumerograms (ALL CAPS ideograms)
SUMEROGRAMS = {
    'DINGIR', 'LUGAL', 'MUNUS', 'LÚ', 'KUR', 'É', 'GIŠ', 'DUG',
    'ÍD', 'URU', 'KÙ.BABBAR', 'KÙ.GI', 'GU₄', 'UDU', 'ITU', 
    'MU', 'AN', 'KI', 'IM', 'A.ŠÀ', 'GEME₂', 'ARAD', 'DAM'
}

# Structural punctuation that MUST be preserved
ALLOWED_PUNCT = set('-:.[]()/ʾʿ')

# Unicode cuneiform block range
CUNEIFORM_BLOCK_START = 0x12000
CUNEIFORM_BLOCK_END = 0x123FF


def to_nfc(s: str) -> str:
    """
    Normalize to NFC (Canonical Composition).
    
    NFC is REQUIRED for Akkadian transliteration to preserve:
    - Combining diacritical marks
    - Base + diacritic as single composed character where possible
    
    Args:
        s: Input string
        
    Returns:
        NFC-normalized string
    """
    if not s:
        return s
    return unicodedata.normalize("NFC", s)


def collapse_spaces_preserve_punct(s: str, allowed_punct: Set[str] = None) -> str:
    """
    Collapse whitespace while preserving all diacritics, subscripts, and structural punctuation.
    
    NON-NEGOTIABLES:
    - Never delete: - : . [ ] ( ) / ʾ ʿ or subscripts
    - Preserve all combining diacritical marks
    - Never merge lines in transliteration blocks
    
    Args:
        s: Input string
        allowed_punct: Set of punctuation characters to preserve (default: ALLOWED_PUNCT)
        
    Returns:
        String with collapsed spaces but all diacritics/punct preserved
    """
    if not s:
        return s
    
    if allowed_punct is None:
        allowed_punct = ALLOWED_PUNCT
    
    # Apply NFC first
    s = to_nfc(s)
    
    # Collapse multiple spaces/tabs to single space
    # BUT preserve newlines (never merge lines)
    s = re.sub(r'[ \t]+', ' ', s)
    
    # Trim leading/trailing space on each line
    lines = s.split('\n')
    lines = [line.strip() for line in lines]
    
    return '\n'.join(lines)


def is_transliteration_line(s: str, threshold: float = 0.03) -> bool:
    """
    Heuristic to detect if a line contains Akkadian/cuneiform transliteration.
    
    Detection criteria (ANY of):
    1. Diacritic density ≥ threshold (default 3%)
    2. Presence of Sumerograms (DINGIR, LUGAL, etc.)
    3. Heavy use of structural punctuation (- : .)
    4. Unicode cuneiform block characters (U+12000–U+123FF)
    
    Args:
        s: Input line
        threshold: Minimum diacritic density (default 0.03 = 3%)
        
    Returns:
        True if line appears to be transliteration
    """
    if not s or len(s) < 5:
        return False
    
    s_nfc = to_nfc(s)
    
    # Criterion 1: Diacritic density
    diacritic_count = sum(1 for c in s_nfc if c in TRANSLIT_DIACRITICS)
    diacritic_density = diacritic_count / len(s_nfc)
    
    if diacritic_density >= threshold:
        return True
    
    # Criterion 2: Sumerograms
    s_upper = s_nfc.upper()
    for sumerogram in SUMEROGRAMS:
        if sumerogram in s_upper:
            return True
    
    # Criterion 3: Heavy structural punctuation
    # Look for patterns like "a-na", "i-sé-er", "DUMu.A-lá"
    hyphen_colon_count = s_nfc.count('-') + s_nfc.count(':') + s_nfc.count('.')
    punct_density = hyphen_colon_count / len(s_nfc)
    
    if punct_density >= 0.1:  # 10% or more hyphens/colons/periods
        return True
    
    # Criterion 4: Unicode cuneiform block
    for char in s_nfc:
        code_point = ord(char)
        if CUNEIFORM_BLOCK_START <= code_point <= CUNEIFORM_BLOCK_END:
            return True
    
    return False


def normalize_transliteration(s: str, config: dict = None) -> str:
    """
    Normalize transliteration text with STRICT preservation rules.
    
    NON-NEGOTIABLES enforced:
    - NFC normalization only
    - Preserve all diacritics, subscripts, structural punctuation
    - Never lowercase
    - Never expand/contract hyphenation
    - Never merge lines
    
    Args:
        s: Input transliteration text
        config: Optional config dict with 'preserve_transliteration' settings
        
    Returns:
        Normalized transliteration (minimal changes)
    """
    if not s:
        return s
    
    # Default config
    if config is None:
        config = {
            'normalize': 'NFC',
            'allowed_punct': '-:.[]()/ʾʿ',
            'keep_subscripts': True
        }
    
    # Step 1: NFC normalization (REQUIRED)
    s = to_nfc(s)
    
    # Step 2: Collapse redundant whitespace only
    allowed_punct_set = set(config.get('allowed_punct', '-:.[]()/ʾʿ'))
    s = collapse_spaces_preserve_punct(s, allowed_punct_set)
    
    # Step 3: NO OTHER TRANSFORMATIONS
    # - Do NOT lowercase
    # - Do NOT strip punctuation
    # - Do NOT expand/contract
    # - Do NOT merge lines
    
    return s


def normalize_prose(s: str) -> str:
    """
    Normalize modern prose (Turkish, German, English, etc.).
    
    For non-transliteration text, we can be more aggressive:
    - Collapse whitespace
    - Strip some punctuation
    - Lowercase (configurable)
    
    But still preserve:
    - Turkish characters (ş, ğ, ı, ö, ü, ç)
    - German umlauts (ä, ö, ü, ß)
    - French accents (é, è, ê, à, ç)
    
    Args:
        s: Input prose text
        
    Returns:
        Normalized prose
    """
    if not s:
        return s
    
    # NFC for consistency
    s = to_nfc(s)
    
    # Collapse whitespace (more aggressive than transliteration)
    s = re.sub(r'\s+', ' ', s)
    s = s.strip()
    
    return s


def smart_normalize(s: str, config: dict = None) -> str:
    """
    Intelligently normalize text based on content type.
    
    Detects whether text is transliteration or prose and applies
    appropriate normalization rules.
    
    Args:
        s: Input text
        config: Optional config dict
        
    Returns:
        Normalized text with appropriate rules
    """
    if not s:
        return s
    
    # Check if this is transliteration
    if is_transliteration_line(s):
        return normalize_transliteration(s, config)
    else:
        return normalize_prose(s)


def validate_preservation(original: str, normalized: str) -> dict:
    """
    Validate that normalization preserved critical elements.
    
    Checks:
    - Diacritic count unchanged (š, ṣ, ṭ, ḫ, etc.)
    - Subscript count unchanged (₁₂₃₄₅₆₇₈₉₀)
    - Structural punctuation count unchanged (- : . [ ] ( ))
    
    Args:
        original: Original text before normalization
        normalized: Text after normalization
        
    Returns:
        Dict with validation results and counts
    """
    orig_nfc = to_nfc(original)
    norm_nfc = to_nfc(normalized)
    
    # Count diacritics
    orig_diacritics = sum(1 for c in orig_nfc if c in TRANSLIT_DIACRITICS)
    norm_diacritics = sum(1 for c in norm_nfc if c in TRANSLIT_DIACRITICS)
    
    # Count subscripts
    subscript_chars = set('₀₁₂₃₄₅₆₇₈₉')
    orig_subscripts = sum(1 for c in orig_nfc if c in subscript_chars)
    norm_subscripts = sum(1 for c in norm_nfc if c in subscript_chars)
    
    # Count structural punctuation
    orig_punct = sum(1 for c in orig_nfc if c in ALLOWED_PUNCT)
    norm_punct = sum(1 for c in norm_nfc if c in ALLOWED_PUNCT)
    
    return {
        'diacritics_preserved': orig_diacritics == norm_diacritics,
        'subscripts_preserved': orig_subscripts == norm_subscripts,
        'punctuation_preserved': orig_punct == norm_punct,
        'counts': {
            'diacritics': {'original': orig_diacritics, 'normalized': norm_diacritics},
            'subscripts': {'original': orig_subscripts, 'normalized': norm_subscripts},
            'punctuation': {'original': orig_punct, 'normalized': norm_punct}
        },
        'all_preserved': (
            orig_diacritics == norm_diacritics and
            orig_subscripts == norm_subscripts and
            orig_punct == norm_punct
        )
    }


if __name__ == '__main__':
    # Test cases
    test_cases = [
        # Akkadian transliteration
        "a-na DINGIR-lí-šu i-qí-a-am",
        "1 ma-na KÙ.BABBAR ša-ru-pá-am",
        "LUGAL-ri ù MUNUS.LUGAL-ti",
        
        # With subscripts
        "GIŠ.TUKUL₂ ša₂ LUGAL",
        
        # Modern Turkish
        "Anadolu'da bulunan eski metinler",
        
        # German
        "Die assyrischen Handelshäuser",
    ]
    
    print("=== Transliteration Normalization Tests ===\n")
    
    for i, text in enumerate(test_cases, 1):
        print(f"Test {i}: {text}")
        is_translit = is_transliteration_line(text)
        print(f"  Detected as transliteration: {is_translit}")
        
        normalized = smart_normalize(text)
        print(f"  Normalized: {normalized}")
        
        validation = validate_preservation(text, normalized)
        print(f"  Preservation check: {validation['all_preserved']}")
        if not validation['all_preserved']:
            print(f"    Details: {validation['counts']}")
        
        print()
