"""
Utility modules for OCR pipeline.
"""

from .shortpath import safe_path, ensure_dir, open_safe, get_safe_basename
from .encoding import has_mojibake, repair_mojibake, apply_fallback_fixes, has_expected_diacritics
from .triggers import should_trigger_llm, get_calibrated_threshold, has_diacritic_mismatch

__all__ = [
    # Path utilities
    'safe_path',
    'ensure_dir', 
    'open_safe',
    'get_safe_basename',
    # Encoding utilities
    'has_mojibake',
    'repair_mojibake',
    'apply_fallback_fixes',
    'has_expected_diacritics',
    # LLM trigger utilities
    'should_trigger_llm',
    'get_calibrated_threshold',
    'has_diacritic_mismatch',
]
