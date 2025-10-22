"""
Configuration parameters for the OCR translation pipeline.
"""
import os
import re
from typing import Set, Dict, List, Tuple, Optional

# Target languages to extract
TARGET_LANGUAGES: Set[str] = {'fr', 'de', 'tr', 'en', 'it'}

# OCR confidence thresholds
CONFIDENCE_THRESHOLD: float = 0.85
QUICK_OCR_CONFIDENCE_THRESHOLD: float = 0.5

# LLaMA correction thresholds
LLAMA_CORRECTION_MIN_CONF: float = 0.70
LLAMA_CORRECTION_MAX_CONF: float = 0.92

# NMS (Non-Maximum Suppression) parameters
NMS_IOU_THRESHOLD: float = 0.3

# Rotation candidates for orientation detection
ROTATION_CANDIDATES: List[int] = [0, 90, 180, 270]

# PDF processing parameters
PDF_DPI: int = 220
PDF_MAX_PAGES: Optional[int] = None  # None = no limit

# LLaMA Configuration  
LLM_PROVIDER: str = os.getenv('LLM_PROVIDER', 'ollama')  # Enable LLM correction
LLM_MODEL: str = os.getenv('LLM_MODEL', 'mistral:latest')  # Use available model
LLM_BASE_URL: str = os.getenv('LLM_BASE_URL', 'http://localhost:11434')
LLM_TIMEOUT: int = int(os.getenv('LLM_TIMEOUT', '30'))  # Increase timeout to accommodate page-level requests

# Output directories
OUTPUT_DIRS = {
    'csv': 'out_csv',
    'audit': 'out_audit', 
    'overlay': 'out_overlay',
    'report': 'out_report'
}

# Image preprocessing parameters
PREPROCESSING = {
    'gamma_correction': 1.2,
    'clahe_clip_limit': 3.0,
    'clahe_grid_size': (8, 8),
    'binarize_window_size': 15,
    'binarize_k': 0.2,
    'dilation_kernel_size': 2,
    'deskew_angle_threshold': 45.0,
}

# Language label patterns (case-insensitive)
LANGUAGE_LABELS: Dict[str, List[str]] = {
    'fr': [r'fr\.', r'frz\.', r'français', r'francais'],
    'de': [r'de\.', r'dt\.', r'deu\.', r'deutsch'],
    'tr': [r'tr\.', r'türkçe', r'turc'],
    'en': [r'en\.', r'engl\.', r'english', r'angl\.'],
    'it': [r'it\.', r'ital\.', r'italiano'],
}

# Compiled regex patterns for translation extraction
LABEL_PATTERNS: Dict[str, re.Pattern] = {}
for lang, patterns in LANGUAGE_LABELS.items():
    combined_pattern = '|'.join(patterns)
    LABEL_PATTERNS[lang] = re.compile(
        rf'({combined_pattern})\s*[:–—]\s*(.+)',
        re.IGNORECASE | re.UNICODE
    )

# Inline series pattern (e.g., "fr.: text ; de.: text ; it.: text")
INLINE_SERIES_PATTERN = re.compile(
    r'(?:' + '|'.join([
        '|'.join(patterns) for patterns in LANGUAGE_LABELS.values()
    ]) + r')\s*[:–—]\s*[^;]+',
    re.IGNORECASE | re.UNICODE
)

# Tesseract configuration
TESSERACT_CONFIG = '--oem 1 --psm 6'
TESSERACT_LANGUAGES = 'deu+fra+tur+eng+ita'
TESSERACT_CMD = r'C:\Users\abdul\Desktop\OCR_pipeline\tesseract_suite\tesseract.exe'
TESSDATA_PREFIX = r'C:\Users\abdul\Desktop\OCR_pipeline\tessdata'

# PaddleOCR configuration
# Note: PaddleOCR will try multiple configurations automatically
PADDLE_CONFIG = {
    'use_textline_orientation': True,  # Updated parameter name
    'lang': 'en',  # Primary language (English works well for European languages)
}

# Quick OCR settings for orientation detection
QUICK_OCR_MAX_WORDS = 60

# DeepSeek-OCR Configuration
DEEPSEEK_MODEL = os.getenv('DEEPSEEK_MODEL', 'deepseek-ai/DeepSeek-OCR')
DEEPSEEK_DEVICE = os.getenv('DEEPSEEK_DEVICE', 'auto')  # 'auto', 'cpu', 'cuda'
DEEPSEEK_PROMPT = os.getenv('DEEPSEEK_PROMPT', '<image>\n<|grounding|>Convert the document to markdown.')
DEEPSEEK_BASE_SIZE = int(os.getenv('DEEPSEEK_BASE_SIZE', '1024'))
DEEPSEEK_IMAGE_SIZE = int(os.getenv('DEEPSEEK_IMAGE_SIZE', '640'))