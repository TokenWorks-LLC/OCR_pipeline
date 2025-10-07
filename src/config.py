"""
Configuration parameters for the OCR translation pipeline.
"""
import os
import re
from typing import Set, Dict, List, Tuple, Optional, Any

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
    'gamma_correction': 0.9,          # Slightly <1.0 lifts shadows, keeps faint text
    'clahe_clip_limit': 3.0,          # Contrast enhancement strength
    'clahe_grid_size': (8, 8),        # Tile size for CLAHE
    'binarize_window_size': 31,       # Odd & >=3 (Sauvola requirement)
    'binarize_k': 0.2,                # Contrast parameter for Sauvola
    'dilation_kernel_size': 1,        # Mild dilation; 1px by default
    'deskew_angle_threshold': 3.0,    # Max angle (degrees) for deskew sweep
    'apply_dilation': False           # Off by default to avoid CER regressions
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

# OCR Engine Configuration
OCR_ENGINE = {
    'engine': 'paddle',          # Default engine: 'paddle', 'doctr', 'mmocr', 'kraken'
    'profile': 'balanced',       # Performance profile: 'fast', 'balanced', 'quality'
    'device': 'auto',           # Device selection: 'auto', 'cpu', 'cuda'
    'fallback_engines': ['paddle'],  # Fallback engines if primary fails
    
    # Engine-specific configurations
    'paddle': {
        'use_textline_orientation': True,
        'lang': 'en',  # Primary language (English works well for European languages)
        'use_gpu': False,  # PaddleOCR GPU usage
    },
    
    'doctr': {
        'det_arch': 'db_resnet50',           # Detection architecture
        'reco_arch': 'sar',                  # Recognition architecture (sar, crnn_vgg16_bn)
        'pretrained': True,                  # Use pretrained models
        'assume_straight_pages': True,      # Assume straight text pages
        'export_as_straight_boxes': True,   # Export as straight boxes
        'word_level': False,                # Line-level by default
    },
    
    'mmocr': {
        'recognizer': 'abinet',             # Recognition model: 'abinet', 'parseq', 'crnn'
        'det_config': 'dbnetpp_resnet50-oclip_fpnc_1200e_icdar2015',  # Detection config
        'batch_mode': False,                # Batch processing mode
    },
    
    'kraken': {
        'model': None,                      # Recognition model (auto-detect if None)
        'segmentation_model': 'blla.mlmodel',  # Segmentation model
        'bidi': True,                       # Bidirectional text support
        'reading_order': True,              # Reading order detection
        'line_level': True,                 # Line-level recognition
    },
}

# Profile-specific configurations
OCR_PROFILES = {
    'fast': {
        'paddle': {'lang': 'en', 'use_gpu': False},
        'doctr': {'det_arch': 'db_resnet50', 'reco_arch': 'crnn_vgg16_bn'},
        'mmocr': {'recognizer': 'parseq', 'det_config': 'dbnet_resnet18_fpnc_1200e_icdar2015'},
        'kraken': {'bidi': False, 'reading_order': False},
    },
    
    'balanced': {
        'paddle': {'lang': 'en', 'use_gpu': False},
        'doctr': {'det_arch': 'db_resnet50', 'reco_arch': 'sar'},
        'mmocr': {'recognizer': 'abinet', 'det_config': 'dbnetpp_resnet50-oclip_fpnc_1200e_icdar2015'},
        'kraken': {'bidi': True, 'reading_order': True},
    },
    
    'quality': {
        'paddle': {'lang': 'en', 'use_gpu': False},
        'doctr': {'det_arch': 'db_resnet50', 'reco_arch': 'sar', 'assume_straight_pages': False},
        'mmocr': {'recognizer': 'abinet', 'det_config': 'dbnetpp_resnet50-oclip_fpnc_1200e_icdar2015'},
        'kraken': {'bidi': True, 'reading_order': True, 'model': None},
    },
}

# Quick OCR settings for orientation detection
QUICK_OCR_MAX_WORDS = 60


# Configuration helper functions
def get_ocr_engine_config(engine: str = None, profile: str = None) -> Dict[str, Any]:
    """
    Get OCR engine configuration with profile-specific overrides.
    
    Args:
        engine: OCR engine name (defaults to OCR_ENGINE['engine'])
        profile: Performance profile (defaults to OCR_ENGINE['profile'])
        
    Returns:
        Merged configuration dictionary
    """
    engine = engine or OCR_ENGINE['engine']
    profile = profile or OCR_ENGINE['profile']
    
    # Start with base engine config
    base_config = OCR_ENGINE.get(engine, {}).copy()
    
    # Apply profile-specific overrides
    if profile in OCR_PROFILES and engine in OCR_PROFILES[profile]:
        profile_config = OCR_PROFILES[profile][engine]
        base_config.update(profile_config)
    
    # Add global settings
    base_config['device'] = OCR_ENGINE.get('device', 'auto')
    
    return base_config


def get_available_ocr_engines() -> List[str]:
    """
    Get list of available OCR engines based on installed dependencies.
    
    Returns:
        List of available engine names
    """
    available = ['paddle']  # Always available
    
    # Check optional engines
    try:
        import doctr
        available.append('doctr')
    except ImportError:
        pass
    
    try:
        import mmocr
        available.append('mmocr')
    except ImportError:
        pass
    
    try:
        import kraken
        available.append('kraken')
    except ImportError:
        pass
    
    return available


def validate_ocr_config(engine: str, profile: str) -> bool:
    """
    Validate OCR engine and profile configuration.
    
    Args:
        engine: OCR engine name
        profile: Performance profile
        
    Returns:
        True if configuration is valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    available_engines = get_available_ocr_engines()
    
    if engine not in available_engines:
        raise ValueError(f"Engine '{engine}' not available. Available: {available_engines}")
    
    if profile not in OCR_PROFILES:
        available_profiles = list(OCR_PROFILES.keys())
        raise ValueError(f"Profile '{profile}' not available. Available: {available_profiles}")
    
    return True


def create_engine_config_for_evaluation(engine: str, profile: str, device: str = "auto") -> Dict[str, Any]:
    """
    Create OCR engine configuration specifically for evaluation runs.
    
    Args:
        engine: OCR engine name
        profile: Performance profile
        device: Device to use
        
    Returns:
        Configuration dictionary optimized for evaluation
    """
    validate_ocr_config(engine, profile)
    
    config = get_ocr_engine_config(engine, profile)
    config['device'] = device
    
    # Evaluation-specific optimizations
    if engine == 'paddle':
        config['use_gpu'] = (device == 'cuda')
    elif engine == 'doctr':
        # Ensure deterministic output for evaluation
        config['assume_straight_pages'] = True
        config['export_as_straight_boxes'] = True
    elif engine == 'mmocr':
        # Disable batch mode for consistent results
        config['batch_mode'] = False
    elif engine == 'kraken':
        # Ensure line-level output for evaluation
        config['line_level'] = True
    
    return config
