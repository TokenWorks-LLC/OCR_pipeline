"""
Language detection and translation extraction from OCR results.
"""
import logging
import re
from typing import List, Dict, Optional, Set

try:
    import langid
    LANGID_AVAILABLE = True
    langdetect = None
except ImportError:
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0  # For consistent results
        LANGID_AVAILABLE = False
        LANGDETECT_AVAILABLE = True
        langid = None
    except ImportError:
        LANGID_AVAILABLE = False
        LANGDETECT_AVAILABLE = False
        langid = None
        langdetect = None

from config import TARGET_LANGUAGES, LABEL_PATTERNS, INLINE_SERIES_PATTERN
from ocr_utils import Line

logger = logging.getLogger(__name__)


def detect_lang(text: str) -> Optional[str]:
    """Detect language of text using langid or langdetect."""
    try:
        # Clean text for language detection
        clean_text = re.sub(r'[^\w\s]', ' ', text).strip()
        if len(clean_text) < 3:
            return None
        
        detected_lang = None
        confidence = 0.0
        
        if LANGID_AVAILABLE and langid:
            lang, confidence = langid.classify(clean_text)
            detected_lang = lang
        elif LANGDETECT_AVAILABLE:
            try:
                detected_lang = detect(clean_text)
                confidence = 0.8  # langdetect doesn't provide confidence, assume decent
            except:
                detected_lang = None
                confidence = 0.0
        else:
            logger.warning("No language detection library available")
            return None
        
        # Only return if confident and in our target languages
        if confidence > 0.7 and detected_lang in TARGET_LANGUAGES:
            return detected_lang
        
        return None
        
    except Exception as e:
        logger.debug(f"Language detection failed for '{text[:50]}': {e}")
        return None


def extract_labeled_translation(text: str) -> List[Dict[str, str]]:
    """Extract translations with explicit language labels."""
    extractions = []
    
    for lang, pattern in LABEL_PATTERNS.items():
        matches = pattern.finditer(text)
        for match in matches:
            label, translation = match.groups()
            translation = translation.strip()
            
            if translation:
                extractions.append({
                    'lang': lang,
                    'text': translation,
                    'method': 'labeled'
                })
                logger.debug(f"Found {lang} labeled translation: '{translation[:50]}'")
    
    return extractions


def extract_inline_series(text: str) -> List[Dict[str, str]]:
    """Extract inline series like 'fr.: text ; de.: text ; it.: text'."""
    extractions = []
    
    # Find all matches in the text
    matches = list(INLINE_SERIES_PATTERN.finditer(text))
    
    for match in matches:
        segment = match.group(0)
        
        # Parse each language segment
        for lang, pattern in LABEL_PATTERNS.items():
            lang_matches = pattern.finditer(segment)
            for lang_match in lang_matches:
                label, translation = lang_match.groups()
                translation = translation.strip().rstrip(';').strip()
                
                if translation:
                    extractions.append({
                        'lang': lang,
                        'text': translation,
                        'method': 'inline_series'
                    })
                    logger.debug(f"Found {lang} inline translation: '{translation[:50]}'")
    
    return extractions


def extract_unlabeled_translation(text: str, detected_lang: str) -> Optional[Dict[str, str]]:
    """Extract unlabeled text if language is clearly one of our targets."""
    if not detected_lang or detected_lang not in TARGET_LANGUAGES:
        return None
    
    # Skip if text contains obvious label patterns (already handled)
    for patterns in LABEL_PATTERNS.values():
        if patterns.search(text):
            return None
    
    # Basic heuristics to avoid Akkadian transliteration
    # (This is optional - could be enhanced with more sophisticated detection)
    akkadian_indicators = [
        r'\b[ŠšḪḫṢṣḶḷṬṭḪḫṢṣḶḷṬṭ]+\b',  # Akkadian characters
        r'\b[A-Z]{2,}\b.*\b[A-Z]{2,}\b',   # Many uppercase (transliteration style)
        r'\d+[a-z]?\.',                     # Line numbers common in transliterations
    ]
    
    for pattern in akkadian_indicators:
        if re.search(pattern, text):
            logger.debug(f"Skipping potential Akkadian text: '{text[:50]}'")
            return None
    
    return {
        'lang': detected_lang,
        'text': text.strip(),
        'method': 'unlabeled'
    }


def normalize_translation_text(text: str) -> str:
    """Normalize translation text by cleaning up formatting issues."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # De-hyphenate obvious line breaks (simple heuristic)
    text = re.sub(r'-\s+', '', text)
    
    # Remove isolated punctuation at start/end
    text = re.sub(r'^[;:,.-]+\s*', '', text)
    text = re.sub(r'\s*[;,.-]+$', '', text)
    
    return text.strip()


def extract_translations(lines: List[Line]) -> List[Dict]:
    """
    Extract translations from OCR lines.
    
    Returns list of dictionaries with keys:
    - lang: language code
    - text: translation text
    - bbox: bounding box tuple
    - conf: confidence score
    - engine: OCR engine used
    - method: extraction method used
    """
    translations = []
    processed_texts = set()  # To avoid duplicates
    
    for line in lines:
        text = line.text.strip()
        if not text or text in processed_texts:
            continue
        
        processed_texts.add(text)
        
        # Method 1: Labeled translations
        labeled = extract_labeled_translation(text)
        for extraction in labeled:
            normalized_text = normalize_translation_text(extraction['text'])
            if normalized_text:
                translations.append({
                    'lang': extraction['lang'],
                    'text': normalized_text,
                    'bbox': line.bbox,
                    'conf': line.conf,
                    'engine': line.engine,
                    'method': extraction['method']
                })
        
        # Method 2: Inline series
        if not labeled:  # Only if no labeled translations found
            inline = extract_inline_series(text)
            for extraction in inline:
                normalized_text = normalize_translation_text(extraction['text'])
                if normalized_text:
                    translations.append({
                        'lang': extraction['lang'],
                        'text': normalized_text,
                        'bbox': line.bbox,
                        'conf': line.conf,
                        'engine': line.engine,
                        'method': extraction['method']
                    })
        
        # Method 3: Unlabeled but clearly in target language
        if not labeled and not inline:  # Only if no other extractions found
            detected_lang = detect_lang(text)
            unlabeled = extract_unlabeled_translation(text, detected_lang)
            
            if unlabeled:
                normalized_text = normalize_translation_text(unlabeled['text'])
                if normalized_text:
                    translations.append({
                        'lang': unlabeled['lang'],
                        'text': normalized_text,
                        'bbox': line.bbox,
                        'conf': line.conf,
                        'engine': line.engine,
                        'method': unlabeled['method']
                    })
    
    logger.info(f"Extracted {len(translations)} translations from {len(lines)} OCR lines")
    
    # Log language distribution
    lang_counts = {}
    for t in translations:
        lang_counts[t['lang']] = lang_counts.get(t['lang'], 0) + 1
    
    logger.info(f"Language distribution: {lang_counts}")
    
    return translations
