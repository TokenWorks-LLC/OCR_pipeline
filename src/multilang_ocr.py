"""
Multi-language OCR support with automatic language detection and lazy instance caching.
Supports tr/de/en/it/fr with fallback to latin for unsupported languages.
"""
import logging
import re
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass

try:
    from paddleocr import PaddleOCR
    import paddle
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class LanguageDetectionResult:
    """Result of language detection analysis."""
    primary_language: str
    confidence: float
    secondary_language: Optional[str] = None
    secondary_confidence: float = 0.0
    character_sets: Dict[str, float] = None

class MultiLanguageOCR:
    """Multi-language OCR with automatic language detection and lazy instance caching."""
    
    # Language mapping from ISO codes to PaddleOCR language codes
    LANGUAGE_MAPPING = {
        'en': 'en',
        'de': 'german',
        'fr': 'french', 
        'it': 'italian',
        'tr': 'tr',
        'latin': 'latin'  # Fallback for unsupported languages
    }
    
    # Character set patterns for language detection
    LANGUAGE_PATTERNS = {
        'tr': re.compile(r'[çğıöşüÇĞIİÖŞÜ]'),
        'de': re.compile(r'[äöüßÄÖÜ]'),
        'fr': re.compile(r'[àâäéèêëïîôùûüÿçÀÂÄÉÈÊËÏÎÔÙÛÜŸÇ]'),
        'it': re.compile(r'[àèéìíîòóùúÀÈÉÌÍÎÒÓÙÚ]'),
        'en': re.compile(r'^[a-zA-Z\s\d\.,;:!?\'"()-]+$')  # Basic Latin only
    }
    
    # Common words for language identification
    COMMON_WORDS = {
        'en': {'the', 'and', 'of', 'to', 'a', 'in', 'is', 'it', 'you', 'that', 'he', 'was', 'for', 'on', 'are', 'as', 'with', 'his', 'they'},
        'de': {'der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich', 'des', 'auf', 'für', 'ist', 'im', 'dem', 'nicht', 'ein', 'eine', 'als'},
        'fr': {'le', 'de', 'et', 'un', 'il', 'être', 'et', 'en', 'avoir', 'que', 'pour', 'dans', 'ce', 'son', 'une', 'sur', 'avec', 'ne', 'se', 'pas'},
        'it': {'il', 'di', 'che', 'e', 'la', 'per', 'un', 'in', 'essere', 'con', 'avere', 'tutto', 'da', 'su', 'come', 'anche', 'fare', 'solo', 'dire', 'anno'},
        'tr': {'ve', 'bir', 'bu', 'da', 'de', 'en', 'ile', 'için', 'o', 'olarak', 'kadar', 'çok', 'daha', 'var', 'gibi', 'ancak', 'sonra', 'üzerine', 'böyle', 'şey'}
    }
    
    def __init__(self, config_languages: List[str] = None, config_params: Dict[str, Dict[str, float]] = None):
        """Initialize multi-language OCR system.
        
        Args:
            config_languages: List of language codes to support (default: ['en','de','fr','it','tr'])
            config_params: Per-language OCR parameters override
        """
        self.supported_languages = config_languages or ['en', 'de', 'fr', 'it', 'tr']
        self.ocr_params = config_params or {}
        self.ocr_by_lang: Dict[str, PaddleOCR] = {}
        
        # Set up device detection
        self.device = "cpu"
        if PADDLE_AVAILABLE:
            try:
                import paddle
                self.device = "gpu" if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0 else "cpu"
                if self.device == "gpu":
                    paddle.device.set_device("gpu")
                logger.info(f"MultiLanguageOCR device: {self.device}")
            except Exception as e:
                logger.warning(f"Device detection failed: {e}")
    
    def detect_language(self, text: str) -> LanguageDetectionResult:
        """Detect language from text using character patterns and common words.
        
        Args:
            text: Input text to analyze
            
        Returns:
            LanguageDetectionResult with primary and optional secondary language
        """
        if not text or len(text.strip()) < 3:
            return LanguageDetectionResult('en', 0.1)  # Default fallback
        
        text_lower = text.lower()
        text_words = set(re.findall(r'\b\w+\b', text_lower))
        total_chars = len(text)
        
        scores = {}
        character_sets = {}
        
        # Score by character patterns
        for lang, pattern in self.LANGUAGE_PATTERNS.items():
            if lang == 'en':
                # English: penalize non-basic Latin characters
                non_basic = len([c for c in text if ord(c) > 127])
                char_score = max(0, 1.0 - (non_basic / total_chars * 2))
            else:
                # Other languages: reward characteristic characters
                matches = len(pattern.findall(text))
                char_score = min(1.0, matches / total_chars * 10)
            
            character_sets[lang] = char_score
            scores[lang] = char_score * 0.4  # Character patterns worth 40%
        
        # Score by common words
        for lang, common_words in self.COMMON_WORDS.items():
            word_matches = len(text_words.intersection(common_words))
            if text_words:
                word_score = word_matches / len(text_words)
                scores[lang] = scores.get(lang, 0) + word_score * 0.6  # Words worth 60%
        
        # Sort by score
        sorted_langs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        if not sorted_langs:
            return LanguageDetectionResult('en', 0.1, character_sets=character_sets)
        
        primary_lang, primary_score = sorted_langs[0]
        secondary_lang, secondary_score = (sorted_langs[1] if len(sorted_langs) > 1 else (None, 0.0))
        
        # Boost confidence if we have clear indicators
        if primary_score > 0.3:
            primary_score = min(1.0, primary_score + 0.2)
        
        return LanguageDetectionResult(
            primary_language=primary_lang,
            confidence=primary_score,
            secondary_language=secondary_lang if secondary_score > 0.2 else None,
            secondary_confidence=secondary_score,
            character_sets=character_sets
        )
    
    def get_ocr_instance(self, language: str) -> Optional[PaddleOCR]:
        """Get or create PaddleOCR instance for a specific language.
        
        Args:
            language: Language code (en, de, fr, it, tr)
            
        Returns:
            PaddleOCR instance or None if creation failed
        """
        if not PADDLE_AVAILABLE:
            logger.error("PaddleOCR not available")
            return None
        
        if language in self.ocr_by_lang:
            return self.ocr_by_lang[language]
        
        # Map language to PaddleOCR language code
        paddle_lang = self.LANGUAGE_MAPPING.get(language, 'latin')
        
        # Get language-specific parameters
        lang_params = self.ocr_params.get(language, {})
        
        try:
            # Base parameters
            params = {
                'use_textline_orientation': True,
                'lang': paddle_lang,
                'det_db_thresh': lang_params.get('det_db_thresh', 0.3),
                'det_db_box_thresh': lang_params.get('det_db_box_thresh', 0.6), 
                'rec_score_thresh': lang_params.get('rec_score_thresh', 0.5)
            }
            
            # Add GPU memory if on GPU
            if self.device == "gpu":
                params['gpu_mem'] = 2000  # Conservative allocation for multiple instances
            
            ocr_instance = PaddleOCR(**params)
            self.ocr_by_lang[language] = ocr_instance
            logger.info(f"Created PaddleOCR instance for {language} ({paddle_lang})")
            return ocr_instance
            
        except Exception as e:
            logger.error(f"Failed to create PaddleOCR instance for {language}: {e}")
            # Try fallback to latin
            if paddle_lang != 'latin':
                try:
                    ocr_instance = PaddleOCR(use_textline_orientation=True, lang='latin')
                    self.ocr_by_lang[language] = ocr_instance
                    logger.warning(f"Using latin fallback for {language}")
                    return ocr_instance
                except Exception as e2:
                    logger.error(f"Latin fallback also failed for {language}: {e2}")
            return None
    
    def process_text_with_language_detection(self, image, initial_text: str = None) -> Tuple[str, str, float, Dict]:
        """Process image with automatic language detection and appropriate OCR.
        
        Args:
            image: Input image (numpy array)
            initial_text: Optional pre-detected text for language analysis
            
        Returns:
            Tuple of (final_text, detected_language, confidence, metadata)
        """
        metadata = {
            'languages_detected': [],
            'ocr_attempts': [],
            'language_scores': {}
        }
        
        # If we have initial text, use it for language detection
        if initial_text and len(initial_text.strip()) > 10:
            lang_result = self.detect_language(initial_text)
            logger.debug(f"Language detection from initial text: {lang_result.primary_language} ({lang_result.confidence:.2f})")
            
            metadata['language_scores'] = lang_result.character_sets or {}
            
            # If confidence is high, use the detected language
            if lang_result.confidence > 0.5:
                target_lang = lang_result.primary_language
                ocr_instance = self.get_ocr_instance(target_lang)
                if ocr_instance:
                    try:
                        results = ocr_instance.ocr(image, cls=True)
                        if results and results[0]:
                            final_text = ' '.join([line[1][0] for line in results[0]])
                            metadata['ocr_attempts'].append({
                                'language': target_lang,
                                'success': True,
                                'text_length': len(final_text)
                            })
                            return final_text, target_lang, lang_result.confidence, metadata
                    except Exception as e:
                        logger.warning(f"OCR failed for detected language {target_lang}: {e}")
                        metadata['ocr_attempts'].append({
                            'language': target_lang,
                            'success': False,
                            'error': str(e)
                        })
        
        # Fallback: try multiple languages if detection was uncertain
        for lang in self.supported_languages:
            ocr_instance = self.get_ocr_instance(lang)
            if ocr_instance:
                try:
                    results = ocr_instance.ocr(image, cls=True)
                    if results and results[0]:
                        text = ' '.join([line[1][0] for line in results[0]])
                        if len(text.strip()) > 5:  # Valid text detected
                            # Re-detect language on the OCR result
                            lang_result = self.detect_language(text)
                            metadata['ocr_attempts'].append({
                                'language': lang,
                                'success': True,
                                'text_length': len(text),
                                'redetected_language': lang_result.primary_language,
                                'redetected_confidence': lang_result.confidence
                            })
                            metadata['language_scores'].update(lang_result.character_sets or {})
                            return text, lang_result.primary_language, lang_result.confidence, metadata
                except Exception as e:
                    logger.debug(f"OCR attempt failed for {lang}: {e}")
                    metadata['ocr_attempts'].append({
                        'language': lang,
                        'success': False, 
                        'error': str(e)
                    })
        
        # Ultimate fallback
        logger.warning("All language-specific OCR attempts failed, using basic latin")
        try:
            if 'latin' not in self.ocr_by_lang:
                self.ocr_by_lang['latin'] = PaddleOCR(use_textline_orientation=True, lang='latin')
            
            results = self.ocr_by_lang['latin'].ocr(image, cls=True)
            if results and results[0]:
                text = ' '.join([line[1][0] for line in results[0]])
                lang_result = self.detect_language(text)
                metadata['ocr_attempts'].append({
                    'language': 'latin',
                    'success': True,
                    'text_length': len(text)
                })
                return text, lang_result.primary_language, lang_result.confidence, metadata
        except Exception as e:
            logger.error(f"Even latin fallback failed: {e}")
            metadata['ocr_attempts'].append({
                'language': 'latin',
                'success': False,
                'error': str(e)
            })
        
        return "", "en", 0.0, metadata
    
    def cleanup(self):
        """Clean up OCR instances to free memory."""
        for lang, instance in self.ocr_by_lang.items():
            try:
                del instance
            except:
                pass
        self.ocr_by_lang.clear()
        logger.info("Cleaned up OCR instances")