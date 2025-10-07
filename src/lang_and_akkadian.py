#!/usr/bin/env python3
"""
Language detection and Akkadian transliteration identification for OCR routing.
"""
import logging
import re
from typing import Dict, Tuple, Optional, Set
import unicodedata

logger = logging.getLogger(__name__)


class LanguageDetector:
    """Lightweight language detection using character inventory and patterns."""
    
    def __init__(self):
        # Character inventories for different languages
        self.char_patterns = {
            'tr': {
                'special_chars': set('çğıöşüÇĞIİÖŞÜ'),
                'common_words': {'bir', 've', 'bu', 'için', 'ile', 'da', 'de', 'ki', 'en', 'olan'},
                'diacritics': set('çğıöşü'),
                'weight': 1.0
            },
            'de': {
                'special_chars': set('äöüßÄÖÜ'),
                'common_words': {'der', 'die', 'das', 'und', 'in', 'zu', 'den', 'von', 'mit', 'ist'},
                'diacritics': set('äöüß'),
                'weight': 1.0
            },
            'fr': {
                'special_chars': set('àâäéèêëïîôöùûüÿçÀÂÄÉÈÊËÏÎÔÖÙÛÜŸÇ'),
                'common_words': {'le', 'de', 'et', 'à', 'un', 'il', 'être', 'ce', 'avoir', 'que'},
                'diacritics': set('àâäéèêëïîôöùûüÿç'),
                'weight': 1.0
            },
            'it': {
                'special_chars': set('àèéìíîòóùúÀÈÉÌÍÎÒÓÙÚ'),
                'common_words': {'il', 'di', 'e', 'la', 'che', 'per', 'in', 'un', 'è', 'con'},
                'diacritics': set('àèéìíîòóùú'),
                'weight': 1.0
            },
            'en': {
                'special_chars': set(),  # English has no special chars
                'common_words': {'the', 'and', 'of', 'to', 'a', 'in', 'for', 'is', 'on', 'that'},
                'diacritics': set(),
                'weight': 0.8  # Lower weight as fallback
            }
        }
        
        # Akkadian transliteration patterns
        self.akkadian_chars = set('šṣṭḫāēīūâêîû')  # Special Akkadian chars
        self.akkadian_patterns = [
            r'[šṣṭḫ]',  # Special consonants
            r'[āēīūâêîû]',  # Long vowels and circumflex
            r'\b\w*[šṣṭḫ]\w*\b',  # Words with special consonants
            r'[0-9]+\.',  # Line numbers (common in transliterations)
        ]
        
        # Unicode ranges for cuneiform
        self.cuneiform_ranges = [
            (0x12000, 0x123FF),  # Cuneiform
            (0x12400, 0x1247F),  # Cuneiform Numbers and Punctuation
        ]
    
    def detect_language(self, text: str, fallback: str = 'en') -> Tuple[str, float]:
        """
        Detect language from text using character inventory and patterns.
        
        Args:
            text: Input text to analyze
            fallback: Default language if detection fails
            
        Returns:
            Tuple of (language_code, confidence_score)
        """
        if not text or len(text.strip()) < 3:
            return fallback, 0.1
        
        text_lower = text.lower()
        text_clean = re.sub(r'[^\w\s]', ' ', text_lower)
        words = text_clean.split()
        
        if not words:
            return fallback, 0.1
        
        scores = {}
        
        for lang, patterns in self.char_patterns.items():
            score = 0.0
            
            # Character-based scoring
            special_count = sum(1 for char in text if char in patterns['special_chars'])
            if len(text) > 0:
                char_ratio = special_count / len(text)
                score += char_ratio * 3.0  # Weight character evidence highly
            
            # Word-based scoring
            word_matches = sum(1 for word in words if word in patterns['common_words'])
            if len(words) > 0:
                word_ratio = word_matches / len(words)
                score += word_ratio * 2.0
            
            # Diacritic density
            diacritic_count = sum(1 for char in text if char in patterns['diacritics'])
            if len(text) > 0:
                diacritic_ratio = diacritic_count / len(text)
                score += diacritic_ratio * 1.5
            
            # Apply language weight
            score *= patterns['weight']
            scores[lang] = score
        
        # Find best match
        if scores:
            best_lang = max(scores, key=scores.get)
            best_score = scores[best_lang]
            
            # Normalize confidence to [0, 1]
            confidence = min(1.0, best_score)
            
            # Minimum confidence threshold
            if confidence < 0.1:
                return fallback, 0.1
            
            logger.debug(f"Language detection: {best_lang} (conf={confidence:.3f}) from scores={scores}")
            return best_lang, confidence
        
        return fallback, 0.1
    
    def is_akkadian_transliteration(self, text: str) -> Tuple[bool, float]:
        """
        Detect if text contains Akkadian transliteration.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Tuple of (is_akkadian, confidence_score)
        """
        if not text:
            return False, 0.0
        
        evidence_count = 0
        total_checks = 0
        
        # Check for special Akkadian characters
        akkadian_char_count = sum(1 for char in text if char in self.akkadian_chars)
        if akkadian_char_count > 0:
            evidence_count += 1
        total_checks += 1
        
        # Check for Akkadian patterns
        pattern_matches = 0
        for pattern in self.akkadian_patterns:
            if re.search(pattern, text):
                pattern_matches += 1
        
        if pattern_matches > 0:
            evidence_count += min(pattern_matches, 2)  # Cap at 2 points
        total_checks += 2
        
        # Check for cuneiform Unicode characters
        cuneiform_found = False
        for char in text:
            char_code = ord(char)
            for start, end in self.cuneiform_ranges:
                if start <= char_code <= end:
                    cuneiform_found = True
                    break
            if cuneiform_found:
                break
        
        if cuneiform_found:
            evidence_count += 2
        total_checks += 2
        
        # Check for transliteration-like patterns
        # Line numbers, scholarly notation
        if re.search(r'\b\d+[ab]?\.\s', text) or re.search(r'\[.*?\]', text):
            evidence_count += 1
        total_checks += 1
        
        # Calculate confidence
        confidence = evidence_count / total_checks if total_checks > 0 else 0.0
        is_akkadian = confidence > 0.3  # Threshold for positive detection
        
        if is_akkadian:
            logger.debug(f"Akkadian detected: evidence={evidence_count}/{total_checks}, conf={confidence:.3f}")
        
        return is_akkadian, confidence
    
    def analyze_text_line(self, text: str) -> Dict[str, any]:
        """
        Complete analysis of a text line for routing decisions.
        
        Args:
            text: Line text to analyze
            
        Returns:
            Dict with language, Akkadian detection, and confidence scores
        """
        if not text:
            return {
                'language': 'en',
                'language_confidence': 0.1,
                'is_akkadian': False,
                'akkadian_confidence': 0.0,
                'route_to_kraken': False
            }
        
        # Language detection
        language, lang_conf = self.detect_language(text)
        
        # Akkadian detection
        is_akkadian, akk_conf = self.is_akkadian_transliteration(text)
        
        # Routing decision
        route_to_kraken = is_akkadian and akk_conf > 0.5
        
        result = {
            'language': language,
            'language_confidence': lang_conf,
            'is_akkadian': is_akkadian,
            'akkadian_confidence': akk_conf,
            'route_to_kraken': route_to_kraken
        }
        
        logger.debug(f"Text analysis: lang={language}({lang_conf:.3f}), akkadian={is_akkadian}({akk_conf:.3f})")
        return result


# Try to import CLD3 for enhanced language detection
try:
    import cld3
    
    class EnhancedLanguageDetector(LanguageDetector):
        """Enhanced language detector using CLD3 + heuristics."""
        
        def detect_language(self, text: str, fallback: str = 'en') -> Tuple[str, float]:
            """Enhanced detection with CLD3 primary, heuristics fallback."""
            if not text or len(text.strip()) < 10:
                # Too short for CLD3, use heuristics
                return super().detect_language(text, fallback)
            
            try:
                # Try CLD3 first
                cld3_result = cld3.get_language(text)
                if cld3_result and cld3_result.probability > 0.8:
                    # Map CLD3 codes to our codes
                    lang_map = {
                        'en': 'en', 'de': 'de', 'fr': 'fr', 'it': 'it', 'tr': 'tr'
                    }
                    mapped_lang = lang_map.get(cld3_result.language, fallback)
                    confidence = cld3_result.probability
                    
                    logger.debug(f"CLD3 detection: {mapped_lang} (conf={confidence:.3f})")
                    return mapped_lang, confidence
                
            except Exception as e:
                logger.debug(f"CLD3 failed, using heuristics: {e}")
            
            # Fallback to heuristic detection
            return super().detect_language(text, fallback)
    
    # Use enhanced detector if CLD3 available
    _default_detector = EnhancedLanguageDetector()
    
except ImportError:
    logger.debug("CLD3 not available, using heuristic language detection")
    _default_detector = LanguageDetector()


# Global convenience functions
def detect_language(text: str, fallback: str = 'en') -> Tuple[str, float]:
    """Detect language from text."""
    return _default_detector.detect_language(text, fallback)


def is_akkadian_transliteration(text: str) -> Tuple[bool, float]:
    """Check if text is Akkadian transliteration."""
    return _default_detector.is_akkadian_transliteration(text)


def analyze_text_line(text: str) -> Dict[str, any]:
    """Complete text line analysis for routing."""
    return _default_detector.analyze_text_line(text)


# Test functionality
if __name__ == "__main__":
    detector = _default_detector
    
    # Test cases
    test_texts = [
        "This is an English text with common words.",
        "Das ist ein deutscher Text mit Umlauten: äöüß.",
        "Ceci est un texte français avec des accents: éàç.",
        "Bu Türkçe bir metindir ve özel karakterler içerir: çğıöşü.",
        "Questo è un testo italiano con caratteri speciali: àèéìòù.",
        "1. ana šarri šūbû 2. kīma awīlū ša šūt libbi 3. šarru ana muhhīšu",
        "a-na LUGAL šu-bu-u ki-ma LU₂ ša šu-ut lib-bi",
    ]
    
    for text in test_texts:
        print(f"\nText: {text}")
        result = detector.analyze_text_line(text)
        print(f"Language: {result['language']} (conf: {result['language_confidence']:.3f})")
        print(f"Akkadian: {result['is_akkadian']} (conf: {result['akkadian_confidence']:.3f})")
        print(f"Route to Kraken: {result['route_to_kraken']}")