"""
Akkadian text detection and translation extraction module.
"""
import re
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

# Akkadian transliteration character classes
AKK_CHARS = "A-Za-zĀĒĪŪāēīūŠšṢṣṬṭḪḫ"
AKK_TOKEN = rf"(?:[{AKK_CHARS}]+(?:\.[{AKK_CHARS}]+)+|[{AKK_CHARS}]*[āēīūŠšṢṣṬṭḪḫ][{AKK_CHARS}]*)"

# Common Akkadian keywords for confidence boosting
AKKADIAN_KEYWORDS = {
    'lugal', 'dingir', 'dumu', 'ki', 'kur', 'inanna', 'enlil', 'nanna',
    'šarru', 'bēlu', 'awīlu', 'māru', 'mārtu', 'šūt', 'šutēšub',
    'ilu', 'ilānu', 'šamû', 'erṣetu', 'nāru', 'alu', 'bītu',
    'šá', 'ša', 'ana', 'itti', 'ina', 'eli', 'ištu', 'adi',
    'mu.an.na', 'itu', 'ud', 'kam', 'šu.si', 'gín', 'še.gur'
}

# Language detection patterns for translation labels
TRANSLATION_LABELS = {
    'turkish': [r'tr\.|türkçe|tercüme|çeviri|anlamı|manası'],
    'german': [r'de\.|deutsch|übersetzung|übers\.|bedeutung'],
    'french': [r'fr\.|français|traduction|signification'],
    'english': [r'en\.|english|translation|meaning|gloss'],
    'italian': [r'it\.|italiano|traduzione|significato'],
}

@dataclass
class TextElement:
    """Text element with position and metadata."""
    text: str
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    conf: float
    lang: str = 'unknown'

@dataclass
class AkkadianEntry:
    """Akkadian term with its translations."""
    akkadian: TextElement
    translations: List[Dict[str, Any]]  # lang, text, bbox, source, conf

@dataclass
class PageEntries:
    """All Akkadian entries for a page."""
    page_id: str
    entries: List[AkkadianEntry]
    rejected: List[Dict[str, Any]]  # Audit trail for rejected candidates

class AkkadianExtractor:
    """Extract Akkadian terms and their translations from OCR results."""
    
    def __init__(self, 
                 min_akk_conf: float = 0.80,
                 min_trans_conf: float = 0.80,
                 max_distance_pixels: int = 100):
        """
        Initialize Akkadian extractor.
        
        Args:
            min_akk_conf: Minimum confidence for Akkadian candidates
            min_trans_conf: Minimum confidence for translation candidates
            max_distance_pixels: Maximum distance for translation targeting
        """
        self.min_akk_conf = min_akk_conf
        self.min_trans_conf = min_trans_conf
        self.max_distance_pixels = max_distance_pixels
        
        # Compile regex patterns
        self.akk_pattern = re.compile(AKK_TOKEN, re.IGNORECASE)
        self.label_patterns = {}
        for lang, patterns in TRANSLATION_LABELS.items():
            self.label_patterns[lang] = [re.compile(p, re.IGNORECASE) for p in patterns]
    
    def detect_akkadian_candidates(self, text_elements: List[Dict]) -> List[TextElement]:
        """
        Detect Akkadian transliteration candidates from OCR results.
        
        Args:
            text_elements: List of OCR text elements with text, bbox, conf
            
        Returns:
            List of TextElement objects for Akkadian candidates
        """
        candidates = []
        
        for elem in text_elements:
            text = elem.get('text', '').strip()
            if not text:
                continue
                
            bbox = elem.get('bbox', (0, 0, 0, 0))
            conf = float(elem.get('conf', 0.0))
            
            # Skip if confidence too low
            if conf < self.min_akk_conf:
                continue
            
            # Check for Akkadian patterns
            is_akkadian = False
            confidence_boost = 0.0
            
            # Primary detection: diacritics or morphological dots
            if self.akk_pattern.search(text):
                is_akkadian = True
                
                # Boost confidence for known Akkadian terms
                text_lower = text.lower()
                for keyword in AKKADIAN_KEYWORDS:
                    if keyword in text_lower:
                        confidence_boost += 0.05
                        
                # Additional boost for common patterns
                if '.' in text and any(c in text for c in 'āēīūŠšṢṣṬṭḪḫ'):
                    confidence_boost += 0.10
            
            if is_akkadian:
                # Apply confidence boost but cap at 1.0
                adjusted_conf = min(1.0, conf + confidence_boost)
                
                candidates.append(TextElement(
                    text=text,
                    bbox=tuple(bbox),
                    conf=adjusted_conf,
                    lang='akkadian'
                ))
                
                logger.debug(f"Akkadian candidate: '{text}' (conf: {conf:.3f} -> {adjusted_conf:.3f})")
        
        return candidates
    
    def find_followed_translations(self, akk_elem: TextElement, 
                                 text_elements: List[Dict]) -> List[Dict[str, Any]]:
        """
        Find translations that follow an Akkadian element (Rule 1).
        
        Args:
            akk_elem: Akkadian text element
            text_elements: All text elements on the page
            
        Returns:
            List of translation candidates with metadata
        """
        translations = []
        akk_x, akk_y, akk_w, akk_h = akk_elem.bbox
        
        # Find elements that could be translations
        for elem in text_elements:
            elem_text = elem.get('text', '').strip()
            if not elem_text:
                continue
                
            elem_bbox = elem.get('bbox', (0, 0, 0, 0))
            elem_conf = float(elem.get('conf', 0.0))
            
            if elem_conf < self.min_trans_conf:
                continue
                
            elem_x, elem_y, elem_w, elem_h = elem_bbox
            
            # Check if element follows the Akkadian term
            # Same column (overlapping x-range) and below
            x_overlap = not (elem_x + elem_w < akk_x or akk_x + akk_w < elem_x)
            is_below = elem_y > akk_y + akk_h
            distance = elem_y - (akk_y + akk_h) if is_below else float('inf')
            
            if x_overlap and is_below and distance <= self.max_distance_pixels:
                # Skip if looks like another Akkadian term
                if self.akk_pattern.search(elem_text):
                    continue
                    
                # Detect language
                detected_lang = self._detect_language(elem_text)
                
                if detected_lang != 'akkadian':
                    translations.append({
                        'lang': detected_lang,
                        'text': elem_text,
                        'bbox': elem_bbox,
                        'source': 'rule:followed',
                        'conf': elem_conf,
                        'distance': distance
                    })
        
        return translations
    
    def find_labeled_translations(self, akk_elem: TextElement, 
                                text_elements: List[Dict]) -> List[Dict[str, Any]]:
        """
        Find translations with explicit language labels (Rule 2).
        
        Args:
            akk_elem: Akkadian text element
            text_elements: All text elements on the page
            
        Returns:
            List of translation candidates with metadata
        """
        translations = []
        akk_x, akk_y, akk_w, akk_h = akk_elem.bbox
        
        for elem in text_elements:
            elem_text = elem.get('text', '').strip()
            if not elem_text:
                continue
                
            elem_bbox = elem.get('bbox', (0, 0, 0, 0))
            elem_conf = float(elem.get('conf', 0.0))
            
            if elem_conf < self.min_trans_conf:
                continue
                
            elem_x, elem_y, elem_w, elem_h = elem_bbox
            
            # Check proximity to Akkadian term
            distance = abs(elem_y - akk_y) + abs(elem_x - akk_x)
            
            if distance > self.max_distance_pixels * 2:  # Larger search radius for labels
                continue
            
            # Check for language labels
            for lang, patterns in self.label_patterns.items():
                for pattern in patterns:
                    if pattern.search(elem_text):
                        # Extract translation text after the label
                        match = pattern.search(elem_text)
                        if match:
                            # Get text after the label
                            trans_text = elem_text[match.end():].strip()
                            trans_text = trans_text.lstrip(':').strip()
                            
                            if trans_text and not self.akk_pattern.search(trans_text):
                                translations.append({
                                    'lang': lang,
                                    'text': trans_text,
                                    'bbox': elem_bbox,
                                    'source': 'rule:labeled',
                                    'conf': elem_conf,
                                    'label_match': match.group()
                                })
                                break
        
        return translations
    
    def find_formatted_translations(self, akk_elem: TextElement, 
                                  text_elements: List[Dict]) -> List[Dict[str, Any]]:
        """
        Find translations with formatting separation (Rule 3).
        
        Args:
            akk_elem: Akkadian text element
            text_elements: All text elements on the page
            
        Returns:
            List of translation candidates with metadata
        """
        translations = []
        akk_text = akk_elem.text
        
        # Look for formatting separators within the same text element first
        separators = [r':', r'—', r'–', r';', r'\|', r'=']
        
        for sep_pattern in separators:
            pattern = rf'{re.escape(akk_text)}\s*{sep_pattern}\s*(.+)'
            
            # Check within same element and nearby elements
            for elem in text_elements:
                elem_text = elem.get('text', '').strip()
                elem_bbox = elem.get('bbox', (0, 0, 0, 0))
                elem_conf = float(elem.get('conf', 0.0))
                
                if elem_conf < self.min_trans_conf:
                    continue
                
                match = re.search(pattern, elem_text, re.IGNORECASE)
                if match:
                    trans_text = match.group(1).strip()
                    
                    if trans_text and not self.akk_pattern.search(trans_text):
                        detected_lang = self._detect_language(trans_text)
                        
                        if detected_lang != 'akkadian':
                            translations.append({
                                'lang': detected_lang,
                                'text': trans_text,
                                'bbox': elem_bbox,
                                'source': 'rule:format',
                                'conf': elem_conf,
                                'separator': sep_pattern
                            })
        
        return translations
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection based on character patterns."""
        text_lower = text.lower()
        
        # Turkish indicators
        if any(c in text for c in 'çğıöşüÇĞIİÖŞÜ'):
            return 'turkish'
        
        # German indicators
        if any(c in text for c in 'äöüßÄÖÜ'):
            return 'german'
        
        # French indicators  
        if any(c in text for c in 'àâäéèêëïîôùûüÿçÀÂÄÉÈÊËÏÎÔÙÛÜŸÇ'):
            return 'french'
        
        # Italian indicators
        if any(c in text for c in 'àèéìíîòóùúÀÈÉÌÍÎÒÓÙÚ'):
            return 'italian'
        
        # Check for language-specific words
        turkish_words = {'ve', 'ile', 'için', 'olan', 'bir', 'bu', 'şu', 'o'}
        german_words = {'und', 'der', 'die', 'das', 'ein', 'eine', 'mit', 'von'}
        french_words = {'et', 'le', 'la', 'les', 'un', 'une', 'de', 'du', 'avec'}
        italian_words = {'e', 'il', 'la', 'lo', 'un', 'una', 'di', 'da', 'con'}
        
        words = set(text_lower.split())
        
        if words & turkish_words:
            return 'turkish'
        elif words & german_words:
            return 'german'
        elif words & french_words:
            return 'french'
        elif words & italian_words:
            return 'italian'
        
        return 'english'  # Default fallback
    
    def deduplicate_translations(self, translations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate translations based on text similarity and bbox overlap."""
        if not translations:
            return []
        
        unique = []
        
        for trans in translations:
            is_duplicate = False
            
            for existing in unique:
                # Check text similarity (simple exact match for now)
                if trans['text'].strip().lower() == existing['text'].strip().lower():
                    is_duplicate = True
                    break
                    
                # Check bbox overlap
                x1, y1, w1, h1 = trans['bbox']
                x2, y2, w2, h2 = existing['bbox']
                
                # Calculate overlap
                x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                
                overlap_area = x_overlap * y_overlap
                area1 = w1 * h1
                area2 = w2 * h2
                
                if area1 > 0 and area2 > 0:
                    overlap_ratio = overlap_area / min(area1, area2)
                    if overlap_ratio > 0.8:  # 80% overlap threshold
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique.append(trans)
        
        return unique
    
    def collect_entries(self, page_id: str, text_elements: List[Dict]) -> PageEntries:
        """
        Collect all Akkadian entries and translations for a page.
        
        Args:
            page_id: Page identifier
            text_elements: OCR text elements from the page
            
        Returns:
            PageEntries object with all found entries and rejected candidates
        """
        logger.info(f"Collecting Akkadian entries for {page_id}")
        
        # Find Akkadian candidates
        akkadian_candidates = self.detect_akkadian_candidates(text_elements)
        
        entries = []
        rejected = []
        
        for akk_elem in akkadian_candidates:
            logger.debug(f"Processing Akkadian candidate: '{akk_elem.text}'")
            
            # Find translations using all three rules
            all_translations = []
            
            # Rule 1: Followed translations
            followed = self.find_followed_translations(akk_elem, text_elements)
            all_translations.extend(followed)
            
            # Rule 2: Labeled translations
            labeled = self.find_labeled_translations(akk_elem, text_elements)
            all_translations.extend(labeled)
            
            # Rule 3: Formatted translations
            formatted = self.find_formatted_translations(akk_elem, text_elements)
            all_translations.extend(formatted)
            
            # Deduplicate translations
            unique_translations = self.deduplicate_translations(all_translations)
            
            if unique_translations:
                entries.append(AkkadianEntry(
                    akkadian=akk_elem,
                    translations=unique_translations
                ))
                logger.debug(f"Found {len(unique_translations)} translations for '{akk_elem.text}'")
            else:
                rejected.append({
                    'akkadian_text': akk_elem.text,
                    'akkadian_bbox': akk_elem.bbox,
                    'reason': 'no_translations_found',
                    'conf': akk_elem.conf
                })
        
        # Log rejected low-confidence candidates
        low_conf_count = sum(1 for elem in text_elements 
                           if elem.get('conf', 0) < self.min_akk_conf 
                           and self.akk_pattern.search(elem.get('text', '')))
        
        if low_conf_count > 0:
            rejected.append({
                'reason': 'low_confidence_akkadian',
                'count': low_conf_count,
                'threshold': self.min_akk_conf
            })
        
        logger.info(f"Page {page_id}: {len(entries)} Akkadian entries, {len(rejected)} rejected")
        
        return PageEntries(
            page_id=page_id,
            entries=entries,
            rejected=rejected
        )
    
    def extract_translations_from_page(self, text_elements: List[Dict], page_num: int) -> List[Dict[str, Any]]:
        """
        Extract Akkadian translations from a page (pipeline-compatible method).
        
        Args:
            text_elements: OCR text elements from the page
            page_num: Page number
            
        Returns:
            List of translation dictionaries compatible with PDF generator
        """
        page_id = f"page_{page_num:03d}"
        entries = self.collect_entries(page_id, text_elements)
        
        translations = []
        
        for entry in entries.entries:
            akkadian_elem = entry.akkadian
            
            for trans in entry.translations:
                # Create translation dict compatible with PDF generator
                translation_dict = {
                    'akkadian_text': akkadian_elem.text,
                    'translation_text': trans['text'],
                    'translation_language': trans['lang'],
                    'strategy': trans['source'].replace('rule:', '').replace('followed', 'followed-by'),
                    'confidence': min(akkadian_elem.conf, trans['conf']),
                    'context': self._build_context(akkadian_elem, text_elements),
                    'akkadian_bbox': list(akkadian_elem.bbox),
                    'translation_bbox': list(trans['bbox'])
                }
                
                translations.append(translation_dict)
        
        return translations
    
    def _build_context(self, akk_elem: TextElement, text_elements: List[Dict], 
                      context_radius: int = 50) -> str:
        """Build context text around an Akkadian element."""
        akk_x, akk_y, akk_w, akk_h = akk_elem.bbox
        center_x, center_y = akk_x + akk_w/2, akk_y + akk_h/2
        
        # Find nearby elements
        nearby = []
        for elem in text_elements:
            elem_bbox = elem.get('bbox', (0, 0, 0, 0))
            elem_x, elem_y, elem_w, elem_h = elem_bbox
            elem_center_x, elem_center_y = elem_x + elem_w/2, elem_y + elem_h/2
            
            distance = ((elem_center_x - center_x)**2 + (elem_center_y - center_y)**2)**0.5
            
            if distance <= context_radius and elem.get('text', '').strip():
                nearby.append((distance, elem.get('text', '')))
        
        # Sort by distance and take closest elements
        nearby.sort(key=lambda x: x[0])
        context_texts = [text for _, text in nearby[:5]]  # Take up to 5 nearby elements
        
        return ' '.join(context_texts)


def extract_akkadian_from_page(page_id: str, text_elements: List[Dict],
                              extractor: AkkadianExtractor = None) -> PageEntries:
    """
    Convenience function to extract Akkadian entries from a page.
    
    Args:
        page_id: Page identifier
        text_elements: OCR results from the page
        extractor: Optional custom extractor (creates default if None)
        
    Returns:
        PageEntries with all found Akkadian-translation pairs
    """
    if extractor is None:
        extractor = AkkadianExtractor()
    
    return extractor.collect_entries(page_id, text_elements)
