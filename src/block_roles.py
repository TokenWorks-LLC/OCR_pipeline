"""
Block Role Tagger - Classify OCR blocks by their semantic role.

Roles:
- akkadian: Cuneiform transliteration/transcription
- translation: Modern language translation (de/tr/en/fr/it)
- reference_meta: Citations, bibliographic references, page numbers
- header_footer: Running heads, page numbers, chapter titles
- figure_caption: Figure/table captions
- other: Unclassified content

This tagging happens AFTER block splitting and BEFORE pairing, allowing
us to filter out reference metadata from translation candidates.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Set
from enum import Enum

logger = logging.getLogger(__name__)


class BlockRole(str, Enum):
    """Enumeration of possible block roles."""
    AKKADIAN = "akkadian"
    TRANSLATION = "translation"
    REFERENCE_META = "reference_meta"
    HEADER_FOOTER = "header_footer"
    FIGURE_CAPTION = "figure_caption"
    OTHER = "other"


class BlockRoleTagger:
    """Assign semantic roles to text blocks based on content analysis."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize role tagger with configuration.
        
        Args:
            config: Optional configuration dict
        """
        self.config = config or {}
        self.enabled = self.config.get('role_tagging', True)
        
        # Translation languages to recognize
        self.translation_langs = {'de', 'tr', 'en', 'fr', 'it', 'es', 'nl'}
        
        # Compile patterns
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for role detection."""
        
        # Reference/bibliographic patterns (EXPANDED for better coverage)
        self.citation_pattern = re.compile(
            r'\b(HW\s+s\.|HW|vgl\.|s\.|pp\.|Bkz\.|cf\.|op\.cit\.|ibid\.|siehe|vid\.|f\.|ff\.)\b',
            re.IGNORECASE
        )
        
        self.year_pattern = re.compile(r'\b(19|20)\d{2}[a-d]?\b')
        
        self.page_range_pattern = re.compile(r'\b\d+\s*[-–—]\s*\d+\b')
        
        # DOI, ISSN patterns
        self.doi_pattern = re.compile(r'\b(DOI|doi):\s*10\.\d+')
        self.issn_pattern = re.compile(r'\bISSN\s*\d{4}-\d{4}')
        
        # Catalog numbers (EXPANDED: added more flexible Kt patterns)
        self.catalog_pattern = re.compile(
            r'\b(AKT|KBo|KUB|BIN|TCL|RA|ICK|CCT|TC|VAT|Mus\.|ATHE|KTS|EL|PBS|YOS|MVAG)\s?'
            r'\d+[:/a-z-]*\s?\d*'
            r'|\bKt\.?\s+[a-z]/k[,\s]*\d*'  # FIXED: Allow optional period after Kt
            r'|\bNr\.?\s?\d+\b'
            r'|\bEnv\.\s?Nr\.\s?\d+',
            re.IGNORECASE
        )
        
        # Museum/archive numbers (EXPANDED: added Müze variant)
        self.museum_pattern = re.compile(
            r'\b(Env\.|Mus\.|Museum|M[uü]ze)\s+(Nr\.|No\.|env\.)?\s*\d+',
            re.IGNORECASE
        )
        
        # NEW: French/multilingual reference markers
        self.french_ref_pattern = re.compile(
            r'\b(sceau|tableau|pl\.|planches?|fig\.|figures?)\b',
            re.IGNORECASE
        )
        
        # NEW: "Notes" section headers in multiple languages
        self.notes_header_pattern = re.compile(
            r'\b(Notlar|Anmerkungen|Notes?|Bemerkungen):\s*$',
            re.IGNORECASE
        )
        
        # Page numbers (standalone)
        self.page_number_pattern = re.compile(r'^\s*\d+\s*$')
        
        # NEW: Author name + page number pattern (e.g., "250 S. BAYRAM-R KÖZOGLU")
        self.author_page_pattern = re.compile(
            r'^\d+\s+[A-ZÇĞİÖŞÜ]\.\s+[A-ZÇĞİÖŞÜ]',
            re.UNICODE
        )
        
        # Header/footer patterns
        self.running_head_pattern = re.compile(
            r'^[A-ZÄÖÜĞİŞÇ][A-Z0-9ÄÖÜĞİŞÇ \-:.]{10,}$'
        )
        
        # Figure/table captions
        self.caption_pattern = re.compile(
            r'^(Fig\.|Figure|Abb\.|Abbildung|Şekil|Resim|Table|Tablo|Tabelle|Map|Karte|Harita)\.?\s+\d+',
            re.IGNORECASE
        )
        
        # Akkadian transliteration markers
        self.akkadian_markers = re.compile(
            r'[šṣṭḫāēīūáéíúàèìùâêîûḫ]|'  # Diacritics
            r'\b(LUGAL|DUMU|ᵈ|ᵐ|ᶠ|KÙ\.BABBAR|AN\.NA|GIŠ|DUG|GÍR|IGI|SAG|KI|URU)\b|'  # Determinatives/sumerograms
            r'[A-Z]{2,}[-.]'  # Caps with hyphens (cuneiform convention)
        )
        
        # Strong Akkadian indicators (multiple tokens with diacritics)
        self.strong_akk_pattern = re.compile(
            r'([a-z]+-[a-z]+-[a-z]+)|'  # Triple-hyphenated tokens
            r'(\b[A-Z]{3,}\b.*\b[A-Z]{3,}\b)'  # Multiple sumerograms
        )
        
        # Reference-heavy brackets (common in citations)
        self.bracket_pattern = re.compile(r'\[[^\]]{5,}\]')
        
    def tag_block(self, block: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assign a role to a block based on content analysis.
        
        Args:
            block: Block dict with 'text' key
            
        Returns:
            Block dict with added 'role', 'role_confidence', 'role_reasons' keys
        """
        if not self.enabled:
            block['role'] = BlockRole.OTHER
            block['role_confidence'] = 0.0
            block['role_reasons'] = ['tagging_disabled']
            return block
        
        text = block.get('text', '').strip()
        if not text:
            block['role'] = BlockRole.OTHER
            block['role_confidence'] = 0.0
            block['role_reasons'] = ['empty_text']
            return block
        
        # Collect evidence for each role
        evidence = {
            BlockRole.REFERENCE_META: [],
            BlockRole.HEADER_FOOTER: [],
            BlockRole.FIGURE_CAPTION: [],
            BlockRole.AKKADIAN: [],
            BlockRole.TRANSLATION: []
        }
        
        # Check for reference/bibliographic indicators
        if self.citation_pattern.search(text):
            evidence[BlockRole.REFERENCE_META].append('citation_markers')
        
        if self.year_pattern.search(text):
            evidence[BlockRole.REFERENCE_META].append('year_patterns')
        
        if self.page_range_pattern.search(text):
            evidence[BlockRole.REFERENCE_META].append('page_ranges')
        
        if self.doi_pattern.search(text) or self.issn_pattern.search(text):
            evidence[BlockRole.REFERENCE_META].append('doi_issn')
        
        if self.catalog_pattern.search(text):
            evidence[BlockRole.REFERENCE_META].append('catalog_numbers')
        
        if self.museum_pattern.search(text):
            evidence[BlockRole.REFERENCE_META].append('museum_numbers')
        
        if self.french_ref_pattern.search(text):
            evidence[BlockRole.REFERENCE_META].append('french_references')
        
        if self.notes_header_pattern.search(text):
            evidence[BlockRole.REFERENCE_META].append('notes_header')
        
        if self.author_page_pattern.match(text):
            evidence[BlockRole.HEADER_FOOTER].append('author_page_header')
        
        if len(self.bracket_pattern.findall(text)) >= 2:
            evidence[BlockRole.REFERENCE_META].append('multiple_brackets')
        
        # Check for header/footer
        if self.page_number_pattern.match(text):
            evidence[BlockRole.HEADER_FOOTER].append('standalone_page_number')
        
        if self.running_head_pattern.match(text):
            evidence[BlockRole.HEADER_FOOTER].append('running_head')
        
        if len(text) < 100 and text.isupper():
            evidence[BlockRole.HEADER_FOOTER].append('short_all_caps')
        
        # Check for figure captions
        if self.caption_pattern.match(text):
            evidence[BlockRole.FIGURE_CAPTION].append('caption_prefix')
        
        # Check for Akkadian transliteration
        akk_marker_count = len(self.akkadian_markers.findall(text))
        if akk_marker_count >= 3:
            evidence[BlockRole.AKKADIAN].append(f'akkadian_markers_x{akk_marker_count}')
        
        if self.strong_akk_pattern.search(text):
            evidence[BlockRole.AKKADIAN].append('strong_akkadian_pattern')
        
        # Check language (if available from previous detection)
        detected_lang = block.get('lang', block.get('trans_lang', None))
        if detected_lang in self.translation_langs:
            evidence[BlockRole.TRANSLATION].append(f'lang_{detected_lang}')
        
        # Decision logic (priority order)
        role = BlockRole.OTHER
        confidence = 0.0
        reasons = []
        
        # High-priority exclusions first
        if len(evidence[BlockRole.HEADER_FOOTER]) >= 1:
            role = BlockRole.HEADER_FOOTER
            confidence = 0.9
            reasons = evidence[BlockRole.HEADER_FOOTER]
        
        elif len(evidence[BlockRole.FIGURE_CAPTION]) >= 1:
            role = BlockRole.FIGURE_CAPTION
            confidence = 0.95
            reasons = evidence[BlockRole.FIGURE_CAPTION]
        
        elif len(evidence[BlockRole.REFERENCE_META]) >= 2:
            # Multiple reference indicators = likely bibliographic
            role = BlockRole.REFERENCE_META
            confidence = min(0.9, 0.5 + 0.1 * len(evidence[BlockRole.REFERENCE_META]))
            reasons = evidence[BlockRole.REFERENCE_META]
        
        # NEW: Single STRONG reference indicators (catalog numbers, museum IDs, citation markers in short text are definitive)
        elif 'catalog_numbers' in evidence[BlockRole.REFERENCE_META] and len(text) < 150:
            # Short text with catalog pattern = definitely reference
            role = BlockRole.REFERENCE_META
            confidence = 0.85
            reasons = evidence[BlockRole.REFERENCE_META]
        
        elif 'citation_markers' in evidence[BlockRole.REFERENCE_META] and len(text) < 100:
            # Short text with "HW s.", "vgl.", "Bkz.", etc. = likely just a reference
            role = BlockRole.REFERENCE_META
            confidence = 0.80
            reasons = evidence[BlockRole.REFERENCE_META]
        
        elif 'museum_numbers' in evidence[BlockRole.REFERENCE_META] and len(text) < 150:
            # Short text with museum pattern = definitely reference
            role = BlockRole.REFERENCE_META
            confidence = 0.85
            reasons = evidence[BlockRole.REFERENCE_META]
        
        elif 'notes_header' in evidence[BlockRole.REFERENCE_META]:
            # "Notlar:" etc are section headers
            role = BlockRole.REFERENCE_META
            confidence = 0.90
            reasons = evidence[BlockRole.REFERENCE_META]
        
        elif 'citation_markers' in evidence[BlockRole.REFERENCE_META] and len(text) < 50:
            # Very short text with HW, vgl., Bkz. = citation
            role = BlockRole.REFERENCE_META
            confidence = 0.80
            reasons = evidence[BlockRole.REFERENCE_META]
        
        elif 'french_references' in evidence[BlockRole.REFERENCE_META] and 'catalog_numbers' in evidence[BlockRole.REFERENCE_META]:
            # Catalog + French term = reference
            role = BlockRole.REFERENCE_META
            confidence = 0.85
            reasons = evidence[BlockRole.REFERENCE_META]
        
        elif len(evidence[BlockRole.AKKADIAN]) >= 1:
            # Akkadian markers present
            role = BlockRole.AKKADIAN
            confidence = min(0.85, 0.6 + 0.1 * len(evidence[BlockRole.AKKADIAN]))
            reasons = evidence[BlockRole.AKKADIAN]
        
        elif len(evidence[BlockRole.TRANSLATION]) >= 1:
            # Has translation language but no other strong indicators
            role = BlockRole.TRANSLATION
            confidence = 0.7
            reasons = evidence[BlockRole.TRANSLATION]
        
        else:
            # Default to OTHER
            role = BlockRole.OTHER
            confidence = 0.3
            reasons = ['no_strong_indicators']
        
        # Store results
        block['role'] = role
        block['role_confidence'] = confidence
        block['role_reasons'] = reasons
        
        logger.debug(f"Block {block.get('block_id', 'unk')}: role={role.value}, "
                    f"confidence={confidence:.2f}, reasons={reasons}")
        
        return block
    
    def tag_blocks(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Tag a list of blocks with roles.
        
        Args:
            blocks: List of block dicts
            
        Returns:
            Same list with role metadata added to each block
        """
        for block in blocks:
            self.tag_block(block)
        
        return blocks


def tag_block_roles(blocks: List[Dict[str, Any]], config: Optional[Dict] = None) -> List[Dict[str, Any]]:
    """
    Convenience function to tag a list of blocks with semantic roles.
    
    Args:
        blocks: List of block dicts with 'text' key
        config: Optional configuration
        
    Returns:
        Blocks with role metadata added
    """
    tagger = BlockRoleTagger(config)
    return tagger.tag_blocks(blocks)


def filter_blocks_by_role(
    blocks: List[Dict[str, Any]],
    include_roles: Optional[Set[str]] = None,
    exclude_roles: Optional[Set[str]] = None
) -> List[Dict[str, Any]]:
    """
    Filter blocks based on their assigned roles.
    
    Args:
        blocks: List of blocks with 'role' key
        include_roles: If provided, only include blocks with these roles
        exclude_roles: If provided, exclude blocks with these roles
        
    Returns:
        Filtered list of blocks
    """
    result = []
    
    for block in blocks:
        role = block.get('role', BlockRole.OTHER)
        if isinstance(role, BlockRole):
            role_str = role.value
        else:
            role_str = str(role)
        
        # Check exclusions first
        if exclude_roles and role_str in exclude_roles:
            continue
        
        # Check inclusions
        if include_roles and role_str not in include_roles:
            continue
        
        result.append(block)
    
    return result
