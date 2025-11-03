"""
Block Splitter - Split mixed-content OCR blocks into clean single-purpose fragments.

This module addresses the systematic issue where OCR creates blocks containing:
- Author names + transliteration + commentary (mixed content)
- Headers, captions, and metadata intermixed with content

Splitting happens BEFORE Akkadian detection and pairing, ensuring cleaner inputs.
"""

import re
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class BlockSplitter:
    """Split OCR blocks on structural boundaries to isolate content types."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize block splitter with configuration.
        
        Args:
            config: Optional configuration dict with split rules
        """
        self.config = config or {}
        self.enabled = self.config.get('split_enabled', True)
        
        # Compile regex patterns for efficiency
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile all regex patterns used for splitting."""
        
        # All-caps headers (8+ chars, letters/numbers/spaces/punctuation)
        self.header_pattern = re.compile(
            r'^[A-ZÄÖÜĞİŞÇ][A-Z0-9ÄÖÜĞİŞÇ \-:.]{7,}$',
            re.MULTILINE
        )
        
        # Author/citation lines with names
        # Matches: "S. BAYRAM – R. KÖZOĞLU", "Michel, Cécile - Title"
        self.author_pattern = re.compile(
            r'^[A-ZÄÖÜĞİŞÇ]\.\s+[A-ZÄÖÜĞİŞÇ][a-zäöüğışç]+.*?[–—-].*?[A-ZÄÖÜĞİŞÇ]',
            re.MULTILINE
        )
        
        # Citation markers (HW, vgl., s., Bkz., cf., etc.)
        self.citation_markers = re.compile(
            r'\b(HW|vgl\.|s\.|Bkz\.|cf\.|op\.cit\.|ibid\.|pp\.|f\.|ff\.|siehe)\b'
        )
        
        # Catalog/numbering: AKT 7a, KBo 12, Kt n/k 1295, Nr. 42
        self.catalog_pattern = re.compile(
            r'\b(AKT|KBo|KUB|BIN|TCL|RA|Kt|ICK|CCT|TC|VAT|Mus\.|ATHE|KTS|EL)\s?'
            r'\d+[:/a-z-]*\s?\d*|\bNr\.?\s?\d+\b|\bEnv\.\s?Nr\.\s?\d+',
            re.IGNORECASE
        )
        
        # Dense punctuation / reference style (many commas, semicolons, page refs)
        self.dense_punct_pattern = re.compile(
            r'^.{0,35}[,;:].{0,35}[,;:].{0,35}[,;:]',
            re.MULTILINE
        )
        
        # Em/en dashes and standalone colons as potential boundaries
        self.dash_boundary = re.compile(r'\s*[—–]\s*|\s*:\s*(?=[A-ZÄÖÜĞİŞÇ])')
        
        # Year patterns for bibliographic references
        self.year_pattern = re.compile(r'\b(19|20)\d{2}\b')
        
    def split_block(self, block_text: str, block_id: str = None) -> List[Dict[str, Any]]:
        """
        Split a block into sub-blocks based on structural boundaries.
        
        Args:
            block_text: The text content of the block
            block_id: Optional block identifier for logging
            
        Returns:
            List of sub-blocks, each with 'text' and 'split_reason' keys
        """
        if not self.enabled:
            return [{'text': block_text, 'split_reason': 'splitting_disabled'}]
        
        if not block_text or not block_text.strip():
            return []
        
        # Start with the full block
        fragments = [{'text': block_text, 'split_reason': 'original'}]
        
        # Apply splitting rules in sequence
        fragments = self._split_on_headers(fragments)
        fragments = self._split_on_authors(fragments)
        fragments = self._split_on_catalogs(fragments)
        fragments = self._split_on_dense_punctuation(fragments)
        fragments = self._split_on_line_length(fragments)
        
        # Clean up fragments
        cleaned = []
        for frag in fragments:
            text = frag['text'].strip()
            if text and len(text) > 5:  # Minimum fragment size
                frag['text'] = text
                cleaned.append(frag)
        
        logger.debug(f"Block {block_id}: split into {len(cleaned)} fragments")
        
        return cleaned if cleaned else [{'text': block_text, 'split_reason': 'no_split_applied'}]
    
    def _split_on_headers(self, fragments: List[Dict]) -> List[Dict]:
        """Split fragments on all-caps header lines."""
        result = []
        for frag in fragments:
            lines = frag['text'].split('\n')
            current = []
            
            for line in lines:
                if self.header_pattern.match(line.strip()):
                    # Header line - flush current fragment and start new
                    if current:
                        result.append({
                            'text': '\n'.join(current),
                            'split_reason': f"{frag['split_reason']}→header_boundary"
                        })
                        current = []
                    # Add header as its own fragment
                    result.append({
                        'text': line,
                        'split_reason': 'header'
                    })
                else:
                    current.append(line)
            
            if current:
                result.append({
                    'text': '\n'.join(current),
                    'split_reason': frag['split_reason']
                })
        
        return result
    
    def _split_on_authors(self, fragments: List[Dict]) -> List[Dict]:
        """Split fragments on author/citation lines."""
        result = []
        for frag in fragments:
            lines = frag['text'].split('\n')
            current = []
            
            for line in lines:
                if self.author_pattern.search(line):
                    # Author line - flush and isolate
                    if current:
                        result.append({
                            'text': '\n'.join(current),
                            'split_reason': f"{frag['split_reason']}→author_boundary"
                        })
                        current = []
                    result.append({
                        'text': line,
                        'split_reason': 'author_line'
                    })
                else:
                    current.append(line)
            
            if current:
                result.append({
                    'text': '\n'.join(current),
                    'split_reason': frag['split_reason']
                })
        
        return result
    
    def _split_on_catalogs(self, fragments: List[Dict]) -> List[Dict]:
        """Split fragments containing catalog numbers into separate blocks."""
        result = []
        for frag in fragments:
            lines = frag['text'].split('\n')
            current = []
            
            for line in lines:
                # If line is ONLY catalog/numbering (short line dominated by catalog pattern)
                if len(line.strip()) < 50 and self.catalog_pattern.search(line):
                    if current:
                        result.append({
                            'text': '\n'.join(current),
                            'split_reason': f"{frag['split_reason']}→catalog_boundary"
                        })
                        current = []
                    result.append({
                        'text': line,
                        'split_reason': 'catalog_line'
                    })
                else:
                    current.append(line)
            
            if current:
                result.append({
                    'text': '\n'.join(current),
                    'split_reason': frag['split_reason']
                })
        
        return result
    
    def _split_on_dense_punctuation(self, fragments: List[Dict]) -> List[Dict]:
        """Split fragments with dense punctuation (likely references)."""
        result = []
        for frag in fragments:
            lines = frag['text'].split('\n')
            current = []
            
            for line in lines:
                # Dense punctuation + citation markers + years = likely reference
                is_reference = (
                    self.dense_punct_pattern.match(line) and
                    (self.citation_markers.search(line) or self.year_pattern.search(line))
                )
                
                if is_reference:
                    if current:
                        result.append({
                            'text': '\n'.join(current),
                            'split_reason': f"{frag['split_reason']}→reference_boundary"
                        })
                        current = []
                    result.append({
                        'text': line,
                        'split_reason': 'reference_line'
                    })
                else:
                    current.append(line)
            
            if current:
                result.append({
                    'text': '\n'.join(current),
                    'split_reason': frag['split_reason']
                })
        
        return result
    
    def _split_on_line_length(self, fragments: List[Dict]) -> List[Dict]:
        """
        Split fragments where very short lines (< 35 chars) are mixed with longer content.
        Short lines with catalog/reference markers are likely metadata.
        """
        result = []
        for frag in fragments:
            lines = frag['text'].split('\n')
            current = []
            
            for line in lines:
                stripped = line.strip()
                is_short_metadata = (
                    len(stripped) < 35 and
                    (self.catalog_pattern.search(stripped) or
                     self.citation_markers.search(stripped) or
                     re.match(r'^\d+\s*$', stripped))  # Just a number
                )
                
                if is_short_metadata and current:
                    # Flush content, isolate metadata
                    result.append({
                        'text': '\n'.join(current),
                        'split_reason': f"{frag['split_reason']}→metadata_boundary"
                    })
                    current = []
                    result.append({
                        'text': line,
                        'split_reason': 'metadata_line'
                    })
                else:
                    current.append(line)
            
            if current:
                result.append({
                    'text': '\n'.join(current),
                    'split_reason': frag['split_reason']
                })
        
        return result


def split_blocks(blocks: List[Dict[str, Any]], config: Optional[Dict] = None) -> List[Dict[str, Any]]:
    """
    Convenience function to split a list of blocks.
    
    Args:
        blocks: List of block dicts with 'text' and 'block_id' keys
        config: Optional configuration for splitter
        
    Returns:
        List of sub-blocks with original block metadata preserved
    """
    splitter = BlockSplitter(config)
    result = []
    
    for block in blocks:
        fragments = splitter.split_block(
            block.get('text', ''),
            block.get('block_id', 'unknown')
        )
        
        # Preserve original block metadata in each fragment
        for i, frag in enumerate(fragments):
            sub_block = block.copy()
            sub_block['text'] = frag['text']
            sub_block['split_reason'] = frag['split_reason']
            sub_block['original_block_id'] = block.get('block_id')
            sub_block['block_id'] = f"{block.get('block_id', 'unk')}_f{i}"
            sub_block['fragment_index'] = i
            sub_block['total_fragments'] = len(fragments)
            result.append(sub_block)
    
    return result
