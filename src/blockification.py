#!/usr/bin/env python3
"""
Text blockification module for post-fusion processing.

Clusters lines into logical blocks using reading order detection,
then tags each block with language and Akkadian detection metadata.
"""

import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import numpy as np

from reading_order import ReadingOrderDetector, TextElement
from lang_and_akkadian import LanguageDetector
from akkadian_extract import is_akkadian_transliteration

logger = logging.getLogger(__name__)


@dataclass
class TextLine:
    """Single line of text with metadata."""
    text: str
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float
    line_id: str


@dataclass
class TextBlock:
    """Logical block of text with language/Akkadian metadata."""
    block_id: str
    page: int
    bbox: Tuple[int, int, int, int]  # Bounding box of entire block
    text: str  # Concatenated text from all lines
    mean_conf: float
    lines: List[TextLine]
    
    # Language detection results
    lang: str  # Detected language code (en, de, tr, fr, etc.)
    is_akk: bool  # True if Akkadian transliteration detected
    akk_conf: float  # Confidence of Akkadian detection (0.0-1.0)
    
    # Additional metadata
    column_index: int = 0
    reading_order: int = 0


class TextBlockifier:
    """
    Converts fusion output lines into structured blocks with language tags.
    
    Workflow:
    1. Group lines into spatial blocks using reading order
    2. Detect language for each block
    3. Detect Akkadian transliteration
    4. Return structured blocks ready for LLM routing
    """
    
    def __init__(
        self,
        line_merge_threshold: float = 1.5,
        block_min_lines: int = 1,
        akkadian_threshold: float = 0.5
    ):
        """
        Initialize blockifier.
        
        Args:
            line_merge_threshold: Vertical distance threshold for merging lines (×line_height)
            block_min_lines: Minimum lines to form a block
            akkadian_threshold: Confidence threshold for Akkadian detection
        """
        self.line_merge_threshold = line_merge_threshold
        self.block_min_lines = block_min_lines
        self.akkadian_threshold = akkadian_threshold
        
        # Initialize detectors
        self.reading_order = ReadingOrderDetector(
            line_height_threshold=line_merge_threshold
        )
        self.lang_detector = LanguageDetector()
        
        logger.info(f"TextBlockifier initialized (merge_threshold={line_merge_threshold})")
    
    def blockify(
        self,
        lines: List[Dict],
        page_num: int,
        page_width: int,
        page_height: int
    ) -> List[TextBlock]:
        """
        Convert fusion output lines into structured blocks.
        
        Args:
            lines: List of line dicts with {text, bbox, confidence, line_id}
            page_num: Page number
            page_width: Page width for column detection
            page_height: Page height
            
        Returns:
            List of TextBlock objects with language/Akkadian tags
        """
        if not lines:
            logger.warning(f"Page {page_num}: No lines to blockify")
            return []
        
        logger.debug(f"Page {page_num}: Blockifying {len(lines)} lines")
        
        # Convert to TextElement for reading order detection
        text_elements = []
        line_map = {}  # Map TextElement to original line data
        
        for i, line in enumerate(lines):
            bbox = line['bbox']  # (x, y, w, h)
            elem = TextElement(
                text=line['text'],
                x=bbox[0],
                y=bbox[1],
                width=bbox[2],
                height=bbox[3],
                conf=line.get('confidence', 0.0),
                bbox=bbox
            )
            text_elements.append(elem)
            line_map[id(elem)] = line
        
        # Detect columns and reading order
        columns = self.reading_order.detect_columns(text_elements, page_width)
        
        logger.debug(f"Page {page_num}: Detected {len(columns)} columns")
        
        # Group lines into blocks by proximity
        blocks = []
        block_counter = 0
        
        for col_idx, column_elements in enumerate(columns):
            # Group lines within column into blocks based on vertical proximity
            current_block_lines = []
            last_y = None
            
            for elem in column_elements:
                if last_y is None:
                    # Start first block
                    current_block_lines.append(elem)
                    last_y = elem.y + elem.height
                else:
                    # Check if line should be in same block
                    line_gap = elem.y - last_y
                    avg_height = (elem.height + current_block_lines[-1].height) / 2
                    
                    if line_gap <= self.line_merge_threshold * avg_height:
                        # Merge into current block
                        current_block_lines.append(elem)
                        last_y = elem.y + elem.height
                    else:
                        # Start new block
                        if len(current_block_lines) >= self.block_min_lines:
                            block = self._create_block(
                                current_block_lines,
                                line_map,
                                page_num,
                                col_idx,
                                block_counter
                            )
                            blocks.append(block)
                            block_counter += 1
                        
                        # Reset for new block
                        current_block_lines = [elem]
                        last_y = elem.y + elem.height
            
            # Add final block in column
            if len(current_block_lines) >= self.block_min_lines:
                block = self._create_block(
                    current_block_lines,
                    line_map,
                    page_num,
                    col_idx,
                    block_counter
                )
                blocks.append(block)
                block_counter += 1
        
        logger.info(
            f"Page {page_num}: Created {len(blocks)} blocks from {len(lines)} lines "
            f"({len(columns)} columns)"
        )
        
        return blocks
    
    def _create_block(
        self,
        elements: List[TextElement],
        line_map: Dict,
        page_num: int,
        col_idx: int,
        block_idx: int
    ) -> TextBlock:
        """
        Create a TextBlock from a group of TextElements.
        
        Args:
            elements: List of TextElement objects
            line_map: Mapping from element ID to original line data
            page_num: Page number
            col_idx: Column index
            block_idx: Block index
            
        Returns:
            TextBlock with language/Akkadian tags
        """
        # Extract lines
        text_lines = []
        confidences = []
        
        for elem in elements:
            original = line_map.get(id(elem))
            if original:
                text_lines.append(TextLine(
                    text=elem.text,
                    bbox=elem.bbox,
                    confidence=elem.conf,
                    line_id=original.get('line_id', f"line_{len(text_lines)}")
                ))
                confidences.append(elem.conf)
        
        # Compute block bbox (union of all line bboxes)
        min_x = min(line.bbox[0] for line in text_lines)
        min_y = min(line.bbox[1] for line in text_lines)
        max_x = max(line.bbox[0] + line.bbox[2] for line in text_lines)
        max_y = max(line.bbox[1] + line.bbox[3] for line in text_lines)
        
        block_bbox = (min_x, min_y, max_x - min_x, max_y - min_y)
        
        # Concatenate text
        block_text = '\n'.join(line.text for line in text_lines)
        
        # Compute mean confidence
        mean_conf = np.mean(confidences) if confidences else 0.0
        
        # Detect language
        lang, lang_conf = self.lang_detector.detect_language(block_text)
        
        # Detect Akkadian
        is_akk = is_akkadian_transliteration(block_text, threshold=self.akkadian_threshold)
        akk_conf = self._compute_akkadian_confidence(block_text)
        
        # Create block
        block_id = f"p{page_num}_c{col_idx}_b{block_idx}"
        
        block = TextBlock(
            block_id=block_id,
            page=page_num,
            bbox=block_bbox,
            text=block_text,
            mean_conf=mean_conf,
            lines=text_lines,
            lang=lang,
            is_akk=is_akk,
            akk_conf=akk_conf,
            column_index=col_idx,
            reading_order=block_idx
        )
        
        logger.debug(
            f"Block {block_id}: {len(text_lines)} lines, "
            f"lang={lang}, is_akk={is_akk}, akk_conf={akk_conf:.2f}, "
            f"mean_conf={mean_conf:.2f}"
        )
        
        return block
    
    def _compute_akkadian_confidence(self, text: str) -> float:
        """
        Compute Akkadian transliteration confidence score.
        
        Uses multiple signals:
        - Diacritic density
        - Determinative presence
        - Syllable separator density
        
        Args:
            text: Text to analyze
            
        Returns:
            Confidence score 0.0-1.0
        """
        if not text:
            return 0.0
        
        score = 0.0
        text_len = len(text)
        
        # Special Akkadian characters
        akk_chars = {'š', 'ṣ', 'ṭ', 'ḫ', 'ā', 'ē', 'ī', 'ū'}
        char_count = sum(1 for c in text if c.lower() in akk_chars)
        
        if char_count > 0:
            density = char_count / text_len
            score += min(0.5, density * 10)  # Cap at 0.5
        
        # Determinatives
        determinatives = {'ᵈ', 'ᵐ', 'ᶠ'}
        if any(c in text for c in determinatives):
            score += 0.3
        
        # Syllable separators (hyphens, dots in context)
        separators = text.count('-') + text.count(':')
        sep_density = separators / text_len
        if sep_density >= 0.05:
            score += 0.2
        
        return min(1.0, score)
    
    def get_statistics(self, blocks: List[TextBlock]) -> Dict:
        """Get blockification statistics."""
        if not blocks:
            return {}
        
        stats = {
            'total_blocks': len(blocks),
            'akkadian_blocks': sum(1 for b in blocks if b.is_akk),
            'non_akkadian_blocks': sum(1 for b in blocks if not b.is_akk),
            'language_breakdown': {},
            'avg_confidence': np.mean([b.mean_conf for b in blocks]),
            'avg_lines_per_block': np.mean([len(b.lines) for b in blocks]),
            'total_lines': sum(len(b.lines) for b in blocks)
        }
        
        # Language breakdown
        for block in blocks:
            lang_key = f"{block.lang}_akk" if block.is_akk else block.lang
            stats['language_breakdown'][lang_key] = stats['language_breakdown'].get(lang_key, 0) + 1
        
        return stats


def blockify_page(
    fusion_lines: List[Dict],
    page_num: int,
    page_width: int,
    page_height: int,
    config: Optional[Dict] = None
) -> List[TextBlock]:
    """
    Convenience function to blockify a single page.
    
    Args:
        fusion_lines: Lines from ROVER fusion
        page_num: Page number
        page_width: Page width in pixels
        page_height: Page height in pixels
        config: Optional configuration dict
        
    Returns:
        List of TextBlock objects
    """
    config = config or {}
    
    blockifier = TextBlockifier(
        line_merge_threshold=config.get('line_merge_threshold', 1.5),
        block_min_lines=config.get('block_min_lines', 1),
        akkadian_threshold=config.get('akkadian_threshold', 0.5)
    )
    
    return blockifier.blockify(fusion_lines, page_num, page_width, page_height)


if __name__ == '__main__':
    # Test blockification
    import sys
    
    # Mock test data
    test_lines = [
        {
            'text': 'lugal.e a.ab.ba ḫé.gál',
            'bbox': (100, 100, 300, 20),
            'confidence': 0.85,
            'line_id': 'l1'
        },
        {
            'text': 'König soll das Meer besitzen',
            'bbox': (100, 125, 320, 18),
            'confidence': 0.92,
            'line_id': 'l2'
        },
        {
            'text': 'The king shall possess the sea',
            'bbox': (100, 148, 310, 18),
            'confidence': 0.94,
            'line_id': 'l3'
        }
    ]
    
    blocks = blockify_page(test_lines, page_num=1, page_width=800, page_height=1200)
    
    print(f"\nCreated {len(blocks)} blocks:")
    for block in blocks:
        print(f"\n{block.block_id}:")
        print(f"  Text: {block.text[:50]}...")
        print(f"  Language: {block.lang}")
        print(f"  Is Akkadian: {block.is_akk} (conf={block.akk_conf:.2f})")
        print(f"  Mean confidence: {block.mean_conf:.2f}")
        print(f"  Lines: {len(block.lines)}")
