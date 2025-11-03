#!/usr/bin/env python3
"""
Translation Pairing Module (Prompt 4)

Pairs Akkadian blocks with their translations using layout-aware scoring.

Scoring function:
    score(A,B) = α*exp(-dist_px/300)
               + β*same_or_adjacent_column
               + γ*lang∈{de,tr,en,fr,it}
               + δ*lexical_markers("Übersetzung|translation|çeviri|transl.")
               + ε*reading_order_consistency (B below/right of A)
               + ζ*font_size_ratio_in_range

Assignment:
    - Hungarian algorithm for optimal 1:1 pairing
    - Multi-target support for interlinear (A/T/A/T)
    - Cross-page continuation detection

Output:
    - translations.csv per PDF
    - HTML overlays with bboxes and arrows
"""

import re
import math
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Set
from pathlib import Path
import csv

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("scipy not available - will use greedy assignment fallback")

from blockification import TextBlock


logger = logging.getLogger(__name__)


@dataclass
class TranslationPair:
    """A paired Akkadian block with its translation."""
    pdf_id: str
    page: int
    
    # Akkadian block
    akk_block_id: str
    akk_text: str
    akk_bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    akk_column: int
    
    # Translation block
    trans_block_id: str
    trans_text: str
    trans_lang: str
    trans_bbox: Tuple[int, int, int, int]
    trans_column: int
    
    # Pairing metadata
    score: float
    distance_px: float
    same_column: bool
    has_marker: bool
    reading_order_ok: bool


@dataclass
class PairingConfig:
    """Configuration for translation pairing with tiered strategy."""
    # Scoring weights (must sum to ~1.0)
    weight_distance: float = 0.30
    weight_column: float = 0.30
    weight_language: float = 0.15
    weight_markers: float = 0.15
    weight_reading_order: float = 0.10
    
    # Distance parameters
    distance_threshold_px: int = 800
    distance_decay: float = 300.0
    
    # Language parameters
    target_languages: Set[str] = None
    
    # Lexical markers (case-insensitive)
    markers: List[str] = None
    
    # Commentary penalty
    commentary_penalty_words: List[str] = None
    commentary_penalty: float = 0.35
    
    # Tiered strategy configuration
    tier_config: Dict = None
    
    # Quality guardrails
    min_translation_chars: int = 25
    max_translation_punct_ratio: float = 0.35
    min_lang_conf: float = 0.30
    
    # Cross-page continuation
    check_next_page: bool = True
    check_prev_page: bool = True
    
    # Assignment method
    use_hungarian: bool = True
    allow_multi_target: bool = True
    interlinear_threshold: int = 50
    
    def __post_init__(self):
        """Set defaults for mutable fields."""
        if self.target_languages is None:
            self.target_languages = {'de', 'tr', 'en', 'fr', 'it'}
        
        if self.markers is None:
            self.markers = [
                'übersetzung', 'translation', 'çeviri', 'transl',
                'traduction', 'traduzione'
            ]
        
        if self.commentary_penalty_words is None:
            self.commentary_penalty_words = [
                'notlar', 'anmerkungen', 'remarks', 'discussion',
                'vgl.', 'bkz.', 'cf.', 'catalog', 'env.', 'sceau'
            ]
        
        if self.tier_config is None:
            self.tier_config = {
                'tier1': {
                    'column_strict': True,
                    'require_below': True,
                    'min_score': 0.55
                },
                'tier2': {
                    'adjacent_ok_with_markers': True,
                    'require_below': True,
                    'min_score': 0.50,
                    'adjacent_markers': ['übersetzung', 'çeviri', 'translation', 'traduction']
                },
                'tier3': {
                    'allow_interlinear': True,
                    'window_px': 60,
                    'min_score': 0.45
                }
            }
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'PairingConfig':
        """Create config from dictionary (e.g., loaded from JSON)."""
        # Map JSON fields to dataclass fields
        kwargs = {}
        
        # Direct mappings
        if 'weights' in config_dict:
            w = config_dict['weights']
            kwargs['weight_distance'] = w.get('distance', 0.30)
            kwargs['weight_column'] = w.get('column', 0.30)
            kwargs['weight_language'] = w.get('language', 0.15)
            kwargs['weight_markers'] = w.get('markers', 0.15)
            kwargs['weight_reading_order'] = w.get('reading_order', 0.10)
        
        if 'max_dist_px' in config_dict:
            kwargs['distance_threshold_px'] = config_dict['max_dist_px']
        
        if 'distance_decay' in config_dict:
            kwargs['distance_decay'] = config_dict['distance_decay']
        
        if 'languages' in config_dict:
            kwargs['target_languages'] = set(config_dict['languages'])
        
        if 'lexical_markers' in config_dict:
            kwargs['markers'] = config_dict['lexical_markers']
        
        if 'commentary_penalty_words' in config_dict:
            kwargs['commentary_penalty_words'] = config_dict['commentary_penalty_words']
        
        if 'commentary_penalty' in config_dict:
            kwargs['commentary_penalty'] = config_dict['commentary_penalty']
        
        if 'strategy' in config_dict:
            kwargs['tier_config'] = config_dict['strategy']
        
        if 'min_translation_chars' in config_dict:
            kwargs['min_translation_chars'] = config_dict['min_translation_chars']
        
        if 'max_translation_punct_ratio' in config_dict:
            kwargs['max_translation_punct_ratio'] = config_dict['max_translation_punct_ratio']
        
        if 'min_lang_conf' in config_dict:
            kwargs['min_lang_conf'] = config_dict['min_lang_conf']
        
        return cls(**kwargs)


class TranslationPairer:
    """
    Pairs Akkadian blocks with translation blocks using layout + language scoring.
    """
    
    def __init__(self, config: Optional[PairingConfig] = None):
        """
        Initialize pairer.
        
        Args:
            config: Pairing configuration (uses defaults if None)
        """
        self.config = config or PairingConfig()
        
        if not SCIPY_AVAILABLE and self.config.use_hungarian:
            logger.warning(
                "scipy not available, falling back to greedy assignment. "
                "Install scipy for optimal pairing: pip install scipy"
            )
            self.config.use_hungarian = False
        
        # Statistics
        self.stats = {
            'pages_processed': 0,
            'akkadian_blocks': 0,
            'translation_blocks': 0,
            'pairs_created': 0,
            'unpaired_akkadian': 0,
            'cross_page_pairs': 0,
            'interlinear_groups': 0
        }
        
        logger.info(
            f"TranslationPairer initialized "
            f"(method={'hungarian' if self.config.use_hungarian else 'greedy'})"
        )
        
        # Page geometry cache for normalized distance
        self.page_diagonal_cache = {}
    
    def _get_page_diagonal(self, blocks: List[TextBlock]) -> float:
        """Compute page diagonal from block bboxes (cached)."""
        if not blocks:
            return 1000.0  # Default fallback
        
        # Find page bounds
        max_x = max(b.bbox[0] + b.bbox[2] for b in blocks)
        max_y = max(b.bbox[1] + b.bbox[3] for b in blocks)
        
        return math.sqrt(max_x**2 + max_y**2)
    
    def _iou_x(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Compute horizontal IoU (intersection over union on x-axis)."""
        x1_start, _, w1, _ = bbox1
        x2_start, _, w2, _ = bbox2
        
        x1_end = x1_start + w1
        x2_end = x2_start + w2
        
        intersection = max(0, min(x1_end, x2_end) - max(x1_start, x2_start))
        union = max(x1_end, x2_end) - min(x1_start, x2_start)
        
        return intersection / max(1, union)
    
    def _overlap_y_ratio(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Compute vertical overlap ratio (intersection / min height)."""
        _, y1_start, _, h1 = bbox1
        _, y2_start, _, h2 = bbox2
        
        y1_end = y1_start + h1
        y2_end = y2_start + h2
        
        intersection = max(0, min(y1_end, y2_end) - max(y1_start, y2_start))
        min_height = min(h1, h2)
        
        return intersection / max(1, min_height)
    
    def _is_below_or_side_by_side(self, akk: TextBlock, trans: TextBlock) -> bool:
        """
        Generalized reading order check for multi-column layouts.
        Returns True if translation is below OR to the right with vertical overlap.
        """
        vertical_overlap = self._overlap_y_ratio(akk.bbox, trans.bbox)
        
        # Center x positions
        akk_cx = akk.bbox[0] + akk.bbox[2] / 2
        trans_cx = trans.bbox[0] + trans.bbox[2] / 2
        
        # Translation is to the right with significant vertical overlap (side-by-side)
        right_of = trans_cx >= akk_cx and vertical_overlap >= 0.50
        
        # Translation is below (with small overlap tolerance)
        below = trans.bbox[1] >= (akk.bbox[1] + akk.bbox[3] - 10)
        
        return below or right_of
    
    def pair_blocks(
        self,
        blocks: List[TextBlock],
        page: int,
        pdf_id: str,
        prev_page_blocks: Optional[List[TextBlock]] = None,
        next_page_blocks: Optional[List[TextBlock]] = None
    ) -> List[TranslationPair]:
        """
        Pair Akkadian blocks with translations on a page.
        
        Args:
            blocks: All blocks on current page
            page: Page number
            pdf_id: PDF identifier
            prev_page_blocks: Blocks from previous page (for continuation)
            next_page_blocks: Blocks from next page (for continuation)
            
        Returns:
            List of TranslationPair objects
        """
        self.stats['pages_processed'] += 1
        
        # Separate Akkadian and translation blocks
        akk_blocks = [b for b in blocks if b.is_akk]
        trans_blocks = [b for b in blocks if not b.is_akk and b.lang in self.config.target_languages]
        
        self.stats['akkadian_blocks'] += len(akk_blocks)
        self.stats['translation_blocks'] += len(trans_blocks)
        
        if not akk_blocks:
            logger.debug(f"Page {page}: No Akkadian blocks found")
            return []
        
        if not trans_blocks:
            logger.warning(f"Page {page}: No translation blocks found for {len(akk_blocks)} Akkadian blocks")
            self.stats['unpaired_akkadian'] += len(akk_blocks)
            return []
        
        # Check for interlinear pattern (A/T/A/T with small gaps)
        is_interlinear = self._detect_interlinear(akk_blocks, trans_blocks)
        
        if is_interlinear and self.config.allow_multi_target:
            logger.info(f"Page {page}: Detected interlinear layout")
            self.stats['interlinear_groups'] += 1
            pairs = self._pair_interlinear(akk_blocks, trans_blocks, page, pdf_id)
        else:
            # Standard 1:1 pairing
            pairs = self._pair_standard(akk_blocks, trans_blocks, page, pdf_id)
        
        # Try cross-page continuation for unpaired blocks
        if prev_page_blocks and self.config.check_prev_page:
            pairs.extend(self._pair_cross_page(
                akk_blocks, prev_page_blocks, page, pdf_id, direction='prev'
            ))
        
        if next_page_blocks and self.config.check_next_page:
            pairs.extend(self._pair_cross_page(
                akk_blocks, next_page_blocks, page, pdf_id, direction='next'
            ))
        
        self.stats['pairs_created'] += len(pairs)
        
        logger.info(
            f"Page {page}: Created {len(pairs)} pairs from "
            f"{len(akk_blocks)} Akkadian / {len(trans_blocks)} translation blocks"
        )
        
        return pairs
    
    def _pair_standard(
        self,
        akk_blocks: List[TextBlock],
        trans_blocks: List[TextBlock],
        page: int,
        pdf_id: str
    ) -> List[TranslationPair]:
        """
        Tiered pairing strategy:
        - Tier1: same-column/below strict
        - Tier2: adjacent columns OK if markers present
        - Tier3: interlinear fallback
        """
        n_akk = len(akk_blocks)
        n_trans = len(trans_blocks)
        
        # Compute page diagonal for normalized distance
        all_blocks = akk_blocks + trans_blocks
        page_diagonal = self._get_page_diagonal(all_blocks)
        logger.debug(f"[PAIR] page={page} page_diagonal={page_diagonal:.1f}")
        
        pairs = []
        
        # Try each tier in sequence
        for tier in ['tier1', 'tier2', 'tier3']:
            logger.info(f"[PAIR] page={page} trying tier={tier}")
            
            # Build cost matrix for this tier
            cost_matrix = []
            for i, akk in enumerate(akk_blocks):
                row = []
                for j, trans in enumerate(trans_blocks):
                    score = self._compute_score(akk, trans, tier=tier, page_diagonal=page_diagonal)
                    row.append(-score if score > 0 else 9999)  # Negative for maximization
                cost_matrix.append(row)
            
            # Solve assignment
            if self.config.use_hungarian and SCIPY_AVAILABLE:
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                assignments = list(zip(row_ind, col_ind))
            else:
                assignments = self._greedy_assignment(cost_matrix)
            
            # Filter by min_score threshold for this tier
            tier_config = getattr(self.config, 'tier_config', {}).get(tier, {})
            min_score = tier_config.get('min_score', 0.45)
            
            tier_pairs = []
            for i, j in assignments:
                score = -cost_matrix[i][j] if cost_matrix[i][j] != 9999 else 0.0
                
                if score < min_score:
                    continue
                
                akk = akk_blocks[i]
                trans = trans_blocks[j]
                pair = self._create_pair(akk, trans, page, pdf_id, score)
                tier_pairs.append(pair)
            
            logger.info(
                f"[PAIR] page={page} tier={tier} → {len(tier_pairs)} pairs "
                f"(≥{min_score} threshold from {n_akk}×{n_trans} candidates)"
            )
            
            if tier_pairs:
                # Found valid pairs, use them and stop
                pairs = tier_pairs
                break
            else:
                logger.info(f"[PAIR] page={page} tier={tier} fallthrough → trying next tier")
        
        if not pairs:
            logger.warning(
                f"[PAIR] page={page} no pairs found after all tiers "
                f"({n_akk} Akkadian, {n_trans} translation blocks)"
            )
        
        return pairs
    
    def _pair_interlinear(
        self,
        akk_blocks: List[TextBlock],
        trans_blocks: List[TextBlock],
        page: int,
        pdf_id: str
    ) -> List[TranslationPair]:
        """
        Pair interlinear layout (A/T/A/T with tight spacing).
        Allows multiple translations per Akkadian block.
        """
        pairs = []
        
        # Sort by vertical position
        akk_sorted = sorted(akk_blocks, key=lambda b: b.bbox[1])
        trans_sorted = sorted(trans_blocks, key=lambda b: b.bbox[1])
        
        # Greedy matching: for each Akkadian, take nearby translations
        for akk in akk_sorted:
            akk_y = akk.bbox[1]
            
            # Find translations within interlinear threshold
            candidates = [
                t for t in trans_sorted
                if abs(t.bbox[1] - akk_y) < self.config.interlinear_threshold
            ]
            
            for trans in candidates:
                score = self._compute_score(akk, trans)
                if score > 0:
                    pair = self._create_pair(akk, trans, page, pdf_id, score)
                    pairs.append(pair)
        
        return pairs
    
    def _pair_cross_page(
        self,
        akk_blocks: List[TextBlock],
        other_page_blocks: List[TextBlock],
        page: int,
        pdf_id: str,
        direction: str
    ) -> List[TranslationPair]:
        """
        Pair Akkadian blocks with translations from adjacent page.
        Only considers header/footer regions.
        """
        # TODO: Implement cross-page logic
        # For now, return empty
        return []
    
    def _base_constraints(self, akk: TextBlock, trans: TextBlock) -> float:
        """
        Check basic hygiene constraints before scoring.
        
        Returns:
            0.0 if valid, -1.0 if should be rejected
        """
        # Role-based hard exclude
        trans_role = str(getattr(trans, 'role', 'other'))
        if trans_role in {'reference_meta', 'header_footer', 'figure_caption'}:
            return -1.0
        
        # Basic text length check
        text = (trans.text or '').strip()
        min_chars = getattr(self.config, 'min_translation_chars', 25)
        if len(text) < min_chars:
            return -1.0
        
        # Punct ratio check (avoid catalog numbers, page refs)
        punct_chars = sum(1 for c in text if c in ',.;:·•|/\\[]()')
        punct_ratio = punct_chars / max(1, len(text))
        max_punct = getattr(self.config, 'max_translation_punct_ratio', 0.35)
        if punct_ratio > max_punct:
            return -1.0
        
        # Language confidence check
        lang_conf = getattr(trans, 'lang_confidence', 1.0)
        min_conf = getattr(self.config, 'min_lang_conf', 0.30)
        if lang_conf < min_conf:
            return -1.0
        
        return 0.0
    
    def _compute_score(self, akk: TextBlock, trans: TextBlock, tier: str = 'tier1', page_diagonal: float = 1000.0) -> float:
        """
        Compute pairing score with tier-specific gates and normalized distance.
        
        Args:
            akk: Akkadian block
            trans: Translation block
            tier: One of 'tier1', 'tier2', 'tier3'
            page_diagonal: Page diagonal for distance normalization
            
        Returns:
            Score in [0, 1], or 0.0 if tier gates reject
        """
        # Check base constraints first
        if self._base_constraints(akk, trans) < 0:
            return 0.0
        
        # Compute geometric features with robust fallbacks
        # Column detection with IoU fallback
        akk_col = getattr(akk, 'column_index', None)
        trans_col = getattr(trans, 'column_index', None)
        
        if akk_col is not None and trans_col is not None:
            same_col = (akk_col == trans_col)
        else:
            # Fallback: horizontal IoU >= 0.60 means same column
            same_col = self._iou_x(akk.bbox, trans.bbox) >= 0.60
        
        # Generalized reading order (handles multi-column)
        reading_order_ok = self._is_below_or_side_by_side(akk, trans)
        
        # Normalized distance
        euclid_dist = self._bbox_distance(akk.bbox, trans.bbox)
        euclid_norm = euclid_dist / max(1.0, page_diagonal)
        
        # Reject if too far
        max_dist_norm = getattr(self.config, 'max_dist_norm', 0.50)
        if euclid_norm > max_dist_norm:
            return 0.0
        
        # TIER GATES
        tier_config = getattr(self.config, 'tier_config', {}).get(tier, {})
        
        if tier == 'tier1':
            # Strict: same column AND reading-order-ok
            if not same_col:
                return 0.0
            if not reading_order_ok:
                return 0.0
        
        elif tier == 'tier2':
            # Adjacent OK if markers present, but reading-order required
            if not reading_order_ok:
                return 0.0
            if not same_col:
                # Adjacent column requires markers
                adjacent_ok = tier_config.get('adjacent_ok_with_markers', False)
                if not adjacent_ok:
                    return 0.0
                # Check for markers
                mark_score = self._marker_score(trans.text)
                if mark_score < 0.5:
                    return 0.0
        
        elif tier == 'tier3':
            # Interlinear: allow vertical proximity OR overlap
            window_px = tier_config.get('window_px', 120)
            vertical_gap = abs(trans.bbox[1] - (akk.bbox[1] + akk.bbox[3]))
            vertical_overlap = self._overlap_y_ratio(akk.bbox, trans.bbox)
            
            if vertical_gap > window_px and vertical_overlap < 0.5:
                return 0.0
        
        # Compute score components
        score = 0.0
        
        # Distance (normalized exponential decay)
        dist_decay = getattr(self.config, 'distance_decay', 0.35)
        dist_score = math.exp(-euclid_norm / dist_decay)
        score += self.config.weight_distance * dist_score
        
        # Column
        col_score = 1.0 if same_col else 0.0
        score += self.config.weight_column * col_score
        
        # Language
        if trans.lang in self.config.target_languages:
            score += self.config.weight_language
        
        # Markers
        mark_score = self._marker_score(trans.text)
        score += self.config.weight_markers * mark_score
        
        # Reading order
        order_score = 1.0 if reading_order_ok else 0.0
        score += self.config.weight_reading_order * order_score
        
        # Commentary penalty
        penalty = self._commentary_penalty(trans.text)
        score -= penalty
        
        return max(score, 0.0)
    
    def _marker_score(self, text: str) -> float:
        """Compute marker presence score."""
        text_lower = (text or '').lower()
        if any(marker in text_lower for marker in self.config.markers):
            return 1.0
        return 0.0
    
    def _commentary_penalty(self, text: str) -> float:
        """Penalize scholarly commentary text."""
        text_lower = (text or '').lower()
        penalty_words = getattr(self.config, 'commentary_penalty_words', [])
        if any(word in text_lower for word in penalty_words):
            return getattr(self.config, 'commentary_penalty', 0.35)
        return 0.0
    
    def _bbox_distance(
        self,
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int]
    ) -> float:
        """
        Compute distance between two bboxes (Euclidean distance between centers).
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        center1_x = x1 + w1 / 2
        center1_y = y1 + h1 / 2
        center2_x = x2 + w2 / 2
        center2_y = y2 + h2 / 2
        
        return math.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
    
    def _check_markers(self, text: str) -> bool:
        """Check if text contains lexical markers."""
        text_lower = text.lower()
        return any(marker in text_lower for marker in self.config.markers)
    
    def _check_reading_order(self, akk: TextBlock, trans: TextBlock) -> bool:
        """
        Check if translation follows Akkadian in reading order.
        Translation should be below or to the right of Akkadian.
        """
        akk_x, akk_y, akk_w, akk_h = akk.bbox
        trans_x, trans_y, trans_w, trans_h = trans.bbox
        
        # Below: trans_y > akk_y + akk_h
        is_below = trans_y >= akk_y
        
        # To the right: trans_x > akk_x + akk_w
        is_right = trans_x >= akk_x
        
        return is_below or is_right
    
    def _detect_interlinear(
        self,
        akk_blocks: List[TextBlock],
        trans_blocks: List[TextBlock]
    ) -> bool:
        """
        Detect if layout is interlinear (A/T/A/T alternating pattern).
        """
        # Simple heuristic: check if multiple A and T blocks with small vertical gaps
        if len(akk_blocks) < 2 or len(trans_blocks) < 2:
            return False
        
        all_blocks = sorted(
            akk_blocks + trans_blocks,
            key=lambda b: b.bbox[1]  # Sort by y position
        )
        
        # Check for A/T alternation
        alternations = 0
        for i in range(len(all_blocks) - 1):
            curr_is_akk = all_blocks[i].is_akk
            next_is_akk = all_blocks[i + 1].is_akk
            
            if curr_is_akk != next_is_akk:
                # Check gap
                curr_bottom = all_blocks[i].bbox[1] + all_blocks[i].bbox[3]
                next_top = all_blocks[i + 1].bbox[1]
                gap = next_top - curr_bottom
                
                if gap < self.config.interlinear_threshold:
                    alternations += 1
        
        # If >50% of transitions are alternating, it's interlinear
        return alternations > len(all_blocks) / 2
    
    def _greedy_assignment(self, cost_matrix: List[List[float]]) -> List[Tuple[int, int]]:
        """
        Greedy assignment fallback (when scipy unavailable).
        Each Akkadian gets its best translation.
        """
        assignments = []
        used_trans = set()
        
        for i, row in enumerate(cost_matrix):
            # Find best unused translation
            best_j = None
            best_cost = float('inf')
            
            for j, cost in enumerate(row):
                if j not in used_trans and cost < best_cost:
                    best_cost = cost
                    best_j = j
            
            if best_j is not None and best_cost < 9999:
                assignments.append((i, best_j))
                used_trans.add(best_j)
        
        return assignments
    
    def _create_pair(
        self,
        akk: TextBlock,
        trans: TextBlock,
        page: int,
        pdf_id: str,
        score: float
    ) -> TranslationPair:
        """Create a TranslationPair from matched blocks."""
        dist = self._bbox_distance(akk.bbox, trans.bbox)
        same_col = akk.column_index == trans.column_index
        has_marker = self._check_markers(trans.text)
        reading_ok = self._check_reading_order(akk, trans)
        
        return TranslationPair(
            pdf_id=pdf_id,
            page=page,
            akk_block_id=akk.block_id,
            akk_text=akk.text,
            akk_bbox=akk.bbox,
            akk_column=akk.column_index,
            trans_block_id=trans.block_id,
            trans_text=trans.text,
            trans_lang=trans.lang,
            trans_bbox=trans.bbox,
            trans_column=trans.column_index,
            score=score,
            distance_px=dist,
            same_column=same_col,
            has_marker=has_marker,
            reading_order_ok=reading_ok
        )
    
    def get_statistics(self) -> Dict:
        """Return pairing statistics."""
        return self.stats.copy()
    
    def save_pairs_csv(
        self,
        pairs: List[TranslationPair],
        output_path: Path
    ):
        """
        Save translation pairs to CSV.
        
        Args:
            pairs: List of translation pairs
            output_path: Output CSV file path
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'pdf_id', 'page',
                'akk_block_id', 'akk_text', 'akk_bbox', 'akk_column',
                'trans_block_id', 'trans_text', 'trans_lang', 'trans_bbox', 'trans_column',
                'score', 'distance_px', 'same_column', 'has_marker', 'reading_order_ok'
            ])
            writer.writeheader()
            
            for pair in pairs:
                row = asdict(pair)
                # Format bboxes as strings
                row['akk_bbox'] = f"{pair.akk_bbox[0]},{pair.akk_bbox[1]},{pair.akk_bbox[2]},{pair.akk_bbox[3]}"
                row['trans_bbox'] = f"{pair.trans_bbox[0]},{pair.trans_bbox[1]},{pair.trans_bbox[2]},{pair.trans_bbox[3]}"
                writer.writerow(row)
        
        logger.info(f"Saved {len(pairs)} pairs to {output_path}")


# ============================================================================
# Test/Demo
# ============================================================================

if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
    from blockification import TextLine
    
    logging.basicConfig(level=logging.INFO)
    
    # Create mock blocks
    akk_line = TextLine(
        text="šarru ᵈUTU ērēbu",
        bbox=(100, 100, 300, 20),
        confidence=0.9,
        line_id="akk_1"
    )
    
    trans_line = TextLine(
        text="König Šamaš tritt ein (Übersetzung)",
        bbox=(100, 150, 350, 20),
        confidence=0.85,
        line_id="trans_1"
    )
    
    akk_block = TextBlock(
        block_id="block_1",
        page=1,
        bbox=(100, 100, 300, 20),
        text="šarru ᵈUTU ērēbu",
        mean_conf=0.9,
        lines=[akk_line],
        lang="tr",
        is_akk=True,
        akk_conf=0.95,
        column_index=0,
        reading_order=0
    )
    
    trans_block = TextBlock(
        block_id="block_2",
        page=1,
        bbox=(100, 150, 350, 20),
        text="König Šamaš tritt ein (Übersetzung)",
        mean_conf=0.85,
        lines=[trans_line],
        lang="de",
        is_akk=False,
        akk_conf=0.0,
        column_index=0,
        reading_order=1
    )
    
    # Test pairer
    pairer = TranslationPairer()
    pairs = pairer.pair_blocks(
        blocks=[akk_block, trans_block],
        page=1,
        pdf_id="test_doc"
    )
    
    print("\n=== Translation Pairing Test ===")
    print(f"Created {len(pairs)} pairs")
    
    for pair in pairs:
        print(f"\nPair (score={pair.score:.3f}):")
        print(f"  Akkadian: {pair.akk_text}")
        print(f"  Translation ({pair.trans_lang}): {pair.trans_text}")
        print(f"  Distance: {pair.distance_px:.1f}px")
        print(f"  Same column: {pair.same_column}")
        print(f"  Has marker: {pair.has_marker}")
        print(f"  Reading order OK: {pair.reading_order_ok}")
    
    # Test CSV export
    output_path = Path("test_pairs.csv")
    pairer.save_pairs_csv(pairs, output_path)
    print(f"\nSaved to {output_path}")
    
    # Show stats
    print("\nStatistics:", pairer.get_statistics())
