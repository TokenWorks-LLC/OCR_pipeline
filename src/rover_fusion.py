#!/usr/bin/env python3
"""
ROVER (Recognizer Output Voting Error Reduction) ensemble fusion.

Implements character-level alignment with confidence calibration and weighted voting
to fuse multiple OCR engine outputs into a superior consensus hypothesis.

REQUIREMENTS:
- Character-level DP alignment
- Per-engine confidence calibration (temperature scaling)
- Weighted voting with rover.weights
- Per-position provenance tracking (which engine won)

Author: Senior OCR Engineer
Date: 2025-10-07
"""

import logging
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from difflib import SequenceMatcher
from collections import defaultdict


@dataclass
class Hypothesis:
    """Single OCR hypothesis from one engine."""
    text: str
    confidence: float
    engine: str
    char_confidences: Optional[List[float]] = None  # Per-character confidences


@dataclass
class AlignedPosition:
    """Aligned character position across multiple hypotheses."""
    position: int
    candidates: Dict[str, Tuple[float, str]]  # char -> (calibrated_conf, engine)
    consensus_char: str
    consensus_conf: float
    winning_engine: str


class ConfidenceCalibrator:
    """
    Calibrate confidence scores using temperature scaling.
    
    Per-engine × language calibration to ensure fair voting.
    Stores calibrated temperatures for reuse.
    """
    
    def __init__(self, calibration_file: Optional[str] = None):
        """
        Initialize calibrator.
        
        Args:
            calibration_file: Path to saved calibration params (JSON)
        """
        self.logger = logging.getLogger(__name__)
        self.temperatures = {}  # (engine, language) -> temperature
        
        if calibration_file:
            self._load_calibration(calibration_file)
        else:
            # Default temperatures (can be refined with calibration data)
            self.temperatures = {
                ('abinet', 'en'): 1.2,
                ('abinet', 'tr'): 1.3,
                ('abinet', 'de'): 1.2,
                ('parseq', 'en'): 1.0,
                ('parseq', 'tr'): 1.1,
                ('parseq', 'de'): 1.0,
                ('doctr_sar', 'en'): 0.9,
                ('doctr_sar', 'tr'): 0.8,
                ('doctr_sar', 'de'): 0.9,
            }
    
    def _load_calibration(self, calibration_file: str):
        """Load calibration parameters from file."""
        import json
        
        try:
            with open(calibration_file, 'r') as f:
                data = json.load(f)
            
            for key, temp in data.items():
                engine, lang = key.split('_')
                self.temperatures[(engine, lang)] = temp
            
            self.logger.info(f"Loaded calibration from {calibration_file}")
        except Exception as e:
            self.logger.warning(f"Failed to load calibration: {e}")
            self.logger.warning("Using default temperatures")
    
    def calibrate(self, confidence: float, engine: str, language: str = 'en') -> float:
        """
        Apply temperature scaling to confidence.
        
        calibrated = sigmoid(logit / temperature)
        
        Args:
            confidence: Raw confidence score [0, 1]
            engine: Engine name
            language: Language code
            
        Returns:
            Calibrated confidence [0, 1]
        """
        # Get temperature (default 1.0 if not found)
        temp = self.temperatures.get((engine, language), 1.0)
        
        # Avoid numerical issues at extremes
        confidence = np.clip(confidence, 1e-7, 1 - 1e-7)
        
        # Convert to logit
        logit = np.log(confidence / (1 - confidence))
        
        # Scale by temperature
        scaled_logit = logit / temp
        
        # Convert back to probability
        calibrated = 1 / (1 + np.exp(-scaled_logit))
        
        return float(calibrated)


class ROVERFusion:
    """
    ROVER ensemble fusion with character-level alignment.
    
    Aligns multiple hypotheses at character level and performs weighted voting.
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None,
                 calibrator: Optional[ConfidenceCalibrator] = None):
        """
        Initialize ROVER fusion.
        
        Args:
            weights: Per-engine weights (default: equal weights)
            calibrator: Confidence calibrator (default: create new)
        """
        self.logger = logging.getLogger(__name__)
        
        self.weights = weights or {
            'abinet': 1.0,
            'parseq': 1.0,
            'doctr_sar': 0.8,
        }
        
        self.calibrator = calibrator or ConfidenceCalibrator()
        
        self.logger.info(f"ROVER initialized with weights: {self.weights}")
    
    def align_hypotheses(self, hypotheses: List[Hypothesis]) -> List[AlignedPosition]:
        """
        Align multiple hypotheses at character level using DP.
        
        Uses progressive alignment:
        1. Align first two hypotheses
        2. Align result with third hypothesis
        3. Continue until all aligned
        
        Args:
            hypotheses: List of hypotheses to align
            
        Returns:
            List of aligned positions
        """
        if not hypotheses:
            return []
        
        if len(hypotheses) == 1:
            # Single hypothesis - no alignment needed
            return self._single_hypothesis_to_aligned(hypotheses[0])
        
        # Start with first hypothesis
        aligned = self._single_hypothesis_to_aligned(hypotheses[0])
        
        # Progressively align remaining hypotheses
        for hyp in hypotheses[1:]:
            aligned = self._align_two(aligned, hyp)
        
        return aligned
    
    def _single_hypothesis_to_aligned(self, hyp: Hypothesis) -> List[AlignedPosition]:
        """Convert single hypothesis to aligned positions."""
        aligned = []
        
        # Get per-character confidences
        if hyp.char_confidences and len(hyp.char_confidences) == len(hyp.text):
            char_confs = hyp.char_confidences
        else:
            # Uniform confidence if not available
            char_confs = [hyp.confidence] * len(hyp.text)
        
        for i, (char, conf) in enumerate(zip(hyp.text, char_confs)):
            # Calibrate confidence
            calibrated_conf = self.calibrator.calibrate(conf, hyp.engine)
            
            # Apply engine weight
            weighted_conf = calibrated_conf * self.weights.get(hyp.engine, 1.0)
            
            pos = AlignedPosition(
                position=i,
                candidates={char: (weighted_conf, hyp.engine)},
                consensus_char=char,
                consensus_conf=weighted_conf,
                winning_engine=hyp.engine
            )
            aligned.append(pos)
        
        return aligned
    
    def _align_two(self, aligned: List[AlignedPosition], 
                   hyp: Hypothesis) -> List[AlignedPosition]:
        """
        Align existing aligned positions with new hypothesis.
        
        Uses SequenceMatcher for character-level alignment.
        """
        # Extract consensus text from aligned positions
        consensus_text = ''.join(pos.consensus_char for pos in aligned)
        
        # Get per-character confidences
        if hyp.char_confidences and len(hyp.char_confidences) == len(hyp.text):
            char_confs = hyp.char_confidences
        else:
            char_confs = [hyp.confidence] * len(hyp.text)
        
        # Align using SequenceMatcher
        matcher = SequenceMatcher(None, consensus_text, hyp.text)
        
        new_aligned = []
        
        for op, i1, i2, j1, j2 in matcher.get_opcodes():
            if op == 'equal':
                # Characters match - add hypothesis vote
                for i, j in zip(range(i1, i2), range(j1, j2)):
                    char = hyp.text[j]
                    conf = char_confs[j]
                    
                    # Calibrate and weight
                    calibrated_conf = self.calibrator.calibrate(conf, hyp.engine)
                    weighted_conf = calibrated_conf * self.weights.get(hyp.engine, 1.0)
                    
                    # Add to existing position
                    pos = aligned[i]
                    
                    # Add candidate vote
                    if char in pos.candidates:
                        # Merge confidences (max)
                        existing_conf, existing_engine = pos.candidates[char]
                        if weighted_conf > existing_conf:
                            pos.candidates[char] = (weighted_conf, hyp.engine)
                    else:
                        pos.candidates[char] = (weighted_conf, hyp.engine)
                    
                    new_aligned.append(pos)
            
            elif op == 'replace':
                # Substitution - take best candidate
                max_len = max(i2 - i1, j2 - j1)
                
                for k in range(max_len):
                    i = i1 + k if k < (i2 - i1) else None
                    j = j1 + k if k < (j2 - j1) else None
                    
                    candidates = {}
                    
                    # Add aligned position candidate (if exists)
                    if i is not None:
                        pos = aligned[i]
                        candidates.update(pos.candidates)
                    
                    # Add hypothesis candidate (if exists)
                    if j is not None:
                        char = hyp.text[j]
                        conf = char_confs[j]
                        calibrated_conf = self.calibrator.calibrate(conf, hyp.engine)
                        weighted_conf = calibrated_conf * self.weights.get(hyp.engine, 1.0)
                        
                        if char in candidates:
                            existing_conf, _ = candidates[char]
                            if weighted_conf > existing_conf:
                                candidates[char] = (weighted_conf, hyp.engine)
                        else:
                            candidates[char] = (weighted_conf, hyp.engine)
                    
                    # Create aligned position with voting
                    new_pos = self._vote_on_position(candidates, len(new_aligned))
                    new_aligned.append(new_pos)
            
            elif op == 'insert':
                # Hypothesis has extra characters
                for j in range(j1, j2):
                    char = hyp.text[j]
                    conf = char_confs[j]
                    calibrated_conf = self.calibrator.calibrate(conf, hyp.engine)
                    weighted_conf = calibrated_conf * self.weights.get(hyp.engine, 1.0)
                    
                    pos = AlignedPosition(
                        position=len(new_aligned),
                        candidates={char: (weighted_conf, hyp.engine)},
                        consensus_char=char,
                        consensus_conf=weighted_conf,
                        winning_engine=hyp.engine
                    )
                    new_aligned.append(pos)
            
            elif op == 'delete':
                # Aligned has extra characters - keep them
                for i in range(i1, i2):
                    new_aligned.append(aligned[i])
        
        return new_aligned
    
    def _vote_on_position(self, candidates: Dict[str, Tuple[float, str]], 
                         position: int) -> AlignedPosition:
        """
        Perform weighted voting on candidates for a position.
        
        Args:
            candidates: Dict of char -> (weighted_conf, engine)
            position: Position index
            
        Returns:
            AlignedPosition with consensus
        """
        if not candidates:
            # Empty position - use space
            return AlignedPosition(
                position=position,
                candidates={},
                consensus_char=' ',
                consensus_conf=0.0,
                winning_engine='none'
            )
        
        # Find highest confidence candidate
        best_char = max(candidates.keys(), key=lambda c: candidates[c][0])
        best_conf, best_engine = candidates[best_char]
        
        return AlignedPosition(
            position=position,
            candidates=candidates,
            consensus_char=best_char,
            consensus_conf=best_conf,
            winning_engine=best_engine
        )
    
    def fuse(self, hypotheses: List[Hypothesis], language: str = 'en') -> Tuple[str, float, List[str]]:
        """
        Fuse multiple hypotheses into consensus.
        
        Args:
            hypotheses: List of hypotheses from different engines
            language: Language for calibration
            
        Returns:
            Tuple of (consensus_text, avg_confidence, per_position_engines)
        """
        if not hypotheses:
            return '', 0.0, []
        
        # Update calibrator language context
        for hyp in hypotheses:
            if hasattr(self.calibrator, 'current_language'):
                self.calibrator.current_language = language
        
        # Align hypotheses
        aligned = self.align_hypotheses(hypotheses)
        
        # Extract consensus
        consensus_text = ''.join(pos.consensus_char for pos in aligned)
        
        # Compute average confidence
        confidences = [pos.consensus_conf for pos in aligned]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        # Extract provenance
        provenance = [pos.winning_engine for pos in aligned]
        
        self.logger.debug(f"ROVER fused {len(hypotheses)} hypotheses -> '{consensus_text[:50]}...'")
        self.logger.debug(f"  Avg confidence: {avg_confidence:.4f}")
        
        # Log engine contributions
        engine_counts = defaultdict(int)
        for engine in provenance:
            engine_counts[engine] += 1
        
        self.logger.debug(f"  Engine contributions: {dict(engine_counts)}")
        
        return consensus_text, float(avg_confidence), provenance


if __name__ == '__main__':
    # Test ROVER fusion
    print("=== ROVER Fusion Test ===\n")
    
    # Create test hypotheses
    hypotheses = [
        Hypothesis(
            text="a-na DINGIR-li-su i-qi-a-am",
            confidence=0.92,
            engine='abinet',
            char_confidences=[0.95] * 10 + [0.85] * 10 + [0.92] * 10
        ),
        Hypothesis(
            text="a-na DINGIR-lí-šu i-qí-a-am",
            confidence=0.88,
            engine='parseq',
            char_confidences=[0.90] * 10 + [0.88] * 10 + [0.87] * 10
        ),
        Hypothesis(
            text="a-na DINGIR-li-su i-qi-a-am",
            confidence=0.75,
            engine='doctr_sar',
            char_confidences=[0.80] * 10 + [0.70] * 10 + [0.75] * 10
        ),
    ]
    
    print("Input hypotheses:")
    for hyp in hypotheses:
        print(f"  [{hyp.engine}] {hyp.text} (conf={hyp.confidence:.3f})")
    
    print()
    
    # Create ROVER fusion
    rover = ROVERFusion(
        weights={'abinet': 1.0, 'parseq': 1.0, 'doctr_sar': 0.8}
    )
    
    # Fuse hypotheses
    consensus, confidence, provenance = rover.fuse(hypotheses, language='en')
    
    print("ROVER consensus:")
    print(f"  Text: {consensus}")
    print(f"  Confidence: {confidence:.4f}")
    print(f"  Provenance: {provenance[:20]}... (showing first 20 chars)")
    
    # Count engine contributions
    from collections import Counter
    engine_counts = Counter(provenance)
    print(f"\nEngine contributions:")
    for engine, count in engine_counts.items():
        pct = count / len(provenance) * 100
        print(f"  {engine}: {count}/{len(provenance)} ({pct:.1f}%)")
    
    print("\n=== Test Complete ===")
