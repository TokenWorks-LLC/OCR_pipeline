#!/usr/bin/env python3
"""
Confusion-aware re-ranking for diacritic confusions.

Implements non-destructive re-ranking based on typical OCR confusion patterns
for Akkadian transliteration diacritics.

REQUIREMENTS:
- Define confusion priors (š↔s, ṣ↔s, ṭ↔t, ḫ↔h, ā↔a, ī↔i, ū↔u)
- Re-ranking only (never alter characters)
- Tie-breaker when candidates within 1-2 log points
- Non-destructive (output characters unchanged)

Author: Senior OCR Engineer
Date: 2025-10-07
"""

import logging
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import math


@dataclass
class Candidate:
    """OCR candidate hypothesis."""
    text: str
    score: float  # Log probability
    metadata: Optional[Dict] = None


class ConfusionPrior:
    """
    Confusion prior for re-ranking OCR hypotheses.
    
    Uses knowledge of typical diacritic confusions to prefer more likely variants
    when candidates are close in score.
    """
    
    # Default confusion priors (probability that diacritic is confused with base)
    DEFAULT_PRIORS = {
        # Akkadian consonants
        ('š', 's'): 0.6,   # š often recognized as s
        ('ṣ', 's'): 0.6,   # ṣ often recognized as s  
        ('ṭ', 't'): 0.6,   # ṭ often recognized as t
        ('ḫ', 'h'): 0.6,   # ḫ often recognized as h
        
        # Long vowels
        ('ā', 'a'): 0.5,   # ā often recognized as a
        ('ē', 'e'): 0.5,   # ē often recognized as e
        ('ī', 'i'): 0.5,   # ī often recognized as i
        ('ū', 'u'): 0.5,   # ū often recognized as u
        
        # Circumflex vowels
        ('â', 'a'): 0.5,
        ('ê', 'e'): 0.5,
        ('î', 'i'): 0.5,
        ('û', 'u'): 0.5,
        
        # Acute vowels
        ('á', 'a'): 0.5,
        ('é', 'e'): 0.5,
        ('í', 'i'): 0.5,
        ('ú', 'u'): 0.5,
    }
    
    def __init__(self, priors: Optional[Dict[Tuple[str, str], float]] = None,
                 tie_threshold: float = 2.0):
        """
        Initialize confusion prior.
        
        Args:
            priors: Custom confusion priors (uses defaults if None)
            tie_threshold: Score difference threshold for tie-breaking (log points)
        """
        self.logger = logging.getLogger(__name__)
        
        self.priors = priors if priors is not None else self.DEFAULT_PRIORS.copy()
        self.tie_threshold = tie_threshold
        
        # Build reverse mapping for bidirectional lookup
        self.reverse_priors = {}
        for (diacritic, base), prob in self.priors.items():
            self.reverse_priors[(base, diacritic)] = prob
        
        self.logger.info(f"Confusion prior initialized with {len(self.priors)} rules")
        self.logger.info(f"Tie threshold: {tie_threshold} log points")
    
    def count_confusions(self, text: str, reference: Optional[str] = None) -> Dict[str, int]:
        """
        Count potential confusions in text.
        
        Args:
            text: Text to analyze
            reference: Optional reference for comparison
            
        Returns:
            Dict of confusion_type -> count
        """
        confusion_counts = {}
        
        for (diacritic, base), _ in self.priors.items():
            # Count base characters (potential confusions)
            base_count = text.count(base)
            diacritic_count = text.count(diacritic)
            
            if base_count > 0:
                confusion_counts[f'{diacritic}→{base}'] = base_count
            
            if diacritic_count > 0:
                confusion_counts[f'correct_{diacritic}'] = diacritic_count
        
        return confusion_counts
    
    def compute_confusion_penalty(self, text: str) -> float:
        """
        Compute penalty for likely confusions in text.
        
        Penalty = sum of log(prior) for each likely confusion
        
        Args:
            text: Text to score
            
        Returns:
            Penalty score (negative log probability)
        """
        penalty = 0.0
        
        # Scan text for potential confusions
        for i, char in enumerate(text):
            # Check if this character is a "base" that might be a confused diacritic
            for (diacritic, base), prior_prob in self.priors.items():
                if char == base:
                    # Check context - if surrounded by diacritics, likely should be diacritic too
                    has_nearby_diacritics = False
                    
                    # Look at neighbors (±2 characters)
                    for j in range(max(0, i-2), min(len(text), i+3)):
                        if j != i and text[j] in [d for d, _ in self.priors.keys()]:
                            has_nearby_diacritics = True
                            break
                    
                    if has_nearby_diacritics:
                        # Likely confusion - add penalty
                        # penalty = -log(P(base|should_be_diacritic))
                        #        = -log(prior_prob)
                        penalty += -math.log(prior_prob)
        
        return penalty
    
    def rerank(self, candidates: List[Candidate]) -> List[Candidate]:
        """
        Re-rank candidates using confusion priors.
        
        Only re-ranks when candidates are within tie_threshold.
        Never modifies the candidate text - only adjusts scores.
        
        Args:
            candidates: List of candidate hypotheses
            
        Returns:
            Re-ranked list of candidates (sorted by adjusted score descending)
        """
        if not candidates:
            return []
        
        if len(candidates) == 1:
            return candidates
        
        # Sort by original score (descending)
        sorted_candidates = sorted(candidates, key=lambda c: c.score, reverse=True)
        
        # Get top candidate
        top_candidate = sorted_candidates[0]
        top_score = top_candidate.score
        
        # Identify ties (candidates within threshold of top)
        tied_candidates = []
        other_candidates = []
        
        for cand in sorted_candidates:
            score_diff = abs(top_score - cand.score)
            
            if score_diff <= self.tie_threshold:
                tied_candidates.append(cand)
            else:
                other_candidates.append(cand)
        
        if len(tied_candidates) <= 1:
            # No ties - return original order
            return sorted_candidates
        
        # Re-rank tied candidates using confusion prior
        reranked_tied = []
        
        for cand in tied_candidates:
            # Compute confusion penalty
            penalty = self.compute_confusion_penalty(cand.text)
            
            # Adjusted score = original score - penalty
            adjusted_score = cand.score - penalty
            
            # Create new candidate with adjusted score
            reranked_cand = Candidate(
                text=cand.text,
                score=adjusted_score,
                metadata={
                    **(cand.metadata or {}),
                    'original_score': cand.score,
                    'confusion_penalty': penalty,
                    'reranked': True
                }
            )
            
            reranked_tied.append(reranked_cand)
        
        # Sort tied candidates by adjusted score
        reranked_tied.sort(key=lambda c: c.score, reverse=True)
        
        # Combine: reranked ties + other candidates
        final_ranking = reranked_tied + other_candidates
        
        # Log re-ranking
        if reranked_tied[0].text != tied_candidates[0].text:
            self.logger.debug(f"Re-ranked: '{tied_candidates[0].text}' → '{reranked_tied[0].text}'")
            self.logger.debug(f"  Original score: {tied_candidates[0].score:.4f}")
            self.logger.debug(f"  Adjusted score: {reranked_tied[0].score:.4f}")
            self.logger.debug(f"  Penalty: {reranked_tied[0].metadata['confusion_penalty']:.4f}")
        
        return final_ranking
    
    def prefer_diacritics(self, text_without: str, text_with: str, score_diff: float) -> str:
        """
        Choose between text with/without diacritics based on score difference.
        
        If score difference is small, prefer the version with diacritics.
        
        Args:
            text_without: Text without diacritics (e.g., "a-na i-na")
            text_with: Text with diacritics (e.g., "a-nā i-nā")
            score_diff: Score difference (with - without)
            
        Returns:
            Preferred text
        """
        if score_diff > self.tie_threshold:
            # Large difference - prefer higher score regardless
            return text_with if score_diff > 0 else text_without
        
        # Small difference - prefer diacritics
        # Count diacritics in each
        diacritic_chars = set(d for d, _ in self.priors.keys())
        
        count_with = sum(1 for c in text_with if c in diacritic_chars)
        count_without = sum(1 for c in text_without if c in diacritic_chars)
        
        if count_with > count_without:
            return text_with
        elif count_without > count_with:
            return text_without
        else:
            # Equal diacritics - return higher score
            return text_with if score_diff >= 0 else text_without


if __name__ == '__main__':
    # Test confusion prior
    print("=== Confusion Prior Test ===\n")
    
    # Create confusion prior
    prior = ConfusionPrior(tie_threshold=2.0)
    
    # Test case 1: Clear winner (no tie)
    print("[1] No tie - clear winner...")
    candidates_no_tie = [
        Candidate(text="a-na DINGIR-lí-šu", score=-10.5),
        Candidate(text="a-na DINGIR-li-su", score=-15.2),
    ]
    
    reranked = prior.rerank(candidates_no_tie)
    
    print("  Input:")
    for c in candidates_no_tie:
        print(f"    {c.text}: {c.score:.4f}")
    
    print("\n  Output (unchanged):")
    for c in reranked:
        print(f"    {c.text}: {c.score:.4f}")
    
    # Test case 2: Tie - should prefer diacritics
    print("\n[2] Tie - prefer diacritics...")
    candidates_tie = [
        Candidate(text="a-na DINGIR-li-su i-qi-a-am", score=-10.5),  # No diacritics
        Candidate(text="a-na DINGIR-lí-šu i-qí-a-am", score=-11.0),  # With diacritics
    ]
    
    reranked = prior.rerank(candidates_tie)
    
    print("  Input:")
    for c in candidates_tie:
        penalty = prior.compute_confusion_penalty(c.text)
        print(f"    {c.text}: {c.score:.4f} (penalty={penalty:.4f})")
    
    print("\n  Output (re-ranked):")
    for c in reranked:
        print(f"    {c.text}: {c.score:.4f}")
        if c.metadata and c.metadata.get('reranked'):
            print(f"      Original: {c.metadata['original_score']:.4f}")
            print(f"      Penalty: {c.metadata['confusion_penalty']:.4f}")
    
    # Test case 3: Confusion analysis
    print("\n[3] Confusion analysis...")
    
    test_texts = [
        "a-na DINGIR-li-su",  # Missing diacritics
        "a-na DINGIR-lí-šu",  # Correct diacritics
        "KÙ.BABBAR ša-ru-pá-am",  # Mixed
    ]
    
    for text in test_texts:
        confusions = prior.count_confusions(text)
        penalty = prior.compute_confusion_penalty(text)
        
        print(f"\n  Text: {text}")
        print(f"  Confusion penalty: {penalty:.4f}")
        print(f"  Confusions: {confusions}")
    
    print("\n=== Test Complete ===")
