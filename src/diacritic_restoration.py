#!/usr/bin/env python3
"""
Post-OCR diacritic restoration for Akkadian transliteration.

Generates candidate variants with diacritics and uses lexicon + confusion prior
to select the most likely correct form.

This solves the "single hypothesis" problem by creating synthetic candidates
from the baseline OCR output.

Author: Senior OCR Engineer
Date: 2025-10-07
"""

import logging
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass
import re


@dataclass
class CandidateWord:
    """A candidate word with its score."""
    word: str
    score: float
    source: str  # 'original', 'lexicon', 'diacritic', 'confusion'


class DiacriticRestorer:
    """
    Restore diacritics to OCR text using lexicon and confusion priors.
    
    Process:
    1. Tokenize OCR output
    2. For each token, generate diacritic variants
    3. Score variants using lexicon
    4. Apply confusion prior
    5. Select best candidate
    """
    
    # Akkadian diacritic mappings (base → diacritic variants)
    DIACRITIC_MAPPINGS = {
        's': ['š', 'ṣ'],      # s can be š or ṣ
        't': ['ṭ'],            # t can be ṭ
        'h': ['ḫ'],            # h can be ḫ
        'a': ['ā', 'á', 'à'],  # a can have macron or accents
        'e': ['ē', 'é', 'è'],  # e can have macron or accents
        'i': ['ī', 'í', 'ì'],  # i can have macron or accents
        'u': ['ū', 'ú', 'ù'],  # u can have macron or accents
    }
    
    def __init__(self, lexicon=None, confusion_prior=None, 
                 max_candidates: int = 20,
                 lexicon_weight: float = 2.0,
                 confusion_weight: float = 1.0):
        """
        Initialize diacritic restorer.
        
        Args:
            lexicon: LexiconBias instance (optional)
            confusion_prior: ConfusionPrior instance (optional)
            max_candidates: Maximum candidates to generate per word
            lexicon_weight: Weight for lexicon matching (higher = prefer lexicon)
            confusion_weight: Weight for confusion prior (higher = prefer likely confusions)
        """
        self.logger = logging.getLogger(__name__)
        self.lexicon = lexicon
        self.confusion_prior = confusion_prior
        self.max_candidates = max_candidates
        self.lexicon_weight = lexicon_weight
        self.confusion_weight = confusion_weight
        
        self.stats = {
            'words_processed': 0,
            'words_changed': 0,
            'diacritics_restored': 0
        }
        
        self.logger.info(f"Diacritic restorer initialized (lexicon={lexicon is not None}, "
                        f"confusion={confusion_prior is not None})")
    
    def generate_variants(self, word: str) -> List[str]:
        """
        Generate diacritic variants for a word.
        
        Args:
            word: Input word (may have some diacritics already)
            
        Returns:
            List of variant strings (including original)
        """
        variants = {word}  # Start with original
        
        # Find positions where diacritics could be added
        for i, char in enumerate(word):
            char_lower = char.lower()
            
            # Check if this character has diacritic variants
            if char_lower in self.DIACRITIC_MAPPINGS:
                diacritic_chars = self.DIACRITIC_MAPPINGS[char_lower]
                
                # Generate variants by substituting this position
                new_variants = set()
                for variant in variants:
                    for diacritic in diacritic_chars:
                        # Preserve case
                        if char.isupper():
                            diacritic = diacritic.upper()
                        
                        # Create new variant with substitution
                        new_word = variant[:i] + diacritic + variant[i+1:]
                        new_variants.add(new_word)
                
                variants.update(new_variants)
                
                # Limit explosion of variants
                if len(variants) > self.max_candidates:
                    break
        
        return list(variants)[:self.max_candidates]
    
    def score_variant(self, variant: str, original: str) -> Tuple[float, str]:
        """
        Score a variant using lexicon and confusion prior.
        
        Args:
            variant: Candidate variant
            original: Original OCR output
            
        Returns:
            (score, source) tuple
        """
        score = 0.0
        source = 'original' if variant == original else 'diacritic'
        
        # Lexicon matching
        if self.lexicon:
            is_match, frequency = self.lexicon.combined_trie.search(variant)
            
            if is_match:
                # Lexicon match - strong signal
                import math
                lexicon_score = math.log(frequency + 1) * self.lexicon_weight
                score += lexicon_score
                source = 'lexicon'
                
                self.logger.debug(f"Lexicon match: '{variant}' (freq={frequency}, score=+{lexicon_score:.2f})")
        
        # Confusion prior scoring
        if self.confusion_prior and variant != original:
            # Count how many confusions we're "undoing"
            confusion_score = 0.0
            
            for i, (orig_char, var_char) in enumerate(zip(original, variant)):
                if orig_char != var_char:
                    # Check if this is a known confusion
                    # var_char (diacritic) was confused as orig_char (base)
                    confusion_pair = (var_char, orig_char)
                    
                    if confusion_pair in self.confusion_prior.priors:
                        # This is a likely confusion - boost score
                        prior_prob = self.confusion_prior.priors[confusion_pair]
                        import math
                        confusion_score += math.log(prior_prob) * self.confusion_weight
            
            score += confusion_score
            
            if confusion_score > 0:
                self.logger.debug(f"Confusion boost: '{original}' → '{variant}' (+{confusion_score:.2f})")
        
        # Prefer original if no evidence for change (tie-breaker)
        if variant == original:
            score += 0.1  # Small boost for not changing
        
        return score, source
    
    def restore_word(self, word: str) -> CandidateWord:
        """
        Restore diacritics for a single word.
        
        Args:
            word: Input word from OCR
            
        Returns:
            Best candidate with score
        """
        # Skip very short words or numbers
        if len(word) <= 1 or word.isdigit():
            return CandidateWord(word=word, score=0.0, source='original')
        
        # Generate variants
        variants = self.generate_variants(word)
        
        # Score each variant
        candidates = []
        for variant in variants:
            score, source = self.score_variant(variant, word)
            candidates.append(CandidateWord(word=variant, score=score, source=source))
        
        # Sort by score (descending)
        candidates.sort(key=lambda c: c.score, reverse=True)
        
        # Return best candidate
        best = candidates[0]
        
        # Update stats
        if best.word != word:
            self.stats['words_changed'] += 1
            # Count diacritics added
            diacritics_added = sum(1 for c in best.word if c in 'šṣṭḫāēīūáéíúàèìù')
            original_diacritics = sum(1 for c in word if c in 'šṣṭḫāēīūáéíúàèìù')
            self.stats['diacritics_restored'] += (diacritics_added - original_diacritics)
        
        return best
    
    def restore_text(self, text: str) -> str:
        """
        Restore diacritics for entire text.
        
        Args:
            text: OCR text output
            
        Returns:
            Text with restored diacritics
        """
        if not text:
            return text
        
        # Tokenize (preserve separators)
        # Split on whitespace and common separators, but keep them
        tokens = re.findall(r'\S+|\s+', text)
        
        restored_tokens = []
        
        for token in tokens:
            # Skip whitespace
            if token.isspace():
                restored_tokens.append(token)
                continue
            
            # Process word token
            self.stats['words_processed'] += 1
            
            # Restore diacritics
            best = self.restore_word(token)
            restored_tokens.append(best.word)
            
            if best.word != token:
                self.logger.debug(f"Restored: '{token}' → '{best.word}' (score={best.score:.2f}, source={best.source})")
        
        return ''.join(restored_tokens)
    
    def get_stats(self) -> dict:
        """Get restoration statistics."""
        stats = self.stats.copy()
        if stats['words_processed'] > 0:
            stats['change_rate'] = stats['words_changed'] / stats['words_processed']
        else:
            stats['change_rate'] = 0.0
        return stats
    
    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            'words_processed': 0,
            'words_changed': 0,
            'diacritics_restored': 0
        }


if __name__ == '__main__':
    # Test diacritic restoration
    print("=== Diacritic Restoration Test ===\n")
    
    # Create mock lexicon for testing
    class MockTrie:
        def __init__(self):
            self.words = {
                'A-šur': 100,      # Assyria
                'A-šur-du10': 50,  # Assyrian name
                'kiˇsib': 80,      # seal
                'DUMU': 200,       # son
                'a-na': 150,       # to/for
                'i-na': 120,       # in/at
            }
        
        def search(self, word):
            if word in self.words:
                return True, self.words[word]
            return False, 0
    
    class MockLexicon:
        def __init__(self):
            self.combined_trie = MockTrie()
    
    class MockConfusionPrior:
        def __init__(self):
            self.priors = {
                ('š', 's'): 0.6,
                ('ṣ', 's'): 0.6,
                ('ṭ', 't'): 0.6,
                ('ḫ', 'h'): 0.6,
            }
    
    # Create restorer
    lexicon = MockLexicon()
    confusion = MockConfusionPrior()
    restorer = DiacriticRestorer(lexicon=lexicon, confusion_prior=confusion)
    
    # Test cases
    test_cases = [
        "A-sur-du10",           # Should become A-šur-du10
        "a-na i-na",            # Should stay (already correct)
        "DUMU A-sur",           # DUMU stays, A-sur → A-šur
        "kisib",                # Could add diacritics if in lexicon
    ]
    
    print("Test Cases:")
    for i, test in enumerate(test_cases, 1):
        print(f"\n{i}. Input:  '{test}'")
        
        # Generate variants
        variants = restorer.generate_variants(test)
        print(f"   Variants: {len(variants)}")
        
        # Restore
        restored = restorer.restore_text(test)
        print(f"   Output: '{restored}'")
        
        if restored != test:
            print(f"   ✓ Changed")
        else:
            print(f"   - No change")
    
    # Show stats
    print("\n" + "="*50)
    print("Statistics:")
    stats = restorer.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n=== Test Complete ===")
