#!/usr/bin/env python3
"""
Lexicon biasing system for OCR decoding.

Implements prefix-trie based lexicon for non-destructive log-boost during decoding.
Supports Sumerograms, Akkadian morphemes, and common function words.

REQUIREMENTS:
- Prefix-trie structure for efficient lookup
- Non-destructive boosting (never force lexicon token)
- Build from gold + repo examples
- Support for:
  * Sumerograms (LUGAL, DINGIR, etc.)
  * Akkadian morphemes (a-na, i-na, etc.)
  * Turkish/German/French function words

Author: Senior OCR Engineer  
Date: 2025-10-07
"""

import logging
import json
from typing import Dict, List, Set, Optional, Tuple
from pathlib import Path
from collections import defaultdict


class TrieNode:
    """Node in prefix trie."""
    
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.frequency = 0
        self.word = None


class LexiconTrie:
    """
    Prefix trie for lexicon storage and lookup.
    
    Supports incremental matching during beam search.
    """
    
    def __init__(self):
        self.root = TrieNode()
        self.word_count = 0
        self.logger = logging.getLogger(__name__)
    
    def insert(self, word: str, frequency: int = 1):
        """
        Insert word into trie.
        
        Args:
            word: Word to insert
            frequency: Word frequency (for boost calculation)
        """
        if not word:
            return
        
        node = self.root
        
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        if not node.is_word:
            self.word_count += 1
        
        node.is_word = True
        node.frequency = frequency
        node.word = word
    
    def search(self, word: str) -> Tuple[bool, int]:
        """
        Search for exact word match.
        
        Args:
            word: Word to search
            
        Returns:
            Tuple of (found, frequency)
        """
        node = self._search_prefix(word)
        
        if node and node.is_word:
            return True, node.frequency
        
        return False, 0
    
    def starts_with(self, prefix: str) -> bool:
        """
        Check if any word starts with prefix.
        
        Args:
            prefix: Prefix to check
            
        Returns:
            True if prefix exists in trie
        """
        return self._search_prefix(prefix) is not None
    
    def _search_prefix(self, prefix: str) -> Optional[TrieNode]:
        """
        Search for prefix in trie.
        
        Args:
            prefix: Prefix to search
            
        Returns:
            TrieNode if found, None otherwise
        """
        node = self.root
        
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        
        return node
    
    def get_completions(self, prefix: str, max_count: int = 10) -> List[Tuple[str, int]]:
        """
        Get possible completions for prefix.
        
        Args:
            prefix: Prefix
            max_count: Maximum completions to return
            
        Returns:
            List of (word, frequency) tuples
        """
        node = self._search_prefix(prefix)
        
        if not node:
            return []
        
        completions = []
        self._collect_words(node, prefix, completions, max_count)
        
        # Sort by frequency (descending)
        completions.sort(key=lambda x: x[1], reverse=True)
        
        return completions[:max_count]
    
    def _collect_words(self, node: TrieNode, current: str, 
                      completions: List[Tuple[str, int]], max_count: int):
        """Recursively collect words from node."""
        if len(completions) >= max_count:
            return
        
        if node.is_word:
            completions.append((current, node.frequency))
        
        for char, child in node.children.items():
            self._collect_words(child, current + char, completions, max_count)


class LexiconBias:
    """
    Lexicon biasing system for OCR decoding.
    
    Provides non-destructive log-boosts for lexicon matches during beam search.
    """
    
    def __init__(self, min_freq: int = 2, bias: float = 0.95, 
                 max_boost_per_token: float = 2.0):
        """
        Initialize lexicon bias.
        
        Args:
            min_freq: Minimum frequency for a word to be included
            bias: Bias strength [0, 1] (0 = no bias, 1 = strong bias)
            max_boost_per_token: Maximum log-boost per token
        """
        self.logger = logging.getLogger(__name__)
        
        self.min_freq = min_freq
        self.bias = bias
        self.max_boost_per_token = max_boost_per_token
        
        # Create separate tries for different categories
        self.sumerograms = LexiconTrie()
        self.akkadian_morphemes = LexiconTrie()
        self.function_words = LexiconTrie()
        
        # Combined trie for efficient lookup
        self.combined_trie = LexiconTrie()
        
        self.logger.info(f"Lexicon bias initialized (min_freq={min_freq}, bias={bias})")
    
    def load_sumerograms(self, sumerogram_list: Optional[List[str]] = None):
        """
        Load Sumerogram lexicon.
        
        Args:
            sumerogram_list: Optional custom list (uses defaults if None)
        """
        if sumerogram_list is None:
            # Default Sumerograms (common ideograms)
            sumerogram_list = [
                'DINGIR', 'LUGAL', 'MUNUS', 'LÚ', 'KUR', 'É', 'GIŠ', 'DUG',
                'ÍD', 'URU', 'KÙ.BABBAR', 'KÙ.GI', 'GU₄', 'UDU', 'ITU',
                'MU', 'AN', 'KI', 'IM', 'A.ŠÀ', 'GEME₂', 'ARAD', 'DAM',
                'DUMU', 'ŠEŠ', 'NIN', 'AMA', 'AD', 'SAL', 'TUR',
                # Compound ideograms
                'AN.ŠÀR', 'EN.LÍL', 'MUNUS.LUGAL', 'LÚ.GEŠTIN',
            ]
        
        for word in sumerogram_list:
            # High frequency for Sumerograms (they're very distinctive)
            self.sumerograms.insert(word, frequency=100)
            self.combined_trie.insert(word, frequency=100)
        
        self.logger.info(f"Loaded {len(sumerogram_list)} Sumerograms")
    
    def load_akkadian_morphemes(self, morpheme_list: Optional[List[str]] = None):
        """
        Load Akkadian morpheme lexicon.
        
        Args:
            morpheme_list: Optional custom list
        """
        if morpheme_list is None:
            # Common Akkadian morphemes and particles
            morpheme_list = [
                'a-na', 'i-na', 'ša', 'ša₂', 'ša₃',
                'ma-a', 'ma', 'ù', 'u₄', 'i-na', 
                'i-šu', 'i-di-in', 'i-lá-qé', 'i-a-qal',
                'ma-na', 'GíN', 'ṭup-pí', 'kù.babbar',
                # Common verbs
                'i-sé-er', 'ú-sa-áb', 'i-qú-ul',
            ]
        
        for word in morpheme_list:
            self.akkadian_morphemes.insert(word, frequency=50)
            self.combined_trie.insert(word, frequency=50)
        
        self.logger.info(f"Loaded {len(morpheme_list)} Akkadian morphemes")
    
    def load_function_words(self, function_word_dict: Optional[Dict[str, List[str]]] = None):
        """
        Load function words for multiple languages.
        
        Args:
            function_word_dict: Dict of language -> word list
        """
        if function_word_dict is None:
            # Default function words
            function_word_dict = {
                'tr': ['ve', 'bir', 'bu', 'için', 'ile', 'olan', 'en', 'da', 'de'],
                'de': ['der', 'die', 'das', 'und', 'von', 'zu', 'in', 'mit', 'für'],
                'en': ['the', 'and', 'of', 'to', 'in', 'a', 'is', 'that', 'for'],
                'fr': ['le', 'la', 'les', 'de', 'et', 'à', 'un', 'une', 'dans'],
                'it': ['il', 'lo', 'la', 'di', 'e', 'a', 'un', 'una', 'in'],
            }
        
        word_count = 0
        for lang, words in function_word_dict.items():
            for word in words:
                self.function_words.insert(word, frequency=80)
                self.combined_trie.insert(word, frequency=80)
                word_count += 1
        
        self.logger.info(f"Loaded {word_count} function words across {len(function_word_dict)} languages")
    
    def load_from_gold(self, gold_texts: List[str]):
        """
        Extract lexicon from gold standard texts.
        
        Args:
            gold_texts: List of gold text samples
        """
        # Count word frequencies
        word_freq = defaultdict(int)
        
        for text in gold_texts:
            # Tokenize (split on whitespace and some punctuation)
            tokens = text.replace('\n', ' ').split()
            
            for token in tokens:
                # Clean token
                token = token.strip('.,;:!?()[]{}')
                
                if token and len(token) >= 2:  # Min length 2
                    word_freq[token] += 1
        
        # Add to trie (only words above min_freq)
        added_count = 0
        for word, freq in word_freq.items():
            if freq >= self.min_freq:
                self.combined_trie.insert(word, frequency=freq)
                added_count += 1
        
        self.logger.info(f"Extracted {added_count} words from gold data (min_freq={self.min_freq})")
    
    def compute_boost(self, partial_text: str, candidate_char: str) -> float:
        """
        Compute log-boost for adding candidate_char to partial_text.
        
        Non-destructive: only provides positive bias if extension matches lexicon.
        
        Args:
            partial_text: Text built so far
            candidate_char: Character being considered
            
        Returns:
            Log-boost value (0 if no match, positive if match)
        """
        # Get last token (word being built)
        tokens = partial_text.split()
        current_token = tokens[-1] if tokens else ''
        
        # Proposed extension
        proposed = current_token + candidate_char
        
        # Check if proposed matches prefix in trie
        if not self.combined_trie.starts_with(proposed):
            return 0.0
        
        # Check if proposed is exact match
        is_match, frequency = self.combined_trie.search(proposed)
        
        if is_match:
            # Exact match - provide boost based on frequency and bias
            # Higher frequency = higher boost
            # Boost = bias * min(log(frequency), max_boost_per_token)
            import math
            
            boost = self.bias * min(math.log(frequency + 1), self.max_boost_per_token)
            return boost
        
        # Prefix match but not exact - provide small encouragement
        # to continue building toward a lexicon word
        return self.bias * 0.1
    
    def score_complete_text(self, text: str) -> float:
        """
        Score already-complete OCR text by checking word matches in lexicon.
        
        This is for post-processing (not beam search). Returns a confidence
        boost based on how many words match the lexicon.
        
        Args:
            text: Complete OCR text to score
            
        Returns:
            Confidence boost (0.0 to 1.0)
        """
        import math
        
        if not text or not text.strip():
            return 0.0
        
        # Tokenize text (split on whitespace and common separators)
        words = text.replace('-', ' ').replace('_', ' ').split()
        
        if not words:
            return 0.0
        
        # Count lexicon matches
        matched_words = 0
        total_boost = 0.0
        
        for word in words:
            # Check exact match
            is_match, frequency = self.combined_trie.search(word)
            
            if is_match:
                matched_words += 1
                # Compute boost for this word
                word_boost = self.bias * min(math.log(frequency + 1), self.max_boost_per_token)
                total_boost += word_boost
        
        # Average boost per word
        if len(words) > 0:
            avg_boost = total_boost / len(words)
            match_rate = matched_words / len(words)
            
            # Return confidence boost (0.0 to 1.0)
            # Higher match rate = higher boost
            confidence_boost = min(0.5, avg_boost) * match_rate
            
            if matched_words > 0:
                self.logger.debug(f"Lexicon scoring: {matched_words}/{len(words)} words matched, boost={confidence_boost:.3f}")
            
            return confidence_boost
        
        return 0.0
    
    def save(self, output_path: str):
        """
        Save lexicon to JSON file.
        
        Args:
            output_path: Output file path
        """
        # Collect all words from combined trie
        words = []
        
        def collect(node, current):
            if node.is_word:
                words.append({'word': current, 'freq': node.frequency})
            for char, child in node.children.items():
                collect(child, current + char)
        
        collect(self.combined_trie.root, '')
        
        # Save to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'config': {
                    'min_freq': self.min_freq,
                    'bias': self.bias,
                    'max_boost_per_token': self.max_boost_per_token
                },
                'words': words
            }, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Saved lexicon ({len(words)} words) to {output_path}")
    
    def load(self, input_path: str):
        """
        Load lexicon from JSON file.
        
        Args:
            input_path: Input file path
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Load config
        config = data.get('config', {})
        self.min_freq = config.get('min_freq', self.min_freq)
        self.bias = config.get('bias', self.bias)
        self.max_boost_per_token = config.get('max_boost_per_token', self.max_boost_per_token)
        
        # Load words
        words = data.get('words', [])
        for entry in words:
            word = entry['word']
            freq = entry['freq']
            self.combined_trie.insert(word, frequency=freq)
        
        self.logger.info(f"Loaded lexicon ({len(words)} words) from {input_path}")


if __name__ == '__main__':
    # Test lexicon bias
    print("=== Lexicon Bias Test ===\n")
    
    # Create lexicon
    lexicon = LexiconBias(min_freq=2, bias=0.95, max_boost_per_token=2.0)
    
    # Load default lexicons
    print("[1] Loading Sumerograms...")
    lexicon.load_sumerograms()
    
    print("[2] Loading Akkadian morphemes...")
    lexicon.load_akkadian_morphemes()
    
    print("[3] Loading function words...")
    lexicon.load_function_words()
    
    # Test prefix matching
    print("\n[4] Testing prefix matching...")
    
    test_cases = [
        ("DIN", "G"),  # Building DINGIR
        ("DINGI", "R"),  # Complete DINGIR
        ("a-n", "a"),  # Building a-na
        ("xyz", "q"),  # Random gibberish
    ]
    
    for partial, char in test_cases:
        boost = lexicon.compute_boost(partial, char)
        proposed = partial + char
        is_match, freq = lexicon.combined_trie.search(proposed)
        
        print(f"  '{partial}' + '{char}' = '{proposed}'")
        print(f"    Boost: {boost:.4f}")
        print(f"    Match: {is_match} (freq={freq})")
        print()
    
    # Test save/load
    print("[5] Testing save/load...")
    lexicon.save('test_lexicon.json')
    
    new_lexicon = LexiconBias()
    new_lexicon.load('test_lexicon.json')
    
    print(f"  Loaded {new_lexicon.combined_trie.word_count} words")
    
    print("\n=== Test Complete ===")
