#!/usr/bin/env python3
"""
Pure Python Character-level Language Model (no kenlm dependency).

A lightweight n-gram character LM for scoring and re-ranking OCR candidates.
Uses simple smoothed probability estimates from training data.

Author: GitHub Copilot
Date: 2025-10-07
"""

import os
import json
import math
import logging
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter


class PythonCharLM:
    """
    Pure Python character-level n-gram language model.
    
    Supports:
    - Character-level n-grams (1-5 grams)
    - Laplace smoothing for unseen n-grams
    - Log probability scoring
    - Perplexity calculation
    """
    
    def __init__(self, order: int = 3, alpha: float = 0.1):
        """
        Initialize character LM.
        
        Args:
            order: N-gram order (default: 3 for trigrams)
            alpha: Smoothing parameter (default: 0.1)
        """
        self.logger = logging.getLogger(__name__)
        self.order = order
        self.alpha = alpha  # Laplace smoothing
        
        # N-gram counts
        self.ngram_counts = defaultdict(Counter)  # {n: {ngram: count}}
        self.context_counts = defaultdict(Counter)  # {n: {context: count}}
        self.vocab = set()  # Character vocabulary
        
        self.total_chars = 0
        self.is_trained = False
    
    def train_from_file(self, train_file: str):
        """
        Train LM from text file.
        
        Args:
            train_file: Path to training text file
        """
        self.logger.info(f"Training character LM from {train_file}")
        
        with open(train_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        self.train_from_text(text)
    
    def train_from_text(self, text: str):
        """
        Train LM from text string.
        
        Args:
            text: Training text
        """
        # Add special tokens
        text = '<s>' + text + '</s>'
        
        # Build vocabulary
        self.vocab = set(text)
        self.vocab.add('<unk>')  # Unknown character token
        
        # Count n-grams
        for n in range(1, self.order + 1):
            for i in range(len(text) - n + 1):
                ngram = text[i:i+n]
                self.ngram_counts[n][ngram] += 1
                
                if n > 1:
                    context = text[i:i+n-1]
                    self.context_counts[n][context] += 1
        
        self.total_chars = len(text)
        self.is_trained = True
        
        self.logger.info(f"Trained on {self.total_chars} characters")
        self.logger.info(f"Vocabulary size: {len(self.vocab)}")
        for n in range(1, self.order + 1):
            self.logger.info(f"  {n}-grams: {len(self.ngram_counts[n])}")
    
    def _get_probability(self, ngram: str, context: Optional[str] = None) -> float:
        """
        Get probability of n-gram with smoothing.
        
        Args:
            ngram: N-gram string
            context: Context (n-1 chars) for conditional probability
            
        Returns:
            Smoothed probability
        """
        n = len(ngram)
        
        if n == 1:
            # Unigram probability with Laplace smoothing
            count = self.ngram_counts[1].get(ngram, 0)
            vocab_size = len(self.vocab)
            return (count + self.alpha) / (self.total_chars + self.alpha * vocab_size)
        
        else:
            # N-gram conditional probability P(char | context)
            if context is None:
                context = ngram[:-1]
            
            ngram_count = self.ngram_counts[n].get(ngram, 0)
            context_count = self.context_counts[n].get(context, 0)
            
            if context_count == 0:
                # Backoff to lower order
                return self._get_probability(ngram[-1], context[1:] if len(context) > 1 else None)
            
            vocab_size = len(self.vocab)
            return (ngram_count + self.alpha) / (context_count + self.alpha * vocab_size)
    
    def score(self, text: str, normalize: bool = False) -> float:
        """
        Score text with LM (log probability).
        
        Args:
            text: Input text to score
            normalize: If True, normalize by text length
            
        Returns:
            Log probability (higher is better)
        """
        if not self.is_trained:
            self.logger.warning("LM not trained, returning neutral score")
            return 0.0
        
        # Add sentence markers
        text = '<s>' + text + '</s>'
        
        log_prob = 0.0
        
        # Score using highest order n-grams available
        for i in range(len(text)):
            # Try from highest order down to unigram
            for n in range(min(self.order, i + 1), 0, -1):
                if i >= n - 1:
                    ngram = text[i-n+1:i+1]
                    context = text[i-n+1:i] if n > 1 else None
                    
                    prob = self._get_probability(ngram, context)
                    log_prob += math.log(prob) if prob > 0 else -20.0  # Cap very low probs
                    break
        
        if normalize and len(text) > 0:
            log_prob /= len(text)
        
        return log_prob
    
    def perplexity(self, text: str) -> float:
        """
        Calculate perplexity on text.
        
        Args:
            text: Test text
            
        Returns:
            Perplexity (lower is better)
        """
        log_prob = self.score(text, normalize=True)
        return math.exp(-log_prob)
    
    def rescore_candidates(self, candidates: List[Tuple[str, float]], 
                          alpha: float = 0.7) -> List[Tuple[str, float]]:
        """
        Re-score OCR candidates using LM.
        
        Args:
            candidates: List of (text, ocr_score) tuples
            alpha: Weight for LM score (1-alpha for OCR score)
            
        Returns:
            Re-ranked list of (text, combined_score) tuples
        """
        rescored = []
        
        for text, ocr_score in candidates:
            lm_score = self.score(text, normalize=True)
            
            # Combine OCR and LM scores
            combined_score = alpha * lm_score + (1 - alpha) * math.log(ocr_score + 1e-10)
            
            rescored.append((text, combined_score))
        
        # Sort by combined score (higher is better)
        rescored.sort(key=lambda x: x[1], reverse=True)
        
        return rescored
    
    def save(self, model_path: str):
        """
        Save model to JSON file.
        
        Args:
            model_path: Path to save model
        """
        model_data = {
            'order': self.order,
            'alpha': self.alpha,
            'vocab': list(self.vocab),
            'total_chars': self.total_chars,
            'ngram_counts': {
                n: dict(counts) for n, counts in self.ngram_counts.items()
            },
            'context_counts': {
                n: dict(counts) for n, counts in self.context_counts.items()
            }
        }
        
        with open(model_path, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Model saved to {model_path}")
    
    def load(self, model_path: str):
        """
        Load model from JSON file.
        
        Args:
            model_path: Path to model file
        """
        with open(model_path, 'r', encoding='utf-8') as f:
            model_data = json.load(f)
        
        self.order = model_data['order']
        self.alpha = model_data['alpha']
        self.vocab = set(model_data['vocab'])
        self.total_chars = model_data['total_chars']
        
        # Convert back to defaultdict(Counter)
        self.ngram_counts = defaultdict(Counter)
        for n_str, counts in model_data['ngram_counts'].items():
            n = int(n_str)
            self.ngram_counts[n] = Counter(counts)
        
        self.context_counts = defaultdict(Counter)
        for n_str, counts in model_data['context_counts'].items():
            n = int(n_str)
            self.context_counts[n] = Counter(counts)
        
        self.is_trained = True
        self.logger.info(f"Model loaded from {model_path}")


if __name__ == '__main__':
    # Test the LM
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    # Create and train model
    lm = PythonCharLM(order=3, alpha=0.1)
    
    # Check if train file exists
    train_file = 'reports/research_assets/lm/train.txt'
    if os.path.exists(train_file):
        lm.train_from_file(train_file)
        
        # Test scoring
        test_texts = [
            "a-bu-šu-ma",
            "DUMU E-ri-ib",
            "hello world",
            "xyz123"
        ]
        
        print("\n" + "="*60)
        print("Testing Character LM")
        print("="*60)
        
        for text in test_texts:
            score = lm.score(text, normalize=True)
            perp = lm.perplexity(text)
            print(f"\nText: {text}")
            print(f"  Score: {score:.4f}")
            print(f"  Perplexity: {perp:.2f}")
        
        # Save model
        model_path = 'reports/research_assets/lm/char_python.json'
        lm.save(model_path)
        print(f"\n✅ Model saved to {model_path}")
        
        # Test loading
        lm2 = PythonCharLM()
        lm2.load(model_path)
        print("✅ Model loaded successfully")
    else:
        print(f"❌ Training file not found: {train_file}")
        sys.exit(1)
