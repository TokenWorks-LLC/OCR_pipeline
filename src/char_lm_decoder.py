#!/usr/bin/env python3
"""
Character-level language model decoder with shallow fusion.

Integrates character LM into OCR decoding for CTC and attention-based engines.

CTC Engines (docTR, CRNN, SAR):
    Use pyctcdecode with KenLM for beam search decoding (if available)
    Otherwise use Python-based LM for re-scoring

Attention Engines (ABINet, PARSeq):
    Add α·logP_LM + β·length_norm during beam expansion

Author: Senior OCR Engineer
Date: 2025-10-06
Updated: 2025-10-07 - Added pure Python LM fallback
"""

import os
import logging
from typing import List, Tuple, Optional, Dict
import numpy as np

# Optional dependencies
try:
    import kenlm
    KENLM_AVAILABLE = True
except ImportError:
    KENLM_AVAILABLE = False

try:
    from pyctcdecode import build_ctcdecoder
    PYCTCDECODE_AVAILABLE = True
except ImportError:
    PYCTCDECODE_AVAILABLE = False

# Pure Python LM (always available)
try:
    from python_char_lm import PythonCharLM
    PYTHON_LM_AVAILABLE = True
except ImportError:
    PYTHON_LM_AVAILABLE = False


class CharacterLM:
    """
    Character-level language model wrapper.
    
    Supports both KenLM (for perplexity/scoring) and pure Python LM (fallback).
    Automatically uses Python LM if KenLM is not available.
    """
    
    def __init__(self, lm_path: str, charset_path: str, 
                 alpha: float = 0.7, beta: float = 0.3):
        """
        Initialize character LM.
        
        Args:
            lm_path: Path to KenLM binary (.klm) or Python LM (.json)
            charset_path: Path to charset.txt
            alpha: LM weight (default: 0.7)
            beta: Length normalization weight (default: 0.3)
        """
        self.logger = logging.getLogger(__name__)
        
        self.lm_path = lm_path
        self.charset_path = charset_path
        self.alpha = alpha
        self.beta = beta
        self.model = None
        self.use_kenlm = False
        
        # Try to load KenLM first
        if KENLM_AVAILABLE and lm_path.endswith('.klm'):
            try:
                self.logger.info(f"Loading KenLM from {lm_path}")
                self.model = kenlm.Model(lm_path)
                self.use_kenlm = True
                self.logger.info("✅ KenLM loaded successfully")
            except Exception as e:
                self.logger.warning(f"KenLM loading failed: {e}, falling back to Python LM")
        
        # Fall back to Python LM
        if self.model is None:
            if not PYTHON_LM_AVAILABLE:
                raise ImportError("Neither kenlm nor python_char_lm available")
            
            # Check for Python LM model file
            python_lm_path = lm_path.replace('.klm', '_python.json')
            if not os.path.exists(python_lm_path):
                python_lm_path = 'reports/research_assets/lm/char_python.json'
            
            if os.path.exists(python_lm_path):
                self.logger.info(f"Loading Python LM from {python_lm_path}")
                self.model = PythonCharLM()
                self.model.load(python_lm_path)
                self.use_kenlm = False
                self.logger.info("✅ Python LM loaded successfully")
            else:
                # Train new Python LM from training data
                train_path = lm_path.replace('.klm', '_train.txt')
                if not os.path.exists(train_path):
                    train_path = 'reports/research_assets/lm/train.txt'
                
                if os.path.exists(train_path):
                    self.logger.info(f"Training new Python LM from {train_path}")
                    self.model = PythonCharLM(order=3, alpha=0.1)
                    self.model.train_from_file(train_path)
                    self.model.save(python_lm_path)
                    self.use_kenlm = False
                    self.logger.info("✅ Python LM trained and saved")
                else:
                    raise FileNotFoundError(f"No LM model or training data found: {lm_path}, {python_lm_path}, {train_path}")
        
        # Load charset
        self.charset = self._load_charset(charset_path)
        self.logger.info(f"Loaded charset with {len(self.charset)} characters")
    
    def _load_charset(self, charset_path: str) -> List[str]:
        """Load charset from file."""
        charset = []
        
        with open(charset_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line == '\\n':
                    charset.append('\n')
                elif line == '\\t':
                    charset.append('\t')
                elif line == '[SPACE]':
                    charset.append(' ')
                elif line:
                    charset.append(line)
        
        return charset
    
    def score(self, text: str, bos: bool = True, eos: bool = True) -> float:
        """
        Score text with LM (log probability).
        
        Args:
            text: Input text
            bos: Add beginning-of-sentence token (for KenLM)
            eos: End-of-sentence token (for KenLM)
            
        Returns:
            Log probability score
        """
        if self.use_kenlm:
            # Convert to space-separated characters for KenLM
            char_seq = ' '.join(text)
            return self.model.score(char_seq, bos=bos, eos=eos)
        else:
            # Use Python LM
            return self.model.score(text, normalize=False)
    
    def perplexity(self, text: str) -> float:
        """
        Compute perplexity of text.
        
        Args:
            text: Input text
            
        Returns:
            Perplexity value
        """
        if self.use_kenlm:
            char_seq = ' '.join(text)
            return self.model.perplexity(char_seq)
        else:
            return self.model.perplexity(text)
    
    def score_char_sequence(self, chars: List[str]) -> float:
        """
        Score a sequence of characters.
        
        Args:
            chars: List of characters
            
        Returns:
            Log probability score
        """
        text = ''.join(chars)
        return self.score(text, bos=False, eos=False)


class CTCDecoderWithLM:
    """
    CTC decoder with character-level LM using pyctcdecode.
    
    For CTC-based engines like docTR SAR, CRNN, etc.
    """
    
    def __init__(self, vocab: List[str], lm_path: str, charset_path: str,
                 alpha: float = 0.7, beta: float = 0.3, 
                 beam_width: int = 100):
        """
        Initialize CTC decoder with LM.
        
        Args:
            vocab: CTC vocabulary (including blank token)
            lm_path: Path to KenLM binary
            charset_path: Path to charset.txt
            alpha: LM weight
            beta: Length normalization weight
            beam_width: Beam search width
        """
        self.logger = logging.getLogger(__name__)
        
        if not PYCTCDECODE_AVAILABLE:
            raise ImportError("pyctcdecode not installed. Run: pip install pyctcdecode")
        
        self.vocab = vocab
        self.alpha = alpha
        self.beta = beta
        self.beam_width = beam_width
        
        # Load character LM
        self.char_lm = CharacterLM(lm_path, charset_path, alpha, beta)
        
        # Build CTC decoder
        self.logger.info("Building CTC decoder with LM...")
        self.decoder = build_ctcdecoder(
            labels=vocab,
            kenlm_model_path=lm_path,
            alpha=alpha,
            beta=beta
        )
        
        self.logger.info(f"CTC decoder ready (beam={beam_width})")
    
    def score(self, text: str) -> float:
        """
        Score text with LM (delegates to underlying CharacterLM).
        
        Args:
            text: Input text
            
        Returns:
            Log probability score
        """
        return self.char_lm.score(text)
    
    def decode(self, logits: np.ndarray, beam_width: Optional[int] = None) -> str:
        """
        Decode CTC logits with LM.
        
        Args:
            logits: CTC logits array (T x V) where T=timesteps, V=vocab size
            beam_width: Beam width (default: use self.beam_width)
            
        Returns:
            Decoded text
        """
        if beam_width is None:
            beam_width = self.beam_width
        
        # Decode with beam search + LM
        text = self.decoder.decode(logits, beam_width=beam_width)
        
        return text
    
    def decode_batch(self, logits_batch: np.ndarray, 
                    beam_width: Optional[int] = None) -> List[str]:
        """
        Decode batch of CTC logits.
        
        Args:
            logits_batch: Batch of logits (B x T x V)
            beam_width: Beam width
            
        Returns:
            List of decoded texts
        """
        if beam_width is None:
            beam_width = self.beam_width
        
        texts = self.decoder.decode_batch(logits_batch, beam_width=beam_width)
        
        return texts


class AttentionDecoderWithLM:
    """
    Attention decoder with LM shallow fusion.
    
    For attention-based engines like ABINet, PARSeq.
    Adds α·logP_LM + β·length_norm during beam search.
    """
    
    def __init__(self, lm_path: str, charset_path: str,
                 alpha: float = 0.7, beta: float = 0.3,
                 length_norm: float = 0.0):
        """
        Initialize attention decoder with LM.
        
        Args:
            lm_path: Path to KenLM binary
            charset_path: Path to charset.txt
            alpha: LM weight
            beta: Length normalization weight (unused if length_norm=0)
            length_norm: Length normalization exponent (0 = disabled)
        """
        self.logger = logging.getLogger(__name__)
        
        self.alpha = alpha
        self.beta = beta
        self.length_norm = length_norm
        
        # Load character LM
        self.char_lm = CharacterLM(lm_path, charset_path, alpha, beta)
        
        self.logger.info(f"Attention decoder with LM (α={alpha}, β={beta}, len_norm={length_norm})")
    
    def score(self, text: str) -> float:
        """
        Score text with LM (delegates to underlying CharacterLM).
        
        Args:
            text: Input text
            
        Returns:
            Log probability score
        """
        return self.char_lm.score(text)
    
    def rescore_hypothesis(self, text: str, model_score: float) -> float:
        """
        Re-score hypothesis by combining model score with LM score.
        
        Combined score = model_score + α·lm_score + β·length_penalty
        
        Args:
            text: Hypothesis text
            model_score: Score from attention model (log probability)
            
        Returns:
            Combined score
        """
        # Get LM score
        lm_score = self.char_lm.score(text, bos=True, eos=True)
        
        # Length penalty (if enabled)
        length_penalty = 0.0
        if self.length_norm > 0:
            length_penalty = len(text) ** self.length_norm
        
        # Combined score
        combined_score = model_score + self.alpha * lm_score + self.beta * length_penalty
        
        return combined_score
    
    def rescore_beam(self, hypotheses: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        Re-score a beam of hypotheses with LM.
        
        Args:
            hypotheses: List of (text, model_score) tuples
            
        Returns:
            List of (text, combined_score) tuples, sorted by score (descending)
        """
        rescored = []
        
        for text, model_score in hypotheses:
            combined_score = self.rescore_hypothesis(text, model_score)
            rescored.append((text, combined_score))
        
        # Sort by score (descending)
        rescored.sort(key=lambda x: x[1], reverse=True)
        
        return rescored


def create_lm_decoder(engine_type: str, lm_path: str, charset_path: str,
                     vocab: Optional[List[str]] = None,
                     config: Optional[Dict] = None) -> object:
    """
    Factory function to create appropriate LM decoder for engine type.
    
    Args:
        engine_type: 'ctc' or 'attention'
        lm_path: Path to KenLM binary
        charset_path: Path to charset.txt
        vocab: CTC vocabulary (required if engine_type='ctc')
        config: Optional config dict with alpha, beta, etc.
        
    Returns:
        CTCDecoderWithLM or AttentionDecoderWithLM instance
    """
    if config is None:
        config = {}
    
    alpha = config.get('alpha', 0.7)
    beta = config.get('beta', 0.3)
    length_norm = config.get('length_norm', 0.0)
    
    if engine_type.lower() == 'ctc':
        if vocab is None:
            raise ValueError("vocab required for CTC decoder")
        
        beam_width = config.get('beam_width', 100)
        
        return CTCDecoderWithLM(
            vocab=vocab,
            lm_path=lm_path,
            charset_path=charset_path,
            alpha=alpha,
            beta=beta,
            beam_width=beam_width
        )
    
    elif engine_type.lower() == 'attention':
        return AttentionDecoderWithLM(
            lm_path=lm_path,
            charset_path=charset_path,
            alpha=alpha,
            beta=beta,
            length_norm=length_norm
        )
    
    else:
        raise ValueError(f"Unknown engine_type: {engine_type}. Use 'ctc' or 'attention'")


if __name__ == '__main__':
    # Test character LM
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python char_lm_decoder.py <lm_path> <charset_path>")
        sys.exit(1)
    
    lm_path = sys.argv[1]
    charset_path = sys.argv[2]
    
    print("=== Character LM Decoder Test ===\n")
    
    # Test CharacterLM
    print("[1] Testing CharacterLM...")
    char_lm = CharacterLM(lm_path, charset_path)
    
    test_texts = [
        "a-na DINGIR-lí-šu",
        "KÙ.BABBAR ša-ru-pá-am",
        "completely random gibberish xyz"
    ]
    
    for text in test_texts:
        score = char_lm.score(text)
        perplexity = char_lm.perplexity(text)
        print(f"  Text: {text}")
        print(f"    Score: {score:.4f}")
        print(f"    Perplexity: {perplexity:.2f}")
        print()
    
    # Test AttentionDecoderWithLM
    print("[2] Testing AttentionDecoderWithLM...")
    attention_decoder = AttentionDecoderWithLM(lm_path, charset_path)
    
    hypotheses = [
        ("a-na DINGIR-li-su", -5.2),  # Missing diacritic
        ("a-na DINGIR-lí-šu", -4.8),  # Correct
    ]
    
    print("  Input beam:")
    for text, score in hypotheses:
        print(f"    {text}: {score:.4f}")
    
    rescored = attention_decoder.rescore_beam(hypotheses)
    
    print("\n  Rescored beam:")
    for text, score in rescored:
        print(f"    {text}: {score:.4f}")
    
    print("\n=== Test Complete ===")
