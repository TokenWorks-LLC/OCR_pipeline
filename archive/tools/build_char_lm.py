#!/usr/bin/env python3
"""
Build a character-level n-gram language model from gold standard data.

This tool builds a KenLM character LM from gold transcriptions to improve
OCR accuracy through shallow fusion during decoding.

REQUIREMENTS:
- Build 5-gram char LM from gold HANDTYPED column
- Save charset.txt, char.arpa, char.klm
- Ensure 0% OOV on gold data (all chars in charset)
- Include diacritics, subscripts, brackets, colon, hyphen

Dependencies:
    pip install kenlm

Author: Senior OCR Engineer
Date: 2025-10-06
"""

import argparse
import os
import sys
import logging
from pathlib import Path
from collections import Counter
from typing import Set, List
import unicodedata

# Add src to path
sys.path.append('src')

try:
    from translit_norm import to_nfc
    TRANSLIT_AVAILABLE = True
except ImportError:
    TRANSLIT_AVAILABLE = False
    def to_nfc(s):
        return unicodedata.normalize("NFC", s)


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_gold_texts(gold_csv_path: str, text_column: str = 'HANDTYPED') -> List[str]:
    """
    Load text data from gold CSV.
    
    Args:
        gold_csv_path: Path to gold standard CSV file
        text_column: Column name containing gold text (default: HANDTYPED)
        
    Returns:
        List of text strings (NFC normalized)
    """
    import pandas as pd
    
    logger = logging.getLogger(__name__)
    
    # Try different encodings
    for encoding in ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']:
        try:
            df = pd.read_csv(gold_csv_path, encoding=encoding)
            logger.info(f"Loaded CSV with {encoding} encoding")
            break
        except Exception as e:
            continue
    else:
        raise ValueError(f"Could not load {gold_csv_path} with any encoding")
    
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found. Available: {df.columns.tolist()}")
    
    # Extract and normalize texts
    texts = []
    for text in df[text_column].dropna():
        text_str = str(text).strip()
        if text_str:
            # Apply NFC normalization
            text_nfc = to_nfc(text_str)
            texts.append(text_nfc)
    
    logger.info(f"Loaded {len(texts)} text samples from gold data")
    
    return texts


def extract_charset(texts: List[str]) -> Set[str]:
    """
    Extract all unique characters from texts.
    
    Includes:
    - All letters (with diacritics)
    - Digits (including subscripts)
    - Punctuation (-, :, ., [, ], (, ), /, etc.)
    - Whitespace (space, newline)
    
    Args:
        texts: List of text strings
        
    Returns:
        Set of unique characters
    """
    charset = set()
    
    for text in texts:
        charset.update(text)
    
    # Always include essential whitespace
    charset.add(' ')
    charset.add('\n')
    
    return charset


def save_charset(charset: Set[str], output_path: str):
    """
    Save charset to file (one character per line).
    
    Args:
        charset: Set of characters
        output_path: Output file path
    """
    logger = logging.getLogger(__name__)
    
    # Sort for reproducibility (by Unicode code point)
    sorted_chars = sorted(charset, key=lambda c: ord(c))
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for char in sorted_chars:
            # Escape newline and other special chars for readability
            if char == '\n':
                f.write('\\n\n')
            elif char == '\t':
                f.write('\\t\n')
            elif char == ' ':
                f.write('[SPACE]\n')
            else:
                f.write(f'{char}\n')
    
    logger.info(f"Saved charset ({len(charset)} chars) to {output_path}")
    
    # Log character categories
    diacritics = sum(1 for c in charset if unicodedata.category(c) == 'Mn')
    letters = sum(1 for c in charset if unicodedata.category(c).startswith('L'))
    digits = sum(1 for c in charset if unicodedata.category(c).startswith('N'))
    punct = sum(1 for c in charset if unicodedata.category(c).startswith('P'))
    
    logger.info(f"  Letters: {letters}, Digits: {digits}, Punct: {punct}, Diacritics: {diacritics}")


def prepare_training_data(texts: List[str], output_path: str):
    """
    Prepare training data for KenLM (character-level).
    
    Format: Each line is a sequence of space-separated characters.
    Example: "a-na" → "a - n a"
    
    Args:
        texts: List of text strings
        output_path: Output file path for training data
    """
    logger = logging.getLogger(__name__)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for text in texts:
            # Split text into lines (preserve line boundaries)
            lines = text.split('\n')
            
            for line in lines:
                if not line.strip():
                    continue
                
                # Convert to space-separated characters
                char_seq = ' '.join(line)
                f.write(char_seq + '\n')
    
    logger.info(f"Prepared training data: {output_path}")


def build_kenlm(train_file: str, output_arpa: str, output_klm: str, 
                order: int = 5, prune_threshold: str = '1e-7'):
    """
    Build KenLM language model.
    
    Args:
        train_file: Path to training data (space-separated chars)
        output_arpa: Output ARPA file path
        output_klm: Output compiled KLM file path
        order: N-gram order (default: 5)
        prune_threshold: Pruning threshold for rare n-grams
    """
    import subprocess
    logger = logging.getLogger(__name__)
    
    # Check if lmplz is available
    try:
        subprocess.run(['lmplz', '--help'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("KenLM 'lmplz' not found. Please install KenLM:")
        logger.error("  Ubuntu: sudo apt-get install kenlm")
        logger.error("  Mac: brew install kenlm")
        logger.error("  Or build from source: https://github.com/kpu/kenlm")
        raise RuntimeError("KenLM not available")
    
    # Build ARPA model
    logger.info(f"Building {order}-gram ARPA model...")
    cmd_arpa = [
        'lmplz',
        '-o', str(order),
        '--discount_fallback',
        '--text', train_file,
        '--arpa', output_arpa,
        '--prune', prune_threshold
    ]
    
    result = subprocess.run(cmd_arpa, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"lmplz failed: {result.stderr}")
        raise RuntimeError("Failed to build ARPA model")
    
    logger.info(f"ARPA model saved: {output_arpa}")
    
    # Compile to binary KLM
    logger.info("Compiling to binary KLM format...")
    cmd_klm = [
        'build_binary',
        output_arpa,
        output_klm
    ]
    
    result = subprocess.run(cmd_klm, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"build_binary failed: {result.stderr}")
        raise RuntimeError("Failed to compile KLM")
    
    logger.info(f"Binary KLM saved: {output_klm}")


def validate_lm(lm_path: str, charset_path: str, gold_texts: List[str]):
    """
    Validate that LM has 0% OOV on gold data.
    
    Args:
        lm_path: Path to KLM file
        charset_path: Path to charset file
        gold_texts: Gold text samples
    """
    import kenlm
    logger = logging.getLogger(__name__)
    
    # Load LM
    model = kenlm.Model(lm_path)
    
    # Load charset
    with open(charset_path, 'r', encoding='utf-8') as f:
        charset_lines = [line.strip() for line in f]
    
    # Reconstitute charset
    charset = set()
    for line in charset_lines:
        if line == '\\n':
            charset.add('\n')
        elif line == '\\t':
            charset.add('\t')
        elif line == '[SPACE]':
            charset.add(' ')
        elif line:
            charset.add(line)
    
    # Check OOV
    oov_chars = set()
    total_chars = 0
    
    for text in gold_texts:
        for char in text:
            total_chars += 1
            if char not in charset:
                oov_chars.add(char)
    
    oov_rate = len(oov_chars) / total_chars if total_chars > 0 else 0.0
    
    logger.info(f"Validation results:")
    logger.info(f"  Total characters in gold: {total_chars}")
    logger.info(f"  Unique OOV characters: {len(oov_chars)}")
    logger.info(f"  OOV rate: {oov_rate*100:.4f}%")
    
    if oov_chars:
        logger.warning(f"  OOV characters found: {sorted(oov_chars)}")
        logger.warning("  This violates the 0% OOV requirement!")
    else:
        logger.info("  ✓ 0% OOV achieved - all gold characters in charset")
    
    # Test perplexity on a sample
    if gold_texts:
        sample = gold_texts[0][:100]  # First 100 chars
        char_seq = ' '.join(sample)
        score = model.score(char_seq, bos=True, eos=True)
        perplexity = model.perplexity(char_seq)
        
        logger.info(f"  Sample perplexity: {perplexity:.2f}")
        logger.info(f"  Sample score: {score:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Build character-level n-gram language model from gold data"
    )
    parser.add_argument(
        '--gold',
        required=True,
        help='Path to gold CSV file'
    )
    parser.add_argument(
        '--out',
        required=True,
        help='Output directory for LM files'
    )
    parser.add_argument(
        '--order',
        type=int,
        default=5,
        help='N-gram order (default: 5)'
    )
    parser.add_argument(
        '--charset',
        help='Output path for charset.txt (default: OUT/charset.txt)'
    )
    parser.add_argument(
        '--column',
        default='HANDTYPED',
        help='CSV column containing gold text (default: HANDTYPED)'
    )
    parser.add_argument(
        '--prune',
        default='1e-7',
        help='Pruning threshold for rare n-grams (default: 1e-7)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    logger = setup_logging(args.verbose)
    
    # Create output directory
    os.makedirs(args.out, exist_ok=True)
    
    # Set output paths
    charset_path = args.charset or os.path.join(args.out, 'charset.txt')
    train_path = os.path.join(args.out, 'train.txt')
    arpa_path = os.path.join(args.out, 'char.arpa')
    klm_path = os.path.join(args.out, 'char.klm')
    
    logger.info("=== Character LM Builder ===")
    logger.info(f"Gold data: {args.gold}")
    logger.info(f"Output dir: {args.out}")
    logger.info(f"N-gram order: {args.order}")
    
    # Step 1: Load gold texts
    logger.info("\n[1/5] Loading gold texts...")
    texts = load_gold_texts(args.gold, args.column)
    
    # Step 2: Extract charset
    logger.info("\n[2/5] Extracting character set...")
    charset = extract_charset(texts)
    save_charset(charset, charset_path)
    
    # Step 3: Prepare training data
    logger.info("\n[3/5] Preparing training data...")
    prepare_training_data(texts, train_path)
    
    # Step 4: Build KenLM
    logger.info("\n[4/5] Building KenLM...")
    build_kenlm(train_path, arpa_path, klm_path, args.order, args.prune)
    
    # Step 5: Validate
    logger.info("\n[5/5] Validating LM...")
    validate_lm(klm_path, charset_path, texts)
    
    logger.info("\n=== LM Build Complete ===")
    logger.info(f"Charset: {charset_path}")
    logger.info(f"ARPA: {arpa_path}")
    logger.info(f"Binary KLM: {klm_path}")


if __name__ == '__main__':
    main()
