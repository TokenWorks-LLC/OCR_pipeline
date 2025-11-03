#!/usr/bin/env python3
"""
Train a PythonCharLM model for Akkadian transliteration detection.

Extracts Akkadian transliterations from gold_text and trains a character-level
n-gram language model for use in lang_and_akkadian.py.

Usage:
    python tools/train_akkadian_lm.py --manifest data/gold/manifest_gold.txt --output models/akkadian_char_lm.json

Author: OCR Pipeline Team
Date: 2025-01-08
"""

import argparse
import sys
import re
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from python_char_lm import PythonCharLM


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def extract_akkadian_patterns(text: str) -> list[str]:
    """
    Extract Akkadian transliteration patterns from mixed text.
    
    Looks for:
    - Sequences with Akkadian special chars: š ṣ ṭ ḫ ā ē ī ū
    - Hyphenated syllabic patterns: a-na, i-ma-at, etc.
    - Cuneiform transliteration markers: [LUGAL], KÙ.BABBAR, etc.
    
    Args:
        text: Input text (may contain mixed languages)
        
    Returns:
        List of extracted Akkadian strings
    """
    akkadian_segments = []
    
    # Pattern 1: Sequences with special diacritics
    # Find runs of characters containing Akkadian special chars
    special_char_pattern = r'[a-zšṣṭḫāēīūâêîûA-ZŠṢṬḪĀĒĪŪÂÊÎÛ0-9\-\.\'\s]{3,}'
    
    for match in re.finditer(special_char_pattern, text):
        segment = match.group(0).strip()
        # Must contain at least one Akkadian special char
        if any(c in segment for c in 'šṣṭḫāēīūâêîû'):
            akkadian_segments.append(segment)
    
    # Pattern 2: Syllabic hyphenated sequences (a-na-kam, i-ma-at, etc.)
    # More aggressive - find sequences of letters/hyphens
    syllabic_pattern = r'\b[a-zšṣṭḫāēīūâêîû]{1,4}(?:-[a-zšṣṭḫāēīūâêîû]{1,4})+\b'
    
    for match in re.finditer(syllabic_pattern, text, re.IGNORECASE):
        segment = match.group(0)
        if segment not in akkadian_segments:  # Avoid duplicates
            akkadian_segments.append(segment)
    
    # Pattern 3: Transliteration markers [WORD], KÙ.BABBAR style
    marker_pattern = r'(\[[A-ZÀ-Ž\s]+\]|[A-ZÀ-Ž]{2,}\.[A-ZÀ-Ž]{2,})'
    
    for match in re.finditer(marker_pattern, text):
        segment = match.group(0)
        if segment not in akkadian_segments:
            akkadian_segments.append(segment)
    
    return akkadian_segments


def load_manifest_and_extract_akkadian(manifest_path: str) -> str:
    """
    Load manifest_gold.txt and extract all Akkadian text.
    
    Args:
        manifest_path: Path to manifest TSV file
        
    Returns:
        Combined Akkadian training text
    """
    logger = logging.getLogger(__name__)
    
    akkadian_corpus = []
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Skip header
    for i, line in enumerate(lines[1:], start=2):
        parts = line.split('\t')
        if len(parts) < 5:
            continue
        
        gold_text = parts[4].strip().strip('"')
        
        # Extract Akkadian segments
        segments = extract_akkadian_patterns(gold_text)
        
        if segments:
            logger.debug(f"Line {i}: Found {len(segments)} Akkadian segments")
            akkadian_corpus.extend(segments)
    
    combined_text = '\n'.join(akkadian_corpus)
    logger.info(f"Extracted {len(akkadian_corpus)} Akkadian segments from manifest")
    logger.info(f"Total corpus size: {len(combined_text)} characters")
    
    return combined_text


def train_and_save_model(corpus_text: str, output_path: str, order: int = 3, alpha: float = 0.1):
    """
    Train PythonCharLM and save to JSON.
    
    Args:
        corpus_text: Training corpus (Akkadian text)
        output_path: Output path for model JSON
        order: N-gram order (default: 3)
        alpha: Laplace smoothing parameter (default: 0.1)
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Training {order}-gram character LM...")
    
    lm = PythonCharLM(order=order, alpha=alpha)
    lm.train_from_text(corpus_text)
    
    logger.info(f"Vocabulary size: {len(lm.vocab)} characters")
    logger.info(f"N-gram count: {sum(len(counts) for counts in lm.ngram_counts.values())} n-grams")
    
    # Test perplexity on sample
    if len(corpus_text) > 100:
        sample = corpus_text[:100]
        perplexity = lm.perplexity(sample)
        logger.info(f"Sample perplexity (on training data): {perplexity:.2f}")
    
    # Save model
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    lm.save(str(output_path))
    logger.info(f"Saved model to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train Akkadian character LM from gold manifest"
    )
    parser.add_argument(
        '--manifest',
        required=True,
        help='Path to manifest_gold.txt file'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output path for trained model (JSON)'
    )
    parser.add_argument(
        '--order',
        type=int,
        default=3,
        help='N-gram order (default: 3)'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.1,
        help='Laplace smoothing parameter (default: 0.1)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    logger = setup_logging(args.verbose)
    
    logger.info("=== Akkadian Character LM Training ===")
    logger.info(f"Manifest: {args.manifest}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Order: {args.order}, Alpha: {args.alpha}")
    
    # Step 1: Extract Akkadian text
    logger.info("\n[1/2] Extracting Akkadian text from manifest...")
    akkadian_corpus = load_manifest_and_extract_akkadian(args.manifest)
    
    if not akkadian_corpus.strip():
        logger.error("No Akkadian text extracted! Check manifest format.")
        sys.exit(1)
    
    # Step 2: Train and save
    logger.info("\n[2/2] Training and saving model...")
    train_and_save_model(akkadian_corpus, args.output, args.order, args.alpha)
    
    logger.info("\n=== Training Complete ===")


if __name__ == '__main__':
    main()
