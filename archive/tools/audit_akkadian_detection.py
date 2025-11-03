#!/usr/bin/env python3
"""
Akkadian Detection Threshold Calibration Tool

Sweeps detection thresholds on known-positive PDFs to find optimal settings.
Implements "any-line" aggregation: block=Akkadian if (qual_lines ≥ 3) OR (qual_ratio ≥ 0.25)
"""

import sys
import csv
import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from lang_and_akkadian import is_akkadian_transliteration
from blockification import TextBlockifier
import fitz  # PyMuPDF

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_manifest(manifest_path: Path) -> List[Tuple[str, int]]:
    """Load manifest TSV file."""
    entries = []
    with open(manifest_path, 'r', encoding='utf-8-sig') as f:  # utf-8-sig strips BOM
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            pdf_path = row['pdf_path']
            page_no = int(row['page_no'])
            entries.append((pdf_path, page_no))
    return entries


def extract_and_blockify(pdf_path: str, page_no: int) -> List[Dict]:
    """Extract text and create blocks from PDF page."""
    doc = fitz.open(pdf_path)
    page = doc[page_no - 1]  # 0-indexed
    text = page.get_text()
    doc.close()
    
    if not text or len(text.strip()) < 50:
        return []
    
    # Create simple blocks (line-based for calibration)
    lines = [line.strip() for line in text.split('\n') if line.strip() and len(line.strip()) > 10]
    
    blocks = []
    for i, line in enumerate(lines):
        blocks.append({
            'text': line,
            'id': f'line_{i}',
            'bbox': (0, i*20, 500, (i+1)*20)  # Dummy bbox
        })
    
    return blocks


def test_threshold(
    blocks: List[Dict],
    threshold: float,
    require_diacritic: bool,
    min_syllabic_tokens: int,
    min_syllabic_ratio: float,
    agg_mode: str
) -> Dict:
    """Test detection at given threshold."""
    config = {
        'threshold': threshold,
        'require_diacritic_or_marker': require_diacritic,
        'min_syllabic_tokens': min_syllabic_tokens,
        'min_syllabic_ratio': min_syllabic_ratio,
        'markers_strict': True
    }
    
    qual_lines = 0
    total_lines = len(blocks)
    detected_lines = []
    
    for block in blocks:
        is_akk, conf = is_akkadian_transliteration(block['text'], config=config)
        if is_akk:
            qual_lines += 1
            detected_lines.append({
                'text': block['text'][:100],
                'conf': conf
            })
    
    # Apply aggregation logic
    if agg_mode == 'any-line':
        # Block is Akkadian if: qual_lines >= 3 OR qual_ratio >= 0.25
        qual_ratio = qual_lines / total_lines if total_lines > 0 else 0.0
        block_is_akkadian = (qual_lines >= 3) or (qual_ratio >= 0.25)
    else:
        # Standard: block is Akkadian if any line qualifies
        block_is_akkadian = qual_lines > 0
    
    return {
        'threshold': threshold,
        'total_lines': total_lines,
        'qual_lines': qual_lines,
        'qual_ratio': qual_lines / total_lines if total_lines > 0 else 0.0,
        'block_detected': block_is_akkadian,
        'detected_lines': detected_lines
    }


def main():
    parser = argparse.ArgumentParser(description='Calibrate Akkadian detection thresholds')
    parser.add_argument('--manifest', required=True, help='Manifest file with known-positive PDFs')
    parser.add_argument('--thresholds', default='0.25,0.30,0.35,0.40,0.45,0.50', 
                       help='Comma-separated thresholds to test')
    parser.add_argument('--require-diacritic-or-marker', action='store_true',
                       help='Require diacritics or markers')
    parser.add_argument('--min-syllabic-tokens', type=int, default=3,
                       help='Minimum syllabic tokens')
    parser.add_argument('--min-syllabic-ratio', type=float, default=0.25,
                       help='Minimum syllabic ratio')
    parser.add_argument('--agg', choices=['any-line', 'standard'], default='any-line',
                       help='Aggregation mode')
    parser.add_argument('--out', required=True, help='Output CSV path')
    
    args = parser.parse_args()
    
    # Parse thresholds
    thresholds = [float(t.strip()) for t in args.thresholds.split(',')]
    
    logger.info(f"Loading manifest: {args.manifest}")
    manifest = load_manifest(Path(args.manifest))
    logger.info(f"Found {len(manifest)} pages to test")
    
    # Prepare output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    # Test each threshold
    for threshold in thresholds:
        logger.info(f"\n{'='*70}")
        logger.info(f"Testing threshold: {threshold}")
        logger.info(f"{'='*70}")
        
        total_blocks = 0
        total_detected = 0
        total_qual_lines = 0
        total_lines = 0
        
        for pdf_path, page_no in manifest:
            logger.info(f"Processing: {Path(pdf_path).name} page {page_no}")
            
            try:
                blocks = extract_and_blockify(pdf_path, page_no)
                if not blocks:
                    logger.warning(f"No blocks extracted from {pdf_path} page {page_no}")
                    continue
                
                result = test_threshold(
                    blocks,
                    threshold,
                    args.require_diacritic_or_marker,
                    args.min_syllabic_tokens,
                    args.min_syllabic_ratio,
                    args.agg
                )
                
                total_blocks += 1
                if result['block_detected']:
                    total_detected += 1
                
                total_qual_lines += result['qual_lines']
                total_lines += result['total_lines']
                
                logger.info(f"  Lines: {result['total_lines']}, Qualified: {result['qual_lines']} "
                          f"({result['qual_ratio']:.2%}), Block detected: {result['block_detected']}")
                
                if result['detected_lines']:
                    logger.info(f"  Sample Akkadian lines:")
                    for line_info in result['detected_lines'][:3]:
                        logger.info(f"    [{line_info['conf']:.3f}] {line_info['text']}")
            
            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {e}")
                continue
        
        # Calculate metrics
        recall = total_detected / total_blocks if total_blocks > 0 else 0.0
        avg_qual_ratio = total_qual_lines / total_lines if total_lines > 0 else 0.0
        
        result_row = {
            'threshold': threshold,
            'require_diacritic': args.require_diacritic_or_marker,
            'min_syllabic_tokens': args.min_syllabic_tokens,
            'min_syllabic_ratio': args.min_syllabic_ratio,
            'agg_mode': args.agg,
            'total_blocks': total_blocks,
            'detected_blocks': total_detected,
            'recall': recall,
            'total_lines': total_lines,
            'qual_lines': total_qual_lines,
            'avg_qual_ratio': avg_qual_ratio
        }
        
        results.append(result_row)
        
        logger.info(f"\nThreshold {threshold} summary:")
        logger.info(f"  Blocks detected: {total_detected}/{total_blocks} (recall={recall:.2%})")
        logger.info(f"  Qualified lines: {total_qual_lines}/{total_lines} ({avg_qual_ratio:.2%})")
    
    # Write results
    logger.info(f"\n{'='*70}")
    logger.info(f"Writing results to: {out_path}")
    logger.info(f"{'='*70}")
    
    with open(out_path, 'w', encoding='utf-8', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    
    # Print recommendation
    logger.info("\nRECOMMENDATION:")
    logger.info("="*70)
    
    # Find best threshold: highest recall, then lowest threshold (for sensitivity)
    best = max(results, key=lambda r: (r['recall'], -r['threshold']))
    
    logger.info(f"Best threshold: {best['threshold']}")
    logger.info(f"  Recall: {best['recall']:.2%}")
    logger.info(f"  Avg qualified ratio: {best['avg_qual_ratio']:.2%}")
    logger.info(f"  Detected blocks: {best['detected_blocks']}/{best['total_blocks']}")
    
    if best['recall'] < 1.0:
        logger.warning(f"⚠️  Recall < 100% - some known-positive blocks not detected!")
        logger.warning(f"Consider lowering threshold to: {best['threshold'] - 0.05:.2f}")
    
    logger.info(f"\n✅ Calibration complete! Results saved to: {out_path}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
