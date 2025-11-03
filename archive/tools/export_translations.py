#!/usr/bin/env python3
"""
Export and consolidate translation pairs from multiple PDFs.
Merges outputs/*/translations.csv → client_translations.csv
"""

import sys
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Set
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TranslationPair:
    """Translation pair for export."""
    pdf_id: str
    page: int
    akk_text: str
    translation_text: str
    translation_lang: str
    akk_bbox: str
    trans_bbox: str
    score: float
    # Optional layout info
    akk_column: int = 0
    trans_column: int = 0
    distance_px: float = 0.0
    same_column: bool = False
    has_marker: bool = False
    reading_order_ok: bool = False


def load_translations_csv(csv_path: Path) -> List[TranslationPair]:
    """Load translations from a single CSV file."""
    pairs = []
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pair = TranslationPair(
                    pdf_id=row['pdf_id'],
                    page=int(row['page']),
                    akk_text=row['akk_text'],
                    translation_text=row['trans_text'],
                    translation_lang=row['trans_lang'],
                    akk_bbox=row['akk_bbox'],
                    trans_bbox=row['trans_bbox'],
                    score=float(row['score']),
                    akk_column=int(row.get('akk_column', 0)),
                    trans_column=int(row.get('trans_column', 0)),
                    distance_px=float(row.get('distance_px', 0.0)),
                    same_column=row.get('same_column', 'False') == 'True',
                    has_marker=row.get('has_marker', 'False') == 'True',
                    reading_order_ok=row.get('reading_order_ok', 'False') == 'True'
                )
                pairs.append(pair)
    except Exception as e:
        logger.error(f"Error loading {csv_path}: {e}")
    
    return pairs


def find_translation_csvs(input_dir: Path) -> List[Path]:
    """Find all translations.csv files in input directory."""
    csv_files = list(input_dir.rglob("translations.csv"))
    logger.info(f"Found {len(csv_files)} translations.csv files in {input_dir}")
    return csv_files


def deduplicate_pairs(pairs: List[TranslationPair]) -> List[TranslationPair]:
    """
    Remove duplicate pairs based on (pdf_id, page, akk_text, translation_text).
    Keep the pair with highest score.
    """
    pairs_dict = {}
    
    for pair in pairs:
        key = (pair.pdf_id, pair.page, pair.akk_text, pair.translation_text)
        
        if key not in pairs_dict or pair.score > pairs_dict[key].score:
            pairs_dict[key] = pair
    
    deduped = list(pairs_dict.values())
    logger.info(f"Deduplicated: {len(pairs)} → {len(deduped)} pairs")
    return deduped


def write_client_csv(
    pairs: List[TranslationPair],
    output_path: Path,
    strip_layout: bool = False,
    excel_friendly: bool = False
):
    """
    Write consolidated translations.csv for client delivery.
    
    Args:
        pairs: Translation pairs to export
        output_path: Output CSV path
        strip_layout: If True, exclude layout metadata (columns, distance, etc.)
        excel_friendly: If True, use UTF-8 BOM and simplified 6-column schema
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Client-friendly format (runbook spec)
    if excel_friendly:
        encoding = 'utf-8-sig'  # UTF-8 with BOM for Excel
        fieldnames = [
            'pdf_name', 'page',
            'akkadian_text', 'translation_text', 'translation_lang',
            'notes'
        ]
    # Define fieldnames
    elif strip_layout:
        encoding = 'utf-8'
        fieldnames = [
            'pdf_id', 'page',
            'akk_text', 'translation_text', 'translation_lang',
            'akk_bbox', 'trans_bbox', 'score'
        ]
    else:
        encoding = 'utf-8'
        fieldnames = [
            'pdf_id', 'page',
            'akk_text', 'translation_text', 'translation_lang',
            'akk_bbox', 'trans_bbox', 'score',
            'akk_column', 'trans_column', 'distance_px',
            'same_column', 'has_marker', 'reading_order_ok'
        ]
    
    with open(output_path, 'w', newline='', encoding=encoding) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for pair in pairs:
            if excel_friendly:
                # Extract PDF filename from pdf_id
                pdf_name = Path(pair.pdf_id).name if '\\' in pair.pdf_id or '/' in pair.pdf_id else pair.pdf_id
                row = {
                    'pdf_name': pdf_name,
                    'page': pair.page,
                    'akkadian_text': pair.akk_text,
                    'translation_text': pair.translation_text,
                    'translation_lang': pair.translation_lang,
                    'notes': ''  # Empty by default
                }
            else:
                row = {
                    'pdf_id': pair.pdf_id,
                    'page': pair.page,
                    'akk_text': pair.akk_text,
                    'translation_text': pair.translation_text,
                    'translation_lang': pair.translation_lang,
                    'akk_bbox': pair.akk_bbox,
                    'trans_bbox': pair.trans_bbox,
                    'score': f"{pair.score:.4f}"
                }
                
                if not strip_layout:
                    row.update({
                        'akk_column': pair.akk_column,
                        'trans_column': pair.trans_column,
                        'distance_px': f"{pair.distance_px:.2f}",
                        'same_column': pair.same_column,
                        'has_marker': pair.has_marker,
                        'reading_order_ok': pair.reading_order_ok
                    })
            
            writer.writerow(row)
    
    logger.info(f"Wrote {len(pairs)} pairs to {output_path} (encoding: {encoding})")


def generate_summary_stats(pairs: List[TranslationPair]) -> Dict:
    """Generate summary statistics for export."""
    if not pairs:
        return {}
    
    # Count by language
    lang_counts = {}
    for pair in pairs:
        lang = pair.translation_lang
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
    
    # Count by PDF
    pdf_counts = {}
    for pair in pairs:
        pdf = pair.pdf_id
        pdf_counts[pdf] = pdf_counts.get(pdf, 0) + 1
    
    # Score statistics
    scores = [pair.score for pair in pairs]
    avg_score = sum(scores) / len(scores)
    min_score = min(scores)
    max_score = max(scores)
    
    return {
        'total_pairs': len(pairs),
        'unique_pdfs': len(pdf_counts),
        'translations_by_language': lang_counts,
        'avg_score': avg_score,
        'min_score': min_score,
        'max_score': max_score
    }


def main():
    parser = argparse.ArgumentParser(description='Export and consolidate translations')
    parser.add_argument('--inputs', required=True,
                       help='Input directory containing outputs/*/translations.csv')
    parser.add_argument('--out', required=True,
                       help='Output client_translations.csv path')
    parser.add_argument('--dedupe', action='store_true',
                       help='Remove duplicate pairs (keep highest score)')
    parser.add_argument('--strip-layout', action='store_true',
                       help='Exclude layout metadata (columns, distances, etc.)')
    parser.add_argument('--excel-friendly', action='store_true',
                       help='Use UTF-8 BOM and simplified 6-column schema for Excel')
    parser.add_argument('--min-score', type=float, default=0.0,
                       help='Minimum score threshold (default: 0.0)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("TRANSLATION EXPORT")
    print("=" * 60)
    
    # Find input CSVs
    input_dir = Path(args.inputs)
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)
    
    csv_paths = find_translation_csvs(input_dir)
    
    if not csv_paths:
        logger.error(f"No translations.csv files found in {input_dir}")
        sys.exit(1)
    
    # Load all pairs
    print()
    logger.info("Loading translation pairs...")
    all_pairs = []
    
    for csv_path in csv_paths:
        pairs = load_translations_csv(csv_path)
        logger.info(f"  {csv_path.parent.name}: {len(pairs)} pairs")
        all_pairs.extend(pairs)
    
    logger.info(f"Total pairs loaded: {len(all_pairs)}")
    
    if not all_pairs:
        logger.error("No pairs loaded!")
        sys.exit(1)
    
    # Filter by score
    if args.min_score > 0:
        before = len(all_pairs)
        all_pairs = [p for p in all_pairs if p.score >= args.min_score]
        logger.info(f"Score filter ({args.min_score}): {before} → {len(all_pairs)} pairs")
    
    # Deduplicate
    if args.dedupe:
        all_pairs = deduplicate_pairs(all_pairs)
    
    # Sort by PDF, page, score
    all_pairs.sort(key=lambda p: (p.pdf_id, p.page, -p.score))
    
    # Write output
    output_path = Path(args.out)
    write_client_csv(all_pairs, output_path, strip_layout=args.strip_layout, excel_friendly=args.excel_friendly)
    
    # Generate stats
    stats = generate_summary_stats(all_pairs)
    
    # Print summary
    print()
    print("=" * 60)
    print("EXPORT SUMMARY")
    print("=" * 60)
    print(f"Input directory:        {input_dir}")
    print(f"CSV files processed:    {len(csv_paths)}")
    print(f"Total pairs:            {stats['total_pairs']}")
    print(f"Unique PDFs:            {stats['unique_pdfs']}")
    print(f"Average score:          {stats['avg_score']:.3f}")
    print(f"Score range:            {stats['min_score']:.3f} - {stats['max_score']:.3f}")
    print()
    print(f"Translations by language:")
    for lang, count in sorted(stats['translations_by_language'].items()):
        print(f"  {lang}: {count}")
    print()
    print(f"Output file:            {output_path}")
    print()
    print("=" * 60)
    print("✅ Export completed successfully!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
