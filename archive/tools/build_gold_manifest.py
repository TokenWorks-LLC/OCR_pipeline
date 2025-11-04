#!/usr/bin/env python3
"""
Build manifest from gold_pages.csv for gold validation.
Maps CSV entries to extracted page PDFs in data/gold_pages_only/
"""

import sys
import csv
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import logging
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def normalize_filename(pdf_name: str, page_no: int) -> str:
    """
    Convert PDF name and page number to expected filename format.
    
    Example:
        "AKT 1, 1990.pdf", 21 -> "AKT_1_1990_page_21.pdf"
    """
    # Remove .pdf extension
    base = pdf_name.replace('.pdf', '').replace('.PDF', '')
    
    # Replace spaces and special chars with underscores
    base = re.sub(r'[,\s]+', '_', base)
    base = re.sub(r'[()]', '', base)
    base = re.sub(r'_+', '_', base)  # Collapse multiple underscores
    
    # Build expected filename
    expected = f"{base}_page_{page_no}.pdf"
    
    return expected


def load_gold_pages_csv(csv_path: Path) -> List[Tuple[str, int, str]]:
    """
    Load gold_pages.csv and return list of (pdf_name, page_no, gold_text) tuples.
    """
    logger.info(f"Loading gold pages from: {csv_path}")
    
    entries = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pdf_name = row['PDF LINK']
            page_str = row['PAGE']
            gold_text = row.get('HANDTYPED', '')
            
            # Handle page ranges like "33-33" or "9"
            if '-' in page_str:
                # Take first page of range
                page_no = int(page_str.split('-')[0])
            else:
                page_no = int(page_str)
            
            entries.append((pdf_name, page_no, gold_text))
    
    logger.info(f"Loaded {len(entries)} gold page entries")
    return entries


def match_extracted_pages(
    gold_entries: List[Tuple[str, int, str]],
    extracted_dir: Path
) -> List[Dict]:
    """
    Match gold CSV entries to extracted page PDFs.
    
    Returns list of dicts with:
        - pdf_path: path to extracted page PDF
        - page_no: page number (always 1 for extracted single pages)
        - original_pdf: original PDF name
        - original_page: original page number
        - gold_text: ground truth text
    """
    logger.info(f"Matching entries to extracted pages in: {extracted_dir}")
    
    # Get all extracted page PDFs
    extracted_files = {f.name: f for f in extracted_dir.glob("*.pdf")}
    logger.info(f"Found {len(extracted_files)} extracted page PDFs")
    
    matched = []
    unmatched = []
    
    for pdf_name, page_no, gold_text in gold_entries:
        expected_name = normalize_filename(pdf_name, page_no)
        
        if expected_name in extracted_files:
            matched.append({
                'pdf_path': str(extracted_files[expected_name]),
                'page_no': 1,  # Extracted pages are single-page PDFs
                'original_pdf': pdf_name,
                'original_page': page_no,
                'gold_text': gold_text
            })
        else:
            unmatched.append((pdf_name, page_no, expected_name))
    
    logger.info(f"Matched: {len(matched)}, Unmatched: {len(unmatched)}")
    
    if unmatched and len(unmatched) <= 10:
        logger.warning("Unmatched entries:")
        for pdf, page, expected in unmatched[:10]:
            logger.warning(f"  {pdf} page {page} -> expected {expected}")
    
    return matched


def write_manifest(
    entries: List[Dict],
    output_path: Path,
    include_gold_text: bool = True
):
    """
    Write manifest TSV file.
    
    Format with gold text:
        pdf_path\tpage_no\toriginal_pdf\toriginal_page\tgold_text
    
    Format without gold text:
        pdf_path\tpage_no
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        
        if include_gold_text:
            writer.writerow(['pdf_path', 'page_no', 'original_pdf', 'original_page', 'gold_text'])
            for entry in entries:
                writer.writerow([
                    entry['pdf_path'],
                    entry['page_no'],
                    entry['original_pdf'],
                    entry['original_page'],
                    entry['gold_text']
                ])
        else:
            writer.writerow(['pdf_path', 'page_no'])
            for entry in entries:
                writer.writerow([entry['pdf_path'], entry['page_no']])
    
    logger.info(f"Manifest written to: {output_path}")
    logger.info(f"Total entries: {len(entries)}")


def main():
    parser = argparse.ArgumentParser(description='Build manifest from gold_pages.csv')
    parser.add_argument('--gold-csv', default='data/gold_data/gold_pages.csv',
                       help='Path to gold_pages.csv (default: data/gold_data/gold_pages.csv)')
    parser.add_argument('--extracted-dir', default='data/gold_pages_only',
                       help='Directory with extracted page PDFs (default: data/gold_pages_only)')
    parser.add_argument('--output', default='data/gold/manifest_gold.txt',
                       help='Output manifest path (default: data/gold/manifest_gold.txt)')
    parser.add_argument('--limit', type=int,
                       help='Limit number of pages (for testing)')
    parser.add_argument('--include-gold-text', action='store_true', default=True,
                       help='Include gold text in manifest (default: True)')
    parser.add_argument('--no-gold-text', dest='include_gold_text', action='store_false',
                       help='Exclude gold text from manifest')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("GOLD MANIFEST BUILDER")
    print("=" * 60)
    
    # Validate inputs
    gold_csv = Path(args.gold_csv)
    if not gold_csv.exists():
        logger.error(f"Gold CSV not found: {gold_csv}")
        sys.exit(1)
    
    extracted_dir = Path(args.extracted_dir)
    if not extracted_dir.exists():
        logger.error(f"Extracted directory not found: {extracted_dir}")
        sys.exit(1)
    
    # Load gold pages
    gold_entries = load_gold_pages_csv(gold_csv)
    
    if args.limit:
        gold_entries = gold_entries[:args.limit]
        logger.info(f"Limited to {len(gold_entries)} entries")
    
    # Match to extracted pages
    matched_entries = match_extracted_pages(gold_entries, extracted_dir)
    
    if not matched_entries:
        logger.error("No entries matched!")
        sys.exit(1)
    
    # Write manifest
    output_path = Path(args.output)
    write_manifest(matched_entries, output_path, include_gold_text=args.include_gold_text)
    
    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Gold CSV: {gold_csv}")
    print(f"Extracted pages dir: {extracted_dir}")
    print(f"Total gold entries: {len(gold_entries)}")
    print(f"Matched pages: {len(matched_entries)}")
    print(f"Match rate: {len(matched_entries)/len(gold_entries)*100:.1f}%")
    print(f"Manifest file: {output_path}")
    print(f"Gold text included: {args.include_gold_text}")
    
    # Show first few entries
    print(f"\nFirst 5 entries:")
    for i, entry in enumerate(matched_entries[:5], 1):
        pdf_name = Path(entry['pdf_path']).name
        print(f"  {i}. {pdf_name} (from {entry['original_pdf']} page {entry['original_page']})")
    
    print()
    print("✅ Manifest created successfully!")
    print()
    print("Next steps:")
    print(f"  1. Review manifest: head -20 {output_path}")
    print(f"  2. Run validation: python tools/run_gold_validation.py --manifest {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
