#!/usr/bin/env python3
"""
Build manifest for gold test set.
Scans PDF directory and creates TSV manifest with pdf_path and page numbers.
"""

import sys
import csv
import argparse
import random
from pathlib import Path
from typing import List, Tuple, Optional
import logging

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_pdf_page_count(pdf_path: Path) -> int:
    """Get number of pages in PDF."""
    if not PYMUPDF_AVAILABLE:
        logger.warning(f"PyMuPDF not available, assuming 10 pages for {pdf_path.name}")
        return 10
    
    try:
        doc = fitz.open(pdf_path)
        page_count = doc.page_count
        doc.close()
        return page_count
    except Exception as e:
        logger.error(f"Error reading {pdf_path.name}: {e}")
        return 0


def scan_pdfs(pdf_dir: Path, limit: Optional[int] = None, sample_pages: int = 0, scan_all_pages: bool = False, recursive: bool = True) -> List[Tuple[str, int]]:
    """
    Scan PDF directory and return list of (pdf_path, page_no) tuples.
    
    Args:
        pdf_dir: Directory containing PDFs
        limit: Limit number of PDFs to process
        sample_pages: If > 0, only include this many pages per PDF (evenly distributed)
        scan_all_pages: If True, generate one entry per page (overrides sample_pages)
        recursive: If True, scan subdirectories recursively
    
    Returns:
        List of (pdf_relative_path, page_no) tuples
    """
    logger.info(f"Scanning directory: {pdf_dir} (recursive={recursive})")
    
    # Find all PDFs (recursive or not)
    if recursive:
        pdf_files = sorted(list(pdf_dir.rglob("*.pdf")) + list(pdf_dir.rglob("*.PDF")))
    else:
        pdf_files = sorted(list(pdf_dir.glob("*.pdf")) + list(pdf_dir.glob("*.PDF")))
    
    if limit:
        pdf_files = pdf_files[:limit]
    
    logger.info(f"Found {len(pdf_files)} PDF files")
    
    manifest_entries = []
    
    for i, pdf_path in enumerate(pdf_files, 1):
        logger.info(f"[{i}/{len(pdf_files)}] Processing: {pdf_path.name}")
        
        page_count = get_pdf_page_count(pdf_path)
        
        if page_count == 0:
            logger.warning(f"Skipping {pdf_path.name} (0 pages)")
            continue
        
        # Determine which pages to include
        if scan_all_pages:
            # Full page-by-page manifest
            pages = list(range(1, page_count + 1))
        elif sample_pages > 0 and sample_pages < page_count:
            # Sample evenly distributed pages
            step = max(1, page_count // sample_pages)
            pages = list(range(1, page_count + 1, step))[:sample_pages]
        else:
            # Include all pages (default behavior)
            pages = list(range(1, page_count + 1))
        
        # Add entries
        for page_no in pages:
            manifest_entries.append((str(pdf_path), page_no))
    
    logger.info(f"Generated {len(manifest_entries)} manifest entries")
    
    return manifest_entries


def write_manifest_tsv(
    entries: List[Tuple[str, int]],
    output_path: Path,
    include_gold_text: bool = False
):
    """
    Write manifest to TSV file.
    
    Format: pdf_path\tpage_no\t[gold_text]
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        
        # Write header
        if include_gold_text:
            writer.writerow(['pdf_path', 'page_no', 'gold_text'])
        else:
            writer.writerow(['pdf_path', 'page_no'])
        
        # Write entries
        for pdf_path, page_no in entries:
            if include_gold_text:
                writer.writerow([pdf_path, page_no, ''])  # Empty gold_text placeholder
            else:
                writer.writerow([pdf_path, page_no])
    
    logger.info(f"Manifest written to: {output_path}")
    logger.info(f"Total entries: {len(entries)}")


def expand_page_ranges(range_str: str) -> List[int]:
    """
    Expand page range string to list of page numbers.
    Examples: "3,7-9,12" -> [3, 7, 8, 9, 12]
    """
    pages = []
    for part in range_str.split(','):
        part = part.strip()
        if '-' in part:
            start, end = part.split('-')
            pages.extend(range(int(start), int(end) + 1))
        else:
            pages.append(int(part))
    return sorted(set(pages))


def load_csv_manifest(csv_path: Path, pdf_root: Path) -> List[Tuple[str, int, str]]:
    """
    Load gold_pages.csv and return list of (pdf_path, page_no, gold_text) tuples.
    
    CSV columns: pdf_name, gold_pages, gold_data
    """
    entries = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pdf_name = row.get('pdf_name', row.get('PDF LINK', '')).strip()
            gold_pages = row.get('gold_pages', row.get('PAGE', '')).strip()
            gold_data = row.get('gold_data', row.get('HANDTYPED', '')).strip()
            
            if not pdf_name or not gold_pages:
                continue
            
            # Expand page ranges
            pages = expand_page_ranges(gold_pages)
            
            # Build full path
            pdf_path = pdf_root / pdf_name
            
            for page_no in pages:
                entries.append((str(pdf_path), page_no, gold_data))
    
    return entries


def main():
    parser = argparse.ArgumentParser(description='Build manifest for gold test set')
    parser.add_argument('--csv', type=str,
                       help='CSV file with columns: pdf_name, gold_pages, gold_data')
    parser.add_argument('--pdf-root', '--pdf-dir', dest='pdf_dir', default='data/input_pdfs',
                       help='Directory containing PDFs (default: data/input_pdfs)')
    parser.add_argument('--inputs', type=str,
                       help='Alias for --pdf-dir (for Drive path scanning)')
    parser.add_argument('--out', '--output', dest='output', default='data/gold/manifest_30.txt',
                       help='Output manifest path (default: data/gold/manifest_30.txt)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of PDFs')
    parser.add_argument('--sample-pages', type=int, default=0,
                       help='Sample N pages per PDF (0=all pages)')
    parser.add_argument('--scan-pages', action='store_true',
                       help='Generate one entry per page (full page-by-page manifest)')
    parser.add_argument('--include-gold-text', action='store_true',
                       help='Add gold_text column (empty placeholder)')
    parser.add_argument('--expand-ranges', action='store_true',
                       help='Expand page ranges like 7-9 to 7,8,9')
    parser.add_argument('--dedupe', action='store_true',
                       help='Remove duplicate page entries')
    parser.add_argument('--shuffle', action='store_true',
                       help='Shuffle manifest entries to avoid alphabetical bias')
    
    args = parser.parse_args()
    
    # Handle --inputs alias
    if args.inputs:
        args.pdf_dir = args.inputs
    
    print("=" * 60)
    print("MANIFEST BUILDER")
    print("=" * 60)
    
    # Validate PDF directory
    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.exists():
        logger.error(f"PDF directory not found: {pdf_dir}")
        sys.exit(1)
    
    # Check if CSV mode
    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            logger.error(f"CSV file not found: {csv_path}")
            sys.exit(1)
        
        logger.info(f"Loading CSV: {csv_path}")
        logger.info(f"PDF root: {pdf_dir}")
        
        # Load from CSV
        csv_entries = load_csv_manifest(csv_path, pdf_dir)
        
        # Dedupe if requested
        if args.dedupe:
            unique = {}
            for pdf_path, page_no, gold_text in csv_entries:
                key = (pdf_path, page_no)
                if key not in unique:
                    unique[key] = gold_text
            csv_entries = [(k[0], k[1], v) for k, v in unique.items()]
            logger.info(f"After deduplication: {len(csv_entries)} entries")
        
        # Shuffle if requested
        if args.shuffle:
            random.shuffle(csv_entries)
            logger.info(f"Shuffled {len(csv_entries)} entries to avoid alphabetical bias")
        
        # Check which PDFs exist
        found = 0
        missing = []
        for pdf_path, _, _ in csv_entries:
            if Path(pdf_path).exists():
                found += 1
            else:
                if pdf_path not in missing:
                    missing.append(pdf_path)
        
        logger.info(f"PDFs found: {found}/{len(set(p for p, _, _ in csv_entries))}")
        if missing:
            logger.warning(f"Missing PDFs: {len(missing)}")
            for m in missing[:5]:
                logger.warning(f"  - {Path(m).name}")
        
        # Write manifest with gold_text
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['pdf_path', 'page_no', 'gold_text'])
            
            for pdf_path, page_no, gold_text in csv_entries:
                writer.writerow([pdf_path, page_no, gold_text])
        
        logger.info(f"Manifest written to: {output_path}")
        logger.info(f"Total entries: {len(csv_entries)}")
        
        # Summary
        print()
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"CSV source: {csv_path}")
        print(f"PDF root: {pdf_dir}")
        print(f"PDFs referenced: {len(set(p for p, _, _ in csv_entries))}")
        print(f"PDFs found: {found}")
        print(f"Total pages: {len(csv_entries)}")
        print(f"Manifest file: {output_path}")
        
        print(f"\nFirst 5 entries:")
        for i, (pdf_path, page_no, _) in enumerate(csv_entries[:5], 1):
            pdf_name = Path(pdf_path).name
            print(f"  {i}. {pdf_name} / page {page_no}")
        
        print()
        print("✅ Manifest created successfully from CSV!")
        
        return 0
    
    # Original scan mode
    # Scan PDFs
    entries = scan_pdfs(
        pdf_dir, 
        limit=args.limit, 
        sample_pages=args.sample_pages,
        scan_all_pages=args.scan_pages,
        recursive=True
    )
    
    if not entries:
        logger.error("No PDF entries generated!")
        sys.exit(1)
    
    # Dedupe if requested (for scan mode)
    if args.dedupe:
        unique = {}
        for pdf_path, page_no in entries:
            key = (pdf_path, page_no)
            if key not in unique:
                unique[key] = True
        entries = list(unique.keys())
        logger.info(f"After deduplication: {len(entries)} entries")
    
    # Shuffle if requested
    if args.shuffle:
        random.shuffle(entries)
        logger.info(f"Shuffled {len(entries)} entries to avoid alphabetical bias")
    
    # Write manifest
    output_path = Path(args.output)
    write_manifest_tsv(entries, output_path, include_gold_text=args.include_gold_text)
    
    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"PDF directory: {pdf_dir}")
    print(f"PDFs processed: {len(set(pdf for pdf, _ in entries))}")
    print(f"Total pages: {len(entries)}")
    print(f"Manifest file: {output_path}")
    
    # Show first few entries
    print(f"\nFirst 5 entries:")
    for i, (pdf_path, page_no) in enumerate(entries[:5], 1):
        pdf_name = Path(pdf_path).name
        print(f"  {i}. {pdf_name} / page {page_no}")
    
    print()
    print("✅ Manifest created successfully!")
    print()
    print("Next steps:")
    print("  1. Review manifest: cat", str(output_path))
    print("  2. Run pipeline: python tools/run_gold_validation.py --manifest", str(output_path))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
