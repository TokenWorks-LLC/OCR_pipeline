"""
Create a reprocessing manifest for PDFs with bad OCR output.

This will create a manifest with ONLY the affected PDFs, and we'll process them
with --force-ocr to ensure OCR runs instead of text extraction.
"""

import sys
from pathlib import Path

def main():
    if len(sys.argv) != 4:
        print("Usage: python create_reprocessing_manifest.py <bad_pdfs_list> <original_manifest> <output_manifest>")
        sys.exit(1)
    
    bad_pdfs_file = sys.argv[1]
    original_manifest = sys.argv[2]
    output_manifest = sys.argv[3]
    
    # Read list of bad PDFs
    print(f"Reading bad PDFs from {bad_pdfs_file}...")
    with open(bad_pdfs_file, 'r', encoding='utf-8') as f:
        bad_pdfs = set()
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                bad_pdfs.add(line)
    
    print(f"Found {len(bad_pdfs):,} bad PDFs")
    
    # Read original manifest and filter to bad PDFs
    print(f"Reading original manifest from {original_manifest}...")
    with open(original_manifest, 'r', encoding='utf-8') as f:
        header = f.readline()
        lines = f.readlines()
    
    print(f"Original manifest has {len(lines):,} entries")
    
    # Filter to bad PDFs only
    filtered_lines = []
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            pdf_path = parts[0]
            pdf_name = Path(pdf_path).name
            
            if pdf_name in bad_pdfs:
                filtered_lines.append(line)
    
    print(f"Filtered to {len(filtered_lines):,} entries for reprocessing")
    
    # Write output manifest
    with open(output_manifest, 'w', encoding='utf-8') as f:
        f.write(header)
        f.writelines(filtered_lines)
    
    print(f"✓ Reprocessing manifest written to: {output_manifest}")
    
    # Count unique PDFs
    unique_pdfs = set()
    total_pages = 0
    for line in filtered_lines:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            pdf_path = parts[0]
            pages = parts[1].split(',')
            unique_pdfs.add(Path(pdf_path).name)
            total_pages += len(pages)
    
    print(f"\nSummary:")
    print(f"  Unique PDFs: {len(unique_pdfs):,}")
    print(f"  Total pages: {total_pages:,}")
    print(f"  Average pages per PDF: {total_pages / len(unique_pdfs):.1f}")

if __name__ == "__main__":
    main()
