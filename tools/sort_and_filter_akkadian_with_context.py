"""
Sort CSV by PDF name and page number, then create filtered version with Akkadian pages + context.

This script:
1. Sorts the full CSV by pdf_name (alphabetical) and page (numerical)
2. Creates a filtered CSV with Akkadian pages plus the page before and after each
   to capture translations that may be offset by ±1 page
"""

import csv
import sys
from pathlib import Path

def natural_sort_key(page_str):
    """Convert page number to integer for proper numerical sorting"""
    try:
        return int(page_str)
    except (ValueError, TypeError):
        return 0

def main():
    if len(sys.argv) != 4:
        print("Usage: python sort_and_filter_akkadian_with_context.py <input_csv> <sorted_output> <akkadian_context_output>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    sorted_output = sys.argv[2]
    akkadian_context_output = sys.argv[3]
    
    print(f"Reading {input_file}...")
    
    # Read all rows
    with open(input_file, 'r', encoding='utf-8-sig', newline='') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        rows = list(reader)
    
    print(f"Read {len(rows):,} rows")
    
    # Sort by pdf_name (alphabetical), then page (numerical)
    print("Sorting by PDF name (alphabetical) and page number (numerical)...")
    rows.sort(key=lambda r: (r['pdf_name'], natural_sort_key(r['page'])))
    
    # Write sorted full CSV
    print(f"Writing sorted CSV to {sorted_output}...")
    with open(sorted_output, 'w', encoding='utf-8-sig', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"✓ Sorted CSV written: {len(rows):,} rows")
    
    # Find all Akkadian pages and build set of pages to include (Akkadian + context)
    print("\nFinding Akkadian pages and building context set...")
    
    akkadian_indices = set()
    for i, row in enumerate(rows):
        if row.get('has_akkadian', '').lower() == 'true':
            akkadian_indices.add(i)
    
    print(f"Found {len(akkadian_indices):,} Akkadian pages")
    
    # Expand to include page before and after each Akkadian page
    pages_to_include = set()
    for idx in akkadian_indices:
        # Add the page before (if exists)
        if idx > 0:
            pages_to_include.add(idx - 1)
        # Add the Akkadian page itself
        pages_to_include.add(idx)
        # Add the page after (if exists)
        if idx < len(rows) - 1:
            pages_to_include.add(idx + 1)
    
    print(f"Expanded to {len(pages_to_include):,} pages (Akkadian + context)")
    
    # Extract rows to include
    filtered_rows = [rows[i] for i in sorted(pages_to_include)]
    
    # Write filtered CSV
    print(f"Writing Akkadian + context CSV to {akkadian_context_output}...")
    with open(akkadian_context_output, 'w', encoding='utf-8-sig', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered_rows)
    
    print(f"✓ Akkadian + context CSV written: {len(filtered_rows):,} rows")
    
    # Statistics
    akkadian_count = sum(1 for row in filtered_rows if row.get('has_akkadian', '').lower() == 'true')
    context_count = len(filtered_rows) - akkadian_count
    
    print(f"\nStatistics:")
    print(f"  Total pages in filtered CSV: {len(filtered_rows):,}")
    print(f"  Akkadian pages: {akkadian_count:,}")
    print(f"  Context pages (before/after): {context_count:,}")
    print(f"  Ratio: {len(filtered_rows) / len(rows) * 100:.2f}% of original dataset")

if __name__ == "__main__":
    main()
