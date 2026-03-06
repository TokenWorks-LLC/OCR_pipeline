"""
Filter page-text CSV to keep only pages with Akkadian text.

Usage:
    python tools/filter_akkadian_only.py input.csv output.csv

Reads the input CSV and writes a new CSV containing only rows where has_akkadian=true
"""
import sys
import csv

if len(sys.argv) != 3:
    print("Usage: python tools/filter_akkadian_only.py input.csv output.csv")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

total_rows = 0
akkadian_rows = 0

with open(input_file, 'r', encoding='utf-8-sig', newline='') as infile:
    reader = csv.DictReader(infile)
    
    with open(output_file, 'w', encoding='utf-8-sig', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
        writer.writeheader()
        
        for row in reader:
            total_rows += 1
            if row.get('has_akkadian', '').lower() == 'true':
                writer.writerow(row)
                akkadian_rows += 1

print(f"Filtered {total_rows:,} total rows")
print(f"Kept {akkadian_rows:,} rows with Akkadian ({akkadian_rows/total_rows*100:.2f}%)")
print(f"Removed {total_rows - akkadian_rows:,} rows without Akkadian")
print(f"Output written to: {output_file}")
