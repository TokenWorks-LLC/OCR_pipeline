"""
Aggregate page-text CSV files from multiple AWS Batch job outputs.

Usage:
    python tools/aggregate_page_text_csv.py "reports/page_text_*_shard_*/client_page_text.csv" output.csv

Combines all matching CSV files into a single UTF-8 BOM file.
"""
import sys
import csv
import glob
import os
import codecs

# usage: python tools/aggregate_page_text_csv.py reports/page_text_*_shard_*/client_page_text.csv out.csv
inputs = sorted(glob.glob(sys.argv[1]))
out = sys.argv[2]
wrote_hdr = False

with open(out, "w", newline="", encoding="utf-8-sig") as fo:
    w = None
    for f in inputs:
        with open(f, newline="", encoding="utf-8-sig") as fi:
            r = csv.DictReader(fi)
            if not wrote_hdr:
                w = csv.DictWriter(fo, fieldnames=r.fieldnames)
                w.writeheader()
                wrote_hdr = True
            for row in r:
                w.writerow(row)

print("combined rows from", len(inputs), "files ->", out)
