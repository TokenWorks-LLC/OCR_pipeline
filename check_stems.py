import csv
import os
import sys

pdf_text_file = 'reports/real_gold_eval_runs/smoke_50/run/client_page_text.csv'
metrics_file = 'reports/source_render_fix_per_page_metrics.csv'

by_pdf_stem = {}
try:
    with open(pdf_text_file, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        # Debug: print fieldnames
        # print(f"Fieldnames: {reader.fieldnames}")
        for row in reader:
            pdf_name = row.get('pdf_name')
            if pdf_name:
                # Use split('.') instead of splitext to handle multiple dots if any, 
                # but splitext is usually correct for .pdf
                stem = pdf_name.replace('.pdf', '')
                by_pdf_stem[stem] = row.get('page_text', '')
except Exception as e:
    print(f"Error reading {pdf_text_file}: {e}")
    sys.exit(1)

def check_stem(stem):
    exists = stem in by_pdf_stem
    length = len(by_pdf_stem[stem]) if exists else -1
    print(f"Stem: {stem} | Exists: {exists} | Length: {length}")

print("Checking 'cord_train_00055_page_1':")
check_stem('cord_train_00055_page_1')

print("\nChecking first 5 rows from metrics file:")
try:
    with open(metrics_file, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            if count >= 5:
                break
            stem = row.get('page_id')
            if stem:
                check_stem(stem)
                count += 1
except Exception as e:
    print(f"Error reading {metrics_file}: {e}")

# Also check for a stem that we saw in the head output
print("\nChecking 'cord_test_00012_page_1':")
check_stem('cord_test_00012_page_1')
