"""
Detect PDFs with bad OCR output that need to be reprocessed.

Patterns of bad output:
1. High ratio of special/garbage characters (ߋ, ݅, ఋ, Ü, etc.)
2. Excessive newlines or broken text flow
3. Very short average word length (garbage has short fragments)
4. High ratio of non-ASCII characters that aren't common diacritics
5. Presence of common OCR garbage patterns like repeating symbols
"""

import csv
import re
import sys
from collections import Counter
from pathlib import Path

def analyze_text_quality(text):
    """
    Analyze text to detect bad OCR output.
    Returns (is_bad, scores_dict)
    """
    if not text or len(text) < 50:
        return False, {}
    
    # Character analysis
    total_chars = len(text)
    
    # Count different character types
    garbage_chars = re.findall(r'[ߋ݅ఋÜÉՙڵڻŅԝԞԟԠȲȳȹ˻˼˽˾˿ÍÙæĭù]', text)
    garbage_ratio = len(garbage_chars) / total_chars if total_chars > 0 else 0
    
    # Count excessive non-ASCII that's not normal diacritics
    non_ascii = sum(1 for c in text if ord(c) > 127)
    non_ascii_ratio = non_ascii / total_chars if total_chars > 0 else 0
    
    # Count newlines (bad OCR often has many broken lines)
    newline_count = text.count('\n')
    newline_ratio = newline_count / total_chars if total_chars > 0 else 0
    
    # Analyze word structure
    words = text.split()
    if words:
        avg_word_length = sum(len(w) for w in words) / len(words)
        # Count "words" that are just garbage (< 2 chars or all non-alpha)
        garbage_words = sum(1 for w in words if len(w) < 2 or not any(c.isalpha() for c in w))
        garbage_word_ratio = garbage_words / len(words)
    else:
        avg_word_length = 0
        garbage_word_ratio = 1.0
    
    # Check for repeating garbage patterns
    repeating_patterns = len(re.findall(r'(.)\1{3,}', text))  # Same char 4+ times
    repeating_ratio = repeating_patterns / total_chars if total_chars > 0 else 0
    
    # Scoring system - flag as bad if any threshold exceeded
    scores = {
        'garbage_char_ratio': garbage_ratio,
        'non_ascii_ratio': non_ascii_ratio,
        'newline_ratio': newline_ratio,
        'avg_word_length': avg_word_length,
        'garbage_word_ratio': garbage_word_ratio,
        'repeating_ratio': repeating_ratio,
        'text_length': total_chars
    }
    
    # Determine if bad (multiple criteria)
    is_bad = (
        garbage_ratio > 0.05 or  # More than 5% garbage chars
        (non_ascii_ratio > 0.3 and garbage_word_ratio > 0.3) or  # High non-ASCII + garbage words
        (avg_word_length < 3 and total_chars > 500) or  # Very short words in long text
        garbage_word_ratio > 0.5 or  # More than 50% garbage words
        repeating_ratio > 0.02  # Excessive repeating chars
    )
    
    return is_bad, scores

def main():
    if len(sys.argv) != 3:
        print("Usage: python detect_bad_ocr_outputs.py <input_csv> <output_bad_pdfs_list>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    print(f"Analyzing {input_file}...")
    
    # Track PDFs and their quality scores
    pdf_scores = {}  # pdf_name -> list of (is_bad, scores) for each page
    
    with open(input_file, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            pdf_name = row['pdf_name']
            page_text = row.get('page_text', '')
            
            if pdf_name not in pdf_scores:
                pdf_scores[pdf_name] = []
            
            is_bad, scores = analyze_text_quality(page_text)
            pdf_scores[pdf_name].append((is_bad, scores))
    
    print(f"Analyzed {len(pdf_scores):,} unique PDFs")
    
    # Determine which PDFs are bad overall
    # A PDF is bad if more than 30% of its pages are flagged as bad
    bad_pdfs = []
    
    for pdf_name, page_analyses in pdf_scores.items():
        total_pages = len(page_analyses)
        bad_pages = sum(1 for is_bad, _ in page_analyses if is_bad)
        bad_ratio = bad_pages / total_pages if total_pages > 0 else 0
        
        if bad_ratio > 0.3:  # More than 30% bad pages
            # Get average scores across all pages (filter out empty scores)
            valid_scores = [s for _, s in page_analyses if s]
            if not valid_scores:
                continue
            
            avg_garbage_ratio = sum(s.get('garbage_char_ratio', 0) for s in valid_scores) / len(valid_scores)
            avg_non_ascii = sum(s.get('non_ascii_ratio', 0) for s in valid_scores) / len(valid_scores)
            avg_word_len = sum(s.get('avg_word_length', 0) for s in valid_scores) / len(valid_scores)
            
            bad_pdfs.append({
                'pdf_name': pdf_name,
                'total_pages': total_pages,
                'bad_pages': bad_pages,
                'bad_ratio': bad_ratio,
                'avg_garbage_ratio': avg_garbage_ratio,
                'avg_non_ascii_ratio': avg_non_ascii,
                'avg_word_length': avg_word_len
            })
    
    # Sort by bad ratio (worst first)
    bad_pdfs.sort(key=lambda x: x['bad_ratio'], reverse=True)
    
    print(f"\nFound {len(bad_pdfs):,} PDFs with bad OCR output")
    
    # Write results
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# PDFs with bad OCR output - need reprocessing\n")
        f.write("# Sorted by severity (worst first)\n\n")
        
        for pdf in bad_pdfs:
            f.write(f"{pdf['pdf_name']}\n")
    
    print(f"✓ Bad PDFs list written to: {output_file}")
    
    # Display top 20 worst
    print(f"\nTop 20 worst PDFs:")
    print(f"{'PDF Name':<50} {'Bad Pages':<12} {'Garbage %':<12} {'Non-ASCII %':<12} {'Avg Word Len':<12}")
    print("=" * 100)
    
    for pdf in bad_pdfs[:20]:
        print(f"{pdf['pdf_name']:<50} "
              f"{pdf['bad_pages']}/{pdf['total_pages']:<10} "
              f"{pdf['avg_garbage_ratio']*100:>6.2f}%     "
              f"{pdf['avg_non_ascii_ratio']*100:>6.2f}%     "
              f"{pdf['avg_word_length']:>6.2f}")
    
    # Summary statistics
    print(f"\n{'='*100}")
    print(f"Summary:")
    print(f"  Total PDFs analyzed: {len(pdf_scores):,}")
    print(f"  PDFs with bad output: {len(bad_pdfs):,} ({len(bad_pdfs)/len(pdf_scores)*100:.2f}%)")
    print(f"  Total pages to reprocess: {sum(p['total_pages'] for p in bad_pdfs):,}")
    
    return bad_pdfs

if __name__ == "__main__":
    main()
