"""
Detect pages with broken cuneiform transliterations.

These pages often have:
- Broken diacritics: ܒ, ì, á, š, ú instead of proper š, ṭ, ā, etc.
- Broken brackets: >@ instead of proper transliteration notation
- Mathematical/symbol chars mixed with text: Ɛ, ␦, ݅, ఋ
- Very high density of special Unicode chars
"""

import csv
import re
import sys
from collections import Counter

def has_broken_cuneiform_transliteration(text):
    """
    Detect if page has broken cuneiform transliteration characters.
    """
    if not text or len(text) < 50:
        return False, {}
    
    # Common broken cuneiform chars (Syriac, Telugu, etc. getting mixed in)
    broken_chars = r'[ܒܐܕܓܘܙܚܛܝܟܠܡܢܣܥܦܨܩܪܫܬݍݎݏݐݑݒݓݔݕݖݗݘݙݚݛݜݝݞݟ]'
    broken_count = len(re.findall(broken_chars, text))
    
    # Mathematical operators misused in transliteration
    math_ops = r'[ߋఋÍݍݎݏݐݑ␦Ɛ]'
    math_count = len(re.findall(math_ops, text))
    
    # Broken bracket patterns like >@ or @> common in bad OCR of transliteration
    broken_brackets = len(re.findall(r'[>@]\w*[>@]', text))
    
    # Look for patterns like mì-ig>ra@ (broken Akkadian transliteration)
    broken_transliteration = len(re.findall(r'\w+-\w+>\w*@', text))
    
    # Very high Unicode range chars (telltale sign)
    very_high_unicode = sum(1 for c in text if ord(c) > 1000)
    
    total_chars = len(text)
    scores = {
        'broken_char_count': broken_count,
        'math_char_count': math_count,
        'broken_brackets': broken_brackets,
        'broken_transliteration': broken_transliteration,
        'very_high_unicode': very_high_unicode,
        'broken_char_ratio': broken_count / total_chars,
        'text_length': total_chars
    }
    
    # Flag as bad if significant presence of broken chars
    is_broken = (
        broken_count > 5 or  # Any Syriac/broken chars
        math_count > 5 or    # Mathematical symbols
        broken_transliteration > 2 or  # Broken transliteration patterns
        very_high_unicode > 10  # High Unicode chars
    )
    
    return is_broken, scores

def main():
    if len(sys.argv) != 3:
        print("Usage: python detect_broken_cuneiform_pages.py <input_csv> <output_list>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    print(f"Analyzing {input_file} for broken cuneiform transliterations...")
    
    broken_items = []  # List of (pdf_name, page, scores)
    pdf_broken_counts = {}  # pdf_name -> count of broken pages
    
    with open(input_file, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        
        row_count = 0
        for row in reader:
            row_count += 1
            pdf_name = row['pdf_name']
            page = row['page']
            page_text = row.get('page_text', '')
            
            is_broken, scores = has_broken_cuneiform_transliteration(page_text)
            
            if is_broken:
                broken_items.append({
                    'pdf_name': pdf_name,
                    'page': page,
                    **scores
                })
                
                if pdf_name not in pdf_broken_counts:
                    pdf_broken_counts[pdf_name] = 0
                pdf_broken_counts[pdf_name] += 1
    
    print(f"Analyzed {row_count:,} pages")
    print(f"Found {len(broken_items):,} pages with broken cuneiform transliterations")
    print(f"Affected PDFs: {len(pdf_broken_counts):,}")
    
    # Sort PDFs by number of broken pages
    sorted_pdfs = sorted(pdf_broken_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Write output
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# PDFs with broken cuneiform transliteration pages\n")
        f.write("# Format: pdf_name (broken_pages)\n\n")
        
        for pdf_name, count in sorted_pdfs:
            f.write(f"{pdf_name}\n")
    
    print(f"✓ PDF list written to: {output_file}")
    
    # Display top 30
    print(f"\nTop 30 PDFs with broken cuneiform pages:")
    print(f"{'PDF Name':<70} {'Broken Pages':<15}")
    print("=" * 85)
    
    for pdf_name, count in sorted_pdfs[:30]:
        print(f"{pdf_name:<70} {count:>5}")
    
    # Show some example broken pages
    print(f"\n{'='*85}")
    print(f"Sample broken pages (first 10):")
    print(f"{'PDF':<40} {'Page':<8} {'Broken Chars':<15} {'High Unicode':<15}")
    print("=" * 85)
    
    for item in broken_items[:10]:
        print(f"{item['pdf_name']:<40} "
              f"{item['page']:<8} "
              f"{item['broken_char_count']:>8}       "
              f"{item['very_high_unicode']:>8}")
    
    return sorted_pdfs

if __name__ == "__main__":
    main()
