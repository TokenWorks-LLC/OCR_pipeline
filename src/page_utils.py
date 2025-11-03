"""
Utilities for parsing page specifications and handling page ranges.
"""

import re
from typing import List


def parse_page_spec(spec: str) -> List[int]:
    """
    Parse page specification string into list of page numbers.
    
    Handles:
    - Single pages: "5" → [5]
    - Ranges: "5-6" → [5, 6]
    - En dash ranges: "5–6" → [5, 6]
    - Em dash ranges: "5—6" → [5, 6]
    
    Args:
        spec: Page specification string
        
    Returns:
        List of page numbers
    """
    s = str(spec).strip().replace("–", "-").replace("—", "-")
    
    # Try to match range pattern
    m = re.match(r"^(\d+)\s*-\s*(\d+)$", s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return list(range(min(a, b), max(a, b) + 1))
    
    # Single page
    try:
        return [int(s)]
    except ValueError:
        raise ValueError(f"Could not parse page specification: {spec}")


def try_adjacent_pages(pdf_path, page_idxs, gold_text, ocr_func, compute_metrics_func):
    """
    Try adjacent pages to find best match when initial CER is very high.
    
    This handles cases where the gold standard references a page range
    but the text actually starts on an adjacent page.
    
    Args:
        pdf_path: Path to PDF file
        page_idxs: Initial list of page indices
        gold_text: Gold standard text
        ocr_func: Function to run OCR on pages: (pdf_path, pages) -> (text, meta)
        compute_metrics_func: Function to compute metrics: (gold, ocr) -> (cer, wer)
        
    Returns:
        Tuple of (cer, wer, delta, text, meta) for best match
        delta is -1 (previous), 0 (original), or +1 (next)
    """
    cands = []
    
    for delta in [0, -1, +1]:
        if delta == 0:
            pages = page_idxs
        else:
            pages = [p + delta for p in page_idxs]
        
        # Skip if pages go out of bounds
        if min(pages) < 1:
            continue
        
        try:
            text, meta = ocr_func(pdf_path, pages)
            cer, wer = compute_metrics_func(gold_text, text)
            cands.append((cer, wer, delta, text, meta))
        except Exception as e:
            # Skip this candidate if OCR fails
            continue
    
    if not cands:
        # No valid candidates, return defaults
        return (999.0, 999.0, 0, "", {})
    
    # Return candidate with lowest CER, then lowest WER
    return sorted(cands, key=lambda x: (x[0], x[1]))[0]
