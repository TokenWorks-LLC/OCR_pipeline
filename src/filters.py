"""
Text filtering utilities for removing headers, footers, line numbers, and apparatus.
"""

import re


# Patterns for detecting non-content lines
HEADER_RX = re.compile(
    r"^\s*(AKT|AfO|EA|JNES|JCS|Or|ZDMG|[A-Z]{2,})(?:\b|[IVXLC]+|\d+)", 
    re.I
)
LINENO_RX = re.compile(r"^\s*\d{1,3}[\.\)]?\s*$")
ROMAN_RX = re.compile(r"^\s*[IVXLC]{1,6}\s*$")


def filter_lines(lines, page_h, cfg):
    """
    Filter out non-content lines (headers, footers, line numbers, apparatus).
    
    Args:
        lines: List of text line objects with .text and .bbox attributes
        page_h: Page height in pixels
        cfg: Configuration dict with filtering parameters
        
    Returns:
        Filtered list of lines containing only main content
    """
    out = []
    
    # Calculate header/footer bands
    hb = int(cfg.get("header_band_pct", 0.06) * page_h)
    fb = int(cfg.get("footer_band_pct", 0.08) * page_h)
    
    for L in lines:
        # Get line position
        y0 = getattr(L.bbox, 'y0', getattr(L.bbox, 'y', 0))
        y1 = getattr(L.bbox, 'y1', y0 + getattr(L.bbox, 'height', 0))
        txt = L.text.strip()
        
        # Remove headers and footers based on position
        if cfg.get("remove_headers_footers", True):
            if y1 <= hb or y0 >= (page_h - fb):
                continue
        
        # Remove line numbers (e.g., "1.", "25)", "108")
        if cfg.get("remove_line_numbers", True):
            if LINENO_RX.match(txt) or ROMAN_RX.match(txt):
                continue
        
        # Remove journal headers (e.g., "AKT IV", "AfO Bh. 10")
        if HEADER_RX.match(txt):
            continue
        
        # Remove very short lines (likely noise)
        min_len = cfg.get("min_line_len", 5)
        if len(txt) < min_len:
            continue
        
        # Remove lines with too many digits/punctuation (apparatus, references)
        punct = sum(c in "[](){}:;.,*/\\" for c in txt)
        digits = sum(c.isdigit() for c in txt)
        ratio = (punct + digits) / max(1, len(txt))
        
        max_ratio = cfg.get("digit_punct_ratio_max", 0.6)
        if ratio > max_ratio:
            continue
        
        # Remove lines with excessive brackets (critical apparatus)
        bracket = sum(c in "[]⟦⟧" for c in txt) / max(1, len(txt))
        max_bracket = cfg.get("bracket_density_max", 0.35)
        
        if bracket > max_bracket:
            continue
        
        out.append(L)
    
    return out
