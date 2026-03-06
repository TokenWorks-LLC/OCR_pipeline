"""
Advanced reading order utilities for multi-column layouts (triage mode).
"""

import numpy as np


def xy_cut_lines_triage(lines, max_columns=3):
    """
    Use XY-cut algorithm with K-means clustering to handle multi-column layouts.
    
    Args:
        lines: List of text line objects with .bbox attributes
        max_columns: Maximum number of columns to detect
        
    Returns:
        Lines sorted in proper reading order (left-to-right, top-to-bottom per column)
    """
    if len(lines) < 12:
        # For simple layouts, use basic top-to-bottom, left-to-right sort
        return sorted(lines, key=lambda l: (l.bbox.y0 if hasattr(l.bbox, 'y0') else l.bbox.y, 
                                            l.bbox.x0 if hasattr(l.bbox, 'x0') else l.bbox.x))
    
    # Extract x-coordinates of line centers
    xs = []
    for l in lines:
        if hasattr(l.bbox, 'x0') and hasattr(l.bbox, 'x1'):
            xs.append((l.bbox.x0 + l.bbox.x1) / 2.0)
        else:
            x = getattr(l.bbox, 'x', 0)
            width = getattr(l.bbox, 'width', 0)
            xs.append(x + width / 2.0)
    
    xs = np.array(xs).reshape(-1, 1)
    
    # Cluster lines into columns
    try:
        from sklearn.cluster import KMeans
        k = min(max_columns, 1 + (len(lines) // 30))
        km = KMeans(n_clusters=k, n_init="auto", random_state=17).fit(xs)
        
        # Group lines by column
        groups = {i: [] for i in range(k)}
        for l, c in zip(lines, km.labels_):
            groups[c].append(l)
        
        # Sort columns left-to-right by median x-position
        cols = sorted(
            groups.values(), 
            key=lambda col: np.median([(l.bbox.x0 + l.bbox.x1) / 2 
                                       if hasattr(l.bbox, 'x0') 
                                       else getattr(l.bbox, 'x', 0) + getattr(l.bbox, 'width', 0) / 2
                                       for l in col])
        )
        
        # Sort lines within each column top-to-bottom
        out = []
        for col in cols:
            sorted_col = sorted(
                col, 
                key=lambda l: (l.bbox.y0 if hasattr(l.bbox, 'y0') else l.bbox.y, 
                              l.bbox.x0 if hasattr(l.bbox, 'x0') else l.bbox.x)
            )
            out.extend(sorted_col)
        
        return out
    
    except ImportError:
        # Fallback if sklearn not available
        return sorted(lines, key=lambda l: (l.bbox.y0 if hasattr(l.bbox, 'y0') else l.bbox.y, 
                                            l.bbox.x0 if hasattr(l.bbox, 'x0') else l.bbox.x))


def projection_reading_order_triage(lines):
    """
    Projection-based reading order for dense single-column text.
    
    Groups lines into horizontal bands and sorts left-to-right within each band.
    
    Args:
        lines: List of text line objects with .bbox attributes
        
    Returns:
        Lines sorted in reading order
    """
    bands = {}
    
    for L in lines:
        # Get y-coordinate
        if hasattr(L.bbox, 'y0') and hasattr(L.bbox, 'y1'):
            y = int((L.bbox.y0 + L.bbox.y1) / 2)
        else:
            y = getattr(L.bbox, 'y', 0)
            height = getattr(L.bbox, 'height', 0)
            y = int(y + height / 2)
        
        # Assign to 25px bands
        band = (y // 25)
        bands.setdefault(band, []).append(L)
    
    # Sort bands top-to-bottom, lines left-to-right
    out = []
    for k in sorted(bands.keys()):
        sorted_band = sorted(
            bands[k], 
            key=lambda l: l.bbox.x0 if hasattr(l.bbox, 'x0') else l.bbox.x
        )
        out.extend(sorted_band)
    
    return out


def apply_reading_order_triage(lines, page_width, cfg):
    """
    Apply configured reading order strategy for triage mode.
    
    Args:
        lines: List of text line objects
        page_width: Page width in pixels
        cfg: Reading order configuration
        
    Returns:
        Lines sorted in reading order
    """
    if not cfg.get("xy_cut", True):
        # Default: simple top-to-bottom, left-to-right
        return sorted(lines, key=lambda l: (l.bbox.y0 if hasattr(l.bbox, 'y0') else l.bbox.y, 
                                            l.bbox.x0 if hasattr(l.bbox, 'x0') else l.bbox.x))
    
    # Try XY-cut first
    max_cols = cfg.get("max_columns", 3)
    ordered = xy_cut_lines_triage(lines, max_columns=max_cols)
    
    # Check if we have a single narrow column (fallback to projection)
    if cfg.get("projection_fallback", True) and len(ordered) > 0:
        widths = []
        for l in ordered:
            if hasattr(l.bbox, 'x0') and hasattr(l.bbox, 'x1'):
                widths.append(l.bbox.x1 - l.bbox.x0)
            else:
                widths.append(getattr(l.bbox, 'width', 0))
        
        median_width = np.median(widths) if widths else 0
        if median_width < 0.35 * page_width:
            # Single narrow column detected, use projection instead
            ordered = projection_reading_order_triage(lines)
    
    return ordered
