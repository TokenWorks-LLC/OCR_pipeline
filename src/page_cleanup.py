"""
Page cleanup utilities for margin trimming and rule-line removal.
"""

import cv2
import numpy as np


def trim_margins(img, pad_px=10):
    """
    Trim white margins from the image while preserving content.
    
    Args:
        img: Input image (BGR)
        pad_px: Padding to add around detected content
        
    Returns:
        Trimmed image
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    inv = 255 - thr
    
    ys, xs = np.where(inv > 0)
    if len(xs) == 0 or len(ys) == 0:
        return img
    
    x0 = max(xs.min() - pad_px, 0)
    x1 = min(xs.max() + pad_px, w - 1)
    y0 = max(ys.min() - pad_px, 0)
    y1 = min(ys.max() + pad_px, h - 1)
    
    return img[y0:y1+1, x0:x1+1]


def remove_rulings(img):
    """
    Remove horizontal and vertical rule lines from the image.
    
    This helps with academic documents that have tables, apparatus markers,
    or column separators that interfere with text recognition.
    
    Args:
        img: Input image (BGR)
        
    Returns:
        Image with rule lines removed
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binv = 255 - cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # Detect horizontal lines (40px wide, 1px tall)
    horiz = cv2.morphologyEx(
        binv, 
        cv2.MORPH_OPEN, 
        cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    )
    
    # Detect vertical lines (1px wide, 40px tall)
    vert = cv2.morphologyEx(
        binv, 
        cv2.MORPH_OPEN, 
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    )
    
    # Remove detected lines
    cleaned = np.clip(binv - np.maximum(horiz, vert), 0, 255).astype(np.uint8)
    merged = 255 - np.minimum(255 - gray, cleaned)
    
    return cv2.cvtColor(merged, cv2.COLOR_GRAY2BGR)
