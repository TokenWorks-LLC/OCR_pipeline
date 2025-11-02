"""
Advanced reading order detection with multi-column support.

Key Improvements in this version:
- Robust Guards: Uses `max_columns` and `min_elements_per_column` to
  prevent layout regressions from noise.
- Footnote/Noise Isolation: Automatically separates "noise" elements
  (footnotes, page numbers) from main content columns.
- Translation Pairing: Includes `pair_translations` method to satisfy
  Week 2/3 deliverables.
- Efficiency: Replaced slow N^2 result re-assembly with a
  high-performance, clean implementation.
- Config-Driven: Pulls parameters from `config.py` (assumes `LAYOUT` section).
"""
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import KMeans
from dataclasses import dataclass

# NEW: Import config to drive parameters
from config import LAYOUT

logger = logging.getLogger(__name__)

@dataclass
class TextElement:
    """
    Text element with position and content.
    NEW: Now stores `original_data` for fast, lossless re-assembly.
    """
    text: str
    x: int
    y: int
    width: int
    height: int
    conf: float
    bbox: Tuple[int, int, int, int]
    original_data: dict  # Store the original dict

class ReadingOrderDetector:
    """Advanced reading order detection with multi-column layout support."""
    
    def __init__(self, 
                 line_height_threshold: float = None,
                 min_column_width: int = None,
                 max_columns: int = None,
                 min_elements_per_column: int = None):
        """
        Initialize reading order detector, pulling defaults from config.
        """
        # --- "TIGHTENED GUARDS" ---
        self.max_columns = int(max_columns or LAYOUT.get('max_columns', 3))
        self.min_elements_per_column = int(min_elements_per_column or LAYOUT.get('min_elements_per_column', 3))
        # ---
        
        self.line_height_threshold = float(line_height_threshold or LAYOUT.get('line_height_threshold', 1.5))
        self.min_column_width = int(min_column_width or LAYOUT.get('min_column_width', 100))
    
    def detect_columns(self, text_elements: List[TextElement], page_width: int) -> Tuple[List[List[TextElement]], List[TextElement]]:
        """
        Detect column layout using k-means clustering on x-coordinates.
        
        IMPROVEMENT: This is now a "guarded" detection.
        1. It defaults to 1 column unless a multi-column layout is
           *significantly* better and passes all checks.
        2. It separates "noise" (footnotes, etc.) from main columns.
        
        Returns:
            Tuple of (main_columns, noise_elements)
            - main_columns: List of columns, sorted left-to-right.
            - noise_elements: Flat list of elements that didn't fit
                              a main column (e.g., footnotes).
        """
        # --- GUARD 1: Handle trivial cases ---
        if len(text_elements) < self.min_elements_per_column:
            logger.debug("Too few elements for column detection, treating as single column.")
            return [sorted(text_elements, key=lambda e: (e.y, e.x))], []

        # Extract x-centers for clustering
        x_centers = np.array([elem.x + elem.width / 2 for elem in text_elements]).reshape(-1, 1)

        # --- GUARD 2: Robustly find best k ---
        
        # Fit k=1 (default)
        kmeans_1 = KMeans(n_clusters=1, random_state=42, n_init=10).fit(x_centers)
        best_kmeans = kmeans_1
        best_n_cols = 1
        
        # Only check k=2, k=3 etc. if they are *significantly* better
        for n_cols in range(2, self.max_columns + 1):
            if len(text_elements) < n_cols:
                break # Not enough elements to even try
                
            kmeans = KMeans(n_clusters=n_cols, random_state=42, n_init=10).fit(x_centers)
            labels = kmeans.labels_

            # Check separation guard
            centers = sorted(kmeans.cluster_centers_.flatten())
            min_separation = min(centers[i+1] - centers[i] for i in range(len(centers)-1))
            if min_separation < self.min_column_width:
                logger.debug(f"k={n_cols} failed: min_separation={min_separation:.0f} < {self.min_column_width}")
                continue # Columns too close

            # Check min elements guard
            if np.min(np.bincount(labels)) < self.min_elements_per_column:
                logger.debug(f"k={n_cols} failed: a column had < {self.min_elements_per_column} elements")
                # Don't `continue` here, this `k` might still be valid
                # if we treat the small cluster as noise.
                pass # This check is now handled below.

            # --- GUARD 3: "Is it worth it?" ---
            # Only switch to this `k` if it reduces variance by at least 40%
            # This "guard" prevents k-means from being overly sensitive to
            # small variations and changing layout on minor input changes.
            if kmeans.inertia_ < best_kmeans.inertia_ * 0.6:
                best_kmeans = kmeans
                best_n_cols = n_cols
            else:
                # Not a significant improvement, stop trying larger k
                break
        
        logger.info(f"Detected {best_n_cols} columns")

        # --- ISOLATE FOOTNOTES/NOISE ---
        labels = best_kmeans.labels_
        columns = [[] for _ in range(best_n_cols)]
        for elem, label in zip(text_elements, labels):
            columns[label].append(elem)
        
        main_columns = []
        noise_elements = []
        
        # Sort columns by their average x-position (left to right)
        column_centers = []
        for i, col in enumerate(columns):
            if not col:
                continue
            
            # --- GUARD 4: Footnote/Noise separation ---
            if len(col) < self.min_elements_per_column:
                logger.debug(f"Column {i} (avg_x={np.mean([e.x for e in col]):.0f}) has {len(col)} elements, treating as noise.")
                noise_elements.extend(col)
            else:
                avg_x = sum(elem.x for elem in col) / len(col)
                column_centers.append((avg_x, col))
        
        column_centers.sort()  # Sort by average x position
        
        # Reorder columns and sort elements within each column
        ordered_columns = []
        for avg_x, column in column_centers:
            # Sort within column by reading order (top to bottom)
            column.sort(key=lambda e: (e.y, e.x))
            ordered_columns.append(column)
        
        return ordered_columns, noise_elements
    
    def group_into_lines(self, text_elements: List[TextElement]) -> List[List[TextElement]]:
        """
        Group text elements into lines based on y-coordinate proximity.
        (This logic was already robust, no changes needed).
        """
        if not text_elements:
            return []
        
        lines = []
        current_line = [text_elements[0]]
        
        # Calculate average text height for line grouping
        heights = [elem.height for elem in text_elements if elem.height > 0]
        avg_height = np.median(heights) if heights else 20
        line_threshold = avg_height * self.line_height_threshold
        
        for elem in text_elements[1:]:
            current_y = current_line[-1].y
            
            if abs(elem.y - current_y) <= line_threshold:
                current_line.append(elem)
            else:
                lines.append(sorted(current_line, key=lambda e: e.x))
                current_line = [elem]
        
        if current_line:
            lines.append(sorted(current_line, key=lambda e: e.x))
        
        return lines
    
    def order_text_elements(self, text_elements: List[TextElement], page_width: int) -> List[TextElement]:
        """
        Order text elements by proper reading order.
        
        IMPROVEMENT: Now correctly handles main columns + noise elements.
        """
        if not text_elements:
            return []
        
        # Detect columns and isolate noise
        main_columns, noise_elements = self.detect_columns(text_elements, page_width)
        
        ordered_elements = []
        
        # 1. Process main columns in left-to-right order
        for column in main_columns:
            lines = self.group_into_lines(column)
            for line in lines:
                ordered_elements.extend(line)
                
        # 2. Add all noise elements at the end, sorted by y-pos
        # This correctly handles footnotes/page numbers.
        if noise_elements:
            logger.debug(f"Appending {len(noise_elements)} noise elements to end of reading order.")
            noise_elements.sort(key=lambda e: (e.y, e.x))
            ordered_elements.extend(noise_elements)
        
        return ordered_elements

    # --- NEW: WEEK 2/3 CATCH-UP ---
    def pair_translations(self, 
                          main_columns: List[List[TextElement]], 
                          row_band_tolerance: float = 0.5
                          ) -> List[Tuple[TextElement, TextElement, str]]:
        """
        Implements translation pairing via row-band and adjacent-column heuristics.
        
        Args:
            main_columns: The left-to-right sorted list of main columns.
            row_band_tolerance: Multiplier of element height to define
                                a "row". 0.5 means +/- 50% of height.
        
        Returns:
            List of (elem_left, elem_right, "pair_method") tuples.
        """
        # This heuristic only works for 2-column layouts
        if len(main_columns) != 2:
            return []
            
        logger.debug(f"Attempting translation pairing on 2-column layout...")
        col1, col2 = main_columns[0], main_columns[1]
        
        # Make a mutable copy of the right-hand column to "consume" matches
        col2_remaining = list(col2)
        pairs = []

        for elem1 in col1:
            avg_height = elem1.height if elem1.height > 0 else 20
            tolerance = avg_height * row_band_tolerance
            
            best_match = None
            min_y_diff = float('inf')
            
            for elem2 in col2_remaining:
                y_diff = abs(elem1.y - elem2.y)
                
                # Check if elem2 is in the same "row band"
                if y_diff <= tolerance and y_diff < min_y_diff:
                    best_match = elem2
                    min_y_diff = y_diff
            
            if best_match:
                pairs.append((elem1, best_match, "row_band_match"))
                # Remove from pool so it can't be matched again
                col2_remaining.remove(best_match)
                
        logger.info(f"Created {len(pairs)} translation pairs.")
        return pairs

    def create_reading_order_report(self, text_elements: List[TextElement], 
                                  ordered_elements: List[TextElement],
                                  page_width: int) -> Dict:
        """
        Create a detailed report of the reading order detection process.
        
        IMPROVEMENT: Updated to reflect new column/noise logic.
        """
        # Re-run detection to get the separated parts
        main_columns, noise_elements = self.detect_columns(text_elements, page_width)
        
        report = {
            'total_elements': len(text_elements),
            'num_columns': len(main_columns),
            'noise_elements_found': len(noise_elements),
            'column_stats': [],
            'processing_quality': {
                'elements_processed': len(ordered_elements),
                'elements_lost': len(text_elements) - len(ordered_elements)
            }
        }
        
        for i, column in enumerate(main_columns):
            lines = self.group_into_lines(column)
            avg_x = sum(elem.x for elem in column) / len(column) if column else 0
            
            report['column_stats'].append({
                'column_id': i,
                'elements': len(column),
                'lines': len(lines),
                'avg_x_position': int(avg_x),
                'x_range': (min(elem.x for elem in column), max(elem.x + elem.width for elem in column)) if column else (0, 0)
            })
        
        return report


def convert_ocr_results_to_elements(ocr_results: List[Dict]) -> List[TextElement]:
    """
    Convert OCR results to TextElement objects.
    
    IMPROVEMENT: Now stores `original_data` for lossless re-assembly.
    """
    elements = []
    
    for result in ocr_results:
        # Use tuple() to ensure hashability and correct type
        bbox = tuple(result.get('bbox', (0, 0, 0, 0)))
        
        if len(bbox) != 4:
            logger.warning(f"Skipping element with malformed bbox: {result.get('text')}")
            continue

        x, y, w, h = bbox
        
        element = TextElement(
            text=str(result.get('text', '')),
            x=int(x),
            y=int(y),
            width=int(w),
            height=int(h),
            conf=float(result.get('conf', 0.0)),
            bbox=bbox,
            original_data=result  # Store the original dict
        )
        elements.append(element)
    
    return elements


def apply_reading_order(ocr_results: List[Dict], page_width: int = 800) -> Tuple[List[Dict], Dict]:
    """
    Apply reading order detection to OCR results.
    
    Args:
        ocr_results: List of OCR result dictionaries
        page_width: Width of the page for column detection
        
    Returns:
        Tuple of (ordered_ocr_results, reading_order_report)
    """
    # Convert to TextElement objects
    elements = convert_ocr_results_to_elements(ocr_results)
    
    # Create reading order detector (pulls from config)
    detector = ReadingOrderDetector()
    
    # Apply reading order
    ordered_elements = detector.order_text_elements(elements, page_width)
    
    # Create report
    report = detector.create_reading_order_report(elements, ordered_elements, page_width)
    
    # --- EFFICIENCY IMPROVEMENT ---
    # Replaced N^2 loop with a simple, fast list comprehension.
    # We just extract the original_data from each ordered element.
    ordered_results = []
    for elem in ordered_elements:
        # Add a flag to show it's been processed
        elem.original_data['engine'] = 'ordered'
        ordered_results.append(elem.original_data)
    
    return ordered_results, report