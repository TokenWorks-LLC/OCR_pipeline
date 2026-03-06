"""
Advanced reading order detection with multi-column support using k-means clustering.
"""
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import KMeans
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TextElement:
    """Text element with position and content."""
    text: str
    x: int
    y: int
    width: int
    height: int
    conf: float
    bbox: Tuple[int, int, int, int]

class ReadingOrderDetector:
    """Advanced reading order detection with multi-column layout support."""
    
    def __init__(self, 
                 column_threshold: float = 0.3,
                 line_height_threshold: float = 1.5,
                 min_column_width: int = 100):
        """
        Initialize reading order detector.
        
        Args:
            column_threshold: Threshold for column separation (relative to page width)
            line_height_threshold: Multiplier for line height to group text lines
            min_column_width: Minimum width to consider as a valid column
        """
        self.column_threshold = column_threshold
        self.line_height_threshold = line_height_threshold
        self.min_column_width = min_column_width
    
    def detect_columns(self, text_elements: List[TextElement], page_width: int) -> List[List[TextElement]]:
        """
        Detect column layout using k-means clustering on x-coordinates.
        
        Args:
            text_elements: List of text elements with positions
            page_width: Width of the page for relative positioning
        
        Returns:
            List of columns, each containing sorted text elements
        """
        if not text_elements:
            return []
        
        if len(text_elements) < 3:
            # Too few elements for column detection, treat as single column
            return [sorted(text_elements, key=lambda e: (e.y, e.x))]
        
        # Extract x-coordinates for clustering
        x_coords = np.array([elem.x + elem.width/2 for elem in text_elements]).reshape(-1, 1)
        
        # Try different numbers of columns (1-4)
        best_columns = 1
        best_score = float('inf')
        
        for n_cols in range(1, min(5, len(text_elements) + 1)):
            try:
                kmeans = KMeans(n_clusters=n_cols, random_state=42, n_init=10)
                labels = kmeans.fit_predict(x_coords)
                
                # Calculate within-cluster sum of squares as quality metric
                score = kmeans.inertia_
                
                # Check if columns are well-separated and wide enough
                centers = sorted(kmeans.cluster_centers_.flatten())
                if n_cols > 1:
                    min_separation = min(centers[i+1] - centers[i] for i in range(len(centers)-1))
                    if min_separation < self.min_column_width:
                        continue  # Columns too close together
                
                if score < best_score:
                    best_score = score
                    best_columns = n_cols
                    
            except Exception as e:
                logger.debug(f"K-means failed for {n_cols} columns: {e}")
                continue
        
        logger.info(f"Detected {best_columns} columns")
        
        # Apply final clustering with best number of columns
        if best_columns == 1:
            return [sorted(text_elements, key=lambda e: (e.y, e.x))]
        
        kmeans = KMeans(n_clusters=best_columns, random_state=42, n_init=10)
        labels = kmeans.fit_predict(x_coords)
        
        # Group elements by column
        columns = [[] for _ in range(best_columns)]
        for elem, label in zip(text_elements, labels):
            columns[label].append(elem)
        
        # Sort columns by their average x-position (left to right)
        column_centers = []
        for i, col in enumerate(columns):
            if col:  # Only non-empty columns
                avg_x = sum(elem.x for elem in col) / len(col)
                column_centers.append((avg_x, i))
        
        column_centers.sort()  # Sort by average x position
        
        # Reorder columns and sort elements within each column
        ordered_columns = []
        for _, col_idx in column_centers:
            column = columns[col_idx]
            if column:  # Only include non-empty columns
                # Sort within column by reading order (top to bottom, left to right)
                column.sort(key=lambda e: (e.y, e.x))
                ordered_columns.append(column)
        
        return ordered_columns
    
    def group_into_lines(self, text_elements: List[TextElement]) -> List[List[TextElement]]:
        """
        Group text elements into lines based on y-coordinate proximity.
        
        Args:
            text_elements: List of text elements sorted by reading order
            
        Returns:
            List of lines, each containing text elements on the same line
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
            # Check if element belongs to current line
            current_y = current_line[-1].y
            
            if abs(elem.y - current_y) <= line_threshold:
                current_line.append(elem)
            else:
                # Start new line
                lines.append(sorted(current_line, key=lambda e: e.x))  # Sort by x within line
                current_line = [elem]
        
        # Add the last line
        if current_line:
            lines.append(sorted(current_line, key=lambda e: e.x))
        
        return lines
    
    def order_text_elements(self, text_elements: List[TextElement], page_width: int) -> List[TextElement]:
        """
        Order text elements by proper reading order with column detection.
        
        Args:
            text_elements: List of text elements with positions
            page_width: Width of the page
            
        Returns:
            List of text elements in proper reading order
        """
        if not text_elements:
            return []
        
        # Detect columns
        columns = self.detect_columns(text_elements, page_width)
        
        ordered_elements = []
        
        # Process each column in left-to-right order
        for column in columns:
            # Group elements in column into lines
            lines = self.group_into_lines(column)
            
            # Add all elements from lines in order
            for line in lines:
                ordered_elements.extend(line)
        
        return ordered_elements
    
    def create_reading_order_report(self, text_elements: List[TextElement], 
                                  ordered_elements: List[TextElement],
                                  page_width: int) -> Dict:
        """
        Create a detailed report of the reading order detection process.
        
        Returns:
            Dictionary with detection statistics and column information
        """
        columns = self.detect_columns(text_elements, page_width)
        
        report = {
            'total_elements': len(text_elements),
            'num_columns': len(columns),
            'column_stats': [],
            'processing_quality': {
                'elements_processed': len(ordered_elements),
                'elements_lost': len(text_elements) - len(ordered_elements)
            }
        }
        
        for i, column in enumerate(columns):
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
    """Convert OCR results to TextElement objects."""
    elements = []
    
    for result in ocr_results:
        bbox = result.get('bbox', (0, 0, 0, 0))
        if len(bbox) == 4:
            x, y, w, h = bbox
        else:
            x, y, w, h = 0, 0, 100, 20
        
        element = TextElement(
            text=result.get('text', ''),
            x=x,
            y=y,
            width=w,
            height=h,
            conf=result.get('conf', 0.0),
            bbox=(x, y, w, h)
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
    
    # Create reading order detector
    detector = ReadingOrderDetector()
    
    # Apply reading order
    ordered_elements = detector.order_text_elements(elements, page_width)
    
    # Create report
    report = detector.create_reading_order_report(elements, ordered_elements, page_width)
    
    # Convert back to OCR result format
    ordered_results = []
    for elem in ordered_elements:
        result = {
            'text': elem.text,
            'bbox': elem.bbox,
            'conf': elem.conf,
            'engine': 'ordered'  # Mark as processed
        }
        # Preserve any additional fields from original results
        for orig_result in ocr_results:
            if orig_result.get('text') == elem.text and orig_result.get('bbox') == elem.bbox:
                for key, value in orig_result.items():
                    if key not in result:
                        result[key] = value
                break
        
        ordered_results.append(result)
    
    return ordered_results, report
