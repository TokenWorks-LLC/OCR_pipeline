"""
Advanced preprocessing module for OCR with deskewing, denoising, column segmentation,
footnote isolation, and reading order analysis.
"""
import logging
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
import math

logger = logging.getLogger(__name__)

@dataclass
class BBox:
    """Bounding box representation."""
    x: int
    y: int
    w: int
    h: int
    conf: float = 1.0
    text: str = ""
    
    @property
    def x2(self) -> int:
        return self.x + self.w
    
    @property
    def y2(self) -> int:
        return self.y + self.h
    
    @property
    def center_x(self) -> float:
        return self.x + self.w / 2
    
    @property
    def center_y(self) -> float:
        return self.y + self.h / 2
    
    @property
    def area(self) -> int:
        return self.w * self.h

@dataclass
class Column:
    """Text column representation."""
    bbox: BBox
    lines: List[BBox]
    column_id: int
    
@dataclass
class PageLayout:
    """Complete page layout analysis result."""
    columns: List[Column]
    footnotes: List[BBox]
    captions: List[BBox]
    reading_order: List[int]  # Order of line indices
    skew_angle: float
    image_processed: np.ndarray

class AdvancedPreprocessor:
    """Advanced preprocessing for OCR with layout analysis."""
    
    def __init__(self, config_features: Dict[str, bool] = None):
        """Initialize preprocessor with feature flags.
        
        Args:
            config_features: Dict of feature flags (deskew, denoise, contrast, etc.)
        """
        self.features = config_features or {
            'deskew': True,
            'denoise': True, 
            'contrast': True,
            'column_detection': True,
            'footnote_detection': True,
            'reading_order': True
        }
        
    def detect_skew_angle(self, image: np.ndarray) -> float:
        """Detect skew angle using Hough transform on text lines.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Skew angle in degrees (-45 to +45)
        """
        if not self.features.get('deskew', False):
            return 0.0
            
        try:
            # Edge detection for line detection
            edges = cv2.Canny(image, 50, 150, apertureSize=3)
            
            # Hough line detection
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is None or len(lines) == 0:
                return 0.0
            
            # Collect angles of detected lines
            angles = []
            for rho, theta in lines[:, 0]:
                angle = theta * 180 / np.pi
                # Convert to skew angle (-45 to +45)
                if angle > 90:
                    angle -= 180
                elif angle < -90:
                    angle += 180
                    
                # Filter out vertical/horizontal lines
                if abs(angle) > 1 and abs(angle) < 45:
                    angles.append(angle)
            
            if not angles:
                return 0.0
                
            # Use median angle to avoid outliers
            skew_angle = np.median(angles)
            logger.debug(f"Detected skew angle: {skew_angle:.2f}°")
            return skew_angle
            
        except Exception as e:
            logger.warning(f"Skew detection failed: {e}")
            return 0.0
    
    def deskew_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Deskew image by rotating by the given angle.
        
        Args:
            image: Input image
            angle: Rotation angle in degrees
            
        Returns:
            Deskewed image
        """
        if abs(angle) < 0.5:  # Skip minimal corrections
            return image
            
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Create rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new image dimensions
        cos_angle = abs(M[0, 0])
        sin_angle = abs(M[0, 1])
        new_w = int((h * sin_angle) + (w * cos_angle))
        new_h = int((h * cos_angle) + (w * sin_angle))
        
        # Adjust translation
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        # Apply rotation
        deskewed = cv2.warpAffine(image, M, (new_w, new_h), 
                                  flags=cv2.INTER_CUBIC, 
                                  borderMode=cv2.BORDER_REPLICATE)
        
        logger.debug(f"Deskewed image by {angle:.2f}°")
        return deskewed
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance contrast using CLAHE while preserving faint characters.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Contrast-enhanced image
        """
        if not self.features.get('contrast', False):
            return image
            
        try:
            # Use CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
            
            # Blend with original to preserve fine details
            alpha = 0.7  # Weight for enhanced image
            result = cv2.addWeighted(enhanced, alpha, image, 1-alpha, 0)
            
            logger.debug("Applied CLAHE contrast enhancement")
            return result
            
        except Exception as e:
            logger.warning(f"Contrast enhancement failed: {e}")
            return image
    
    def denoise_image(self, image: np.ndarray) -> np.ndarray:
        """Apply bilateral filtering for noise reduction while preserving edges.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Denoised image
        """
        if not self.features.get('denoise', False):
            return image
            
        try:
            # Bilateral filter preserves edges while reducing noise
            denoised = cv2.bilateralFilter(image, 9, 75, 75)
            logger.debug("Applied bilateral filter denoising")
            return denoised
            
        except Exception as e:
            logger.warning(f"Denoising failed: {e}")
            return image
    
    def detect_columns_xy_cut(self, bboxes: List[BBox], image_width: int) -> List[Column]:
        """Detect columns using XY-cut algorithm.
        
        Args:
            bboxes: List of detected text bboxes
            image_width: Width of the image
            
        Returns:
            List of detected columns
        """
        if not self.features.get('column_detection', False) or not bboxes:
            # Single column fallback
            column = Column(
                bbox=BBox(0, 0, image_width, max(bbox.y2 for bbox in bboxes) if bboxes else 100),
                lines=bboxes,
                column_id=0
            )
            return [column]
        
        try:
            # Sort bboxes by y-coordinate for horizontal projection
            sorted_bboxes = sorted(bboxes, key=lambda b: b.y)
            
            # Create horizontal projection (sum of bbox widths per y-line)
            y_projection = {}
            for bbox in sorted_bboxes:
                for y in range(bbox.y, bbox.y2):
                    y_projection[y] = y_projection.get(y, 0) + bbox.w
            
            # Find column boundaries using vertical white space
            x_coords = [bbox.center_x for bbox in bboxes]
            x_coords.sort()
            
            # Simple column detection: look for gaps in x-coordinates
            columns = []
            current_column_bboxes = []
            
            # Group bboxes by x-position (simple left/right column detection)
            if len(set(int(x/50) for x in x_coords)) > 1:  # Multiple x-regions detected
                mid_x = image_width / 2
                left_bboxes = [b for b in bboxes if b.center_x < mid_x]
                right_bboxes = [b for b in bboxes if b.center_x >= mid_x]
                
                if left_bboxes:
                    left_column = Column(
                        bbox=BBox(0, 0, mid_x, image_width),
                        lines=sorted(left_bboxes, key=lambda b: b.y),
                        column_id=0
                    )
                    columns.append(left_column)
                
                if right_bboxes:
                    right_column = Column(
                        bbox=BBox(mid_x, 0, image_width - mid_x, image_width),
                        lines=sorted(right_bboxes, key=lambda b: b.y),
                        column_id=1
                    )
                    columns.append(right_column)
            else:
                # Single column
                column = Column(
                    bbox=BBox(0, 0, image_width, max(bbox.y2 for bbox in bboxes)),
                    lines=sorted(bboxes, key=lambda b: b.y),
                    column_id=0
                )
                columns.append(column)
            
            logger.debug(f"Detected {len(columns)} columns")
            return columns
            
        except Exception as e:
            logger.warning(f"Column detection failed: {e}")
            # Fallback to single column
            column = Column(
                bbox=BBox(0, 0, image_width, max(bbox.y2 for bbox in bboxes) if bboxes else 100),
                lines=bboxes,
                column_id=0
            )
            return [column]
    
    def detect_footnotes_and_captions(self, bboxes: List[BBox], image_height: int) -> Tuple[List[BBox], List[BBox]]:
        """Detect footnotes and captions based on position and font size.
        
        Args:
            bboxes: List of detected text bboxes
            image_height: Height of the image
            
        Returns:
            Tuple of (footnotes, captions)
        """
        footnotes = []
        captions = []
        
        if not self.features.get('footnote_detection', False) or not bboxes:
            return footnotes, captions
        
        try:
            # Calculate average line height
            line_heights = [bbox.h for bbox in bboxes if bbox.h > 5]
            avg_height = np.median(line_heights) if line_heights else 20
            small_text_threshold = avg_height * 0.7
            
            # Bottom 20% of page for footnotes
            footnote_zone_y = image_height * 0.8
            
            for bbox in bboxes:
                # Small text near bottom = footnote
                if bbox.h < small_text_threshold and bbox.y > footnote_zone_y:
                    footnotes.append(bbox)
                # Small text with specific patterns = caption
                elif bbox.h < small_text_threshold and any(word in bbox.text.lower() 
                    for word in ['fig', 'table', 'plate', 'image'] if bbox.text):
                    captions.append(bbox)
            
            logger.debug(f"Detected {len(footnotes)} footnotes and {len(captions)} captions")
            
        except Exception as e:
            logger.warning(f"Footnote/caption detection failed: {e}")
        
        return footnotes, captions
    
    def calculate_reading_order(self, columns: List[Column]) -> List[int]:
        """Calculate reading order within and across columns.
        
        Args:
            columns: List of detected columns
            
        Returns:
            List of line indices in reading order
        """
        if not self.features.get('reading_order', False):
            # Simple fallback: top-to-bottom order
            all_lines = []
            for col in columns:
                all_lines.extend(col.lines)
            return list(range(len(all_lines)))
        
        try:
            reading_order = []
            
            # Sort columns by x-position (left to right)
            sorted_columns = sorted(columns, key=lambda c: c.bbox.x)
            
            # For each column, add lines in y-order
            global_line_idx = 0
            for column in sorted_columns:
                # Sort lines in this column by y-coordinate
                column_lines = sorted(column.lines, key=lambda line: line.y)
                for line in column_lines:
                    reading_order.append(global_line_idx)
                    global_line_idx += 1
            
            logger.debug(f"Calculated reading order for {len(reading_order)} lines")
            return reading_order
            
        except Exception as e:
            logger.warning(f"Reading order calculation failed: {e}")
            # Fallback to sequential order
            total_lines = sum(len(col.lines) for col in columns)
            return list(range(total_lines))
    
    def process_image(self, image: np.ndarray, detected_bboxes: List[BBox] = None) -> PageLayout:
        """Perform complete image preprocessing and layout analysis.
        
        Args:
            image: Input image (color or grayscale)
            detected_bboxes: Optional pre-detected text bboxes for layout analysis
            
        Returns:
            PageLayout with processed image and layout information
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Step 1: Detect and correct skew
        skew_angle = self.detect_skew_angle(gray)
        if abs(skew_angle) > 0.5:
            gray = self.deskew_image(gray, skew_angle)
        
        # Step 2: Enhance contrast
        gray = self.enhance_contrast(gray)
        
        # Step 3: Denoise
        gray = self.denoise_image(gray)
        
        # Step 4: Layout analysis (if bboxes provided)
        columns = []
        footnotes = []
        captions = []
        reading_order = []
        
        if detected_bboxes:
            h, w = gray.shape
            columns = self.detect_columns_xy_cut(detected_bboxes, w)
            footnotes, captions = self.detect_footnotes_and_captions(detected_bboxes, h)
            reading_order = self.calculate_reading_order(columns)
        
        layout = PageLayout(
            columns=columns,
            footnotes=footnotes,
            captions=captions,
            reading_order=reading_order,
            skew_angle=skew_angle,
            image_processed=gray
        )
        
        logger.info(f"Preprocessing complete: {len(columns)} columns, "
                   f"{len(footnotes)} footnotes, {len(captions)} captions, "
                   f"skew: {skew_angle:.2f}°")
        
        return layout