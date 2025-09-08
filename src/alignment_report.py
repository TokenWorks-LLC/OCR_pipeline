"""
Alignment Quality Assurance system with HTML reports and overlay visualization.
"""
import json
import logging
import os
import base64
from datetime import datetime
from io import BytesIO
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from config import OUTPUT_DIRS
from ocr_utils import Line

logger = logging.getLogger(__name__)


@dataclass
class AlignmentMetrics:
    """Metrics for OCR alignment quality."""
    total_lines: int
    column_gaps: List[Tuple[int, float]]  # (gap_position, confidence_score)
    reading_pattern: str  # 'single_column', 'two_column', 'multi_column'
    text_flow_score: float  # 0-1, how well text flows in reading order
    overlap_count: int  # Number of overlapping bounding boxes
    gap_consistency: float  # 0-1, consistency of column gaps
    line_alignment: float  # 0-1, how well lines align horizontally
    confidence_distribution: Dict[str, float]  # confidence ranges
    language_distribution: Dict[str, int]  # detected languages


def calculate_alignment_metrics(lines: List[Line], img_shape: Tuple[int, int]) -> AlignmentMetrics:
    """Calculate comprehensive alignment metrics."""
    if not lines:
        return AlignmentMetrics(
            total_lines=0, column_gaps=[], reading_pattern='none',
            text_flow_score=0.0, overlap_count=0, gap_consistency=0.0,
            line_alignment=0.0, confidence_distribution={}, language_distribution={}
        )
    
    img_height, img_width = img_shape[:2]
    
    # Extract line properties
    centers = [(bbox[0] + bbox[2]//2, bbox[1] + bbox[3]//2) for bbox in [line.bbox for line in lines]]
    x_centers = [c[0] for c in centers]
    y_centers = [c[1] for c in centers]
    
    # Detect column gaps
    column_gaps = detect_column_gaps(x_centers, img_width)
    
    # Determine reading pattern
    reading_pattern = determine_reading_pattern(column_gaps, img_width)
    
    # Calculate text flow score
    text_flow_score = calculate_text_flow_score(lines, column_gaps)
    
    # Count overlaps
    overlap_count = count_overlapping_boxes([line.bbox for line in lines])
    
    # Gap consistency
    gap_consistency = calculate_gap_consistency(column_gaps)
    
    # Line alignment score
    line_alignment = calculate_line_alignment(lines)
    
    # Confidence distribution
    confidence_distribution = analyze_confidence_distribution([line.conf for line in lines])
    
    # Language distribution (simplified - would need langid integration)
    language_distribution = {'unknown': len(lines)}  # Placeholder
    
    return AlignmentMetrics(
        total_lines=len(lines),
        column_gaps=column_gaps,
        reading_pattern=reading_pattern,
        text_flow_score=text_flow_score,
        overlap_count=overlap_count,
        gap_consistency=gap_consistency,
        line_alignment=line_alignment,
        confidence_distribution=confidence_distribution,
        language_distribution=language_distribution
    )


def detect_column_gaps(x_centers: List[int], img_width: int) -> List[Tuple[int, float]]:
    """Detect vertical gaps that indicate column boundaries."""
    if not x_centers:
        return []
    
    # Create histogram of x-positions
    hist_bins = min(100, img_width // 10)
    hist, bin_edges = np.histogram(x_centers, bins=hist_bins, range=(0, img_width))
    
    # Find gaps (low-density regions between high-density regions)
    gaps = []
    in_content = False
    gap_start = 0
    threshold = max(1, len(x_centers) // 20)  # Adaptive threshold
    
    for i, count in enumerate(hist):
        bin_center = (bin_edges[i] + bin_edges[i+1]) / 2
        
        if count >= threshold and not in_content:
            # Start of content region
            if gap_start > 0:
                # We were in a gap
                gap_center = (gap_start + bin_center) / 2
                gap_width = bin_center - gap_start
                confidence = min(1.0, gap_width / (img_width * 0.1))  # Wider gaps = higher confidence
                gaps.append((int(gap_center), confidence))
            in_content = True
        elif count < threshold and in_content:
            # Start of gap
            gap_start = bin_center
            in_content = False
    
    # Sort by confidence and return top gaps
    gaps.sort(key=lambda x: x[1], reverse=True)
    return gaps[:3]  # Maximum 3 column gaps


def determine_reading_pattern(gaps: List[Tuple[int, float]], img_width: int) -> str:
    """Determine the reading pattern based on column gaps."""
    significant_gaps = [g for g in gaps if g[1] > 0.3]  # Confidence > 0.3
    
    if len(significant_gaps) == 0:
        return 'single_column'
    elif len(significant_gaps) == 1:
        return 'two_column'
    else:
        return 'multi_column'


def calculate_text_flow_score(lines: List[Line], gaps: List[Tuple[int, float]]) -> float:
    """Calculate how well text flows in expected reading order."""
    if len(lines) < 2:
        return 1.0
    
    # Sort lines by reading order (top to bottom, left to right within columns)
    if not gaps:
        # Single column - sort by y-coordinate
        sorted_lines = sorted(lines, key=lambda line: line.bbox[1])
    else:
        # Multi-column - group by column, then sort by y within each column
        sorted_lines = sort_lines_by_reading_order(lines, gaps)
    
    # Calculate flow score based on coordinate progression
    flow_violations = 0
    total_transitions = len(sorted_lines) - 1
    
    for i in range(total_transitions):
        curr_line = sorted_lines[i]
        next_line = sorted_lines[i + 1]
        
        curr_y = curr_line.bbox[1]
        next_y = next_line.bbox[1]
        
        # In proper reading flow, y should generally increase (or stay similar for same row)
        if next_y < curr_y - 20:  # Allow some tolerance
            flow_violations += 1
    
    return 1.0 - (flow_violations / total_transitions) if total_transitions > 0 else 1.0


def sort_lines_by_reading_order(lines: List[Line], gaps: List[Tuple[int, float]]) -> List[Line]:
    """Sort lines in proper reading order considering columns."""
    if not gaps:
        return sorted(lines, key=lambda line: line.bbox[1])
    
    # Use the most confident gap to determine column boundary
    primary_gap = gaps[0][0]
    
    # Separate into left and right columns
    left_column = []
    right_column = []
    
    for line in lines:
        line_center_x = line.bbox[0] + line.bbox[2] // 2
        if line_center_x < primary_gap:
            left_column.append(line)
        else:
            right_column.append(line)
    
    # Sort each column by y-coordinate
    left_column.sort(key=lambda line: line.bbox[1])
    right_column.sort(key=lambda line: line.bbox[1])
    
    # Combine: all of left column, then all of right column
    return left_column + right_column


def count_overlapping_boxes(bboxes: List[Tuple[int, int, int, int]]) -> int:
    """Count overlapping bounding boxes."""
    overlaps = 0
    for i, bbox1 in enumerate(bboxes):
        for bbox2 in bboxes[i+1:]:
            if boxes_overlap(bbox1, bbox2):
                overlaps += 1
    return overlaps


def boxes_overlap(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> bool:
    """Check if two bounding boxes overlap."""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)


def calculate_gap_consistency(gaps: List[Tuple[int, float]]) -> float:
    """Calculate consistency of column gaps."""
    if len(gaps) < 2:
        return 1.0
    
    # Gaps should be roughly evenly spaced for consistent columns
    positions = [g[0] for g in gaps]
    positions.sort()
    
    if len(positions) == 1:
        return 1.0
    
    # Calculate spacing between consecutive gaps
    spacings = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
    
    if not spacings:
        return 1.0
    
    # Measure consistency as inverse of coefficient of variation
    mean_spacing = np.mean(spacings)
    std_spacing = np.std(spacings)
    
    if mean_spacing == 0:
        return 1.0
    
    cv = std_spacing / mean_spacing
    return max(0.0, 1.0 - cv)  # Higher consistency = lower coefficient of variation


def calculate_line_alignment(lines: List[Line]) -> float:
    """Calculate how well lines are aligned horizontally."""
    if len(lines) < 2:
        return 1.0
    
    # Group lines by approximate y-coordinate (same row)
    rows = {}
    tolerance = 20  # pixels
    
    for line in lines:
        y = line.bbox[1]
        found_row = False
        for row_y in rows.keys():
            if abs(y - row_y) <= tolerance:
                rows[row_y].append(line)
                found_row = True
                break
        if not found_row:
            rows[y] = [line]
    
    # Calculate alignment score for rows with multiple lines
    alignment_scores = []
    for row_lines in rows.values():
        if len(row_lines) > 1:
            # Calculate standard deviation of y-coordinates in this row
            y_coords = [line.bbox[1] for line in row_lines]
            std_dev = np.std(y_coords)
            # Convert to score (lower std dev = better alignment)
            alignment_score = max(0.0, 1.0 - std_dev / tolerance)
            alignment_scores.append(alignment_score)
    
    return np.mean(alignment_scores) if alignment_scores else 1.0


def analyze_confidence_distribution(confidences: List[float]) -> Dict[str, float]:
    """Analyze the distribution of OCR confidence scores."""
    if not confidences:
        return {}
    
    confidences = np.array(confidences)
    
    return {
        'mean': float(np.mean(confidences)),
        'std': float(np.std(confidences)),
        'min': float(np.min(confidences)),
        'max': float(np.max(confidences)),
        'high_conf_pct': float(np.sum(confidences >= 0.8) / len(confidences)),
        'low_conf_pct': float(np.sum(confidences < 0.5) / len(confidences))
    }


def create_overlay_visualization(img: np.ndarray, lines: List[Line], 
                               metrics: AlignmentMetrics) -> np.ndarray:
    """Create visualization overlay showing OCR results and alignment analysis."""
    overlay = img.copy()
    h, w = overlay.shape[:2]
    
    # Draw bounding boxes
    for i, line in enumerate(lines):
        x, y, box_w, box_h = line.bbox
        
        # Color based on confidence
        if line.conf >= 0.8:
            color = (0, 255, 0)  # Green for high confidence
        elif line.conf >= 0.5:
            color = (0, 255, 255)  # Yellow for medium confidence
        else:
            color = (0, 0, 255)  # Red for low confidence
        
        # Draw bounding box
        cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), color, 2)
        
        # Draw confidence score
        conf_text = f"{line.conf:.2f}"
        cv2.putText(overlay, conf_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Draw column gaps
    for gap_pos, confidence in metrics.column_gaps:
        color = (255, 0, 255)  # Magenta for column boundaries
        thickness = max(1, int(confidence * 5))
        cv2.line(overlay, (gap_pos, 0), (gap_pos, h), color, thickness)
        
        # Label the gap
        cv2.putText(overlay, f"Gap: {confidence:.2f}", (gap_pos + 5, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Add reading order arrows for two-column layout
    if metrics.reading_pattern == 'two_column' and metrics.column_gaps:
        gap_pos = metrics.column_gaps[0][0]
        
        # Left column arrow (top to bottom)
        cv2.arrowedLine(overlay, (gap_pos // 2, 50), (gap_pos // 2, h - 50), 
                       (128, 255, 128), 3, tipLength=0.1)
        cv2.putText(overlay, "1st", (gap_pos // 2 - 15, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 255, 128), 2)
        
        # Right column arrow (top to bottom)
        right_center = gap_pos + (w - gap_pos) // 2
        cv2.arrowedLine(overlay, (right_center, 50), (right_center, h - 50), 
                       (128, 255, 128), 3, tipLength=0.1)
        cv2.putText(overlay, "2nd", (right_center - 15, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 255, 128), 2)
    
    return overlay


def generate_alignment_report(img: np.ndarray, lines: List[Line], 
                            metrics: AlignmentMetrics, output_path: str) -> str:
    """Generate comprehensive HTML alignment report."""
    
    # Create overlay visualization
    overlay_img = create_overlay_visualization(img, lines, metrics)
    
    # Convert images to base64 for embedding
    original_b64 = img_to_base64(img)
    overlay_b64 = img_to_base64(overlay_img)
    
    # Generate HTML report
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OCR Alignment Quality Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; border-bottom: 2px solid #ddd; padding-bottom: 20px; margin-bottom: 30px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .metric-card {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; border-left: 4px solid #007bff; }}
        .metric-title {{ font-weight: bold; color: #333; margin-bottom: 8px; }}
        .metric-value {{ font-size: 1.2em; color: #007bff; }}
        .quality-indicator {{ display: inline-block; padding: 4px 8px; border-radius: 12px; font-size: 0.8em; font-weight: bold; }}
        .quality-excellent {{ background-color: #d4edda; color: #155724; }}
        .quality-good {{ background-color: #d1ecf1; color: #0c5460; }}
        .quality-warning {{ background-color: #fff3cd; color: #856404; }}
        .quality-poor {{ background-color: #f8d7da; color: #721c24; }}
        .images-container {{ display: flex; gap: 20px; margin-bottom: 30px; }}
        .image-panel {{ flex: 1; text-align: center; }}
        .image-panel img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }}
        .image-title {{ font-weight: bold; margin-bottom: 10px; color: #333; }}
        .details-section {{ margin-top: 30px; }}
        .details-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 30px; }}
        .legend {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .legend-item {{ display: flex; align-items: center; margin-bottom: 8px; }}
        .color-box {{ width: 20px; height: 20px; margin-right: 10px; border: 1px solid #ccc; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f8f9fa; font-weight: bold; }}
        .footer {{ margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd; text-align: center; color: #666; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>OCR Alignment Quality Report</h1>
            <p>Generated on {timestamp}</p>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-title">Total Lines Detected</div>
                <div class="metric-value">{metrics.total_lines}</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-title">Reading Pattern</div>
                <div class="metric-value">{metrics.reading_pattern.replace('_', ' ').title()}</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-title">Text Flow Quality</div>
                <div class="metric-value">{metrics.text_flow_score:.3f} {get_quality_indicator(metrics.text_flow_score)}</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-title">Line Alignment</div>
                <div class="metric-value">{metrics.line_alignment:.3f} {get_quality_indicator(metrics.line_alignment)}</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-title">Gap Consistency</div>
                <div class="metric-value">{metrics.gap_consistency:.3f} {get_quality_indicator(metrics.gap_consistency)}</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-title">Overlapping Boxes</div>
                <div class="metric-value">{metrics.overlap_count} {get_overlap_indicator(metrics.overlap_count)}</div>
            </div>
        </div>
        
        <div class="legend">
            <h3>Visualization Legend</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                <div>
                    <div class="legend-item">
                        <div class="color-box" style="background-color: #00ff00;"></div>
                        <span>High Confidence (≥0.8)</span>
                    </div>
                    <div class="legend-item">
                        <div class="color-box" style="background-color: #ffff00;"></div>
                        <span>Medium Confidence (0.5-0.8)</span>
                    </div>
                    <div class="legend-item">
                        <div class="color-box" style="background-color: #ff0000;"></div>
                        <span>Low Confidence (<0.5)</span>
                    </div>
                </div>
                <div>
                    <div class="legend-item">
                        <div class="color-box" style="background-color: #ff00ff;"></div>
                        <span>Column Boundaries</span>
                    </div>
                    <div class="legend-item">
                        <div class="color-box" style="background-color: #80ff80;"></div>
                        <span>Reading Order Arrows</span>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="images-container">
            <div class="image-panel">
                <div class="image-title">Original Image</div>
                <img src="data:image/png;base64,{original_b64}" alt="Original Image">
            </div>
            <div class="image-panel">
                <div class="image-title">Alignment Analysis Overlay</div>
                <img src="data:image/png;base64,{overlay_b64}" alt="Overlay Analysis">
            </div>
        </div>
        
        <div class="details-section">
            <h2>Detailed Analysis</h2>
            <div class="details-grid">
                <div>
                    <h3>Column Detection</h3>
                    {generate_column_details_html(metrics.column_gaps)}
                    
                    <h3>Confidence Distribution</h3>
                    {generate_confidence_details_html(metrics.confidence_distribution)}
                </div>
                <div>
                    <h3>Quality Metrics</h3>
                    {generate_quality_details_html(metrics)}
                    
                    <h3>Recommendations</h3>
                    {generate_recommendations_html(metrics)}
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>OCR Pipeline Alignment QA Report • Advanced Text Layout Analysis</p>
        </div>
    </div>
</body>
</html>
"""
    
    # Write HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"Alignment report generated: {output_path}")
    return output_path


def img_to_base64(img: np.ndarray) -> str:
    """Convert OpenCV image to base64 string for HTML embedding."""
    # Convert BGR to RGB
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img
    
    # Convert to PIL Image
    pil_img = Image.fromarray(img_rgb)
    
    # Save to base64
    buffer = BytesIO()
    pil_img.save(buffer, format='PNG')
    img_b64 = base64.b64encode(buffer.getvalue()).decode()
    
    return img_b64


def get_quality_indicator(score: float) -> str:
    """Get HTML quality indicator based on score."""
    if score >= 0.9:
        return '<span class="quality-indicator quality-excellent">Excellent</span>'
    elif score >= 0.7:
        return '<span class="quality-indicator quality-good">Good</span>'
    elif score >= 0.5:
        return '<span class="quality-indicator quality-warning">Needs Attention</span>'
    else:
        return '<span class="quality-indicator quality-poor">Poor</span>'


def get_overlap_indicator(count: int) -> str:
    """Get HTML overlap indicator based on count."""
    if count == 0:
        return '<span class="quality-indicator quality-excellent">None</span>'
    elif count <= 2:
        return '<span class="quality-indicator quality-good">Minimal</span>'
    elif count <= 5:
        return '<span class="quality-indicator quality-warning">Some</span>'
    else:
        return '<span class="quality-indicator quality-poor">Many</span>'


def generate_column_details_html(gaps: List[Tuple[int, float]]) -> str:
    """Generate HTML for column detection details."""
    if not gaps:
        return "<p>No column boundaries detected (single column layout)</p>"
    
    html = "<table><thead><tr><th>Gap Position</th><th>Confidence</th><th>Quality</th></tr></thead><tbody>"
    for pos, conf in gaps:
        quality = get_quality_indicator(conf)
        html += f"<tr><td>{pos}px</td><td>{conf:.3f}</td><td>{quality}</td></tr>"
    html += "</tbody></table>"
    return html


def generate_confidence_details_html(conf_dist: Dict[str, float]) -> str:
    """Generate HTML for confidence distribution details."""
    if not conf_dist:
        return "<p>No confidence data available</p>"
    
    html = "<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>"
    html += f"<tr><td>Mean Confidence</td><td>{conf_dist.get('mean', 0):.3f}</td></tr>"
    html += f"<tr><td>Standard Deviation</td><td>{conf_dist.get('std', 0):.3f}</td></tr>"
    html += f"<tr><td>High Confidence %</td><td>{conf_dist.get('high_conf_pct', 0)*100:.1f}%</td></tr>"
    html += f"<tr><td>Low Confidence %</td><td>{conf_dist.get('low_conf_pct', 0)*100:.1f}%</td></tr>"
    html += "</tbody></table>"
    return html


def generate_quality_details_html(metrics: AlignmentMetrics) -> str:
    """Generate HTML for quality metrics details."""
    overall_score = (metrics.text_flow_score + metrics.line_alignment + metrics.gap_consistency) / 3
    
    html = f"""
    <table>
        <thead><tr><th>Metric</th><th>Score</th><th>Assessment</th></tr></thead>
        <tbody>
            <tr><td>Overall Quality</td><td>{overall_score:.3f}</td><td>{get_quality_indicator(overall_score)}</td></tr>
            <tr><td>Text Flow</td><td>{metrics.text_flow_score:.3f}</td><td>{get_quality_indicator(metrics.text_flow_score)}</td></tr>
            <tr><td>Line Alignment</td><td>{metrics.line_alignment:.3f}</td><td>{get_quality_indicator(metrics.line_alignment)}</td></tr>
            <tr><td>Gap Consistency</td><td>{metrics.gap_consistency:.3f}</td><td>{get_quality_indicator(metrics.gap_consistency)}</td></tr>
        </tbody>
    </table>
    """
    return html


def generate_recommendations_html(metrics: AlignmentMetrics) -> str:
    """Generate HTML recommendations based on metrics."""
    recommendations = []
    
    if metrics.text_flow_score < 0.7:
        recommendations.append("Consider improving column detection or text ordering algorithm")
    
    if metrics.line_alignment < 0.7:
        recommendations.append("Text lines may benefit from better horizontal alignment detection")
    
    if metrics.overlap_count > 5:
        recommendations.append("High number of overlapping boxes - consider improving NMS or OCR parameters")
    
    if not metrics.column_gaps and metrics.total_lines > 20:
        recommendations.append("Large document without column detection - verify if multi-column layout")
    
    if metrics.gap_consistency < 0.5:
        recommendations.append("Inconsistent column gaps detected - review column boundary detection")
    
    if not recommendations:
        recommendations.append("OCR alignment quality looks good! No specific recommendations.")
    
    html = "<ul>"
    for rec in recommendations:
        html += f"<li>{rec}</li>"
    html += "</ul>"
    
    return html


def create_alignment_report(img: np.ndarray, lines: List[Line], 
                          output_filename: str = None) -> Tuple[str, AlignmentMetrics]:
    """Create comprehensive alignment QA report."""
    # Calculate metrics
    metrics = calculate_alignment_metrics(lines, img.shape)
    
    # Generate output filename if not provided
    if output_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"alignment_report_{timestamp}.html"
    
    # Ensure output directory exists
    report_dir = OUTPUT_DIRS['report']
    os.makedirs(report_dir, exist_ok=True)
    
    output_path = os.path.join(report_dir, output_filename)
    
    # Generate report
    report_path = generate_alignment_report(img, lines, metrics, output_path)
    
    logger.info(f"Alignment QA complete - Report: {report_path}")
    logger.info(f"Quality Summary: Flow={metrics.text_flow_score:.3f}, "
               f"Alignment={metrics.line_alignment:.3f}, "
               f"Pattern={metrics.reading_pattern}")
    
    return report_path, metrics
