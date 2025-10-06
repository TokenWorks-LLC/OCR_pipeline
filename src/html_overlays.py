"""
HTML overlay generation for OCR QA visualization.
Creates interactive HTML overlays with bboxes and confidence tooltips.
"""
import base64
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import io

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

logger = logging.getLogger(__name__)

class HTMLOverlayGenerator:
    """Generates HTML overlays for OCR results visualization."""
    
    def __init__(self, output_dir: str = "reports/overlays"):
        """Initialize overlay generator.
        
        Args:
            output_dir: Directory to save HTML overlays
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 data URL.
        
        Args:
            image: PIL Image
            
        Returns:
            Base64 data URL string
        """
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        img_bytes = buffer.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"
    
    def generate_bbox_html(self, bbox: Dict[str, Any], page_width: int, page_height: int) -> str:
        """Generate HTML for a single bounding box.
        
        Args:
            bbox: Bounding box with x, y, w, h, confidence, text
            page_width: Page width for scaling
            page_height: Page height for scaling
            
        Returns:
            HTML div element for the bbox
        """
        x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
        confidence = bbox.get('confidence', 0.0)
        text = bbox.get('text', '')
        bbox_type = bbox.get('type', 'text')  # text, footnote, caption, etc.
        
        # Calculate percentage positions for responsive design
        left_pct = (x / page_width) * 100
        top_pct = (y / page_height) * 100
        width_pct = (w / page_width) * 100
        height_pct = (h / page_height) * 100
        
        # Color coding by confidence
        if confidence >= 0.8:
            border_color = "#4CAF50"  # Green - high confidence
            bg_color = "rgba(76, 175, 80, 0.1)"
        elif confidence >= 0.6:
            border_color = "#FF9800"  # Orange - medium confidence  
            bg_color = "rgba(255, 152, 0, 0.1)"
        else:
            border_color = "#F44336"  # Red - low confidence
            bg_color = "rgba(244, 67, 54, 0.1)"
        
        # Special styling for different types
        if bbox_type == 'footnote':
            border_color = "#9C27B0"  # Purple for footnotes
            bg_color = "rgba(156, 39, 176, 0.1)"
        elif bbox_type == 'caption':
            border_color = "#2196F3"  # Blue for captions
            bg_color = "rgba(33, 150, 243, 0.1)"
        
        # Escape text for HTML
        escaped_text = (text.replace('&', '&amp;')
                           .replace('<', '&lt;')
                           .replace('>', '&gt;')
                           .replace('"', '&quot;')
                           .replace("'", '&#x27;'))
        
        # Truncate very long text for tooltip
        display_text = escaped_text[:200] + "..." if len(escaped_text) > 200 else escaped_text
        
        html = f'''
        <div class="ocr-bbox ocr-{bbox_type}" 
             style="position: absolute; 
                    left: {left_pct:.2f}%; 
                    top: {top_pct:.2f}%; 
                    width: {width_pct:.2f}%; 
                    height: {height_pct:.2f}%;
                    border: 2px solid {border_color};
                    background-color: {bg_color};
                    cursor: pointer;"
             data-confidence="{confidence:.3f}"
             data-type="{bbox_type}"
             data-text="{escaped_text}"
             title="Type: {bbox_type} | Confidence: {confidence:.3f} | Text: {display_text}">
        </div>'''
        
        return html
    
    def generate_page_overlay(self, page_image: Image.Image, bboxes: List[Dict[str, Any]], 
                            page_info: Dict[str, Any]) -> str:
        """Generate complete HTML overlay for a page.
        
        Args:
            page_image: PIL Image of the page
            bboxes: List of bounding boxes with metadata
            page_info: Page metadata (pdf_name, page_num, etc.)
            
        Returns:
            Complete HTML document string
        """
        # Convert image to base64
        img_data_url = self.image_to_base64(page_image)
        page_width, page_height = page_image.size
        
        # Generate bbox HTML
        bbox_html = ""
        for bbox in bboxes:
            bbox_html += self.generate_bbox_html(bbox, page_width, page_height)
        
        # Calculate statistics
        total_bboxes = len(bboxes)
        avg_confidence = sum(bbox.get('confidence', 0) for bbox in bboxes) / max(1, total_bboxes)
        high_conf_count = sum(1 for bbox in bboxes if bbox.get('confidence', 0) >= 0.8)
        low_conf_count = sum(1 for bbox in bboxes if bbox.get('confidence', 0) < 0.6)
        
        # Count by type
        type_counts = {}
        for bbox in bboxes:
            bbox_type = bbox.get('type', 'text')
            type_counts[bbox_type] = type_counts.get(bbox_type, 0) + 1
        
        type_summary = " | ".join([f"{t}: {c}" for t, c in type_counts.items()])
        
        # Generate HTML document
        html_content = f'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OCR Overlay - {page_info.get('pdf_name', 'Unknown')} Page {page_info.get('page_num', '?')}</title>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            font-family: Arial, sans-serif;
            background-color: #f5f5f5;
        }}
        .header {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0 0 10px 0;
            color: #333;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }}
        .stat {{
            background: #f8f9fa;
            padding: 8px 12px;
            border-radius: 4px;
            border-left: 4px solid #007bff;
        }}
        .page-container {{
            position: relative;
            display: inline-block;
            background: white;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .page-image {{
            display: block;
            max-width: 100%;
            height: auto;
        }}
        .ocr-bbox {{
            transition: all 0.3s ease;
            box-sizing: border-box;
        }}
        .ocr-bbox:hover {{
            border-width: 3px;
            z-index: 1000;
            transform: scale(1.02);
        }}
        .controls {{
            position: fixed;
            top: 20px;
            right: 20px;
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            min-width: 200px;
            z-index: 1001;
        }}
        .controls h3 {{
            margin: 0 0 10px 0;
            color: #333;
        }}
        .control-group {{
            margin-bottom: 10px;
        }}
        .control-group label {{
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }}
        input[type="range"] {{
            width: 100%;
        }}
        .legend {{
            margin-top: 15px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin-bottom: 5px;
        }}
        .legend-color {{
            width: 20px;
            height: 20px;
            margin-right: 8px;
            border-radius: 3px;
            border: 2px solid;
        }}
        .high-conf {{ border-color: #4CAF50; background-color: rgba(76, 175, 80, 0.2); }}
        .med-conf {{ border-color: #FF9800; background-color: rgba(255, 152, 0, 0.2); }}
        .low-conf {{ border-color: #F44336; background-color: rgba(244, 67, 54, 0.2); }}
        .footnote {{ border-color: #9C27B0; background-color: rgba(156, 39, 176, 0.2); }}
        .caption {{ border-color: #2196F3; background-color: rgba(33, 150, 243, 0.2); }}
        
        .tooltip {{
            position: fixed;
            background: rgba(0, 0, 0, 0.9);
            color: white;
            padding: 8px 12px;
            border-radius: 4px;
            font-size: 12px;
            max-width: 300px;
            z-index: 1002;
            pointer-events: none;
            word-wrap: break-word;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>OCR Overlay: {page_info.get('pdf_name', 'Unknown PDF')}</h1>
        <p><strong>Page:</strong> {page_info.get('page_num', '?')} | 
           <strong>Profile:</strong> {page_info.get('profile', 'unknown')} | 
           <strong>Generated:</strong> {page_info.get('timestamp', 'unknown')}</p>
        
        <div class="stats">
            <div class="stat">
                <strong>Total Boxes:</strong> {total_bboxes}
            </div>
            <div class="stat">
                <strong>Average Confidence:</strong> {avg_confidence:.3f}
            </div>
            <div class="stat">
                <strong>High Confidence:</strong> {high_conf_count}
            </div>
            <div class="stat">
                <strong>Low Confidence:</strong> {low_conf_count}
            </div>
            <div class="stat">
                <strong>Types:</strong> {type_summary}
            </div>
        </div>
    </div>
    
    <div class="controls">
        <h3>Display Controls</h3>
        
        <div class="control-group">
            <label for="opacity">Box Opacity:</label>
            <input type="range" id="opacity" min="0" max="100" value="80">
        </div>
        
        <div class="control-group">
            <label for="confidence-filter">Min Confidence:</label>
            <input type="range" id="confidence-filter" min="0" max="100" value="0">
            <span id="confidence-value">0.000</span>
        </div>
        
        <div class="control-group">
            <label>
                <input type="checkbox" id="show-text" checked> Show Text Boxes
            </label>
        </div>
        
        <div class="control-group">
            <label>
                <input type="checkbox" id="show-footnotes" checked> Show Footnotes
            </label>
        </div>
        
        <div class="control-group">
            <label>
                <input type="checkbox" id="show-captions" checked> Show Captions
            </label>
        </div>
        
        <div class="legend">
            <h4>Legend</h4>
            <div class="legend-item">
                <div class="legend-color high-conf"></div>
                <span>High Confidence (≥0.8)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color med-conf"></div>
                <span>Medium Confidence (0.6-0.8)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color low-conf"></div>
                <span>Low Confidence (<0.6)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color footnote"></div>
                <span>Footnotes</span>
            </div>
            <div class="legend-item">
                <div class="legend-color caption"></div>
                <span>Captions</span>
            </div>
        </div>
    </div>
    
    <div class="page-container">
        <img src="{img_data_url}" alt="Page Image" class="page-image">
        {bbox_html}
    </div>
    
    <div class="tooltip" id="tooltip" style="display: none;"></div>
    
    <script>
        // Control handlers
        const opacitySlider = document.getElementById('opacity');
        const confidenceSlider = document.getElementById('confidence-filter');
        const confidenceValue = document.getElementById('confidence-value');
        const showText = document.getElementById('show-text');
        const showFootnotes = document.getElementById('show-footnotes');
        const showCaptions = document.getElementById('show-captions');
        const tooltip = document.getElementById('tooltip');
        
        function updateDisplay() {{
            const opacity = opacitySlider.value / 100;
            const minConfidence = confidenceSlider.value / 100;
            confidenceValue.textContent = minConfidence.toFixed(3);
            
            document.querySelectorAll('.ocr-bbox').forEach(bbox => {{
                const confidence = parseFloat(bbox.dataset.confidence);
                const type = bbox.dataset.type;
                
                let visible = confidence >= minConfidence;
                
                if (type === 'text' && !showText.checked) visible = false;
                if (type === 'footnote' && !showFootnotes.checked) visible = false;
                if (type === 'caption' && !showCaptions.checked) visible = false;
                
                bbox.style.display = visible ? 'block' : 'none';
                bbox.style.opacity = opacity;
            }});
        }}
        
        // Event listeners
        opacitySlider.addEventListener('input', updateDisplay);
        confidenceSlider.addEventListener('input', updateDisplay);
        showText.addEventListener('change', updateDisplay);
        showFootnotes.addEventListener('change', updateDisplay);
        showCaptions.addEventListener('change', updateDisplay);
        
        // Tooltip functionality
        document.querySelectorAll('.ocr-bbox').forEach(bbox => {{
            bbox.addEventListener('mouseenter', (e) => {{
                const text = e.target.dataset.text;
                const confidence = e.target.dataset.confidence;
                const type = e.target.dataset.type;
                
                tooltip.innerHTML = `
                    <strong>Type:</strong> ${{type}}<br>
                    <strong>Confidence:</strong> ${{confidence}}<br>
                    <strong>Text:</strong> ${{text.substring(0, 150)}}${{text.length > 150 ? '...' : ''}}
                `;
                tooltip.style.display = 'block';
            }});
            
            bbox.addEventListener('mousemove', (e) => {{
                tooltip.style.left = e.clientX + 10 + 'px';
                tooltip.style.top = e.clientY + 10 + 'px';
            }});
            
            bbox.addEventListener('mouseleave', () => {{
                tooltip.style.display = 'none';
            }});
        }});
        
        // Initialize display
        updateDisplay();
    </script>
</body>
</html>'''
        
        return html_content
    
    def save_overlay(self, html_content: str, filename: str) -> Path:
        """Save HTML overlay to file.
        
        Args:
            html_content: HTML content string
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Saved HTML overlay to {output_path}")
        return output_path
    
    def generate_overlays_for_results(self, ocr_results: List[Dict[str, Any]], 
                                    max_files: int = 10) -> List[Path]:
        """Generate HTML overlays for a list of OCR results.
        
        Args:
            ocr_results: List of OCR result dictionaries
            max_files: Maximum number of overlays to generate
            
        Returns:
            List of paths to generated overlay files
        """
        if not PIL_AVAILABLE:
            logger.error("PIL not available, cannot generate overlays")
            return []
        
        overlay_paths = []
        
        for i, result in enumerate(ocr_results[:max_files]):
            try:
                # Extract required data from result
                page_image = result.get('page_image')  # PIL Image
                bboxes = result.get('bboxes', [])      # List of bbox dicts
                page_info = result.get('page_info', {}) # Metadata
                
                if not page_image or not bboxes:
                    logger.warning(f"Skipping result {i}: missing image or bboxes")
                    continue
                
                # Generate overlay
                html_content = self.generate_page_overlay(page_image, bboxes, page_info)
                
                # Generate filename
                pdf_name = page_info.get('pdf_name', f'document_{i}')
                page_num = page_info.get('page_num', i+1)
                safe_pdf_name = "".join(c for c in pdf_name if c.isalnum() or c in (' ', '-', '_')).strip()
                filename = f"{safe_pdf_name}_page_{page_num}.html"
                
                # Save overlay
                overlay_path = self.save_overlay(html_content, filename)
                overlay_paths.append(overlay_path)
                
            except Exception as e:
                logger.error(f"Failed to generate overlay for result {i}: {e}")
        
        logger.info(f"Generated {len(overlay_paths)} HTML overlays")
        return overlay_paths

def generate_overlays(input_dir: str, output_dir: str, max_files: int = 5):
    """Convenience function to generate overlays from input directory.
    
    Args:
        input_dir: Directory containing PDF files
        output_dir: Output directory for overlays
        max_files: Maximum number of files to process
    """
    try:
        # This would typically integrate with the main pipeline
        # For now, create a placeholder implementation
        logger.info(f"Generating overlays from {input_dir} to {output_dir} (max {max_files} files)")
        
        generator = HTMLOverlayGenerator(output_dir)
        
        # In a real implementation, this would:
        # 1. Scan input_dir for PDFs
        # 2. Run OCR on each with bbox extraction
        # 3. Generate overlays using generator.generate_overlays_for_results()
        
        logger.info("Overlay generation placeholder completed")
        
    except Exception as e:
        logger.error(f"Overlay generation failed: {e}")

if __name__ == "__main__":
    # Test/demo functionality
    logging.basicConfig(level=logging.INFO)
    
    # Generate sample overlays
    generate_overlays("data/input_pdfs", "reports/overlays", max_files=3)