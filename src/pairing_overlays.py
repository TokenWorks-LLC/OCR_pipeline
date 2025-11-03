#!/usr/bin/env python3
"""
Translation Pairing Overlay Renderer (Prompt 4)

Generates visual overlays showing:
- Blue boxes: Akkadian blocks
- Green boxes: Translation blocks  
- Red arrows: Pairing connections
- Score labels

Outputs:
- HTML files with embedded images
- JPEG/PNG overlays
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional
from pathlib import Path
import base64
from io import BytesIO

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logging.warning("PIL not available - overlays will not render. Install: pip install Pillow")

from translation_pairing import TranslationPair


logger = logging.getLogger(__name__)


@dataclass
class OverlayConfig:
    """Configuration for overlay rendering."""
    # Colors (RGBA)
    color_akkadian: Tuple[int, int, int, int] = (0, 0, 255, 80)  # Blue, semi-transparent
    color_translation: Tuple[int, int, int, int] = (0, 255, 0, 80)  # Green, semi-transparent
    color_arrow: Tuple[int, int, int] = (255, 0, 0)  # Red
    color_text: Tuple[int, int, int] = (0, 0, 0)  # Black
    
    # Line widths
    box_width: int = 3
    arrow_width: int = 2
    
    # Font sizes
    font_size_label: int = 12
    font_size_score: int = 10
    
    # Arrow style
    arrow_head_size: int = 10
    
    # Image quality
    image_quality: int = 95  # JPEG quality (1-100)


class PairingOverlayRenderer:
    """
    Renders visual overlays of translation pairings on page images.
    """
    
    def __init__(self, config: Optional[OverlayConfig] = None):
        """
        Initialize renderer.
        
        Args:
            config: Overlay configuration
        """
        if not PIL_AVAILABLE:
            raise ImportError(
                "PIL/Pillow is required for overlay rendering. "
                "Install with: pip install Pillow"
            )
        
        self.config = config or OverlayConfig()
        
        # Try to load a default font
        try:
            self.font_label = ImageFont.truetype("arial.ttf", self.config.font_size_label)
            self.font_score = ImageFont.truetype("arial.ttf", self.config.font_size_score)
        except:
            # Fallback to default font
            self.font_label = ImageFont.load_default()
            self.font_score = ImageFont.load_default()
        
        logger.info("PairingOverlayRenderer initialized")
    
    def render_page_overlay(
        self,
        page_image: Image.Image,
        pairs: List[TranslationPair],
        output_path: Optional[Path] = None
    ) -> Image.Image:
        """
        Render overlay on a page image.
        
        Args:
            page_image: PIL Image of the page
            pairs: List of translation pairs to visualize
            output_path: Optional path to save rendered image
            
        Returns:
            PIL Image with overlay
        """
        # Create a copy to draw on
        img = page_image.copy().convert('RGBA')
        
        # Create overlay layer
        overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Draw bboxes and arrows
        for pair in pairs:
            # Draw Akkadian bbox (blue)
            self._draw_bbox(
                draw,
                pair.akk_bbox,
                self.config.color_akkadian,
                label=f"AKK: {pair.akk_block_id}",
                text_preview=pair.akk_text[:30] + "..." if len(pair.akk_text) > 30 else pair.akk_text
            )
            
            # Draw translation bbox (green)
            self._draw_bbox(
                draw,
                pair.trans_bbox,
                self.config.color_translation,
                label=f"TRANS ({pair.trans_lang}): {pair.trans_block_id}",
                text_preview=pair.trans_text[:40] + "..." if len(pair.trans_text) > 40 else pair.trans_text
            )
            
            # Draw arrow connecting them (red)
            self._draw_arrow(
                draw,
                pair.akk_bbox,
                pair.trans_bbox,
                score=pair.score
            )
        
        # Composite overlay onto image
        result = Image.alpha_composite(img, overlay)
        result = result.convert('RGB')  # Convert back to RGB for saving
        
        # Save if output path provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result.save(output_path, quality=self.config.image_quality)
            logger.info(f"Saved overlay to {output_path}")
        
        return result
    
    def _draw_bbox(
        self,
        draw: ImageDraw.ImageDraw,
        bbox: Tuple[int, int, int, int],
        color: Tuple[int, int, int, int],
        label: str = "",
        text_preview: str = ""
    ):
        """Draw a bounding box with label."""
        x, y, w, h = bbox
        
        # Draw filled rectangle
        draw.rectangle(
            [(x, y), (x + w, y + h)],
            fill=color,
            outline=(color[0], color[1], color[2], 255),  # Solid outline
            width=self.config.box_width
        )
        
        # Draw label above box
        if label:
            label_y = max(0, y - 15)  # Position above bbox
            draw.text(
                (x, label_y),
                label,
                fill=self.config.color_text,
                font=self.font_label
            )
        
        # Draw text preview inside box (if space permits)
        if text_preview and h > 20:
            preview_y = y + 5
            draw.text(
                (x + 5, preview_y),
                text_preview,
                fill=self.config.color_text,
                font=self.font_score
            )
    
    def _draw_arrow(
        self,
        draw: ImageDraw.ImageDraw,
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int],
        score: float
    ):
        """Draw arrow from bbox1 center to bbox2 center with score label."""
        # Calculate centers
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        center1_x = x1 + w1 // 2
        center1_y = y1 + h1 // 2
        center2_x = x2 + w2 // 2
        center2_y = y2 + h2 // 2
        
        # Draw line
        draw.line(
            [(center1_x, center1_y), (center2_x, center2_y)],
            fill=self.config.color_arrow,
            width=self.config.arrow_width
        )
        
        # Draw arrowhead
        self._draw_arrowhead(
            draw,
            (center1_x, center1_y),
            (center2_x, center2_y)
        )
        
        # Draw score label at midpoint
        mid_x = (center1_x + center2_x) // 2
        mid_y = (center1_y + center2_y) // 2
        
        score_text = f"{score:.2f}"
        draw.text(
            (mid_x + 5, mid_y - 10),
            score_text,
            fill=self.config.color_arrow,
            font=self.font_score
        )
    
    def _draw_arrowhead(
        self,
        draw: ImageDraw.ImageDraw,
        start: Tuple[int, int],
        end: Tuple[int, int]
    ):
        """Draw arrowhead at end point."""
        import math
        
        x1, y1 = start
        x2, y2 = end
        
        # Calculate angle
        angle = math.atan2(y2 - y1, x2 - x1)
        
        # Arrowhead points
        size = self.config.arrow_head_size
        angle1 = angle + math.pi * 5 / 6
        angle2 = angle - math.pi * 5 / 6
        
        p1 = (
            x2 + size * math.cos(angle1),
            y2 + size * math.sin(angle1)
        )
        p2 = (
            x2 + size * math.cos(angle2),
            y2 + size * math.sin(angle2)
        )
        
        # Draw triangle
        draw.polygon(
            [end, p1, p2],
            fill=self.config.color_arrow,
            outline=self.config.color_arrow
        )
    
    def generate_html(
        self,
        page_images_with_overlays: List[Tuple[int, Image.Image]],
        output_path: Path,
        title: str = "Translation Pairing Overlays"
    ):
        """
        Generate HTML file with embedded overlay images.
        
        Args:
            page_images_with_overlays: List of (page_num, PIL Image) tuples
            output_path: Output HTML file path
            title: HTML page title
        """
        html_parts = []
        
        # HTML header
        html_parts.append(f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
        }}
        .page {{
            margin-bottom: 40px;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .page h2 {{
            color: #666;
            margin-top: 0;
        }}
        .page img {{
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        .legend {{
            margin: 20px 0;
            padding: 15px;
            background: #fff;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .legend-item {{
            display: inline-block;
            margin-right: 20px;
        }}
        .legend-box {{
            display: inline-block;
            width: 30px;
            height: 20px;
            border: 2px solid;
            margin-right: 5px;
            vertical-align: middle;
        }}
        .akk-box {{
            background-color: rgba(0, 0, 255, 0.3);
            border-color: blue;
        }}
        .trans-box {{
            background-color: rgba(0, 255, 0, 0.3);
            border-color: green;
        }}
        .arrow {{
            color: red;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    
    <div class="legend">
        <div class="legend-item">
            <span class="legend-box akk-box"></span>
            Akkadian Block
        </div>
        <div class="legend-item">
            <span class="legend-box trans-box"></span>
            Translation Block
        </div>
        <div class="legend-item">
            <span class="arrow">→</span>
            Pairing Arrow (with score)
        </div>
    </div>
""")
        
        # Add each page
        for page_num, img in page_images_with_overlays:
            # Convert image to base64
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=self.config.image_quality)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            html_parts.append(f"""
    <div class="page">
        <h2>Page {page_num}</h2>
        <img src="data:image/jpeg;base64,{img_str}" alt="Page {page_num} with overlays">
    </div>
""")
        
        # HTML footer
        html_parts.append("""
</body>
</html>
""")
        
        # Write HTML file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(''.join(html_parts))
        
        logger.info(f"Generated HTML overlay: {output_path}")


# ============================================================================
# Test/Demo
# ============================================================================

if __name__ == '__main__':
    import sys
    from pathlib import Path
    
    logging.basicConfig(level=logging.INFO)
    
    if not PIL_AVAILABLE:
        print("ERROR: PIL/Pillow required. Install with: pip install Pillow")
        sys.exit(1)
    
    # Create a test image
    img = Image.new('RGB', (1000, 1400), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw some fake text regions
    draw.rectangle([(100, 100), (400, 140)], outline='gray', width=1)
    draw.text((110, 110), "sharru dUTU erebu maru", fill='black')
    
    draw.rectangle([(100, 180), (500, 240)], outline='gray', width=1)
    draw.text((110, 190), "Konig Shamash tritt ein, Sohn (Ubersetzung)", fill='black')
    
    draw.rectangle([(100, 300), (420, 340)], outline='gray', width=1)
    draw.text((110, 310), "harranu tuppu sabu", fill='black')
    
    draw.rectangle([(100, 380), (480, 430)], outline='gray', width=1)
    draw.text((110, 390), "Weg, Tafel, Truppen (traduction)", fill='black')
    
    # Create mock pairs
    sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
    from translation_pairing import TranslationPair
    
    pairs = [
        TranslationPair(
            pdf_id="test_doc",
            page=1,
            akk_block_id="akk_1",
            akk_text="sharru dUTU erebu maru",
            akk_bbox=(100, 100, 300, 40),
            akk_column=0,
            trans_block_id="trans_1",
            trans_text="Konig Shamash tritt ein, Sohn (Ubersetzung)",
            trans_lang="de",
            trans_bbox=(100, 180, 400, 60),
            trans_column=0,
            score=0.95,
            distance_px=80.0,
            same_column=True,
            has_marker=True,
            reading_order_ok=True
        ),
        TranslationPair(
            pdf_id="test_doc",
            page=1,
            akk_block_id="akk_2",
            akk_text="harranu tuppu sabu",
            akk_bbox=(100, 300, 320, 40),
            akk_column=0,
            trans_block_id="trans_2",
            trans_text="Weg, Tafel, Truppen (traduction)",
            trans_lang="fr",
            trans_bbox=(100, 380, 380, 50),
            trans_column=0,
            score=0.88,
            distance_px=80.0,
            same_column=True,
            has_marker=True,
            reading_order_ok=True
        )
    ]
    
    # Render overlay
    renderer = PairingOverlayRenderer()
    overlay_img = renderer.render_page_overlay(
        img,
        pairs,
        output_path=Path("test_pairing_overlay.jpg")
    )
    
    print("\n=== Pairing Overlay Rendering Test ===")
    print(f"Rendered {len(pairs)} pairs on test image")
    print(f"Saved to: test_pairing_overlay.jpg")
    
    # Generate HTML
    renderer.generate_html(
        [(1, overlay_img)],
        Path("test_pairing_overlay.html"),
        title="Test Translation Pairing Overlays"
    )
    
    print(f"Generated HTML: test_pairing_overlay.html")
    print("\nOpen test_pairing_overlay.html in a web browser to view the result!")
