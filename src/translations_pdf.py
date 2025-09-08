"""
Akkadian Translation PDF Generator
Creates formatted PDF reports showing Akkadian terms with their translations
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.colors import Color, black, darkblue, darkred, darkgreen
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.platypus.flowables import PageBreak
from reportlab.lib import colors


class TranslationsPDFGenerator:
    """Generates formatted PDF reports of Akkadian translations"""
    
    def __init__(self):
        self.page_width, self.page_height = A4
        self.margin = 0.75 * inch
        self.content_width = self.page_width - 2 * self.margin
        self.styles = getSampleStyleSheet()
        
        # Custom styles for Akkadian content
        self.akkadian_style = ParagraphStyle(
            'AkkadianText',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=darkred,
            fontName='Helvetica-Bold',
            spaceAfter=6
        )
        
        self.translation_style = ParagraphStyle(
            'TranslationText', 
            parent=self.styles['Normal'],
            fontSize=11,
            textColor=darkblue,
            leftIndent=20,
            spaceAfter=12
        )
        
        self.page_header_style = ParagraphStyle(
            'PageHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=darkgreen,
            spaceAfter=18,
            alignment=1  # Center
        )
    
    def generate_translations_pdf(
        self,
        translations_by_page: Dict[int, List[Dict[str, Any]]],
        output_path: str,
        source_pdf_name: str = "Document"
    ) -> bool:
        """
        Generate complete translations PDF report
        
        Args:
            translations_by_page: Dict mapping page numbers to translation entries
            output_path: Path for output PDF file
            source_pdf_name: Name of source PDF for header
            
        Returns:
            bool: Success status
        """
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            doc = SimpleDocTemplate(
                output_path,
                pagesize=A4,
                rightMargin=self.margin,
                leftMargin=self.margin,
                topMargin=self.margin,
                bottomMargin=self.margin
            )
            
            story = []
            
            # Title page
            story.extend(self._create_title_page(source_pdf_name, translations_by_page))
            story.append(PageBreak())
            
            # Process each page
            for page_num in sorted(translations_by_page.keys()):
                translations = translations_by_page[page_num]
                if translations:  # Only include pages with translations
                    story.extend(self._create_page_section(page_num, translations))
                    story.append(Spacer(1, 0.3 * inch))
            
            # Build the PDF
            doc.build(story)
            
            print(f"✓ Translations PDF generated: {output_path}")
            return True
            
        except Exception as e:
            print(f"✗ Error generating translations PDF: {e}")
            return False
    
    def _create_title_page(
        self, 
        source_pdf_name: str, 
        translations_by_page: Dict[int, List[Dict[str, Any]]]
    ) -> List:
        """Create title page with summary statistics"""
        story = []
        
        # Main title
        title_style = ParagraphStyle(
            'Title',
            parent=self.styles['Title'],
            fontSize=20,
            spaceAfter=30,
            alignment=1  # Center
        )
        
        story.append(Paragraph(
            f"Akkadian Translation Extractions", 
            title_style
        ))
        
        # Source document info
        subtitle_style = ParagraphStyle(
            'Subtitle',
            parent=self.styles['Normal'],
            fontSize=14,
            spaceAfter=20,
            alignment=1,
            textColor=darkblue
        )
        
        story.append(Paragraph(
            f"Source: {source_pdf_name}",
            subtitle_style
        ))
        
        # Generation timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        story.append(Paragraph(
            f"Generated: {timestamp}",
            subtitle_style
        ))
        
        story.append(Spacer(1, 0.5 * inch))
        
        # Summary statistics
        total_translations = sum(len(translations) for translations in translations_by_page.values())
        pages_with_translations = len([p for p, t in translations_by_page.items() if t])
        
        # Strategy breakdown
        strategy_counts = {
            'followed-by': 0,
            'labeled': 0, 
            'formatting-similar': 0
        }
        
        confidence_stats = {'high': 0, 'medium': 0, 'low': 0}
        language_stats = {}
        
        for translations in translations_by_page.values():
            for trans in translations:
                strategy_counts[trans.get('strategy', 'unknown')] += 1
                
                conf = trans.get('confidence', 0)
                if conf >= 0.9:
                    confidence_stats['high'] += 1
                elif conf >= 0.8:
                    confidence_stats['medium'] += 1
                else:
                    confidence_stats['low'] += 1
                
                lang = trans.get('translation_language', 'unknown')
                language_stats[lang] = language_stats.get(lang, 0) + 1
        
        # Summary table
        summary_data = [
            ['Metric', 'Count'],
            ['Total Translations Found', str(total_translations)],
            ['Pages with Translations', str(pages_with_translations)],
            ['', ''],
            ['Detection Strategy', ''],
            ['  Followed-by patterns', str(strategy_counts['followed-by'])],
            ['  Labeled translations', str(strategy_counts['labeled'])],
            ['  Formatting similarities', str(strategy_counts['formatting-similar'])],
            ['', ''],
            ['Confidence Distribution', ''],
            ['  High (≥0.9)', str(confidence_stats['high'])],
            ['  Medium (≥0.8)', str(confidence_stats['medium'])],
            ['  Low (<0.8)', str(confidence_stats['low'])]
        ]
        
        # Add language breakdown
        if language_stats:
            summary_data.extend([['', ''], ['Languages Found', '']])
            for lang, count in sorted(language_stats.items()):
                summary_data.append([f'  {lang.upper()}', str(count)])
        
        table = Table(summary_data, colWidths=[2.5 * inch, 1.5 * inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(table)
        
        return story
    
    def _create_page_section(self, page_num: int, translations: List[Dict[str, Any]]) -> List:
        """Create section for a single page's translations"""
        story = []
        
        # Page header
        story.append(Paragraph(
            f"Page {page_num} - {len(translations)} Translation(s) Found",
            self.page_header_style
        ))
        
        # Process each translation on this page
        for i, trans in enumerate(translations, 1):
            story.extend(self._format_translation_entry(trans, i))
            
            # Add separator between translations
            if i < len(translations):
                story.append(Spacer(1, 0.2 * inch))
        
        return story
    
    def _format_translation_entry(self, trans: Dict[str, Any], entry_num: int) -> List:
        """Format a single translation entry"""
        story = []
        
        # Entry header with confidence and strategy info
        header_style = ParagraphStyle(
            'EntryHeader',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.grey,
            spaceAfter=6
        )
        
        confidence = trans.get('confidence', 0)
        strategy = trans.get('strategy', 'unknown')
        
        confidence_text = f"High ({confidence:.2f})" if confidence >= 0.9 else \
                         f"Medium ({confidence:.2f})" if confidence >= 0.8 else \
                         f"Low ({confidence:.2f})"
        
        story.append(Paragraph(
            f"Translation #{entry_num} | Strategy: {strategy} | Confidence: {confidence_text}",
            header_style
        ))
        
        # Akkadian term
        akkadian_text = trans.get('akkadian_text', 'N/A')
        story.append(Paragraph(
            f"<b>Akkadian:</b> {self._escape_html(akkadian_text)}",
            self.akkadian_style
        ))
        
        # Translation
        translation_text = trans.get('translation_text', 'N/A')
        translation_lang = trans.get('translation_language', 'unknown').upper()
        
        story.append(Paragraph(
            f"<b>Translation ({translation_lang}):</b> {self._escape_html(translation_text)}",
            self.translation_style
        ))
        
        # Context information if available
        if trans.get('context'):
            context_style = ParagraphStyle(
                'Context',
                parent=self.styles['Normal'],
                fontSize=9,
                textColor=colors.grey,
                leftIndent=40,
                spaceAfter=8,
                fontName='Helvetica-Oblique'
            )
            
            story.append(Paragraph(
                f"Context: {self._escape_html(trans['context'][:200])}{'...' if len(trans['context']) > 200 else ''}",
                context_style
            ))
        
        # Bounding box information
        bbox_style = ParagraphStyle(
            'BBoxInfo',
            parent=self.styles['Normal'],
            fontSize=8,
            textColor=colors.lightslategray,
            leftIndent=40,
            spaceAfter=12
        )
        
        akkadian_bbox = trans.get('akkadian_bbox', [])
        translation_bbox = trans.get('translation_bbox', [])
        
        bbox_text = f"Position - Akkadian: {self._format_bbox(akkadian_bbox)} | Translation: {self._format_bbox(translation_bbox)}"
        story.append(Paragraph(bbox_text, bbox_style))
        
        return story
    
    def _format_bbox(self, bbox: List) -> str:
        """Format bounding box coordinates for display"""
        if len(bbox) >= 4:
            return f"({bbox[0]:.0f},{bbox[1]:.0f})-({bbox[2]:.0f},{bbox[3]:.0f})"
        return "N/A"
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML characters in text for ReportLab"""
        if not text:
            return ""
        
        return (text.replace('&', '&amp;')
                   .replace('<', '&lt;')
                   .replace('>', '&gt;')
                   .replace('"', '&quot;')
                   .replace("'", '&#x27;'))
    
    def create_empty_report(self, output_path: str, source_pdf_name: str = "Document") -> bool:
        """Create PDF report indicating no translations were found"""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            doc = SimpleDocTemplate(
                output_path,
                pagesize=A4,
                rightMargin=self.margin,
                leftMargin=self.margin, 
                topMargin=self.margin,
                bottomMargin=self.margin
            )
            
            story = []
            
            # Title
            title_style = ParagraphStyle(
                'Title',
                parent=self.styles['Title'],
                fontSize=18,
                spaceAfter=30,
                alignment=1
            )
            
            story.append(Paragraph("Akkadian Translation Extractions", title_style))
            
            # Source info
            subtitle_style = ParagraphStyle(
                'Subtitle',
                parent=self.styles['Normal'],
                fontSize=12,
                spaceAfter=20,
                alignment=1,
                textColor=darkblue
            )
            
            story.append(Paragraph(f"Source: {source_pdf_name}", subtitle_style))
            story.append(Paragraph(
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                subtitle_style
            ))
            
            story.append(Spacer(1, 1 * inch))
            
            # No results message
            no_results_style = ParagraphStyle(
                'NoResults',
                parent=self.styles['Normal'],
                fontSize=14,
                alignment=1,
                textColor=colors.darkred,
                spaceAfter=20
            )
            
            story.append(Paragraph(
                "No Akkadian translations were detected in this document.",
                no_results_style
            ))
            
            # Explanation
            explanation_style = ParagraphStyle(
                'Explanation',
                parent=self.styles['Normal'],
                fontSize=11,
                alignment=0,
                textColor=colors.grey,
                leftIndent=40,
                rightIndent=40
            )
            
            story.append(Paragraph(
                "This may occur if:<br/>"
                "• The document contains no Akkadian transliterations<br/>"
                "• Akkadian terms are present but no translations were found nearby<br/>"
                "• Translation confidence scores were below the minimum threshold (0.8)<br/>"
                "• The OCR quality was insufficient for accurate detection",
                explanation_style
            ))
            
            doc.build(story)
            
            print(f"✓ Empty translations PDF generated: {output_path}")
            return True
            
        except Exception as e:
            print(f"✗ Error generating empty translations PDF: {e}")
            return False


def generate_translations_report(
    translations_by_page: Dict[int, List[Dict[str, Any]]],
    output_path: str,
    source_pdf_name: str = "Document"
) -> bool:
    """
    Convenience function to generate translations PDF report
    
    Args:
        translations_by_page: Dict mapping page numbers to translation entries
        output_path: Path for output PDF file  
        source_pdf_name: Name of source PDF for header
        
    Returns:
        bool: Success status
    """
    generator = TranslationsPDFGenerator()
    
    # Check if we have any translations
    total_translations = sum(len(translations) for translations in translations_by_page.values())
    
    if total_translations == 0:
        return generator.create_empty_report(output_path, source_pdf_name)
    else:
        return generator.generate_translations_pdf(translations_by_page, output_path, source_pdf_name)


if __name__ == "__main__":
    # Test with sample data
    sample_translations = {
        1: [
            {
                'akkadian_text': 'lugal',
                'translation_text': 'king',
                'translation_language': 'en',
                'strategy': 'followed-by',
                'confidence': 0.95,
                'context': 'The term lugal appears in the context of Mesopotamian rulers.',
                'akkadian_bbox': [100, 200, 150, 220],
                'translation_bbox': [160, 200, 190, 220]
            }
        ],
        2: [
            {
                'akkadian_text': 'dingir',
                'translation_text': 'Gott',
                'translation_language': 'de', 
                'strategy': 'labeled',
                'confidence': 0.88,
                'context': 'Religious terminology section.',
                'akkadian_bbox': [50, 300, 100, 320],
                'translation_bbox': [110, 300, 140, 320]
            }
        ]
    }
    
    success = generate_translations_report(
        sample_translations,
        "test_translations_report.pdf",
        "Sample Akkadian Document"
    )
    
    print(f"Test report generation: {'✓ Success' if success else '✗ Failed'}")
