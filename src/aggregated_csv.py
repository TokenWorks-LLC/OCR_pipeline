"""
New CSV writer for single-row-per-page format with aggregated content.
"""
import csv
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class PageResult:
    """Single page result with aggregated content."""
    page_id: str
    page_text: str
    conf_mean: float
    lines_count: int
    warnings: str
    
    # Additional metadata (optional, not written to CSV)
    language: Optional[str] = None
    correction_stats: Optional[Dict] = None
    reading_order_stats: Optional[Dict] = None
    raw_ocr_results: Optional[List[Dict]] = None  # Store raw OCR for Akkadian extraction

class AggregatedCSVWriter:
    """CSV writer that aggregates multiple text elements into single page rows."""
    
    def __init__(self):
        self.pages_data = {}
    
    def add_page_content(self, page_id: str, text_elements: List[Dict], 
                        language: str = 'unknown',
                        correction_stats: Dict = None,
                        reading_order_stats: Dict = None):
        """
        Add content for a page from multiple text elements.
        
        Args:
            page_id: Page identifier
            text_elements: List of OCR text elements with 'text', 'conf', etc.
            language: Detected language for the page
            correction_stats: LLM correction statistics
            reading_order_stats: Reading order detection statistics
        """
        if not text_elements:
            self.pages_data[page_id] = PageResult(
                page_id=page_id,
                page_text="",
                conf_mean=0.0,
                lines_count=0,
                warnings="No text detected",
                language=language,
                correction_stats=correction_stats,
                reading_order_stats=reading_order_stats,
                raw_ocr_results=[]
            )
            return
        
        # Aggregate text content
        text_parts = []
        confidences = []
        warnings = []
        
        for element in text_elements:
            text = element.get('text', '').strip()
            if text:
                text_parts.append(text)
                
                # Extract confidence
                conf = element.get('conf', 0.0)
                if isinstance(conf, (int, float)):
                    confidences.append(float(conf))
                else:
                    confidences.append(0.5)  # Default confidence
                    warnings.append(f"Invalid confidence for text: {text[:20]}")
        
        # Join text with appropriate spacing
        page_text = self._join_text_intelligently(text_parts)
        
        # Calculate statistics
        conf_mean = sum(confidences) / len(confidences) if confidences else 0.0
        lines_count = len(text_parts)
        
        # Additional warnings
        if conf_mean < 0.5:
            warnings.append("Low average confidence")
        
        if lines_count < 5:
            warnings.append("Very few text elements detected")
        
        warnings_str = "; ".join(warnings) if warnings else ""
        
        self.pages_data[page_id] = PageResult(
            page_id=page_id,
            page_text=page_text,
            conf_mean=round(conf_mean, 3),
            lines_count=lines_count,
            warnings=warnings_str,
            language=language,
            correction_stats=correction_stats,
            reading_order_stats=reading_order_stats,
            raw_ocr_results=text_elements.copy()  # Store raw OCR results
        )
    
    def _join_text_intelligently(self, text_parts: List[str]) -> str:
        """
        Join text parts with intelligent spacing and punctuation handling.
        """
        if not text_parts:
            return ""
        
        if len(text_parts) == 1:
            return text_parts[0]
        
        result = []
        
        for i, part in enumerate(text_parts):
            if i == 0:
                result.append(part)
                continue
            
            prev_part = text_parts[i-1]
            
            # Determine spacing based on punctuation and content
            needs_space = True
            
            # No space after opening punctuation
            if prev_part.endswith(('(', '[', '"', "'")):
                needs_space = False
            
            # No space before closing punctuation
            elif part.startswith((')', ']', '.', ',', ';', ':', '!', '?', '"', "'")):
                needs_space = False
            
            # No space for hyphenated words
            elif prev_part.endswith('-') or part.startswith('-'):
                needs_space = False
            
            # Add appropriate spacing
            if needs_space:
                result.append(' ')
            
            result.append(part)
        
        return ''.join(result)
    
    def write_csv(self, filepath: str, include_metadata: bool = False):
        """
        Write aggregated page data to CSV file.
        
        Args:
            filepath: Output CSV file path
            include_metadata: Include additional metadata columns
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.pages_data:
            logger.warning(f"No page data to write to {filepath}")
            return
        
        # Define CSV columns
        fieldnames = ['page_id', 'page_text', 'conf_mean', 'lines_count', 'warnings']
        
        if include_metadata:
            fieldnames.extend(['language', 'correction_stats', 'reading_order_stats'])
        
        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
                
                # Write header
                writer.writeheader()
                
                # Write page data sorted by page_id
                sorted_pages = sorted(self.pages_data.items(), key=lambda x: self._natural_sort_key(x[0]))
                
                for page_id, page_data in sorted_pages:
                    row = {
                        'page_id': page_data.page_id,
                        'page_text': page_data.page_text,
                        'conf_mean': page_data.conf_mean,
                        'lines_count': page_data.lines_count,
                        'warnings': page_data.warnings
                    }
                    
                    if include_metadata:
                        row.update({
                            'language': page_data.language,
                            'correction_stats': json.dumps(page_data.correction_stats) if page_data.correction_stats else "",
                            'reading_order_stats': json.dumps(page_data.reading_order_stats) if page_data.reading_order_stats else ""
                        })
                    
                    writer.writerow(row)
            
            logger.info(f"Wrote {len(self.pages_data)} pages to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to write CSV {filepath}: {e}")
            raise
    
    def _natural_sort_key(self, text: str):
        """Natural sorting key for page IDs (handles numbers correctly)."""
        import re
        
        parts = re.split(r'(\d+)', text)
        result = []
        
        for part in parts:
            if part.isdigit():
                result.append(int(part))
            else:
                result.append(part)
        
        return result
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for all processed pages."""
        if not self.pages_data:
            return {}
        
        total_pages = len(self.pages_data)
        total_lines = sum(page.lines_count for page in self.pages_data.values())
        total_chars = sum(len(page.page_text) for page in self.pages_data.values())
        
        confidences = [page.conf_mean for page in self.pages_data.values()]
        avg_confidence = sum(confidences) / len(confidences)
        
        pages_with_warnings = sum(1 for page in self.pages_data.values() if page.warnings)
        
        # Language distribution
        languages = {}
        for page in self.pages_data.values():
            if page.language:
                languages[page.language] = languages.get(page.language, 0) + 1
        
        return {
            'total_pages': total_pages,
            'total_text_lines': total_lines,
            'total_characters': total_chars,
            'avg_lines_per_page': total_lines / total_pages,
            'avg_chars_per_page': total_chars / total_pages,
            'avg_confidence': round(avg_confidence, 3),
            'pages_with_warnings': pages_with_warnings,
            'warning_rate': pages_with_warnings / total_pages,
            'language_distribution': languages
        }
    
    def clear(self):
        """Clear all page data."""
        self.pages_data.clear()


def write_aggregated_csv(page_results: Dict[str, List[Dict]], filepath: str, 
                        include_metadata: bool = False) -> Dict[str, Any]:
    """
    Convenience function to write aggregated CSV from page results.
    
    Args:
        page_results: Dict mapping page_id to list of text elements
        filepath: Output CSV file path
        include_metadata: Include additional metadata columns
    
    Returns:
        Summary statistics
    """
    writer = AggregatedCSVWriter()
    
    for page_id, text_elements in page_results.items():
        writer.add_page_content(page_id, text_elements)
    
    writer.write_csv(filepath, include_metadata)
    
    return writer.get_summary_stats()
