"""
Comprehensive OCR pipeline with reading order, LLM correction, and aggregated CSV output.
"""
import os
import sys
import logging
import json
import time
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict

import cv2
import numpy as np
import fitz

# Resource monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from aggregated_csv import AggregatedCSVWriter, PageResult
from llm_correction import LLMCorrector, CorrectionResult
from llm_correction_v3 import initialize_llm_v3, get_llm_v3_corrector, cleanup_llm_v3
from akkadian_extract import AkkadianExtractor
from translations_pdf import generate_translations_report

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """Configuration for the comprehensive OCR pipeline."""
    # LLM settings
    llm_provider: str = "ollama"  # 'ollama', 'llamacpp', or 'none'
    llm_model: str = "llama3.2:latest"  # Updated to use available model
    llm_base_url: str = "http://localhost:11434"
    llm_timeout: int = 30
    
    # OCR settings
    dpi: int = 300
    paddle_use_gpu: bool = False
    
    # Processing settings
    enable_reading_order: bool = True
    enable_llm_correction: bool = True
    enable_llm_v3: bool = False  # Enable LLM-in-the-loop V3 for evaluation
    max_concurrent_corrections: int = 3
    
    # Akkadian translation extraction
    enable_akkadian_extraction: bool = True
    generate_translations_pdf: bool = True
    akkadian_confidence_threshold: float = 0.8
    
    # Output settings
    output_csv_metadata: bool = True
    create_html_overlay: bool = True
    create_visualization: bool = True

class ComprehensivePipeline:
    """
    Advanced OCR pipeline with reading order detection, LLM correction, 
    and comprehensive reporting.
    """
    
    def __init__(self, config: PipelineConfig = None):
        """Initialize the pipeline with configuration."""
        self.config = config or PipelineConfig()
        self.paddle_ocr = None
        self.llm_corrector = None
        self.llm_v3_corrector = None
        self.akkadian_extractor = None
        self.csv_writer = AggregatedCSVWriter()
        self.processing_stats = {}
        
        self._initialize_engines()

    def cleanup(self):
        """Cleanup resources including LLM V3 system."""
        try:
            cleanup_llm_v3()
            logger.info("LLM V3 system cleaned up")
        except Exception as e:
            logger.warning(f"Error during LLM V3 cleanup: {e}")
    
    def _initialize_engines(self):
        """Initialize OCR and LLM engines."""
        # Initialize PaddleOCR
        try:
            from paddleocr import PaddleOCR
            self.paddle_ocr = PaddleOCR(
                use_textline_orientation=True, 
                lang='en'
            )
            logger.info("PaddleOCR initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            raise
        
        # Initialize LLM corrector (legacy mode)
        enable_llm_v3 = getattr(self.config, 'enable_llm_v3', False)
        if self.config.enable_llm_correction and not enable_llm_v3 and self.config.llm_provider != 'none':
            try:
                self.llm_corrector = LLMCorrector(
                    provider=self.config.llm_provider,
                    model=self.config.llm_model,
                    base_url=self.config.llm_base_url,
                    timeout=self.config.llm_timeout,
                    max_workers=self.config.max_concurrent_corrections
                )
                logger.info(f"LLM corrector initialized: {self.config.llm_provider}")
            except Exception as e:
                logger.warning(f"LLM corrector initialization failed: {e}")
                self.llm_corrector = None
        elif enable_llm_v3:
            # Initialize LLM V3 system for evaluation mode
            try:
                # Initialize LLM V3 system with config values
                llm_v3_section = getattr(self.config, 'llm_v3', {})
                v3_config = {
                    'llm': {
                        'llm_enabled': True,
                        'kill_switch': getattr(self.config, 'kill_switch', False),
                        'model_id': self.config.llm_model,
                        'prompt_version': getattr(self.config, 'prompt_version', 'v3_strict_typo_only'),
                        'cache_enabled': getattr(self.config, 'cache_enabled', True),
                        'max_workers': self.config.max_concurrent_corrections,
                        'timeout': self.config.llm_timeout,
                        'enable_telemetry': getattr(self.config, 'enable_telemetry', True)
                    }
                }

                if initialize_llm_v3(v3_config):
                    self.llm_v3_corrector = get_llm_v3_corrector()
                    logger.info("LLM V3 corrector initialized for evaluation mode")
                else:
                    logger.error("Failed to initialize LLM V3 corrector")
                    self.llm_v3_corrector = None
            except Exception as e:
                logger.error(f"LLM V3 initialization failed: {e}")
                self.llm_v3_corrector = None
        else:
            logger.info("LLM correction disabled")
        
        # Initialize Akkadian extractor
        if self.config.enable_akkadian_extraction:
            try:
                self.akkadian_extractor = AkkadianExtractor(
                    min_akk_conf=self.config.akkadian_confidence_threshold,
                    min_trans_conf=self.config.akkadian_confidence_threshold
                )
                logger.info("Akkadian extractor initialized")
            except Exception as e:
                logger.warning(f"Akkadian extractor initialization failed: {e}")
                self.akkadian_extractor = None
        else:
            logger.info("Akkadian extraction disabled")
    
    def _extract_text_from_paddle_result(self, paddle_result):
        """Extract text data from PaddleOCR result format."""
        try:
            if not paddle_result or len(paddle_result) == 0:
                return []
            
            result_obj = paddle_result[0]
            
            # Handle different PaddleOCR result formats
            if isinstance(result_obj, dict):
                # New format with dict
                texts = result_obj.get('rec_texts', [])
                scores = result_obj.get('rec_scores', [])
                polys = result_obj.get('rec_polys', [])
            else:
                # Old format with list of tuples
                texts = []
                scores = []
                polys = []
                for item in paddle_result:
                    if len(item) >= 2:
                        bbox, (text, conf) = item[0], item[1]
                        texts.append(text)
                        scores.append(conf)
                        polys.append(bbox)
            
            logger.debug(f"Extracted {len(texts)} text elements from PaddleOCR")
            
            ocr_results = []
            for i, text in enumerate(texts):
                if not text or not text.strip():
                    continue
                
                # Get confidence score
                conf = scores[i] if i < len(scores) else 0.5
                
                # Get bounding box from polygon
                if i < len(polys) and polys[i] is not None:
                    poly = polys[i]
                    if hasattr(poly, 'tolist'):
                        poly = poly.tolist()
                    
                    # Calculate bounding box from polygon
                    x_coords = [point[0] for point in poly]
                    y_coords = [point[1] for point in poly]
                    x, y = int(min(x_coords)), int(min(y_coords))
                    w, h = int(max(x_coords) - min(x_coords)), int(max(y_coords) - min(y_coords))
                else:
                    # Default bbox if no polygon data
                    x, y, w, h = 0, i * 25, 200, 20
                
                # Clean and normalize text
                cleaned_text = text.strip()
                if cleaned_text:
                    ocr_results.append({
                        'text': cleaned_text,
                        'bbox': (x, y, w, h),
                        'conf': float(conf),
                        'engine': 'paddle'
                    })
            
            return ocr_results
            
        except Exception as e:
            logger.error(f"Error extracting text from PaddleOCR result: {e}")
            return []
    
    def _apply_simple_reading_order(self, text_elements: List[Dict], page_width: int = 800) -> List[Dict]:
        """
        Apply simple reading order without sklearn dependency.
        Handles basic column detection and top-to-bottom, left-to-right ordering.
        """
        if not text_elements:
            return []
        
        # Simple column detection based on x-coordinate clustering
        x_positions = [elem['bbox'][0] + elem['bbox'][2]/2 for elem in text_elements]
        x_positions.sort()
        
        # Find potential column boundaries
        gaps = []
        for i in range(1, len(x_positions)):
            gap = x_positions[i] - x_positions[i-1]
            if gap > 50:  # Minimum gap to consider column boundary
                gaps.append((gap, x_positions[i-1] + gap/2))
        
        # Use largest gap as column separator if significant
        column_threshold = None
        if gaps:
            largest_gap = max(gaps, key=lambda x: x[0])
            if largest_gap[0] > page_width * 0.1:  # At least 10% of page width
                column_threshold = largest_gap[1]
                logger.debug(f"Detected column boundary at x={column_threshold:.0f}")
        
        if column_threshold:
            # Split into columns
            left_column = [elem for elem in text_elements 
                          if elem['bbox'][0] + elem['bbox'][2]/2 < column_threshold]
            right_column = [elem for elem in text_elements 
                           if elem['bbox'][0] + elem['bbox'][2]/2 >= column_threshold]
            
            # Sort each column by reading order (y first, then x)
            left_sorted = sorted(left_column, key=lambda e: (e['bbox'][1], e['bbox'][0]))
            right_sorted = sorted(right_column, key=lambda e: (e['bbox'][1], e['bbox'][0]))
            
            # Combine columns (left first, then right)
            ordered_elements = left_sorted + right_sorted
            
            logger.debug(f"Applied column ordering: {len(left_sorted)} left, {len(right_sorted)} right")
            
        else:
            # Single column - sort by reading order
            ordered_elements = sorted(text_elements, key=lambda e: (e['bbox'][1], e['bbox'][0]))
            logger.debug("Applied single-column reading order")
        
        return ordered_elements
    
    def _detect_page_language(self, text_elements: List[Dict]) -> str:
        """Detect the primary language of the page."""
        all_text = ' '.join([elem['text'] for elem in text_elements])
        
        if not all_text:
            return 'unknown'
        
        # Simple language detection based on character patterns
        text_lower = all_text.lower()
        
        # Count language-specific characters
        char_counts = {
            'turkish': len([c for c in all_text if c in 'çğıöşüÇĞIİÖŞÜ']),
            'german': len([c for c in all_text if c in 'äöüßÄÖÜ']),
            'french': len([c for c in all_text if c in 'àâäéèêëïîôùûüÿçÀÂÄÉÈÊËÏÎÔÙÛÜŸÇ']),
            'italian': len([c for c in all_text if c in 'àèéìíîòóùúÀÈÉÌÍÎÒÓÙÚ'])
        }
        
        # Find language with most characteristic characters
        max_chars = max(char_counts.values())
        if max_chars > len(all_text) * 0.02:  # At least 2% special characters
            for lang, count in char_counts.items():
                if count == max_chars:
                    return lang
        
        return 'english'
    
    def _count_words_and_tokens(self, text_elements: List[Dict], detected_language: str) -> Dict[str, int]:
        """
        Count words and tokens from text elements.
        
        Args:
            text_elements: List of text elements with 'text' field
            detected_language: Detected language code
            
        Returns:
            Dictionary with word_count, token_count, and text_elements_count
        """
        if not text_elements:
            return {'word_count': 0, 'token_count': 0, 'text_elements_count': 0}
        
        # Count text elements (used as token count for Akkadian/limited languages)
        text_elements_count = len(text_elements)
        
        # Count words for modern languages only
        word_count = 0
        if detected_language in ['english', 'russian', 'arabic', 'chinese', 'german', 'french', 'turkish']:
            all_text = ' '.join([elem.get('text', '') for elem in text_elements])
            # Simple word counting - split on whitespace and filter empty strings
            words = [word for word in re.split(r'\s+', all_text) if word.strip()]
            word_count = len(words)
        
        # Use text_elements as token count (as requested)
        token_count = text_elements_count
        
        return {
            'word_count': word_count,
            'token_count': token_count,
            'text_elements_count': text_elements_count
        }
    
    def _monitor_resources(self) -> Dict[str, Any]:
        """
        Monitor system resources during processing.
        
        Returns:
            Dictionary with resource usage metrics
        """
        if not PSUTIL_AVAILABLE:
            return {
                'cpu_percent': 0.0,
                'memory_mb': 0.0,
                'timestamp': time.time(),
                'available': False
            }
        
        try:
            process = psutil.Process()
            # CPU percentage with interval parameter for accurate measurement
            cpu_percent = process.cpu_percent(interval=0.1)
            
            return {
                'cpu_percent': cpu_percent,
                'memory_mb': process.memory_info().rss / 1024 / 1024,
                'timestamp': time.time(),
                'available': True
            }
        except Exception as e:
            logger.warning(f"Resource monitoring failed: {e}")
            return {
                'cpu_percent': 0.0,
                'memory_mb': 0.0,
                'timestamp': time.time(),
                'available': False
            }
    
    def process_single_page(self, pdf_path: str, page_num: int, output_dir: str, 
                          base_filename: str) -> Tuple[Optional[PageResult], Dict[str, Any]]:
        """
        Process a single PDF page with comprehensive pipeline.
        
        Returns:
            Tuple of (PageResult, processing_stats)
        """
        start_time = time.time()
        
        try:
            # Load PDF page
            doc = fitz.open(pdf_path)
            page = doc.load_page(page_num - 1)  # fitz uses 0-based indexing
            
            # Render page to image
            mat = fitz.Matrix(self.config.dpi / 72, self.config.dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            img_array = np.frombuffer(pix.tobytes("ppm"), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            page_width = img.shape[1]
            doc.close()
            
            # Run OCR
            ocr_start = time.time()
            paddle_result = self.paddle_ocr.predict(img)
            ocr_time = time.time() - ocr_start
            
            # Extract text elements
            text_elements = self._extract_text_from_paddle_result(paddle_result)
            
            if not text_elements:
                logger.warning(f"No text detected on page {page_num}")
                # Monitor resources even for failed pages
                resource_usage = self._monitor_resources()
                return None, {
                    'page_num': page_num,
                    'processing_time': time.time() - start_time,
                    'ocr_time': ocr_time,
                    'text_elements': 0,
                    'language': 'unknown',
                    'corrections_made': 0,
                    'word_count': 0,
                    'token_count': 0,
                    'text_elements_count': 0,
                    'resource_usage': resource_usage
                }
            
            # Apply reading order
            reading_order_start = time.time()
            if self.config.enable_reading_order:
                ordered_elements = self._apply_simple_reading_order(text_elements, page_width)
            else:
                ordered_elements = sorted(text_elements, key=lambda e: (e['bbox'][1], e['bbox'][0]))
            reading_order_time = time.time() - reading_order_start
            
            # Detect language
            detected_language = self._detect_page_language(ordered_elements)

            # Apply LLM corrections
            correction_start = time.time()
            corrections_made = 0
            correction_stats = None

            if self.llm_v3_corrector and ordered_elements:
                # Use LLM V3 system for evaluation mode
                spans = []
                for i, elem in enumerate(ordered_elements):
                    spans.append({
                        'text': elem['text'],
                        'confidence': elem.get('confidence', 0.8),  # Default confidence if not available
                        'bbox': elem.get('bbox', []),
                        'id': f"span_{i}"
                    })

                corrected_spans, v3_stats = self.llm_v3_corrector.correct_spans(spans)

                # Apply corrections to elements
                for i, (elem, corrected_span) in enumerate(zip(ordered_elements, corrected_spans)):
                    if corrected_span['text'] != corrected_span.get('original_text', ''):
                        ordered_elements[i]['text'] = corrected_span['text']
                        ordered_elements[i]['original_text'] = corrected_span.get('original_text', '')
                        ordered_elements[i]['corrections'] = corrected_span.get('corrections', [])
                        ordered_elements[i]['llm_language'] = corrected_span.get('llm_language', 'unknown')
                        ordered_elements[i]['llm_processing_time'] = corrected_span.get('llm_processing_time', 0.0)
                        ordered_elements[i]['cache_hit'] = corrected_span.get('cache_hit', False)
                        corrections_made += 1

                correction_stats = v3_stats

            elif self.llm_corrector and ordered_elements:
                # Legacy LLM correction
                texts_to_correct = [elem['text'] for elem in ordered_elements]
                correction_results = self.llm_corrector.correct_multiple_texts(
                    texts_to_correct, detected_language
                )

                # Apply corrections to elements
                for i, correction in enumerate(correction_results):
                    if i < len(ordered_elements) and correction.corrected_text != correction.original_text:
                        ordered_elements[i]['text'] = correction.corrected_text
                        ordered_elements[i]['original_text'] = correction.original_text
                        ordered_elements[i]['corrections'] = correction.corrections_made
                        corrections_made += 1

                correction_stats = self.llm_corrector.get_correction_stats()

            correction_time = time.time() - correction_start

            # Count words and tokens
            word_token_counts = self._count_words_and_tokens(ordered_elements, detected_language)
            
            # Monitor resources
            resource_usage = self._monitor_resources()
            
            # Create page result
            page_id = f"page_{page_num:03d}"
            
            # Add to CSV writer
            self.csv_writer.add_page_content(
                page_id=page_id,
                text_elements=ordered_elements,
                language=detected_language,
                correction_stats=correction_stats,
                reading_order_stats={
                    'processing_time': reading_order_time,
                    'elements_processed': len(ordered_elements),
                    'columns_detected': 2 if len(set(elem['bbox'][0] for elem in ordered_elements)) > page_width * 0.3 else 1
                }
            )
            
            # Get page result from CSV writer
            page_result = self.csv_writer.pages_data.get(page_id)
            
            # Create processing stats
            processing_stats = {
                'page_num': page_num,
                'processing_time': time.time() - start_time,
                'ocr_time': ocr_time,
                'reading_order_time': reading_order_time,
                'correction_time': correction_time,
                'text_elements': len(ordered_elements),
                'language': detected_language,
                'corrections_made': corrections_made,
                'avg_confidence': page_result.conf_mean if page_result else 0.0,
                # New metrics for cost of compute
                'word_count': word_token_counts['word_count'],
                'token_count': word_token_counts['token_count'],
                'text_elements_count': word_token_counts['text_elements_count'],
                'resource_usage': resource_usage
            }
            
            logger.info(f"Processed page {page_num}: {len(ordered_elements)} elements, "
                       f"lang={detected_language}, corrections={corrections_made}")
            
            return page_result, processing_stats
            
        except Exception as e:
            logger.error(f"Error processing page {page_num}: {e}")
            import traceback
            traceback.print_exc()
            
            # Monitor resources even for error cases
            resource_usage = self._monitor_resources()
            return None, {
                'page_num': page_num,
                'processing_time': time.time() - start_time,
                'error': str(e),
                'word_count': 0,
                'token_count': 0,
                'text_elements_count': 0,
                'resource_usage': resource_usage
            }
    
    def process_pdf(self, pdf_path: str, output_dir: str, 
                   start_page: int = 1, end_page: Optional[int] = None) -> Dict[str, Any]:
        """
        Process entire PDF with comprehensive pipeline.
        
        Returns:
            Processing summary with statistics
        """
        start_time = time.time()
        
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        base_filename = pdf_path.stem
        
        logger.info(f"Starting comprehensive OCR pipeline on {pdf_path}")
        logger.info(f"Config: LLM={self.config.llm_provider}, Reading order={self.config.enable_reading_order}")
        
        # Get PDF info
        try:
            doc = fitz.open(pdf_path)
            total_pages = doc.page_count
            doc.close()
        except Exception as e:
            logger.error(f"Cannot open PDF: {e}")
            return {'error': str(e)}
        
        # Determine page range
        if end_page is None:
            end_page = total_pages
        end_page = min(end_page, total_pages)
        
        if start_page > end_page:
            logger.error(f"Invalid page range: {start_page}-{end_page}")
            return {'error': 'Invalid page range'}
        
        logger.info(f"Processing pages {start_page}-{end_page} of {total_pages}")
        
        # Clear CSV writer for new document
        self.csv_writer.clear()
        
        # Process pages
        page_results = []
        processing_stats = []
        akkadian_translations_by_page = {}  # Store Akkadian translations for PDF report
        
        for page_num in range(start_page, end_page + 1):
            logger.info(f"Processing page {page_num}/{end_page}")
            
            page_result, stats = self.process_single_page(
                str(pdf_path), page_num, str(output_dir), base_filename
            )
            
            if page_result:
                page_results.append(page_result)
                
                # Extract Akkadian translations if enabled
                if self.akkadian_extractor and self.config.enable_akkadian_extraction:
                    try:
                        # Get OCR results from PageResult
                        ocr_results = getattr(page_result, 'raw_ocr_results', [])
                        
                        translations = self.akkadian_extractor.extract_translations_from_page(
                            ocr_results, page_num
                        )
                        
                        if translations:
                            akkadian_translations_by_page[page_num] = translations
                            logger.info(f"Found {len(translations)} Akkadian translations on page {page_num}")
                        else:
                            logger.debug(f"No Akkadian translations found on page {page_num}")
                            
                    except Exception as e:
                        logger.warning(f"Akkadian extraction failed for page {page_num}: {e}")
            
            processing_stats.append(stats)
        
        # Write CSV output
        csv_path = output_dir / "comprehensive_results.csv"
        try:
            self.csv_writer.write_csv(str(csv_path), include_metadata=self.config.output_csv_metadata)
        except Exception as e:
            logger.error(f"Failed to write CSV: {e}")
            # Try with simpler path
            csv_path = output_dir / "results.csv"
            self.csv_writer.write_csv(str(csv_path), include_metadata=False)
        
        # Generate Akkadian translations PDF if enabled and translations found
        translations_pdf_path = None
        if (self.config.enable_akkadian_extraction and 
            self.config.generate_translations_pdf and 
            akkadian_translations_by_page):
            
            translations_pdf_path = output_dir / f"{base_filename}_akkadian_translations.pdf"
            try:
                success = generate_translations_report(
                    akkadian_translations_by_page,
                    str(translations_pdf_path),
                    pdf_path.name
                )
                
                if success:
                    logger.info(f"Akkadian translations PDF generated: {translations_pdf_path}")
                else:
                    logger.warning("Failed to generate Akkadian translations PDF")
                    translations_pdf_path = None
                    
            except Exception as e:
                logger.error(f"Error generating Akkadian translations PDF: {e}")
                translations_pdf_path = None
        
        elif (self.config.enable_akkadian_extraction and 
              self.config.generate_translations_pdf and 
              not akkadian_translations_by_page):
            
            # Create empty report indicating no translations found
            translations_pdf_path = output_dir / f"{base_filename}_akkadian_translations.pdf"
            try:
                from translations_pdf import TranslationsPDFGenerator
                generator = TranslationsPDFGenerator()
                success = generator.create_empty_report(str(translations_pdf_path), pdf_path.name)
                
                if success:
                    logger.info(f"Empty Akkadian translations PDF generated: {translations_pdf_path}")
                else:
                    translations_pdf_path = None
                    
            except Exception as e:
                logger.error(f"Error generating empty Akkadian translations PDF: {e}")
                translations_pdf_path = None
        
        # Create summary report
        summary_stats = self.csv_writer.get_summary_stats()
        
        # Add processing statistics
        total_time = time.time() - start_time
        pages_processed = len([s for s in processing_stats if 'error' not in s])
        total_akkadian_translations = sum(len(trans) for trans in akkadian_translations_by_page.values())
        
        # Calculate new metrics for cost of compute
        total_text_elements = sum(s.get('text_elements', 0) for s in processing_stats if 'error' not in s)
        total_word_count = sum(s.get('word_count', 0) for s in processing_stats if 'error' not in s)
        total_token_count = sum(s.get('token_count', 0) for s in processing_stats if 'error' not in s)
        
        # Calculate resource usage averages
        resource_usage_list = [s.get('resource_usage', {}) for s in processing_stats if 'resource_usage' in s]
        avg_cpu_percent = 0.0
        avg_memory_mb = 0.0
        if resource_usage_list:
            cpu_values = [r.get('cpu_percent', 0) for r in resource_usage_list if r.get('available', False)]
            memory_values = [r.get('memory_mb', 0) for r in resource_usage_list if r.get('available', False)]
            avg_cpu_percent = sum(cpu_values) / len(cpu_values) if cpu_values else 0.0
            avg_memory_mb = sum(memory_values) / len(memory_values) if memory_values else 0.0
        
        summary_report = {
            'pdf_path': str(pdf_path),
            'pages_processed': pages_processed,
            'pages_requested': end_page - start_page + 1,
            'total_processing_time': round(total_time, 2),
            'avg_time_per_page': round(total_time / pages_processed, 2) if pages_processed > 0 else 0,
            'output_csv': str(csv_path),
            'akkadian_translations_found': total_akkadian_translations,
            'akkadian_translations_pdf': str(translations_pdf_path) if translations_pdf_path else None,
            'pipeline_config': asdict(self.config),
            # New cost of compute metrics
            'total_text_elements': total_text_elements,
            'total_word_count': total_word_count,
            'total_token_count': total_token_count,
            'avg_text_elements_per_page': round(total_text_elements / pages_processed, 2) if pages_processed > 0 else 0,
            'avg_word_count_per_page': round(total_word_count / pages_processed, 2) if pages_processed > 0 else 0,
            'avg_token_count_per_page': round(total_token_count / pages_processed, 2) if pages_processed > 0 else 0,
            'time_per_text_element': round(total_time / total_text_elements, 4) if total_text_elements > 0 else 0,
            'time_per_word': round(total_time / total_word_count, 4) if total_word_count > 0 else 0,
            'time_per_token': round(total_time / total_token_count, 4) if total_token_count > 0 else 0,
            'avg_cpu_percent': round(avg_cpu_percent, 2),
            'avg_memory_mb': round(avg_memory_mb, 2),
            **summary_stats
        }
        
        # Add individual page stats
        summary_report['page_statistics'] = processing_stats
        
        # Save summary report
        report_path = output_dir / f"{base_filename}_comprehensive_report.json"
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(summary_report, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Could not save report to {report_path}: {e}")
            # Try with simpler filename
            simple_report_path = output_dir / "comprehensive_report.json"
            try:
                with open(simple_report_path, 'w', encoding='utf-8') as f:
                    json.dump(summary_report, f, indent=2, ensure_ascii=False)
                report_path = simple_report_path
            except Exception as e2:
                logger.error(f"Failed to save report: {e2}")
                report_path = None
        
        logger.info(f"Pipeline completed: {pages_processed} pages in {total_time:.1f}s")
        logger.info(f"Output: {csv_path}")
        logger.info(f"Report: {report_path}")
        
        return summary_report

    def process_image(self, image_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Process a single image file with comprehensive pipeline.
        
        Args:
            image_path: Path to the image file
            output_dir: Output directory for results
            
        Returns:
            Processing summary with statistics
        """
        start_time = time.time()
        
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        base_filename = image_path.stem
        
        logger.info(f"Starting image processing: {image_path}")
        logger.info(f"Output directory: {output_dir}")
        
        # Reset CSV writer for new processing
        self.csv_writer = AggregatedCSVWriter()
        
        try:
            # Load image
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            page_width = img.shape[1]
            
            # Run OCR
            ocr_start = time.time()
            paddle_result = self.paddle_ocr.predict(img)
            ocr_time = time.time() - ocr_start
            
            # Extract text elements
            text_elements = self._extract_text_from_paddle_result(paddle_result)
            
            if not text_elements:
                logger.warning(f"No text detected in image")
                return {
                    'image_path': str(image_path),
                    'pages_processed': 0,
                    'total_processing_time': time.time() - start_time,
                    'error': 'No text detected',
                    'output_csv': None
                }
            
            # Apply reading order
            reading_order_start = time.time()
            if self.config.enable_reading_order:
                ordered_elements = self._apply_simple_reading_order(text_elements, page_width)
            else:
                ordered_elements = sorted(text_elements, key=lambda e: (e['bbox'][1], e['bbox'][0]))
            reading_order_time = time.time() - reading_order_start
            
            # Detect language
            detected_language = self._detect_page_language(ordered_elements)

            # Apply LLM corrections
            correction_start = time.time()
            corrections_made = 0
            correction_stats = None

            if self.llm_v3_corrector and ordered_elements:
                # Use LLM V3 system for evaluation mode
                spans = []
                for i, elem in enumerate(ordered_elements):
                    spans.append({
                        'text': elem['text'],
                        'confidence': elem.get('confidence', 0.8),  # Default confidence if not available
                        'bbox': elem.get('bbox', []),
                        'id': f"span_{i}"
                    })

                corrected_spans, v3_stats = self.llm_v3_corrector.correct_spans(spans)

                # Apply corrections to elements
                for i, (elem, corrected_span) in enumerate(zip(ordered_elements, corrected_spans)):
                    if corrected_span['text'] != corrected_span.get('original_text', ''):
                        ordered_elements[i]['text'] = corrected_span['text']
                        ordered_elements[i]['original_text'] = corrected_span.get('original_text', '')
                        ordered_elements[i]['corrections'] = corrected_span.get('corrections', [])
                        ordered_elements[i]['llm_language'] = corrected_span.get('llm_language', 'unknown')
                        ordered_elements[i]['llm_processing_time'] = corrected_span.get('llm_processing_time', 0.0)
                        ordered_elements[i]['cache_hit'] = corrected_span.get('cache_hit', False)
                        corrections_made += 1

                correction_stats = v3_stats

            elif self.llm_corrector and ordered_elements:
                # Legacy LLM correction
                texts_to_correct = [elem['text'] for elem in ordered_elements]
                correction_results = self.llm_corrector.correct_multiple_texts(
                    texts_to_correct, detected_language
                )

                # Apply corrections to elements
                for i, correction in enumerate(correction_results):
                    if i < len(ordered_elements) and correction.corrected_text != correction.original_text:
                        ordered_elements[i]['text'] = correction.corrected_text
                        ordered_elements[i]['original_text'] = correction.original_text
                        ordered_elements[i]['corrections'] = correction.corrections_made
                        corrections_made += 1

                correction_stats = self.llm_corrector.get_correction_stats()

            correction_time = time.time() - correction_start

            # Create page result
            page_id = "image_001"
            
            # Add to CSV writer
            self.csv_writer.add_page_content(
                page_id=page_id,
                text_elements=ordered_elements,
                language=detected_language,
                correction_stats=correction_stats,
                reading_order_stats={
                    'processing_time': reading_order_time,
                    'elements_processed': len(ordered_elements),
                    'columns_detected': 2 if len(set(elem['bbox'][0] for elem in ordered_elements)) > page_width * 0.3 else 1
                }
            )
            
            # Get page result from CSV writer
            page_result = self.csv_writer.pages_data.get(page_id)
            
            # Process Akkadian translations if enabled
            akkadian_translations_by_page = {}
            if self.config.enable_akkadian_extraction and self.akkadian_extractor:
                try:
                    # Get OCR results from PageResult
                    ocr_results = getattr(page_result, 'raw_ocr_results', [])
                    
                    translations = self.akkadian_extractor.extract_translations_from_page(
                        ocr_results, 1  # Treat as page 1
                    )
                    
                    if translations:
                        akkadian_translations_by_page[1] = translations
                        logger.info(f"Found {len(translations)} Akkadian translations in image")
                    else:
                        logger.debug(f"No Akkadian translations found in image")
                        
                except Exception as e:
                    logger.warning(f"Akkadian extraction failed for image: {e}")
            
            # Write CSV output
            csv_path = output_dir / "comprehensive_results.csv"
            try:
                self.csv_writer.write_csv(str(csv_path), include_metadata=self.config.output_csv_metadata)
            except Exception as e:
                logger.error(f"Failed to write CSV: {e}")
                # Try with simpler path
                csv_path = output_dir / "results.csv"
                self.csv_writer.write_csv(str(csv_path), include_metadata=False)
            
            # Generate Akkadian translations PDF if enabled and translations found
            translations_pdf_path = None
            if (self.config.enable_akkadian_extraction and 
                self.config.generate_translations_pdf and 
                akkadian_translations_by_page):
                
                translations_pdf_path = output_dir / f"{base_filename}_akkadian_translations.pdf"
                try:
                    success = generate_translations_report(
                        akkadian_translations_by_page,
                        str(translations_pdf_path),
                        image_path.name
                    )
                    
                    if success:
                        logger.info(f"Akkadian translations PDF generated: {translations_pdf_path}")
                    else:
                        logger.warning("Failed to generate Akkadian translations PDF")
                        translations_pdf_path = None
                        
                except Exception as e:
                    logger.error(f"Error generating Akkadian translations PDF: {e}")
                    translations_pdf_path = None
            
            # Create summary report
            summary_stats = self.csv_writer.get_summary_stats()
            
            # Add processing statistics
            total_time = time.time() - start_time
            total_akkadian_translations = sum(len(trans) for trans in akkadian_translations_by_page.values())
            
            summary_report = {
                'image_path': str(image_path),
                'pages_processed': 1,
                'total_processing_time': round(total_time, 2),
                'output_csv': str(csv_path),
                'akkadian_translations_found': total_akkadian_translations,
                'akkadian_translations_pdf': str(translations_pdf_path) if translations_pdf_path else None,
                'pipeline_config': asdict(self.config),
                'processing_stats': {
                    'ocr_time': ocr_time,
                    'reading_order_time': reading_order_time,
                    'correction_time': correction_time,
                    'text_elements': len(ordered_elements),
                    'language': detected_language,
                    'corrections_made': corrections_made,
                    'avg_confidence': page_result.conf_mean if page_result else 0.0
                },
                **summary_stats
            }
            
            # Save summary report
            report_path = output_dir / f"{base_filename}_comprehensive_report.json"
            try:
                with open(report_path, 'w', encoding='utf-8') as f:
                    json.dump(summary_report, f, indent=2, ensure_ascii=False)
                    
                logger.info(f"Summary report saved: {report_path}")
            except Exception as e:
                logger.warning(f"Could not save summary report: {e}")
            
            logger.info(f"Image processing completed successfully in {total_time:.2f}s")
            
            return summary_report
            
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'image_path': str(image_path),
                'pages_processed': 0,
                'total_processing_time': time.time() - start_time,
                'error': str(e),
                'output_csv': None
            }


def main():
    """Main entry point for the comprehensive OCR pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive OCR Pipeline with LLM correction')
    parser.add_argument('pdf_path', help='Path to PDF file')
    parser.add_argument('-o', '--output', help='Output directory')
    parser.add_argument('-s', '--start-page', type=int, default=1, help='Start page (default: 1)')
    parser.add_argument('-e', '--end-page', type=int, help='End page (default: all pages)')
    
    # Pipeline configuration
    parser.add_argument('--llm-provider', choices=['ollama', 'llamacpp', 'none'], 
                       default='ollama', help='LLM provider for corrections')
    parser.add_argument('--llm-model', default='llama3.2:latest', help='LLM model name')
    parser.add_argument('--disable-corrections', action='store_true', 
                       help='Disable LLM corrections')
    parser.add_argument('--disable-reading-order', action='store_true',
                       help='Disable reading order detection')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for PDF rendering')
    
    args = parser.parse_args()
    
    # Create configuration
    config = PipelineConfig(
        llm_provider='none' if args.disable_corrections else args.llm_provider,
        llm_model=args.llm_model,
        enable_llm_correction=not args.disable_corrections,
        enable_reading_order=not args.disable_reading_order,
        dpi=args.dpi
    )
    
    # Set output directory
    if args.output:
        output_dir = args.output
    else:
        pdf_path = Path(args.pdf_path)
        output_dir = pdf_path.parent / f"{pdf_path.stem}_comprehensive_ocr"
    
    # Create and run pipeline
    pipeline = ComprehensivePipeline(config)
    
    try:
        result = pipeline.process_pdf(
            pdf_path=args.pdf_path,
            output_dir=output_dir,
            start_page=args.start_page,
            end_page=args.end_page
        )
        
        if 'error' in result:
            logger.error(f"Pipeline failed: {result['error']}")
            sys.exit(1)
        else:
            print(f"\nPipeline completed successfully!")
            print(f"Processed: {result['pages_processed']} pages")
            print(f"Time: {result['total_processing_time']}s")
            print(f"Output: {result['output_csv']}")
            
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
