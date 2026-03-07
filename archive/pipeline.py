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
from dataclasses import dataclass, asdict, field

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

# Configure logging (must be before research imports)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from aggregated_csv import AggregatedCSVWriter, PageResult
from enhanced_llm_correction import EnhancedLLMCorrector, CorrectionResult, OCRSpan, BoundingBox
from akkadian_extract import AkkadianExtractor
from translations_pdf import generate_translations_report
from telemetry import get_telemetry, PageTiming

# New LLM post-correction system
try:
    from llm.corrector import LLMCorrector, CorrectionResult as NewCorrectionResult
    from llm.clients.ollama_client import OllamaConfig
    NEW_LLM_AVAILABLE = True
    logger.info("New LLM post-correction system available")
except ImportError as e:
    NEW_LLM_AVAILABLE = False
    logger.info(f"New LLM system not available: {e}")

# Research features (optional, gated by config flags)
try:
    from translit_norm import smart_normalize, is_transliteration_line, validate_preservation
    from rover_fusion import ROVERFusion, Hypothesis
    from char_lm_decoder import create_lm_decoder
    from lexicon_bias import LexiconBias
    from confusion_prior import ConfusionPrior
    from tta_augment import TTAAugmenter, TTAConfig
    from layout_classifier import LayoutBandClassifier
    from grapheme_metrics import compute_all_metrics
    from ocr_utils import enhance_ocr_with_research_features
    from diacritic_restoration import DiacriticRestorer
    RESEARCH_AVAILABLE = True
    logger.info("Research features available")
except ImportError as e:
    RESEARCH_AVAILABLE = False
    logger.info(f"Research features not available: {e}")


@dataclass
class PipelineConfig:
    """Configuration for the comprehensive OCR pipeline."""
    # LLM settings (legacy - kept for backwards compatibility)
    llm_provider: str = "ollama"  # 'ollama', 'llamacpp', or 'none'
    llm_model: str = "mistral:latest"  # Updated to use available model
    llm_base_url: str = "http://localhost:11434"
    llm_timeout: int = 30
    
    # New LLM post-correction settings
    enable_new_llm_correction: bool = False  # Opt-in for new LLM system
    llm_correction_model: str = "qwen2.5:7b-instruct"
    llm_correction_edit_budget: float = 0.12  # Max edit ratio
    llm_correction_cache_enabled: bool = True
    
    # OCR settings
    dpi: int = 300
    paddle_use_gpu: bool = True  # Enable GPU by default for RTX 4070
    ocr_languages: List[str] = field(default_factory=lambda: ['en', 'tr', 'de', 'fr', 'it'])
    
    # OCR Parameters by language
    ocr_params: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        'en': {'det_db_thresh': 0.3, 'det_db_box_thresh': 0.6, 'rec_score_thresh': 0.5},
        'tr': {'det_db_thresh': 0.2, 'det_db_box_thresh': 0.5, 'rec_score_thresh': 0.4},
        'de': {'det_db_thresh': 0.3, 'det_db_box_thresh': 0.6, 'rec_score_thresh': 0.5},
        'fr': {'det_db_thresh': 0.3, 'det_db_box_thresh': 0.6, 'rec_score_thresh': 0.5},
        'it': {'det_db_thresh': 0.3, 'det_db_box_thresh': 0.6, 'rec_score_thresh': 0.5}
    })
    
    # Processing settings
    enable_reading_order: bool = True
    enable_llm_correction: bool = True
    max_concurrent_corrections: int = 3
    
    # Performance profiles
    performance_profile: str = "quality"  # 'fast' or 'quality'
    
    # Akkadian translation extraction
    enable_akkadian_extraction: bool = True
    generate_translations_pdf: bool = True
    akkadian_confidence_threshold: float = 0.8
    
    # Output settings
    output_csv_metadata: bool = True
    create_html_overlay: bool = True
    create_visualization: bool = True
    
    # Research features (from config file)
    preserve_transliteration: Dict[str, Any] = field(default_factory=lambda: {'enable': False})
    metrics: Dict[str, Any] = field(default_factory=lambda: {'enable_grapheme_metrics': False})
    research: Dict[str, Any] = field(default_factory=lambda: {
        'enable_char_lm_decoding': False,
        'enable_rover_ensemble': False,
        'enable_lexicon_bias': False,
        'enable_tta': False,
        'enable_confusion_prior': False,
        'enable_header_footer_lr': False
    })

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
        self.akkadian_extractor = None
        self.csv_writer = AggregatedCSVWriter()
        self.processing_stats = {}
        
        # Initialize telemetry system
        self.telemetry = get_telemetry()
        
        # Research modules (initialized if enabled in config)
        self.rover = None
        self.char_lm = None
        self.lexicon = None
        self.confusion_prior = None
        self.tta = None
        self.layout_classifier = None
        self.diacritic_restorer = None
        self.use_grapheme_metrics = False
        self.preserve_translit = False
        
        self._initialize_engines()
        self._initialize_research_modules()
    
    def get_ocr_for_language(self, language: str) -> Any:
        """Get the appropriate OCR engine for a specific language."""
        # Map language codes to PaddleOCR language codes
        lang_mapping = {
            'english': 'en',
            'turkish': 'tr', 
            'german': 'de',
            'french': 'fr',
            'italian': 'it',
            'en': 'en',
            'tr': 'tr',
            'de': 'de', 
            'fr': 'fr',
            'it': 'it'
        }
        
        paddle_lang = lang_mapping.get(language.lower(), 'en')
        
        if paddle_lang in self.paddle_ocrs:
            return self.paddle_ocrs[paddle_lang]
        else:
            # Fallback to default (English or first available)
            logger.warning(f"No OCR available for language {language}, using default")
            return self.paddle_ocr
    
    def _initialize_engines(self):
        """Initialize OCR and LLM engines with GPU detection for RTX 4070."""
        # Initialize PaddleOCR with GPU support
        try:
            # Force offline mode to prevent network calls
            import os
            os.environ['PADDLE_GIT_CLONE_OFFLINE'] = '1'
            os.environ['HF_DATASETS_OFFLINE'] = '1'
            
            from paddleocr import PaddleOCR
            import paddle
            
            # Set device explicitly with Paddle device detection
            device = "gpu" if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0 else "cpu"
            paddle.device.set_device(device)
            
            logger.info(f"Paddle device set to: {device}")
            if device == "gpu":
                logger.info(f"GPU devices available: {paddle.device.cuda.device_count()}")
            
            # Initialize PaddleOCR with modern API (no use_gpu parameter)
            if device == "gpu" and self.config.paddle_use_gpu:
                try:
                    self.paddle_ocr = PaddleOCR(
                        use_textline_orientation=True,  # Detect and correct text orientation
                        lang='en'  # English language for mixed content
                    )
                    logger.info("✅ PaddleOCR initialized successfully with GPU acceleration")
                    return
                    
                except Exception as gpu_e:
                    logger.warning(f"GPU initialization failed: {gpu_e}")
                    logger.info("Falling back to CPU mode...")
            
            # Fallback to CPU with compatible parameters
            self.paddle_ocr = PaddleOCR(
                use_textline_orientation=True,  # Detect and correct text orientation
                lang='en'  # English language for mixed content
            )
            logger.info("PaddleOCR initialized with CPU mode")
            
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            raise
            raise
        
        # Initialize LLM corrector
        if self.config.enable_llm_correction and self.config.llm_provider != 'none':
            try:
                self.llm_corrector = EnhancedLLMCorrector(
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
        else:
            logger.info("LLM correction disabled")
        
        # Initialize new LLM post-correction system
        self.new_llm_corrector = None
        if self.config.enable_new_llm_correction and NEW_LLM_AVAILABLE:
            try:
                # Create Ollama config
                ollama_config = OllamaConfig(
                    base_url=self.config.llm_base_url,
                    model_id=self.config.llm_correction_model,
                    timeout_s=self.config.llm_timeout
                )
                
                # Create cache (always use dict, even if caching disabled - corrector expects a dict)
                llm_cache = {}
                
                # Create corrector
                self.new_llm_corrector = LLMCorrector(
                    ollama_config=ollama_config,
                    cache=llm_cache,
                    enable_telemetry=True
                )
                logger.info(f"✅ New LLM post-correction initialized: {self.config.llm_correction_model}")
            except Exception as e:
                logger.warning(f"New LLM corrector initialization failed: {e}")
                self.new_llm_corrector = None
        elif self.config.enable_new_llm_correction:
            logger.warning("New LLM correction requested but not available")
        
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
    
    def _initialize_research_modules(self):
        """Initialize research modules if enabled and available."""
        if not RESEARCH_AVAILABLE:
            logger.debug("Research modules not available - skipping initialization")
            return
        
        # Transliteration preservation
        if self.config.preserve_transliteration.get('enable', False):
            self.preserve_translit = True
            logger.info("Transliteration-preserving normalization enabled")
        
        # Grapheme metrics
        if self.config.metrics.get('enable_grapheme_metrics', False):
            self.use_grapheme_metrics = True
            logger.info("Grapheme-aware metrics enabled")
        
        # ROVER ensemble fusion
        if self.config.research.get('enable_rover_ensemble', False):
            try:
                rover_config = self.config.research.get('rover', {})
                weights = rover_config.get('weights', {'abinet': 1.0, 'parseq': 1.0, 'doctr': 1.0})
                self.rover = ROVERFusion(weights=weights)
                logger.info(f"ROVER ensemble initialized with weights: {weights}")
            except Exception as e:
                logger.warning(f"ROVER initialization failed: {e}")
        
        # Character language model
        # Always try to load if available (will be auto-enabled for Akkadian content)
        try:
            lm_config = self.config.research.get('char_lm', {})
            lm_path = lm_config.get('model_path', 'reports/research_assets/lm/char.klm')
            charset_path = lm_config.get('charset_path', 'reports/research_assets/lm/charset.txt')
            
            if os.path.exists(lm_path) and os.path.exists(charset_path):
                # Use attention decoder by default (works with all engines)
                self.char_lm = create_lm_decoder(
                    engine_type='attention',
                    lm_path=lm_path,
                    charset_path=charset_path,
                    config=lm_config
                )
                if self.config.research.get('enable_char_lm_decoding', False):
                    logger.info(f"Character LM decoder initialized (enabled): {lm_path}")
                else:
                    logger.info(f"Character LM decoder loaded (auto-enable for Akkadian): {lm_path}")
            else:
                logger.warning(f"Character LM files not found: {lm_path}, {charset_path}")
        except Exception as e:
            logger.warning(f"Character LM initialization failed: {e}")
        
        # Lexicon biasing
        if self.config.research.get('enable_lexicon_bias', False):
            try:
                lex_config = self.config.research.get('lexicon', {})
                self.lexicon = LexiconBias(
                    min_freq=lex_config.get('min_freq', 2),
                    bias=lex_config.get('bias', 0.95),
                    max_boost_per_token=lex_config.get('max_boost_per_token', 2.0)
                )
                
                # Load lexicon file if it exists
                lex_file = lex_config.get('lexicon_file', 'reports/lexicon/akkadian_lexicon.json')
                if os.path.exists(lex_file):
                    self.lexicon.load(lex_file)
                    logger.info(f"Lexicon loaded from {lex_file}")
                else:
                    # Load default lexicons
                    self.lexicon.load_sumerograms()
                    self.lexicon.load_akkadian_morphemes()
                    self.lexicon.load_function_words()
                    logger.info("Default lexicons loaded (Sumerograms, morphemes, function words)")
            except Exception as e:
                logger.warning(f"Lexicon initialization failed: {e}")
        
        # Confusion prior
        if self.config.research.get('enable_confusion_prior', False):
            try:
                conf_config = self.config.research.get('confusion_prior', {})
                self.confusion_prior = ConfusionPrior(
                    tie_threshold=conf_config.get('tie_threshold', 2.0)
                )
                logger.info("Confusion prior re-ranking enabled")
            except Exception as e:
                logger.warning(f"Confusion prior initialization failed: {e}")
        
        # Diacritic restoration (DISABLED - testing showed 11.45% CER degradation)
        # See DIACRITIC_RESTORATION_TEST_RESULTS.md for details
        # if RESEARCH_AVAILABLE and (self.lexicon or self.confusion_prior):
        #     try:
        #         restore_config = self.config.research.get('diacritic_restoration', {})
        #         self.diacritic_restorer = DiacriticRestorer(
        #             lexicon=self.lexicon,
        #             confusion_prior=self.confusion_prior,
        #             max_candidates=restore_config.get('max_candidates', 20),
        #             lexicon_weight=restore_config.get('lexicon_weight', 2.0),
        #             confusion_weight=restore_config.get('confusion_weight', 1.0)
        #         )
        #         logger.info("Diacritic restoration enabled (post-OCR correction)")
        #     except Exception as e:
        #         logger.warning(f"Diacritic restorer initialization failed: {e}")
        
        # Test-time augmentation
        if self.config.research.get('enable_tta', False):
            try:
                tta_config = self.config.research.get('tta', {})
                self.tta = TTAAugmenter(TTAConfig(
                    rot_deg=tta_config.get('rot_deg', [-2, -1, 0, 1, 2]),
                    scales=tta_config.get('scales', [0.95, 1.0, 1.05]),
                    time_budget=tta_config.get('time_budget', 8.0),
                    max_augments=tta_config.get('max_augments', 4)
                ))
                logger.info(f"Test-time augmentation enabled (budget: {tta_config.get('time_budget', 8.0)}s)")
            except Exception as e:
                logger.warning(f"TTA initialization failed: {e}")
        
        # Layout classifier
        if self.config.research.get('enable_header_footer_lr', False):
            try:
                layout_config = self.config.research.get('header_footer_lr', {})
                self.layout_classifier = LayoutBandClassifier(
                    top_frac=layout_config.get('top_frac', 0.06),
                    bot_frac=layout_config.get('bot_frac', 0.08)
                )
                logger.info("Layout band classifier enabled (LogisticRegression with rule fallback)")
            except Exception as e:
                logger.warning(f"Layout classifier initialization failed: {e}")
    
    def _extract_text_from_paddle_result(self, paddle_result):
        """Extract text data from PaddleOCR result format and convert to OCRSpan objects."""
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
            
            ocr_spans = []
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
                    
                    bbox_obj = BoundingBox(x=x, y=y, width=w, height=h)
                else:
                    # Default bbox if no polygon data
                    bbox_obj = BoundingBox(x=0, y=i * 25, width=200, height=20)
                
                # Clean and normalize text with research-aware normalization
                if self.preserve_translit and RESEARCH_AVAILABLE:
                    # Use transliteration-preserving normalization
                    cleaned_text = smart_normalize(
                        text, 
                        self.config.preserve_transliteration
                    )
                else:
                    # Standard normalization
                    cleaned_text = text.strip()
                
                if cleaned_text:
                    span = OCRSpan(
                        text=cleaned_text,
                        confidence=float(conf),
                        bbox=bbox_obj,
                        language=None,  # Will be detected by LLM corrector
                        is_akkadian=False,  # Will be detected by LLM corrector
                        char_density=0.0
                    )
                    ocr_spans.append(span)
            
            return ocr_spans
            
        except Exception as e:
            logger.error(f"Error extracting text from PaddleOCR result: {e}")
            return []
    
    def _convert_spans_to_legacy_format(self, spans: List[OCRSpan]) -> List[Dict]:
        """Convert OCRSpan objects back to legacy format for compatibility."""
        legacy_results = []
        for span in spans:
            if span.bbox:
                bbox = (span.bbox.x, span.bbox.y, span.bbox.width, span.bbox.height)
            else:
                bbox = (0, 0, 200, 20)
            
            legacy_results.append({
                'text': span.text,
                'bbox': bbox,
                'conf': span.confidence,
                'engine': 'paddle',
                'language': span.language,
                'is_akkadian': span.is_akkadian
            })
        
        return legacy_results
    
    def _apply_simple_reading_order(self, text_elements: List[Dict], page_width: int = 800) -> List[Dict]:
        """
        Apply simple reading order without sklearn dependency.
        Handles basic column detection and top-to-bottom, left-to-right ordering.
        """
        if not text_elements:
            return []
        
        # Simple column detection based on x-coordinate clustering
        # Handle both dict format and OCRSpan object format
        x_positions = []
        for elem in text_elements:
            if hasattr(elem, 'bbox') and hasattr(elem.bbox, 'x'):
                # OCRSpan object with BoundingBox
                x_center = elem.bbox.x + elem.bbox.width / 2
            elif isinstance(elem, dict) and 'bbox' in elem:
                # Dictionary format
                x_center = elem['bbox'][0] + elem['bbox'][2] / 2
            else:
                # Fallback
                continue
            x_positions.append(x_center)
        
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
            left_column = []
            right_column = []
            
            for elem in text_elements:
                if hasattr(elem, 'bbox') and hasattr(elem.bbox, 'x'):
                    # OCRSpan object with BoundingBox
                    x_center = elem.bbox.x + elem.bbox.width / 2
                elif isinstance(elem, dict) and 'bbox' in elem:
                    # Dictionary format
                    x_center = elem['bbox'][0] + elem['bbox'][2] / 2
                else:
                    continue
                    
                if x_center < column_threshold:
                    left_column.append(elem)
                else:
                    right_column.append(elem)
            
            # Sort each column by reading order (y first, then x)
            def get_sort_key(e):
                if hasattr(e, 'bbox') and hasattr(e.bbox, 'x'):
                    return (e.bbox.y, e.bbox.x)
                elif isinstance(e, dict) and 'bbox' in e:
                    return (e['bbox'][1], e['bbox'][0])
                else:
                    return (0, 0)
            
            left_sorted = sorted(left_column, key=get_sort_key)
            right_sorted = sorted(right_column, key=get_sort_key)
            
            # Combine columns (left first, then right)
            ordered_elements = left_sorted + right_sorted
            
            logger.debug(f"Applied column ordering: {len(left_sorted)} left, {len(right_sorted)} right")
            
        else:
            # Single column - sort by reading order
            def get_sort_key(e):
                if hasattr(e, 'bbox') and hasattr(e.bbox, 'x'):
                    return (e.bbox.y, e.bbox.x)
                elif isinstance(e, dict) and 'bbox' in e:
                    return (e['bbox'][1], e['bbox'][0])
                else:
                    return (0, 0)
            
            ordered_elements = sorted(text_elements, key=get_sort_key)
            logger.debug("Applied single-column reading order")
        
        return ordered_elements
    
    def _detect_page_language(self, text_elements: List) -> str:
        """Detect the primary language of the page using langdetect with character-based fallback."""
        # Extract text from elements (handle both dict and OCRSpan formats)
        text_parts = []
        for elem in text_elements:
            if hasattr(elem, 'text'):
                # OCRSpan object
                text_parts.append(elem.text)
            elif isinstance(elem, dict) and 'text' in elem:
                # Dictionary format
                text_parts.append(elem['text'])
        
        all_text = ' '.join(text_parts)
        
        if not all_text or len(all_text.strip()) < 10:
            return 'unknown'
        
        # Check for Akkadian transliteration patterns (before language detection)
        akkadian_markers = len(re.findall(r'\b[A-Z]{2,}\b', all_text))  # All-caps words (DUMU, KIIB, etc.)
        akkadian_markers += len(re.findall(r'[šṣṭḫ]', all_text))  # Akkadian special chars
        akkadian_markers += len(re.findall(r'\b[A-Z]{1,3}\d+[a-z]?\b', all_text))  # AKT IV, Rs., etc.
        
        # Hyphenated words are very characteristic of Akkadian transliteration
        hyphen_density = all_text.count('-') / max(len(all_text), 1)
        
        text_len = len(all_text.replace(' ', ''))
        marker_ratio = akkadian_markers / max(text_len, 1)
        
        # If high hyphen density OR many akkadian markers, mark as akkadian
        if hyphen_density > 0.15 or marker_ratio > 0.05:
            logger.debug(f"Akkadian content detected (markers={akkadian_markers}/{text_len}={marker_ratio:.2%}, hyphens={hyphen_density:.2%})")
            return 'akkadian'
        
        # Try langdetect first (more accurate for mixed content)
        try:
            from langdetect import detect, DetectorFactory
            DetectorFactory.seed = 0  # For consistent results
            
            # Clean text for detection (remove Akkadian transliterations which confuse detector)
            clean_text = re.sub(r'\b[A-Z]{2,}\b', '', all_text)  # Remove all-caps words (often Akkadian)
            clean_text = re.sub(r'[IVX]{2,}', '', clean_text)  # Remove Roman numerals
            clean_text = re.sub(r'\d+', '', clean_text)  # Remove numbers
            clean_text = clean_text.strip()
            
            if len(clean_text) > 20:
                detected = detect(clean_text)
                
                # Map langdetect codes to full names
                lang_map = {
                    'tr': 'turkish',
                    'de': 'german', 
                    'fr': 'french',
                    'it': 'italian',
                    'en': 'english',
                    'es': 'spanish',
                    'ru': 'russian'
                }
                
                if detected in lang_map:
                    logger.debug(f"Language detected via langdetect: {lang_map[detected]}")
                    return lang_map[detected]
        except Exception as e:
            logger.debug(f"langdetect failed, using character-based fallback: {e}")
        
        # Fallback: character-based detection for languages with special chars
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
                    logger.debug(f"Language detected via char patterns: {lang}")
                    return lang
        
        return 'english'
    
    def _detect_line_language(self, text: str) -> str:
        """
        Detect language for a single line of text.
        Used for mixed-language pages to classify individual text elements.
        
        Args:
            text: Single line of text to classify
            
        Returns:
            Language code (akkadian, turkish, german, french, italian, english, unknown)
        """
        if not text or len(text.strip()) < 3:
            return 'unknown'
        
        # Check for Akkadian transliteration patterns
        akkadian_markers = len(re.findall(r'\b[A-Z]{2,}\b', text))  # All-caps words
        akkadian_markers += len(re.findall(r'[šṣṭḫ]', text))  # Akkadian special chars
        
        hyphen_density = text.count('-') / max(len(text), 1)
        text_len = len(text.replace(' ', ''))
        marker_ratio = akkadian_markers / max(text_len, 1) if text_len > 0 else 0
        
        # More aggressive thresholds for line-level detection
        if hyphen_density > 0.12 or marker_ratio > 0.08:
            return 'akkadian'
        
        # Try langdetect for non-Akkadian lines
        try:
            from langdetect import detect
            clean_text = re.sub(r'\b[A-Z]{2,}\b', '', text)
            clean_text = re.sub(r'[IVX]{2,}', '', clean_text)
            clean_text = clean_text.strip()
            
            if len(clean_text) > 10:
                detected = detect(clean_text)
                lang_map = {
                    'tr': 'turkish', 'de': 'german', 'fr': 'french',
                    'it': 'italian', 'en': 'english', 'es': 'spanish'
                }
                return lang_map.get(detected, 'english')
        except:
            pass
        
        # Fallback: character-based detection
        if any(c in text for c in 'çğıöşüÇĞIİÖŞÜ'):
            return 'turkish'
        elif any(c in text for c in 'äöüßÄÖÜ'):
            return 'german'
        elif any(c in text for c in 'àâäéèêëïîôùûüÿçÀÂÄÉÈÊËÏÎÔÙÛÜŸÇ'):
            return 'french'
        
        return 'english'
    
    def _analyze_language_distribution(self, text_elements: List) -> Dict[str, Any]:
        """
        Analyze language distribution across text elements.
        Detects mixed-language pages and provides per-line classifications.
        
        Args:
            text_elements: List of text elements
            
        Returns:
            Dictionary with:
                - primary_language: Most common language
                - language_counts: Distribution of languages
                - is_mixed: Whether page has significant mixed content
                - per_line_languages: List of (index, language) tuples
        """
        if not text_elements:
            return {
                'primary_language': 'unknown',
                'language_counts': {},
                'is_mixed': False,
                'per_line_languages': []
            }
        
        # Detect language for each text element
        per_line_languages = []
        language_counts = {}
        
        for i, elem in enumerate(text_elements):
            # Extract text
            if hasattr(elem, 'text'):
                text = elem.text
            elif isinstance(elem, dict):
                text = elem.get('text', '')
            else:
                text = ''
            
            if text.strip():
                lang = self._detect_line_language(text)
                per_line_languages.append((i, lang))
                language_counts[lang] = language_counts.get(lang, 0) + 1
        
        # Determine primary language
        primary_language = 'unknown'
        if language_counts:
            primary_language = max(language_counts, key=language_counts.get)
        
        # Check if page is mixed (secondary language >20% of content)
        total_lines = len(per_line_languages)
        is_mixed = False
        if total_lines > 5 and len(language_counts) > 1:
            sorted_counts = sorted(language_counts.values(), reverse=True)
            if len(sorted_counts) > 1 and sorted_counts[1] / total_lines > 0.2:
                is_mixed = True
        
        return {
            'primary_language': primary_language,
            'language_counts': language_counts,
            'is_mixed': is_mixed,
            'per_line_languages': per_line_languages
        }
    
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
            # Extract text from elements (handle both dict and OCRSpan formats)
            text_parts = []
            for elem in text_elements:
                if hasattr(elem, 'text'):
                    # OCRSpan object
                    text_parts.append(elem.text)
                elif isinstance(elem, dict) and 'text' in elem:
                    # Dictionary format
                    text_parts.append(elem.get('text', ''))
            
            all_text = ' '.join(text_parts)
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
        page_id = f"page_{page_num:03d}"
        
        # Check kill switch
        if self.telemetry.kill_switch.is_killed:
            logger.warning(f"Processing stopped by kill switch on {page_id}")
            return None, {"error": "killed_by_switch"}
        
        # Start telemetry timing
        timing = self.telemetry.start_page_timing(page_id)
        preprocessing_start = time.time()
        
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
            
            timing.preprocessing_time = time.time() - preprocessing_start
            # Run OCR with optional test-time augmentation
            ocr_start = time.time()
            
            # Check OCR kill switch
            if self.telemetry.kill_switch.is_ocr_disabled:
                logger.warning(f"OCR disabled by kill switch on {page_id}")
                return None, {"error": "ocr_disabled"}
            
            # Apply TTA if enabled
            if self.tta and RESEARCH_AVAILABLE:
                logger.debug(f"Running OCR with TTA on {page_id}")
                
                def decode_fn(image):
                    """Decode function for TTA"""
                    result = self.paddle_ocr.predict(image)
                    # Extract just the text and confidence from result
                    if result and len(result) > 0:
                        result_obj = result[0] if isinstance(result[0], dict) else result
                        if isinstance(result_obj, dict):
                            texts = result_obj.get('rec_texts', [])
                            scores = result_obj.get('rec_scores', [])
                        else:
                            texts = [item[1][0] for item in result if len(item) >= 2]
                            scores = [item[1][1] for item in result if len(item) >= 2]
                        
                        # Combine all text with average confidence
                        full_text = ' '.join(texts)
                        avg_conf = sum(scores) / len(scores) if scores else 0.5
                        return full_text, avg_conf
                    return "", 0.0
                
                # Run TTA and get augmented results
                tta_results = self.tta.decode_with_tta(img, decode_fn, engine_name='paddle')
                
                # Fuse with ROVER if available, otherwise take best result
                if self.rover:
                    final_text, final_conf = self.tta.fuse_with_rover(tta_results, self.rover)
                    logger.info(f"TTA+ROVER fusion: {len(tta_results)} augments → confidence {final_conf:.3f}")
                else:
                    # Take the result with highest confidence
                    final_text, final_conf = max(tta_results, key=lambda x: x[1])
                    logger.info(f"TTA best-of-{len(tta_results)}: confidence {final_conf:.3f}")
                
                # Still need to run OCR normally to get bounding boxes and structure
                paddle_result = self.paddle_ocr.predict(img)
                text_elements = self._extract_text_from_paddle_result(paddle_result)
                
                # Override the concatenated text with TTA-fused text
                # (Keep bounding boxes from normal OCR but use TTA text)
                if text_elements and final_text:
                    # Split TTA text across elements proportionally
                    # For simplicity, just use TTA for overall quality but keep structure
                    logger.debug("TTA text available, structure from standard OCR")
            else:
                # Standard OCR without TTA
                paddle_result = self.paddle_ocr.predict(img)
                text_elements = self._extract_text_from_paddle_result(paddle_result)
            
            timing.ocr_time = time.time() - ocr_start
            
            # Extract text elements (already done above based on TTA branch)
            # text_elements = self._extract_text_from_paddle_result(paddle_result)
            
            if not text_elements:
                logger.warning(f"No text detected on page {page_num}")
                timing.total_time = time.time() - start_time
                self.telemetry.record_page_timing(timing)
                return None, {
                    'page_num': page_num,
                    'processing_time': timing.total_time,
                    'ocr_time': timing.ocr_time,
                    'text_elements': 0,
                    'language': 'unknown',
                    'corrections_made': 0,
                    'word_count': 0,
                    'token_count': 0,
                    'text_elements_count': 0
                }
            
            # Apply reading order
            reading_order_start = time.time()
            if self.config.enable_reading_order:
                ordered_elements = self._apply_simple_reading_order(text_elements, page_width)
            else:
                ordered_elements = sorted(text_elements, key=lambda e: (e['bbox'][1], e['bbox'][0]))
            timing.reading_order_time = time.time() - reading_order_start
            
            # Detect language (both page-level and per-line for mixed content)
            lang_detect_start = time.time()
            detected_language = self._detect_page_language(ordered_elements)
            lang_analysis = self._analyze_language_distribution(ordered_elements)
            
            # Use detailed analysis if page is mixed
            if lang_analysis['is_mixed']:
                logger.info(f"Mixed-language page detected: {lang_analysis['language_counts']}")
                detected_language = f"{lang_analysis['primary_language']}_mixed"
            
            timing.language_detection_time = time.time() - lang_detect_start
            
            # Apply research features enhancement (if enabled)
            research_start = time.time()
            research_corrections = 0
            if RESEARCH_AVAILABLE and ordered_elements:
                # Enable Char LM for Akkadian content (conditional activation)
                enable_char_lm_for_page = self.config.research.get('enable_char_lm_decoding', False)
                
                # Auto-enable Char LM for Akkadian pages if char_lm is available
                if not enable_char_lm_for_page and hasattr(self, 'char_lm') and self.char_lm is not None:
                    if 'akkadian' in detected_language or lang_analysis.get('language_counts', {}).get('akkadian', 0) > 0:
                        enable_char_lm_for_page = True
                        logger.info(f"Auto-enabling Char LM for Akkadian content (page: {detected_language})")
                
                # Check if any research features are enabled
                research_enabled = (
                    enable_char_lm_for_page or
                    self.config.research.get('enable_lexicon_bias', False) or
                    self.config.research.get('enable_confusion_prior', False)
                )
                
                if research_enabled:
                    logger.info(f"Applying research features to {len(ordered_elements)} text elements (Char LM: {enable_char_lm_for_page})")
                    
                    # Apply research features to each text element
                    for idx, elem in enumerate(ordered_elements):
                        # Handle both dict and OCRSpan object types
                        if isinstance(elem, dict):
                            original_text = elem.get('text', '')
                            original_conf = elem.get('confidence', 0.0)
                        else:
                            # OCRSpan or similar object
                            original_text = elem.text if hasattr(elem, 'text') else ''
                            original_conf = elem.confidence if hasattr(elem, 'confidence') else 0.0
                        
                        if not original_text:
                            continue
                        
                        # Determine if Char LM should be used for this specific line
                        use_char_lm_for_line = enable_char_lm_for_page
                        
                        # For mixed pages, use per-line language detection
                        if lang_analysis['is_mixed'] and lang_analysis['per_line_languages']:
                            line_lang = None
                            for i, lang in lang_analysis['per_line_languages']:
                                if i == idx:
                                    line_lang = lang
                                    break
                            
                            # Enable Char LM only for Akkadian lines
                            if line_lang == 'akkadian' and hasattr(self, 'char_lm') and self.char_lm is not None:
                                use_char_lm_for_line = True
                                logger.debug(f"Line {idx}: Akkadian detected, enabling Char LM")
                            elif line_lang != 'akkadian':
                                use_char_lm_for_line = False
                        
                        # Create temp config for this line
                        temp_config = asdict(self.config)
                        temp_config['research']['enable_char_lm_decoding'] = use_char_lm_for_line
                        
                        # Enhance with research features
                        enhanced_text, enhanced_conf = enhance_ocr_with_research_features(
                            text=original_text,
                            confidence=original_conf,
                            config=temp_config,
                            lexicon=self.lexicon if hasattr(self, 'lexicon') else None,
                            char_lm=self.char_lm if use_char_lm_for_line and hasattr(self, 'char_lm') else None,
                            confusion_prior=self.confusion_prior if hasattr(self, 'confusion_prior') else None
                        )
                        
                        # Update element if text changed
                        if enhanced_text != original_text:
                            if isinstance(elem, dict):
                                elem['original_text'] = original_text
                                elem['text'] = enhanced_text
                                elem['confidence'] = enhanced_conf
                                elem['research_enhanced'] = True
                            else:
                                # For OCRSpan objects, update the text attribute
                                elem.text = enhanced_text
                                elem.confidence = enhanced_conf
                                if not hasattr(elem, 'original_text'):
                                    elem.original_text = original_text
                            research_corrections += 1
                    
                    if research_corrections > 0:
                        logger.info(f"Research features enhanced {research_corrections}/{len(ordered_elements)} elements")
            
            timing.research_enhancement_time = time.time() - research_start
            
            # Apply diacritic restoration (DISABLED - see line 327)
            diacritic_start = time.time()
            diacritic_corrections = 0
            # if self.diacritic_restorer and ordered_elements:
            #     logger.info(f"Applying diacritic restoration to {len(ordered_elements)} text elements")
            #     
            #     for elem in ordered_elements:
            #         # Get original text
            #         if isinstance(elem, dict):
            #             original_text = elem.get('text', '')
            #         else:
            #             original_text = elem.text if hasattr(elem, 'text') else ''
            #         
            #         if not original_text:
            #             continue
            #         
            #         # Restore diacritics
            #         restored_text = self.diacritic_restorer.restore_text(original_text)
            #         
            #         # Update element if text changed
            #         if restored_text != original_text:
            #             if isinstance(elem, dict):
            #                 if 'original_text' not in elem:
            #                     elem['original_text'] = original_text
            #                 elem['text'] = restored_text
            #                 elem['diacritic_restored'] = True
            #             else:
            #                 if not hasattr(elem, 'original_text'):
            #                     elem.original_text = original_text
            #                 elem.text = restored_text
            #             diacritic_corrections += 1
            #     
            #     if diacritic_corrections > 0:
            #         stats = self.diacritic_restorer.get_stats()
            #         logger.info(f"Diacritic restoration: {diacritic_corrections}/{len(ordered_elements)} elements changed, "
            #                    f"{stats['diacritics_restored']} diacritics restored")
            
            timing.diacritic_restoration_time = time.time() - diacritic_start
            
            # Apply LLM corrections
            correction_start = time.time()
            corrections_made = 0
            correction_stats = None
            new_llm_corrections = 0
            
            # Use new LLM post-correction system if enabled
            if self.new_llm_corrector and ordered_elements and not self.telemetry.kill_switch.is_llm_disabled:
                logger.info(f"Applying new LLM post-correction to {len(ordered_elements)} elements")
                
                # Process each text element
                for idx, elem in enumerate(ordered_elements):
                    # Get text and confidence
                    if isinstance(elem, dict):
                        text = elem.get('text', '')
                        conf = elem.get('confidence', 0.0)
                        lang = elem.get('language', detected_language)
                    else:
                        text = elem.text if hasattr(elem, 'text') else ''
                        conf = elem.confidence if hasattr(elem, 'confidence') else 0.0
                        lang = elem.language if hasattr(elem, 'language') else detected_language
                    
                    if not text or len(text) < 8:  # Skip very short lines
                        continue
                    
                    # Get context (previous and next lines)
                    prev_line = None
                    next_line = None
                    if idx > 0:
                        prev_elem = ordered_elements[idx - 1]
                        prev_line = prev_elem.get('text', '') if isinstance(prev_elem, dict) else prev_elem.text if hasattr(prev_elem, 'text') else ''
                    if idx < len(ordered_elements) - 1:
                        next_elem = ordered_elements[idx + 1]
                        next_line = next_elem.get('text', '') if isinstance(next_elem, dict) else next_elem.text if hasattr(next_elem, 'text') else ''
                    
                    # Run correction
                    try:
                        # Handle language (could be None)
                        if lang is None:
                            lang = detected_language or 'unknown'
                        lang_clean = lang.split('_')[0] if '_' in str(lang) else str(lang)
                        
                        result = self.new_llm_corrector.correct_line(
                            text=text,
                            lang=lang_clean,
                            confidence=conf,
                            prev_line=prev_line,
                            next_line=next_line,
                            span_id=f"{page_id}_line_{idx}"
                        )
                        
                        # Apply correction if successful
                        if result.applied and result.corrected_text != text:
                            if isinstance(elem, dict):
                                elem['original_text'] = text
                                elem['text'] = result.corrected_text
                                elem['llm_corrected'] = True
                                elem['llm_edit_ratio'] = result.edit_ratio
                            else:
                                elem.original_text = text
                                elem.text = result.corrected_text
                            
                            new_llm_corrections += 1
                            logger.debug(f"Line {idx}: '{text[:50]}...' → '{result.corrected_text[:50]}...' (edit_ratio: {result.edit_ratio:.2%})")
                        
                    except Exception as e:
                        logger.warning(f"LLM correction failed for line {idx}: {e}")
                
                if new_llm_corrections > 0:
                    logger.info(f"✅ New LLM corrected {new_llm_corrections}/{len(ordered_elements)} lines")
                
                # Get telemetry from new LLM corrector
                llm_telemetry = self.new_llm_corrector.get_telemetry()
                correction_stats = {
                    'corrections_applied': new_llm_corrections,
                    'spans_attempted': llm_telemetry.get('llm_spans_attempted', 0),
                    'spans_rejected': llm_telemetry.get('llm_spans_rejected', 0),
                    'cache_hits': llm_telemetry.get('llm_cache_hits', 0),
                    'cache_misses': llm_telemetry.get('llm_cache_misses', 0),
                    'avg_latency_ms': llm_telemetry.get('llm_avg_latency_ms', 0),
                    'guardrail_violations': llm_telemetry.get('guardrail_violations', {})
                }
                
                corrections_made = new_llm_corrections
            
            # Fallback to old LLM corrector if new one not enabled
            elif self.llm_corrector and ordered_elements and not self.telemetry.kill_switch.is_llm_disabled:
                # Check LLM timeout
                if self.telemetry.kill_switch.check_llm_timeout(correction_start, page_id):
                    logger.warning(f"LLM timeout on {page_id} - skipping corrections")
                    correction_stats = None
                else:
                    # Convert elements to OCRSpan objects for enhanced LLM processing
                    ocr_spans = []
                    for elem in ordered_elements:
                        # Skip if already an OCRSpan
                        if isinstance(elem, OCRSpan):
                            ocr_spans.append(elem)
                            continue
                            
                        # Create bounding box from element coordinates (for dict format)
                        bbox = BoundingBox(
                            left=elem.get('x', 0) if isinstance(elem, dict) else 0,
                            top=elem.get('y', 0) if isinstance(elem, dict) else 0,
                            right=(elem.get('x', 0) + elem.get('width', 0)) if isinstance(elem, dict) else 0,
                            bottom=(elem.get('y', 0) + elem.get('height', 0)) if isinstance(elem, dict) else 0
                        )
                        ocr_span = OCRSpan(
                            text=elem.text if hasattr(elem, 'text') else elem.get('text', ''),
                            confidence=elem.confidence if hasattr(elem, 'confidence') else elem.get('confidence', 0.0),
                            bounding_box=bbox
                        )
                        ocr_spans.append(ocr_span)
                    
                    # Apply enhanced LLM corrections
                    correction_results = self.llm_corrector.correct_spans(
                        ocr_spans, detected_language
                    )
                    
                    # Apply corrections to elements
                    for i, corrected_result in enumerate(correction_results):
                        if i < len(ordered_elements) and corrected_result.corrected_text != ocr_spans[i].text:
                            ordered_elements[i]['text'] = corrected_result.corrected_text
                            ordered_elements[i]['original_text'] = ocr_spans[i].text
                            ordered_elements[i]['corrections'] = [f"Enhanced LLM correction applied"]
                            corrections_made += 1
                    
                    correction_stats = self.llm_corrector.get_correction_stats()
            
            timing.llm_correction_time = time.time() - correction_start
            
            # Post-processing
            post_processing_start = time.time()
            
            # Count words and tokens
            word_token_counts = self._count_words_and_tokens(ordered_elements, detected_language)
            
            # Calculate telemetry metrics
            timing.lines_processed = len(ordered_elements)
            timing.characters_processed = sum(
                len(elem.text) if hasattr(elem, 'text') else len(elem['text']) 
                for elem in ordered_elements
            )
            timing.corrections_made = corrections_made + research_corrections + diacritic_corrections
            timing.post_processing_time = time.time() - post_processing_start
            timing.total_time = time.time() - start_time
            
            # Check for timeout
            if self.telemetry.kill_switch.check_page_timeout(start_time, page_id):
                return None, {"error": "page_timeout"}
            
            # Create page result
            page_id = f"page_{page_num:03d}"
            
            # Add to CSV writer
            self.csv_writer.add_page_content(
                page_id=page_id,
                text_elements=ordered_elements,
                language=detected_language,
                correction_stats=correction_stats,
                reading_order_stats={
                    'processing_time': timing.reading_order_time,
                    'elements_processed': len(ordered_elements),
                    'columns_detected': 2 if len(set(
                        elem.bbox.x if hasattr(elem, 'bbox') and hasattr(elem.bbox, 'x') 
                        else elem['bbox'][0] if isinstance(elem, dict) and 'bbox' in elem 
                        else 0 
                        for elem in ordered_elements
                    )) > page_width * 0.3 else 1
                }
            )
            
            # Record telemetry
            self.telemetry.record_page_timing(timing)
            
            # Get page result from CSV writer
            page_result = self.csv_writer.pages_data.get(page_id)
            
            # Create processing stats
            processing_stats = {
                'page_num': page_num,
                'processing_time': timing.total_time,
                'ocr_time': timing.ocr_time,
                'reading_order_time': timing.reading_order_time,
                'research_enhancement_time': timing.research_enhancement_time,
                'diacritic_restoration_time': timing.diacritic_restoration_time,
                'correction_time': timing.llm_correction_time,
                'text_elements': len(ordered_elements),
                'language': detected_language,
                'language_analysis': lang_analysis,  # Add detailed language breakdown
                'corrections_made': corrections_made,
                'research_corrections': research_corrections,
                'diacritic_corrections': diacritic_corrections,
                'avg_confidence': page_result.conf_mean if page_result else 0.0,
                # Research features status (actually applied)
                'research_features_applied': {
                    'char_lm': enable_char_lm_for_page if 'enable_char_lm_for_page' in locals() else False,
                    'lexicon_bias': self.config.research.get('enable_lexicon_bias', False) and hasattr(self, 'lexicon') and self.lexicon is not None,
                    'confusion_prior': self.config.research.get('enable_confusion_prior', False) and hasattr(self, 'confusion_prior') and self.confusion_prior is not None,
                    'rover_ensemble': self.config.research.get('enable_rover_ensemble', False) and hasattr(self, 'rover') and self.rover is not None,
                    'diacritic_restoration': hasattr(self, 'diacritic_restorer') and self.diacritic_restorer is not None
                },
                # New metrics for cost of compute
                'word_count': word_token_counts['word_count'],
                'token_count': word_token_counts['token_count'],
                'text_elements_count': word_token_counts['text_elements_count'],
                'telemetry': {
                    'lines_per_second': timing.lines_per_second,
                    'characters_per_second': timing.characters_per_second,
                    'ms_per_page': timing.ms_per_page
                }
            }
            
            logger.info(f"Processed page {page_num}: {len(ordered_elements)} elements, "
                       f"lang={detected_language}, corrections={corrections_made}")
            
            return page_result, processing_stats
            
        except Exception as e:
            logger.error(f"Error processing page {page_num}: {e}")
            import traceback
            traceback.print_exc()
            
            # Record failed timing
            timing.total_time = time.time() - start_time
            self.telemetry.record_page_timing(timing)
            
            return None, {
                'page_num': page_num,
                'processing_time': timing.total_time,
                'error': str(e),
                'word_count': 0,
                'token_count': 0,
                'text_elements_count': 0
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
        
        # DO NOT clear CSV writer here - we want to accumulate across multiple PDFs in batch mode
        # The CSV writer now uses append mode to properly accumulate results
        # self.csv_writer.clear()  # DISABLED for batch processing
        
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
        
        # Write CSV output (append mode to accumulate across multiple PDFs)
        csv_path = output_dir / "comprehensive_results.csv"
        try:
            self.csv_writer.write_csv(str(csv_path), include_metadata=self.config.output_csv_metadata, append=True)
        except Exception as e:
            logger.error(f"Failed to write CSV: {e}")
            # Try with simpler path
            csv_path = output_dir / "results.csv"
            self.csv_writer.write_csv(str(csv_path), include_metadata=False, append=True)
        
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
            
            if self.llm_corrector and ordered_elements:
                # Convert elements to OCRSpan objects for enhanced LLM processing
                ocr_spans = []
                for elem in ordered_elements:
                    # Skip if already an OCRSpan
                    if isinstance(elem, OCRSpan):
                        ocr_spans.append(elem)
                        continue
                        
                    # Create bounding box from element coordinates (for dict format)
                    bbox = BoundingBox(
                        left=elem.get('x', 0) if isinstance(elem, dict) else 0,
                        top=elem.get('y', 0) if isinstance(elem, dict) else 0,
                        right=(elem.get('x', 0) + elem.get('width', 0)) if isinstance(elem, dict) else 0,
                        bottom=(elem.get('y', 0) + elem.get('height', 0)) if isinstance(elem, dict) else 0
                    )
                    ocr_span = OCRSpan(
                        text=elem.text if hasattr(elem, 'text') else elem.get('text', ''),
                        confidence=elem.confidence if hasattr(elem, 'confidence') else elem.get('confidence', 0.0),
                        bounding_box=bbox
                    )
                    ocr_spans.append(ocr_span)
                
                # Apply enhanced LLM corrections
                correction_results = self.llm_corrector.correct_spans(
                    ocr_spans, detected_language
                )
                
                # Apply corrections to elements
                for i, corrected_span in enumerate(correction_results):
                    if i < len(ordered_elements) and corrected_span.text != ocr_spans[i].text:
                        ordered_elements[i]['text'] = corrected_span.text
                        ordered_elements[i]['original_text'] = ocr_spans[i].text
                        ordered_elements[i]['corrections'] = [f"Enhanced LLM correction applied"]
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
                    'columns_detected': 2 if len(set(
                        elem.bbox.x if hasattr(elem, 'bbox') and hasattr(elem.bbox, 'x') 
                        else elem['bbox'][0] if isinstance(elem, dict) and 'bbox' in elem 
                        else 0 
                        for elem in ordered_elements
                    )) > page_width * 0.3 else 1
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
            
            # Write CSV output (append mode for batch processing)
            csv_path = output_dir / "comprehensive_results.csv"
            try:
                self.csv_writer.write_csv(str(csv_path), include_metadata=self.config.output_csv_metadata, append=True)
            except Exception as e:
                logger.error(f"Failed to write CSV: {e}")
                # Try with simpler path
                csv_path = output_dir / "results.csv"
                self.csv_writer.write_csv(str(csv_path), include_metadata=False, append=True)
            
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
    
    def export_telemetry(self, output_dir: Path) -> None:
        """Export telemetry data and performance metrics."""
        try:
            telemetry_path = output_dir / "telemetry_report.json"
            self.telemetry.export_metrics(str(telemetry_path))
            
            # Also save LLM cache
            if self.llm_corrector:
                self.llm_corrector.save_cache()
                logger.info("LLM cache saved to disk")
            
            logger.info(f"Telemetry exported to {telemetry_path}")
            
        except Exception as e:
            logger.error(f"Failed to export telemetry: {e}")


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
    parser.add_argument('--llm-model', default='mistral:latest', help='LLM model name')
    parser.add_argument('--disable-corrections', action='store_true', 
                       help='Disable LLM corrections')
    parser.add_argument('--disable-reading-order', action='store_true',
                       help='Disable reading order detection')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for PDF rendering')
    
    # New LLM correction system
    parser.add_argument('--enable-new-llm-correction', action='store_true',
                       help='Enable new guardrail-based LLM correction system')
    parser.add_argument('--llm-correction-model', default='qwen2.5:7b-instruct',
                       help='Model for new LLM correction (default: qwen2.5:7b-instruct)')
    parser.add_argument('--llm-correction-edit-budget', type=float, default=0.12,
                       help='Maximum edit ratio for LLM corrections (default: 0.12)')
    parser.add_argument('--llm-correction-cache', action='store_true', default=True,
                       help='Enable caching for LLM corrections (default: enabled)')
    parser.add_argument('--no-llm-correction-cache', dest='llm_correction_cache', action='store_false',
                       help='Disable LLM correction cache')
    
    args = parser.parse_args()
    
    # Create configuration
    config = PipelineConfig(
        llm_provider='none' if args.disable_corrections else args.llm_provider,
        llm_model=args.llm_model,
        enable_llm_correction=not args.disable_corrections,
        enable_reading_order=not args.disable_reading_order,
        dpi=args.dpi,
        # New LLM correction settings
        enable_new_llm_correction=args.enable_new_llm_correction,
        llm_correction_model=args.llm_correction_model,
        llm_correction_edit_budget=args.llm_correction_edit_budget,
        llm_correction_cache_enabled=args.llm_correction_cache
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
