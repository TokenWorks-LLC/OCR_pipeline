#!/usr/bin/env python3
"""
OCR Pipeline - Unified Entry Point
Comprehensive OCR system with ensemble support, LLM correction, and evaluation

Usage:
    # Single page OCR
    python ocr_pipeline.py single --pdf path/to/file.pdf --page 1
    
    # Batch processing
    python ocr_pipeline.py batch --input data/input_pdfs --output data/output
    
    # Gold standard evaluation
    python ocr_pipeline.py eval --mode single    # Single engine (PaddleOCR)
    python ocr_pipeline.py eval --ensemble       # Multi-engine voting
    python ocr_pipeline.py eval --ensemble --llm # Ensemble + LLM correction
    
    # Engine testing
    python ocr_pipeline.py test-engines
"""

import sys
import csv
import json
import time
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import unicodedata
import re

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# IMPORTS - OCR Engines and Pipeline
# =============================================================================

try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    logger.warning("PaddleOCR not available")

try:
    import pytesseract
    from config import TESSERACT_CMD
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("Tesseract not available")

try:
    from engines.easyocr_engine import EasyOCREngine
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logger.warning("EasyOCR not available")

try:
    from enhanced_llm_correction import EnhancedLLMCorrector, create_enhanced_llm_corrector
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    logger.warning("LLM correction not available")

try:
    import fitz  # PyMuPDF
    import cv2
    import numpy as np
except ImportError as e:
    logger.error(f"Missing required dependency: {e}")
    sys.exit(1)

# Import grapheme metrics for evaluation
try:
    from grapheme_metrics import compute_cer_wer, compute_grapheme_cer_wer
except ImportError:
    logger.error("grapheme_metrics.py not found. Please ensure src/grapheme_metrics.py exists.")
    sys.exit(1)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class GoldPage:
    """Gold standard page information."""
    pdf_name: str
    page_num: int
    gold_text: str


@dataclass
class OCRResult:
    """OCR result with metadata."""
    text: str
    boxes: int
    confidence: float
    processing_time: float
    engine: str


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for OCR quality."""
    cer: float  # Character Error Rate
    accuracy: float  # 1 - CER
    ocr_length: int
    gold_length: int
    boxes_detected: int
    processing_time: float


# =============================================================================
# TEXT NORMALIZATION AND CLEANING
# =============================================================================

def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    # Convert combining characters to composed Unicode
    text = unicodedata.normalize('NFC', text)
    
    # Common replacements for Akkadian texts
    replacements = {
        'sˇ': 'š', 'Sˇ': 'Š', 'ˇs': 'š', 'ˇS': 'Š', 's ̌': 'š', 'S ̌': 'Š',
        's.': 'ṣ', 'S.': 'Ṣ', 't.': 'ṭ', 'T.': 'Ṭ',
        'h ̆': 'ḫ', 'H ̆': 'Ḫ',
        'a ́': 'á', 'e ́': 'é', 'i ́': 'í', 'u ́': 'ú',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Remove combining marks
    text = ''.join(c for c in text if not unicodedata.category(c).startswith('M'))
    
    return text


def clean_ocr_output(text: str) -> str:
    """Clean OCR output by removing headers and metadata."""
    # Remove common headers
    text = re.sub(r'\bAKT\s+[IVX]+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bpage\s+\d+\b', '', text, flags=re.IGNORECASE)
    
    # Remove line numbers
    text = re.sub(r'^\s*\[\d+\+?\]\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+\[\d+\+?\]\s+', ' ', text)
    
    # Remove common markers
    text = re.sub(r'^\s*Rs\.\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*o\.K\.\s*', '', text, flags=re.MULTILINE)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


# =============================================================================
# OCR ENGINES
# =============================================================================

class OCREngine:
    """Base OCR Engine interface."""
    
    def __init__(self):
        self.name = "base"
    
    def initialize(self):
        """Initialize the engine."""
        raise NotImplementedError
    
    def process_image(self, image: np.ndarray) -> OCRResult:
        """Process an image and return OCR result."""
        raise NotImplementedError


class PaddleOCREngine(OCREngine):
    """PaddleOCR engine wrapper."""
    
    def __init__(self):
        super().__init__()
        self.name = "paddle"
        self.engine = None
    
    def initialize(self):
        """Initialize PaddleOCR."""
        if not PADDLE_AVAILABLE:
            raise RuntimeError("PaddleOCR not available")
        
        self.engine = PaddleOCR(use_textline_orientation=True, lang='en')
        logger.info("PaddleOCR initialized")
    
    def process_image(self, image: np.ndarray) -> OCRResult:
        """Process image with PaddleOCR."""
        start_time = time.time()
        
        result = self.engine.ocr(image)
        
        # Extract text and metadata
        text_parts = []
        num_boxes = 0
        total_conf = 0.0
        
        if result and len(result) > 0:
            page_result = result[0]
            if hasattr(page_result, 'keys'):
                rec_texts = page_result.get('rec_texts', [])
                rec_scores = page_result.get('rec_scores', [])
                
                text_parts = rec_texts
                num_boxes = len(rec_texts)
                total_conf = sum(rec_scores) / max(len(rec_scores), 1)
        
        processing_time = time.time() - start_time
        
        return OCRResult(
            text=' '.join(text_parts),
            boxes=num_boxes,
            confidence=total_conf,
            processing_time=processing_time,
            engine=self.name
        )


class TesseractEngine(OCREngine):
    """Tesseract OCR engine wrapper."""
    
    def __init__(self):
        super().__init__()
        self.name = "tesseract"
    
    def initialize(self):
        """Initialize Tesseract."""
        if not TESSERACT_AVAILABLE:
            raise RuntimeError("Tesseract not available")
        
        # Test Tesseract
        version = pytesseract.get_tesseract_version()
        logger.info(f"Tesseract initialized: v{version}")
    
    def process_image(self, image: np.ndarray) -> OCRResult:
        """Process image with Tesseract."""
        start_time = time.time()
        
        # Convert to numpy array if needed
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        # Get detailed data
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        
        # Extract text with confidence > 30%
        text_parts = []
        confidences = []
        
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            conf = float(data['conf'][i]) / 100.0 if data['conf'][i] != -1 else 0.0
            
            if text and conf > 0.3:
                text_parts.append(text)
                confidences.append(conf)
        
        processing_time = time.time() - start_time
        avg_conf = sum(confidences) / max(len(confidences), 1)
        
        return OCRResult(
            text=' '.join(text_parts),
            boxes=len(text_parts),
            confidence=avg_conf,
            processing_time=processing_time,
            engine=self.name
        )


class EasyOCREngineWrapper(OCREngine):
    """EasyOCR engine wrapper."""
    
    def __init__(self):
        super().__init__()
        self.name = "easyocr"
        self.engine = None
    
    def initialize(self):
        """Initialize EasyOCR."""
        if not EASYOCR_AVAILABLE:
            raise RuntimeError("EasyOCR not available")
        
        self.engine = EasyOCREngine({'lang_list': ['en'], 'gpu': True})
        logger.info("EasyOCR initialized")
    
    def process_image(self, image: np.ndarray) -> OCRResult:
        """Process image with EasyOCR."""
        start_time = time.time()
        
        result = self.engine.ocr(image)
        
        # Extract text and metadata
        text_parts = []
        confidences = []
        
        for detection in result:
            text_parts.append(detection['text'])
            confidences.append(detection['confidence'])
        
        processing_time = time.time() - start_time
        avg_conf = sum(confidences) / max(len(confidences), 1)
        
        return OCRResult(
            text=' '.join(text_parts),
            boxes=len(text_parts),
            confidence=avg_conf,
            processing_time=processing_time,
            engine=self.name
        )


# =============================================================================
# ENSEMBLE OCR
# =============================================================================

class EnsembleOCR:
    """Multi-engine OCR with voting/fusion."""
    
    def __init__(self, engines: List[OCREngine]):
        self.engines = engines
        logger.info(f"Ensemble initialized with {len(engines)} engines: {[e.name for e in engines]}")
    
    def process_image(self, image: np.ndarray) -> OCRResult:
        """Process image with all engines and combine results."""
        start_time = time.time()
        
        all_results = []
        all_detections = []
        
        # Run all engines sequentially to reduce memory pressure
        for engine in self.engines:
            try:
                logger.info(f"Running {engine.name}...")
                result = engine.process_image(image)
                all_results.append(result)
                
                # Store individual detections for voting
                if result.text:
                    all_detections.append({
                        'text': result.text,
                        'conf': result.confidence,
                        'engine': engine.name,
                        'boxes': result.boxes
                    })
                
                # Clear GPU memory if available
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except:
                    pass
                    
            except Exception as e:
                logger.error(f"{engine.name} failed: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if not all_results:
            logger.warning("All engines failed, returning empty result")
            return OCRResult("", 0, 0.0, time.time() - start_time, "ensemble")
        
        # Simple fusion: use highest confidence result
        best_result = max(all_results, key=lambda x: x.confidence * len(x.text))
        
        # Combine metadata
        total_boxes = sum(r.boxes for r in all_results)
        avg_conf = sum(r.confidence for r in all_results) / len(all_results)
        processing_time = time.time() - start_time
        
        return OCRResult(
            text=best_result.text,
            boxes=total_boxes,
            confidence=avg_conf,
            processing_time=processing_time,
            engine=f"ensemble({','.join([r.engine for r in all_results])})"
        )


# =============================================================================
# IMAGE PROCESSING
# =============================================================================

def render_pdf_page(pdf_path: Path, page_num: int, dpi: int = 300) -> np.ndarray:
    """Render PDF page to image at 300 DPI for quality OCR."""
    doc = fitz.open(str(pdf_path))
    page = doc[page_num - 1]
    pix = page.get_pixmap(dpi=dpi)
    
    img_bytes = pix.tobytes("ppm")
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    doc.close()
    
    # Resize if too large (PaddleOCR 4000px limit)
    h, w = image.shape[:2]
    max_dim = max(h, w)
    if max_dim > 3500:
        scale = 3500 / max_dim
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        logger.info(f"Resized from {w}x{h} to {new_w}x{new_h}")
    
    return image


# =============================================================================
# GOLD DATA LOADING
# =============================================================================

def load_gold_pages(gold_csv: Path) -> List[GoldPage]:
    """Load gold standard pages from CSV."""
    gold_pages = []
    
    with open(gold_csv, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        
        for row in reader:
            if len(row) >= 3:
                pdf_name = row[0].strip()
                try:
                    page_num = int(row[1].strip())
                    gold_text = row[2].strip()
                    
                    # Skip Secondary Sources
                    if 'Secondary Sources' not in pdf_name and pdf_name:
                        gold_pages.append(GoldPage(pdf_name, page_num, gold_text))
                except ValueError:
                    continue
    
    logger.info(f"Loaded {len(gold_pages)} gold pages")
    return gold_pages


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_ocr(ocr_text: str, gold_text: str, ocr_result: OCRResult, 
                 llm_corrector=None, use_llm: bool = False) -> EvaluationMetrics:
    """Evaluate OCR quality against gold standard."""
    # Optionally apply LLM correction
    if use_llm and llm_corrector and LLM_AVAILABLE:
        try:
            logger.info("Applying LLM correction...")
            corrected_text = llm_corrector.correct_page(ocr_text, language='en', avg_confidence=ocr_result.confidence)
            ocr_text = corrected_text.corrected_text if hasattr(corrected_text, 'corrected_text') else corrected_text
        except Exception as e:
            logger.warning(f"LLM correction failed: {e}")
    
    # Clean and normalize
    ocr_clean = clean_ocr_output(normalize_text(ocr_text))
    gold_clean = normalize_text(gold_text)
    
    # Calculate CER using grapheme metrics
    metrics = compute_cer_wer(gold_clean, ocr_clean)
    cer = metrics['cer']
    accuracy = max(0, 1 - cer) * 100
    
    return EvaluationMetrics(
        cer=cer,
        accuracy=accuracy,
        ocr_length=len(ocr_clean),
        gold_length=len(gold_clean),
        boxes_detected=ocr_result.boxes,
        processing_time=ocr_result.processing_time
    )


def run_evaluation(mode: str = 'single', use_llm: bool = False,
                   input_dir: Path = Path('data/input_pdfs'),
                   gold_csv: Path = Path('data/gold_data/gold_pages.csv'),
                   output_csv: Path = Path('data/evaluation_results.csv')) -> Dict[str, Any]:
    """Run evaluation on gold pages."""
    
    print("=" * 80)
    print(f"OCR EVALUATION - {mode.upper()} MODE" + (" + LLM" if use_llm else ""))
    print("=" * 80)
    
    # Load gold pages
    gold_pages = load_gold_pages(gold_csv)
    
    # Initialize LLM corrector if requested
    llm_corrector = None
    if use_llm and LLM_AVAILABLE:
        print("\nInitializing LLM corrector...")
        try:
            llm_corrector = create_enhanced_llm_corrector()
            print("[OK] LLM corrector initialized with guardrails:")
            print("  - Edit budget: ≤12% for modern text, ≤3% for transliteration")
            print("  - Bracket preservation enforced")
            print("  - Vocabulary explosion detection")
            print("  - Confidence-based routing")
        except Exception as e:
            logger.error(f"Failed to initialize LLM corrector: {e}")
            use_llm = False
    elif use_llm and not LLM_AVAILABLE:
        print("\n[WARNING] LLM correction requested but not available")
        use_llm = False
    
    # Initialize engine(s)
    if mode == 'single':
        print("\nInitializing PaddleOCR...")
        engine = PaddleOCREngine()
        engine.initialize()
        engines = [engine]
        processor = engine
    elif mode == 'ensemble':
        print("\nInitializing ensemble engines...")
        engines = []
        
        if PADDLE_AVAILABLE:
            paddle = PaddleOCREngine()
            paddle.initialize()
            engines.append(paddle)
        
        if TESSERACT_AVAILABLE:
            tess = TesseractEngine()
            tess.initialize()
            engines.append(tess)
        
        if EASYOCR_AVAILABLE:
            easy = EasyOCREngineWrapper()
            easy.initialize()
            engines.append(easy)
        
        if not engines:
            raise RuntimeError("No OCR engines available")
        
        processor = EnsembleOCR(engines)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    print(f"\nProcessing {len(gold_pages)} gold pages...\n")
    
    # Process pages
    results = []
    start_time = time.time()
    save_interval = 5  # Save results every 5 pages
    
    # Create output directory
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    for idx, gold_page in enumerate(gold_pages, 1):
        print(f"[{idx}/{len(gold_pages)}] {gold_page.pdf_name}, page {gold_page.page_num}")
        
        pdf_path = input_dir / gold_page.pdf_name
        
        if not pdf_path.exists():
            print(f"  [SKIP] PDF not found")
            continue
        
        try:
            # Render page
            image = render_pdf_page(pdf_path, gold_page.page_num)
            
            # OCR
            ocr_result = processor.process_image(image)
            
            # Evaluate
            metrics = evaluate_ocr(ocr_result.text, gold_page.gold_text, ocr_result, 
                                 llm_corrector=llm_corrector, use_llm=use_llm)
            
            results.append({
                'pdf': gold_page.pdf_name,
                'page': gold_page.page_num,
                'engine': ocr_result.engine,
                'time': metrics.processing_time,
                'boxes': metrics.boxes_detected,
                'cer': metrics.cer,
                'accuracy': metrics.accuracy,
                'ocr_len': metrics.ocr_length,
                'gold_len': metrics.gold_length
            })
            
            print(f"  ✓ {metrics.processing_time:.1f}s | {metrics.boxes_detected} boxes | Accuracy: {metrics.accuracy:.1f}%")
            
            # Save periodically to prevent data loss
            if idx % save_interval == 0 and results:
                try:
                    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=results[0].keys())
                        writer.writeheader()
                        writer.writerows(results)
                    print(f"  [CHECKPOINT] Saved {len(results)} results to {output_csv}")
                except Exception as save_error:
                    logger.warning(f"Failed to save checkpoint: {save_error}")
            
            # Aggressive memory cleanup after each page
            del image
            del ocr_result
            
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except:
                pass
            
            # Force garbage collection
            import gc
            gc.collect()
            
        except KeyboardInterrupt:
            print("\n\n[!] Interrupted by user")
            break
        except Exception as e:
            print(f"  ✗ Error: {str(e)[:100]}")
            import traceback
            traceback.print_exc()
            
            # Save what we have so far
            if results:
                try:
                    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=results[0].keys())
                        writer.writeheader()
                        writer.writerows(results)
                    print(f"  [EMERGENCY SAVE] Saved {len(results)} results before crash")
                except:
                    pass
            
            # Try to clear memory and continue
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except:
                pass
            
            import gc
            gc.collect()
            continue
    
    # Summary
    total_time = time.time() - start_time
    
    # Final save
    if results:
        try:
            with open(output_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
            print(f"\n[FINAL SAVE] Results saved to: {output_csv}")
        except Exception as e:
            logger.error(f"Failed to save final results: {e}")
    
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"Pages processed: {len(results)}/{len(gold_pages)}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    
    if results:
        avg_time = sum(r['time'] for r in results) / len(results)
        avg_acc = sum(r['accuracy'] for r in results) / len(results)
        avg_cer = sum(r['cer'] for r in results) / len(results)
        avg_boxes = sum(r['boxes'] for r in results) / len(results)
        
        print(f"Average time/page: {avg_time:.1f}s")
        print(f"Average accuracy: {avg_acc:.1f}%")
        print(f"Average CER: {avg_cer:.2%}")
        print(f"Average boxes: {avg_boxes:.0f}")
        
        # Best and worst
        best = max(results, key=lambda x: x['accuracy'])
        worst = min(results, key=lambda x: x['accuracy'])
        
        print(f"\n🏆 Best: {best['pdf']} p{best['page']} - {best['accuracy']:.1f}%")
        print(f"⚠️  Worst: {worst['pdf']} p{worst['page']} - {worst['accuracy']:.1f}%")
    
    print("=" * 80)
    
    return {
        'mode': mode,
        'total_pages': len(gold_pages),
        'processed_pages': len(results),
        'avg_accuracy': avg_acc if results else 0,
        'avg_time': avg_time if results else 0,
        'total_time': total_time
    }


# =============================================================================
# ENGINE TESTING
# =============================================================================

def test_engines():
    """Test all available OCR engines."""
    print("=" * 80)
    print("OCR ENGINE TEST")
    print("=" * 80)
    
    engines_status = {
        'PaddleOCR': PADDLE_AVAILABLE,
        'Tesseract': TESSERACT_AVAILABLE,
        'EasyOCR': EASYOCR_AVAILABLE
    }
    
    print("\nEngine Availability:")
    for name, available in engines_status.items():
        status = "✓ Available" if available else "✗ Not Available"
        print(f"  {name}: {status}")
    
    # Test GPU
    try:
        import torch
        print(f"\nGPU Status:")
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  Device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("\nPyTorch not available")
    
    # Test PaddleOCR GPU
    if PADDLE_AVAILABLE:
        try:
            import paddle
            print(f"\nPaddlePaddle:")
            print(f"  Version: {paddle.__version__}")
            print(f"  CUDA compiled: {paddle.device.is_compiled_with_cuda()}")
        except:
            pass
    
    print("=" * 80)


# =============================================================================
# SINGLE PAGE OCR
# =============================================================================

def process_single_page(pdf_path: Path, page_num: int, mode: str = 'single'):
    """Process a single PDF page."""
    print("=" * 80)
    print(f"SINGLE PAGE OCR - {mode.upper()} MODE")
    print("=" * 80)
    print(f"PDF: {pdf_path}")
    print(f"Page: {page_num}\n")
    
    # Initialize engine
    if mode == 'single':
        print("Initializing PaddleOCR...")
        engine = PaddleOCREngine()
        engine.initialize()
        processor = engine
    elif mode == 'ensemble':
        print("Initializing ensemble...")
        engines = []
        if PADDLE_AVAILABLE:
            paddle = PaddleOCREngine()
            paddle.initialize()
            engines.append(paddle)
        if TESSERACT_AVAILABLE:
            tess = TesseractEngine()
            tess.initialize()
            engines.append(tess)
        if EASYOCR_AVAILABLE:
            easy = EasyOCREngineWrapper()
            easy.initialize()
            engines.append(easy)
        processor = EnsembleOCR(engines)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Render and process
    print("\nRendering page...")
    image = render_pdf_page(pdf_path, page_num)
    
    print("Processing with OCR...")
    result = processor.process_image(image)
    
    # Display results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Engine: {result.engine}")
    print(f"Processing time: {result.processing_time:.2f}s")
    print(f"Boxes detected: {result.boxes}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"\nExtracted text ({len(result.text)} chars):")
    print("-" * 80)
    print(result.text[:500])
    if len(result.text) > 500:
        print(f"... ({len(result.text) - 500} more chars)")
    print("=" * 80)


# =============================================================================
# MAIN CLI
# =============================================================================

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='OCR Pipeline - Unified Entry Point',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test engines
  python ocr_pipeline.py test-engines
  
  # Single page OCR
  python ocr_pipeline.py single --pdf data/input_pdfs/test.pdf --page 1
  
  # Single page with ensemble
  python ocr_pipeline.py single --pdf data/input_pdfs/test.pdf --page 1 --ensemble
  
  # Evaluate on gold pages (single engine)
  python ocr_pipeline.py eval
  
  # Evaluate with ensemble
  python ocr_pipeline.py eval --ensemble
  
  # With cache invalidation
  python ocr_pipeline.py eval --invalidate llm
        """
    )
    
    # Global arguments
    parser.add_argument(
        '--invalidate',
        type=str,
        choices=['render', 'ocr', 'fusion', 'llm', 'all'],
        help='Invalidate cache stage before processing'
    )
    parser.add_argument(
        '--profile',
        type=str,
        help='Path to profile JSON (e.g., profiles/akkadian_strict.json)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Test engines
    subparsers.add_parser('test-engines', help='Test available OCR engines')
    
    # Single page
    single_parser = subparsers.add_parser('single', help='Process a single PDF page')
    single_parser.add_argument('--pdf', type=str, required=True, help='Path to PDF file')
    single_parser.add_argument('--page', type=int, required=True, help='Page number (1-indexed)')
    single_parser.add_argument('--ensemble', action='store_true', help='Use ensemble mode')
    
    # Evaluation
    eval_parser = subparsers.add_parser('eval', help='Evaluate on gold standard pages')
    eval_parser.add_argument('--ensemble', action='store_true', help='Use ensemble mode')
    eval_parser.add_argument('--llm', action='store_true', help='Enable LLM post-correction')
    eval_parser.add_argument('--input', type=str, default='data/input_pdfs', help='Input PDFs directory')
    eval_parser.add_argument('--gold', type=str, default='data/gold_data/gold_pages.csv', help='Gold CSV path')
    eval_parser.add_argument('--output', type=str, default='data/evaluation_results.csv', help='Output CSV path')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Handle cache invalidation if requested
    if args.invalidate:
        try:
            from cache_store import CacheStore
            cache = CacheStore(cache_dir='cache/pipeline', enabled=True)
            count = cache.invalidate(args.invalidate)
            logger.info(f"✓ Invalidated {count} cache entries for stage: {args.invalidate}")
        except Exception as e:
            logger.warning(f"Cache invalidation failed: {e}")
    
    try:
        if args.command == 'test-engines':
            test_engines()
        
        elif args.command == 'single':
            mode = 'ensemble' if args.ensemble else 'single'
            process_single_page(Path(args.pdf), args.page, mode)
        
        elif args.command == 'eval':
            mode = 'ensemble' if args.ensemble else 'single'
            suffix = f"_{mode}"
            if args.llm:
                suffix += "_llm"
            output_name = f"eval{suffix}_results.csv"
            run_evaluation(
                mode=mode,
                use_llm=args.llm,
                input_dir=Path(args.input),
                gold_csv=Path(args.gold),
                output_csv=Path('data') / output_name
            )
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
