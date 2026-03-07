#!/usr/bin/env python3
"""
Page-level text extraction with Akkadian detection.

Extracts page-level text from PDFs (with optional OCR fallback),
detects Akkadian content per page, and outputs a 4-column CSV:

    pdf_name,page,page_text,has_akkadian

Usage:
    python tools/run_page_text.py \\
        --manifest data\\gold\\manifest.txt \\
        --output-root reports\\page_text_20251009 \\
        --prefer-text-layer \\
        --ocr-fallback paddle \\
        --status-bar \\
        --progress-csv reports\\page_text_20251009\\progress.csv
        
    OR:
    
    python tools/run_page_text.py \\
        --inputs "G:\\Shared drives\\Secondary Sources" \\
        --output-root reports\\page_text_20251009 \\
        --prefer-text-layer
"""
import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Core dependencies
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
    PDFMINER_AVAILABLE = True
except ImportError:
    PDFMINER_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AkkadianDetector:
    """Page-level Akkadian detection with any-line aggregation."""
    
    def __init__(self, profile_path: Optional[str] = None):
        """Initialize detector with profile configuration."""
        self.config = self._load_profile(profile_path)
        self.akkadian_lm = None
        
        # Try to load char LM if available
        try:
            from python_char_lm import PythonCharLM
            lm_path = os.environ.get('AKKADIAN_LM_PATH', 'models/akkadian_char_lm.json')
            if os.path.exists(lm_path):
                self.akkadian_lm = PythonCharLM()
                self.akkadian_lm.load(lm_path)
                logger.info(f"Loaded Akkadian char LM from {lm_path}")
        except (ImportError, Exception) as e:
            logger.debug(f"Akkadian char LM not available: {e}")
    
    def _load_profile(self, profile_path: Optional[str]) -> Dict:
        """Load detection profile from JSON."""
        default_config = {
            "threshold": 0.25,
            "require_diacritic_or_marker": True,
            "min_diacritics_per_line": 1,
            "min_syllabic_tokens": 3,
            "min_syllabic_ratio": 0.25,
            "aggregation_mode": "any-line",
            "aggregation_qual_lines_min": 3,
            "aggregation_qual_ratio_min": 0.25,
            "markers_strict": True,
            "ppl_boosts": {"lt20": 0.3, "lt40": 0.1},
            "negative_lexicon": [
                "der", "die", "das", "und", "den", "des", "dem", "im", "vom", "zum", "zur",
                "für", "mit", "nach", "bei", "über", "auf", "aus", "nicht", "auch", "nur", "sich",
                "ve", "ile", "için", "bu", "bir", "veya", "de", "da", "olarak", "gibi", "ki", "mi",
                "the", "and", "of", "to", "in", "a", "is", "was", "are", "were", "been",
                "being", "have", "has", "had", "do", "does", "did", "will", "would", "should",
                "could", "may", "might", "must", "can"
            ],
            "neg_penalty_cap": 0.15
        }
        
        if profile_path and os.path.exists(profile_path):
            try:
                with open(profile_path, 'r', encoding='utf-8') as f:
                    profile = json.load(f)
                    if 'akkadian_detection' in profile:
                        default_config.update(profile['akkadian_detection'])
                        logger.info(f"Loaded profile from {profile_path}")
            except Exception as e:
                logger.warning(f"Failed to load profile {profile_path}: {e}, using defaults")
        
        return default_config
    
    def detect_line(self, line: str) -> Tuple[bool, float]:
        """Detect if a single line contains Akkadian transliteration."""
        if not line or len(line.strip()) < 3:
            return False, 0.0
        
        cfg = self.config
        
        # Akkadian markers (determinatives, logograms)
        STRICT_MARKERS = {
            "DUMU", "LUGAL", "KÙ.BABBAR", "KUBABBAR", "KU.BABBAR", "URU", 
            "É", "É.GAL", "EGAL", "KUR", "LU₂", "LÚ", "MUNUS", "MÍ",
            "GIŠ", "ᵈ", "ᵐ", "ᶠ"
        }
        
        # Diacritic chars
        DIACRITIC_CHARS = "āēīūŠšṢṣṬṭḪḫáéíóúàèìù"
        
        # Count diacritics
        num_diacritics = sum(1 for char in line if char in DIACRITIC_CHARS)
        has_diacritic = num_diacritics > 0
        
        # Check for markers
        line_upper = line.upper()
        has_marker = any(m in line_upper for m in STRICT_MARKERS)
        
        # Check for syllabic pattern (hyphenated transliteration)
        syllabic_pattern = re.compile(
            r'\b[a-zšṣṭḫāēīū]{1,4}(?:[-—][a-zšṣṭḫāēīū]{1,4}){2,}\b', 
            re.IGNORECASE
        )
        syllabic_matches = syllabic_pattern.findall(line)
        
        # Count tokens
        all_tokens = re.findall(
            r'\b[a-zšṣṭḫāēīū]+(?:[-—][a-zšṣṭḫāēīū]+)*\b', 
            line, 
            re.IGNORECASE
        )
        total_tokens = len(all_tokens)
        syllabic_token_count = len(syllabic_matches)
        syllabic_ratio = syllabic_token_count / total_tokens if total_tokens > 0 else 0.0
        
        # Apply density gates
        min_syllabic_tokens = cfg.get("min_syllabic_tokens", 3)
        min_syllabic_ratio = cfg.get("min_syllabic_ratio", 0.25)
        
        has_syllabic = (
            syllabic_token_count >= min_syllabic_tokens and 
            syllabic_ratio >= min_syllabic_ratio
        )
        
        # Require diacritics or markers when syllabic pattern present
        if cfg.get("require_diacritic_or_marker", True) and has_syllabic:
            min_diac = cfg.get("min_diacritics_per_line", 1)
            has_syllabic = (num_diacritics >= min_diac) or has_marker
        
        # Negative lexicon penalty
        neg_lexicon = set(cfg.get("negative_lexicon", []))
        tokens_lower = [t.lower() for t in all_tokens]
        neg_count = sum(1 for t in tokens_lower if t in neg_lexicon)
        neg_penalty_cap = cfg.get("neg_penalty_cap", 0.15)
        neg_penalty = min(neg_penalty_cap, 0.03 * neg_count)
        
        # Calculate score
        score = 0.0
        if has_syllabic:
            score += 0.45
        if has_diacritic:
            score += 0.20
        if has_marker:
            score += 0.15
        
        # Char LM boost
        if self.akkadian_lm:
            try:
                ppl = self.akkadian_lm.perplexity(line)
                ppl_boosts = cfg.get("ppl_boosts", {})
                if ppl < 20:
                    score += ppl_boosts.get("lt20", 0.3)
                elif ppl < 40:
                    score += ppl_boosts.get("lt40", 0.1)
            except Exception:
                pass
        
        score -= neg_penalty
        score = max(0.0, min(1.0, score))
        
        threshold = cfg.get("threshold", 0.25)
        is_akkadian = score >= threshold
        
        return is_akkadian, score
    
    def detect_page(self, text: str) -> Tuple[bool, Dict]:
        """
        Detect Akkadian at page level using any-line aggregation.
        
        Returns:
            Tuple of (has_akkadian, metadata_dict)
        """
        if not text or len(text.strip()) < 10:
            return False, {"qualified_lines": 0, "total_lines": 0, "ratio": 0.0}
        
        lines = text.split('\n')
        qualified_lines = 0
        line_scores = []
        
        for line in lines:
            line = line.strip()
            if len(line) < 3:
                continue
            
            is_akk, score = self.detect_line(line)
            line_scores.append(score)
            if is_akk:
                qualified_lines += 1
        
        total_lines = len([l for l in lines if len(l.strip()) >= 3])
        qual_ratio = qualified_lines / total_lines if total_lines > 0 else 0.0
        
        # Any-line aggregation
        min_qual_lines = self.config.get("aggregation_qual_lines_min", 3)
        min_qual_ratio = self.config.get("aggregation_qual_ratio_min", 0.25)
        
        has_akkadian = (qualified_lines >= min_qual_lines) or (qual_ratio >= min_qual_ratio)
        
        metadata = {
            "qualified_lines": qualified_lines,
            "total_lines": total_lines,
            "ratio": qual_ratio,
            "max_score": max(line_scores) if line_scores else 0.0
        }
        
        return has_akkadian, metadata


class PDFTextExtractor:
    """Extract text from PDFs with optional OCR fallback."""
    
    def __init__(self, prefer_text_layer: bool = True, ocr_fallback: Optional[str] = None):
        """
        Initialize PDF text extractor.
        
        Args:
            prefer_text_layer: Try text layer extraction first
            ocr_fallback: OCR engine to use for fallback ('paddle' or None)
        """
        self.prefer_text_layer = prefer_text_layer
        self.ocr_fallback = ocr_fallback
        self.ocr_engine = None
        
        if ocr_fallback == 'paddle':
            self._init_paddle_ocr()
    
    def _init_paddle_ocr(self):
        """Initialize PaddleOCR engine."""
        try:
            from paddleocr import PaddleOCR
            self.ocr_engine = PaddleOCR(lang='en')
            logger.info("Initialized PaddleOCR for fallback")
        except Exception as e:
            logger.warning(f"Failed to initialize PaddleOCR: {e}")
            self.ocr_fallback = None
    
    def extract_page_text(self, pdf_path: str, page_num: int) -> Tuple[str, bool, Dict]:
        """
        Extract text from a PDF page.
        
        Args:
            pdf_path: Path to PDF file
            page_num: Page number (0-based)
        
        Returns:
            Tuple of (text, used_text_layer, metadata)
        """
        metadata = {"method": "unknown", "char_count": 0}
        
        # Try text layer first
        if self.prefer_text_layer:
            text, success = self._extract_text_layer(pdf_path, page_num)
            if success and len(text.strip()) >= 16:
                metadata["method"] = "text_layer"
                metadata["char_count"] = len(text)
                return text, True, metadata
        
        # Fallback to OCR if enabled
        if self.ocr_fallback:
            text, success = self._extract_via_ocr(pdf_path, page_num)
            if success:
                metadata["method"] = f"ocr_{self.ocr_fallback}"
                metadata["char_count"] = len(text)
                return text, False, metadata
        
        # Return empty if all methods failed
        return "", False, {"method": "failed", "char_count": 0}
    
    def _extract_text_layer(self, pdf_path: str, page_num: int) -> Tuple[str, bool]:
        """Extract text from PDF text layer using PyMuPDF."""
        if not PYMUPDF_AVAILABLE:
            return "", False
        
        try:
            doc = fitz.open(pdf_path)
            if page_num >= len(doc):
                return "", False
            
            page = doc[page_num]
            text = page.get_text()
            doc.close()
            
            # Normalize whitespace
            text = self._normalize_whitespace(text)
            
            return text, True
        except Exception as e:
            logger.debug(f"Text layer extraction failed for {pdf_path} page {page_num}: {e}")
            return "", False
    
    def _extract_via_ocr(self, pdf_path: str, page_num: int) -> Tuple[str, bool]:
        """Extract text via OCR (PaddleOCR)."""
        if not self.ocr_engine:
            return "", False
        
        try:
            # Render page to image at 300 DPI
            doc = fitz.open(pdf_path)
            if page_num >= len(doc):
                return "", False
            
            page = doc[page_num]
            # 300 DPI: matrix scale = 300/72 = 4.166...
            mat = fitz.Matrix(4.166, 4.166)
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to numpy array
            import numpy as np
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            
            # Run OCR
            result = self.ocr_engine.ocr(img, cls=True)
            
            # Extract text
            lines = []
            if result and result[0]:
                for line in result[0]:
                    if line and len(line) >= 2:
                        text = line[1][0]
                        lines.append(text)
            
            text = '\n'.join(lines)
            text = self._normalize_whitespace(text)
            
            doc.close()
            return text, True
            
        except Exception as e:
            logger.debug(f"OCR extraction failed for {pdf_path} page {page_num}: {e}")
            return "", False
    
    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Normalize whitespace while preserving newlines."""
        # Collapse >2 spaces to single space on each line
        lines = text.split('\n')
        normalized_lines = []
        for line in lines:
            # Collapse multiple spaces
            line = re.sub(r'  +', ' ', line)
            normalized_lines.append(line.strip())
        
        # Remove empty lines but keep paragraph breaks
        result = '\n'.join(normalized_lines)
        # Collapse >2 consecutive newlines to 2
        result = re.sub(r'\n{3,}', '\n\n', result)
        
        return result.strip()


class PageTextPipeline:
    """Main pipeline for page-level text extraction."""
    
    def __init__(self, args):
        """Initialize pipeline with CLI arguments."""
        self.args = args
        self.output_root = Path(args.output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.extractor = PDFTextExtractor(
            prefer_text_layer=args.prefer_text_layer,
            ocr_fallback=args.ocr_fallback if args.ocr_fallback != 'none' else None
        )
        
        profile_path = args.profile or 'profiles/akkadian_strict.json'
        self.detector = AkkadianDetector(profile_path)
        
        # Output CSV paths
        self.output_csv = self.output_root / "client_page_text.csv"
        self.progress_csv = args.progress_csv or (self.output_root / "progress.csv")
        
        # Stats
        self.stats = {
            "pages_processed": 0,
            "pages_with_akkadian": 0,
            "text_layer_used": 0,
            "ocr_used": 0,
            "errors": 0
        }
    
    def run(self):
        """Run the pipeline."""
        logger.info("Starting page text extraction pipeline")
        logger.info(f"Output: {self.output_csv}")
        
        # Get PDF pages to process
        pages = self._collect_pages()
        logger.info(f"Found {len(pages)} pages to process")
        
        if not pages:
            logger.warning("No pages to process")
            return
        
        # Initialize output CSVs
        self._init_output_csv()
        self._init_progress_csv()
        
        # Process pages
        iterator = tqdm(pages, desc="Processing pages") if TQDM_AVAILABLE and self.args.status_bar else pages
        
        for pdf_path, page_num in iterator:
            self._process_page(pdf_path, page_num)
        
        # Report stats
        self._report_stats()
    
    def _collect_pages(self) -> List[Tuple[str, int]]:
        """Collect PDF pages from manifest or inputs directory."""
        pages = []
        
        if self.args.manifest:
            # Read manifest TSV
            with open(self.args.manifest, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # Skip header line
                    if i == 0 and line.startswith('pdf_path'):
                        continue
                    
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        pdf_path = parts[0]
                        try:
                            page_num = int(parts[1]) - 1  # Convert to 0-based
                            # Trust manifest - don't check existence (too slow for large manifests)
                            pages.append((pdf_path, page_num))
                        except ValueError:
                            # Skip lines with non-numeric page numbers (like headers)
                            continue
        
        elif self.args.inputs:
            # Recursively find PDFs
            input_path = Path(self.args.inputs)
            pdf_files = list(input_path.rglob("*.pdf"))
            
            for pdf_path in pdf_files:
                # Get page count
                try:
                    if PYMUPDF_AVAILABLE:
                        doc = fitz.open(str(pdf_path))
                        page_count = len(doc)
                        doc.close()
                        
                        for page_num in range(page_count):
                            pages.append((str(pdf_path), page_num))
                except Exception as e:
                    logger.error(f"Failed to open {pdf_path}: {e}")
        
        return pages
    
    def _init_output_csv(self):
        """Initialize output CSV with UTF-8 BOM."""
        with open(self.output_csv, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['pdf_name', 'page', 'page_text', 'has_akkadian'])
    
    def _init_progress_csv(self):
        """Initialize progress CSV."""
        with open(self.progress_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'pdf_name', 'page', 'ms', 'used_text_layer', 
                'has_akkadian', 'timestamp'
            ])
    
    def _process_page(self, pdf_path: str, page_num: int):
        """Process a single PDF page."""
        start_time = time.time()
        pdf_name = os.path.basename(pdf_path)
        page_1based = page_num + 1
        
        try:
            # Extract text
            text, used_text_layer, extract_meta = self.extractor.extract_page_text(pdf_path, page_num)
            
            if not text:
                logger.warning(f"No text extracted from {pdf_name} page {page_1based}")
                self.stats["errors"] += 1
                return
            
            # Detect Akkadian
            has_akkadian, detect_meta = self.detector.detect_page(text)
            
            # Write to output CSV
            self._append_output(pdf_name, page_1based, text, has_akkadian)
            
            # Update stats
            self.stats["pages_processed"] += 1
            if has_akkadian:
                self.stats["pages_with_akkadian"] += 1
            if used_text_layer:
                self.stats["text_layer_used"] += 1
            else:
                self.stats["ocr_used"] += 1
            
            # Write progress
            elapsed_ms = int((time.time() - start_time) * 1000)
            self._append_progress(pdf_name, page_1based, elapsed_ms, used_text_layer, has_akkadian)
            
        except Exception as e:
            logger.error(f"Error processing {pdf_name} page {page_1based}: {e}")
            self.stats["errors"] += 1
    
    def _append_output(self, pdf_name: str, page: int, text: str, has_akkadian: bool):
        """Append row to output CSV."""
        with open(self.output_csv, 'a', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                pdf_name,
                page,
                text.replace('\n', '\\n'),  # Escape newlines for CSV
                'true' if has_akkadian else 'false'
            ])
            f.flush()
    
    def _append_progress(self, pdf_name: str, page: int, ms: int,
                        used_text_layer: bool, has_akkadian: bool):
        """Append row to progress CSV."""
        with open(self.progress_csv, 'a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                pdf_name,
                page,
                ms,
                used_text_layer,
                has_akkadian,
                datetime.now().isoformat()
            ])
            f.flush()
    
    def _report_stats(self):
        """Report final statistics."""
        logger.info("=" * 60)
        logger.info("Pipeline completed")
        logger.info(f"Pages processed: {self.stats['pages_processed']}")
        logger.info(f"Pages with Akkadian: {self.stats['pages_with_akkadian']}")
        logger.info(f"Text layer used: {self.stats['text_layer_used']}")
        logger.info(f"OCR used: {self.stats['ocr_used']}")
        logger.info(f"Errors: {self.stats['errors']}")
        logger.info(f"Output CSV: {self.output_csv}")
        logger.info(f"Progress CSV: {self.progress_csv}")
        logger.info("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Page-level text extraction with Akkadian detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Input source (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--manifest', type=str,
                            help='Path to manifest TSV (pdf_path<TAB>page_no)')
    input_group.add_argument('--inputs', type=str,
                            help='Directory to scan for PDFs recursively')
    
    # Output
    parser.add_argument('--output-root', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--progress-csv', type=str,
                       help='Path to progress CSV (default: output-root/progress.csv)')
    
    # Text extraction
    parser.add_argument('--prefer-text-layer', action='store_true', default=False,
                       help='Prefer PDF text layer extraction')
    parser.add_argument('--ocr-fallback', type=str, choices=['paddle', 'none'], default='none',
                       help='OCR engine for fallback when text layer fails')
    
    # Akkadian detection
    parser.add_argument('--profile', type=str,
                       help='Path to detection profile JSON (default: profiles/akkadian_strict.json)')
    
    # UI
    parser.add_argument('--status-bar', action='store_true', default=False,
                       help='Show progress bar')
    
    args = parser.parse_args()
    
    # Validate dependencies
    if not PYMUPDF_AVAILABLE:
        logger.error("PyMuPDF (fitz) is required. Install: pip install PyMuPDF")
        sys.exit(1)
    
    if args.ocr_fallback == 'paddle':
        try:
            import paddleocr
        except ImportError:
            logger.error("PaddleOCR is required for OCR fallback. Install: pip install paddleocr")
            sys.exit(1)
    
    # Run pipeline
    pipeline = PageTextPipeline(args)
    pipeline.run()


if __name__ == '__main__':
    main()
