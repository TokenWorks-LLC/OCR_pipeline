#!/usr/bin/env python3
"""
Page-level text extraction with Akkadian detection and LLM typo correction.

Extracts page-level text from PDFs (with optional OCR fallback),
detects Akkadian content per page, applies LLM typo correction while
preserving Akkadian spans, and outputs a 4-column CSV:

    pdf_name,page,page_text,has_akkadian

Usage:
    python tools/run_page_text.py \\
        --manifest data\\gold\\manifest.txt \\
        --output-root reports\\page_text_20251009 \\
        --prefer-text-layer \\
        --ocr-fallback paddle \\
        --llm-on \\
        --status-bar \\
        --progress-csv reports\\page_text_20251009\\progress.csv
        
    OR:
    
    python tools/run_page_text.py \\
        --inputs "G:\\Shared drives\\Secondary Sources" \\
        --output-root reports\\page_text_20251009 \\
        --prefer-text-layer \\
        --llm-on
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


class AkkadianSpanProtector:
    """Protect Akkadian spans during LLM correction."""
    
    # Akkadian diacritics and markers
    AKKADIAN_CHARS = set('šṣṭḫāēīūâêîûᵈᵐᶠ')
    MARKERS = ['DUMU', 'LUGAL', 'KÙ.BABBAR', 'KUBABBAR', 'URU', 'É', 'GIŠ', 'KUR', 'LÚ', 'LU₂']
    
    def __init__(self):
        """Initialize span protector."""
        # Build pattern for Akkadian markers
        marker_pattern = '|'.join(re.escape(m) for m in self.MARKERS)
        
        # Pattern components:
        # 1. Syllabic with diacritics (a-na-ku, šar-ru-um, etc.)
        # 2. Markers (LUGAL, DUMU, etc.)
        # 3. Words with determinatives
        # 4. Words containing Akkadian diacritics
        self.akk_pattern = re.compile(
            r'\b[a-zšṣṭḫāēīū]+(?:[-—][a-zšṣṭḫāēīū]+)+\b|'  # Syllabic (hyphenated)
            rf'\b(?:{marker_pattern})\b|'  # Markers
            r'[ᵈᵐᶠ][A-Za-zšṣṭḫāēīū-]+|'  # Determinative prefixes
            r'\b[a-zšṣṭḫāēīū]*[šṣṭḫāēīū][a-zšṣṭḫāēīū]*\b',  # Words with diacritics
            re.IGNORECASE
        )
    
    def find_akkadian_spans(self, text: str) -> List[Tuple[int, int, str]]:
        """
        Find all Akkadian spans in text.
        
        Returns:
            List of (start, end, span_text) tuples
        """
        spans = []
        for match in self.akk_pattern.finditer(text):
            span_text = match.group()
            # Verify span has Akkadian characteristics
            # Must have either diacritics, markers, or determinatives
            has_diacritic = any(c in span_text for c in self.AKKADIAN_CHARS)
            has_marker = any(m.lower() in span_text.lower() for m in self.MARKERS)
            has_hyphen_structure = '-' in span_text or '—' in span_text
            
            if has_diacritic or has_marker or (has_hyphen_structure and len(span_text) > 4):
                spans.append((match.start(), match.end(), span_text))
        return spans
    
    def protect_spans(self, text: str) -> Tuple[str, List[str]]:
        """
        Wrap Akkadian spans with <AKK>...</AKK> tags.
        
        Returns:
            Tuple of (protected_text, list_of_original_spans)
        """
        spans = self.find_akkadian_spans(text)
        if not spans:
            return text, []
        
        # Sort spans by position (descending) to avoid offset issues
        spans.sort(key=lambda x: x[0], reverse=True)
        
        protected_text = text
        original_spans = []
        
        for start, end, span_text in spans:
            original_spans.append(span_text)
            protected_text = (
                protected_text[:start] + 
                f"<AKK>{span_text}</AKK>" + 
                protected_text[end:]
            )
        
        # Reverse to match original order
        original_spans.reverse()
        
        return protected_text, original_spans
    
    def validate_protection(self, original_spans: List[str], corrected_text: str) -> Tuple[bool, str]:
        """
        Validate that all protected spans are unchanged.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Extract all <AKK>...</AKK> spans from corrected text
        akk_tag_pattern = re.compile(r'<AKK>(.*?)</AKK>', re.DOTALL)
        found_spans = akk_tag_pattern.findall(corrected_text)
        
        if len(found_spans) != len(original_spans):
            return False, f"Span count mismatch: expected {len(original_spans)}, found {len(found_spans)}"
        
        for i, (orig, found) in enumerate(zip(original_spans, found_spans)):
            if orig != found:
                return False, f"Span {i} altered: '{orig}' → '{found}'"
        
        return True, ""
    
    def unprotect_spans(self, protected_text: str) -> str:
        """Remove <AKK>...</AKK> tags from text."""
        return re.sub(r'<AKK>(.*?)</AKK>', r'\1', protected_text, flags=re.DOTALL)


class SimpleLLMCorrector:
    """Simple LLM corrector for typo fixes with Akkadian protection."""
    
    def __init__(self, 
                 provider: str = "ollama",
                 model: str = "qwen2.5:7b-instruct",
                 base_url: str = "http://localhost:11434",
                 temperature: float = 0.2,
                 top_p: float = 0.2,
                 timeout: int = 120):
        """Initialize LLM corrector."""
        self.provider = provider
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self.enabled = os.getenv('LLM_ENABLED', 'true').lower() == 'true'
        self.client = None
        
        if self.enabled:
            self._init_client()
        
        self.protector = AkkadianSpanProtector()
    
    def _init_client(self):
        """Initialize LLM client."""
        try:
            if self.provider == "ollama":
                import ollama
                self.client = ollama.Client(host=self.base_url)
                logger.info(f"Initialized Ollama client for {self.model}")
            else:
                logger.warning(f"Unknown provider: {self.provider}, disabling LLM")
                self.enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            self.enabled = False
    
    def correct_text(self, text: str) -> Tuple[str, Dict]:
        """
        Correct text with Akkadian span protection.
        
        Returns:
            Tuple of (corrected_text, metadata)
        """
        if not self.enabled or not text or len(text.strip()) < 10:
            return text, {"applied": False, "reason": "disabled or too short"}
        
        # Protect Akkadian spans
        protected_text, original_spans = self.protector.protect_spans(text)
        
        # If no spans to protect, check if text is worth correcting
        if not original_spans:
            # Check if text looks clean (few obvious typos)
            if len(text) < 20:
                return text, {"applied": False, "reason": "too short"}
        
        # Build prompt
        prompt = self._build_prompt(protected_text)
        
        # Call LLM
        try:
            start_time = time.time()
            
            response = self.client.chat(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a text correction assistant. Fix only obvious OCR typos (spacing, obvious misspellings). Do NOT alter text inside <AKK> tags. Return only the corrected plain text with <AKK> tags preserved."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                options={
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "num_predict": 2048
                }
            )
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            corrected_protected = response['message']['content'].strip()
            
            # Validate protection
            is_valid, error_msg = self.protector.validate_protection(original_spans, corrected_protected)
            
            if not is_valid:
                logger.warning(f"Protection validation failed: {error_msg}")
                return text, {
                    "applied": False,
                    "reason": "protection_violated",
                    "error": error_msg,
                    "latency_ms": latency_ms
                }
            
            # Unprotect spans
            corrected_text = self.protector.unprotect_spans(corrected_protected)
            
            # Calculate edit ratio
            edit_ratio = self._edit_distance(text, corrected_text) / len(text) if len(text) > 0 else 0.0
            
            # Apply edit budget (15% for non-Akkadian)
            if edit_ratio > 0.15:
                logger.warning(f"Edit ratio {edit_ratio:.2%} exceeds budget, rejecting")
                return text, {
                    "applied": False,
                    "reason": "edit_budget_exceeded",
                    "edit_ratio": edit_ratio,
                    "latency_ms": latency_ms
                }
            
            return corrected_text, {
                "applied": True,
                "edit_ratio": edit_ratio,
                "protected_spans": len(original_spans),
                "latency_ms": latency_ms
            }
            
        except Exception as e:
            logger.error(f"LLM correction failed: {e}")
            return text, {"applied": False, "reason": "error", "error": str(e)}
    
    def _build_prompt(self, protected_text: str) -> str:
        """Build correction prompt."""
        return f"""Normalize spacing and fix obvious OCR typos in the following text. 

CRITICAL RULES:
1. Do NOT change any text inside <AKK>...</AKK> tags
2. Keep all <AKK> tags exactly as they appear
3. Only fix obvious typos (wrong spacing, common OCR errors)
4. Do NOT add, remove, or rewrite content
5. Preserve all newlines and paragraph structure
6. Return ONLY the corrected text with <AKK> tags intact

Text:
{protected_text}

Corrected text:"""
    
    @staticmethod
    def _edit_distance(s1: str, s2: str) -> int:
        """Calculate Levenshtein edit distance."""
        if len(s1) < len(s2):
            return SimpleLLMCorrector._edit_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]


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
        
        self.corrector = None
        if args.llm_on:
            self.corrector = SimpleLLMCorrector(
                provider=args.llm_provider,
                model=args.llm_model,
                base_url=args.llm_base_url,
                temperature=args.llm_temperature,
                top_p=args.llm_top_p,
                timeout=args.llm_timeout
            )
        
        # Output CSV paths
        self.output_csv = self.output_root / "client_page_text.csv"
        self.progress_csv = args.progress_csv or (self.output_root / "progress.csv")
        
        # Stats
        self.stats = {
            "pages_processed": 0,
            "pages_with_akkadian": 0,
            "pages_with_llm": 0,
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
                'has_akkadian', 'llm_applied', 'timestamp'
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
            
            # Apply LLM correction if enabled
            llm_applied = False
            if self.corrector and not has_akkadian:  # Only correct non-Akkadian pages
                corrected_text, llm_meta = self.corrector.correct_text(text)
                if llm_meta.get("applied", False):
                    text = corrected_text
                    llm_applied = True
                    self.stats["pages_with_llm"] += 1
            
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
            self._append_progress(pdf_name, page_1based, elapsed_ms, used_text_layer, has_akkadian, llm_applied)
            
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
                        used_text_layer: bool, has_akkadian: bool, llm_applied: bool):
        """Append row to progress CSV."""
        with open(self.progress_csv, 'a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                pdf_name,
                page,
                ms,
                used_text_layer,
                has_akkadian,
                llm_applied,
                datetime.now().isoformat()
            ])
            f.flush()
    
    def _report_stats(self):
        """Report final statistics."""
        logger.info("=" * 60)
        logger.info("Pipeline completed")
        logger.info(f"Pages processed: {self.stats['pages_processed']}")
        logger.info(f"Pages with Akkadian: {self.stats['pages_with_akkadian']}")
        logger.info(f"Pages with LLM correction: {self.stats['pages_with_llm']}")
        logger.info(f"Text layer used: {self.stats['text_layer_used']}")
        logger.info(f"OCR used: {self.stats['ocr_used']}")
        logger.info(f"Errors: {self.stats['errors']}")
        logger.info(f"Output CSV: {self.output_csv}")
        logger.info(f"Progress CSV: {self.progress_csv}")
        logger.info("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Page-level text extraction with Akkadian detection and LLM correction",
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
    
    # LLM correction
    parser.add_argument('--llm-on', action='store_true', default=False,
                       help='Enable LLM typo correction')
    parser.add_argument('--llm-off', dest='llm_on', action='store_false',
                       help='Disable LLM typo correction')
    parser.add_argument('--llm-provider', type=str, default='ollama',
                       help='LLM provider (default: ollama)')
    parser.add_argument('--llm-model', type=str, default='qwen2.5:7b-instruct',
                       help='LLM model name')
    parser.add_argument('--llm-base-url', type=str, default='http://localhost:11434',
                       help='LLM API base URL')
    parser.add_argument('--llm-temperature', type=float, default=0.2,
                       help='LLM temperature')
    parser.add_argument('--llm-top-p', type=float, default=0.2,
                       help='LLM top-p')
    parser.add_argument('--llm-timeout', type=int, default=120,
                       help='LLM timeout in seconds')
    
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
    
    if args.llm_on:
        try:
            import ollama
        except ImportError:
            logger.error("Ollama client is required for LLM correction. Install: pip install ollama")
            sys.exit(1)
    
    # Run pipeline
    pipeline = PageTextPipeline(args)
    pipeline.run()


if __name__ == '__main__':
    main()
