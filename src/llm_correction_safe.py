"""
LLM-in-the-loop OCR correction with strict fix-typos-only JSON protocol.
Features span-level routing, caching, and language-aware processing.
"""
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
import os

from llm_cache import LLMCache, CacheEntry

logger = logging.getLogger(__name__)

@dataclass 
class OCRSpan:
    """Represents a span of OCR text with confidence information."""
    text: str
    confidence: float
    span_id: str
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    language: str = "en"
    page_id: str = ""
    block_id: str = ""
    line_id: str = ""

@dataclass
class CorrectionResult:
    """Result of LLM correction for a text span."""
    original_text: str
    corrected_text: str
    applied_edits: List[Dict[str, Any]]
    span_id: str
    confidence_improvement: float
    notes: str
    cached: bool = False
    latency_ms: int = 0

class FixTyposLLMCorrector:
    """LLM corrector with strict fix-typos-only protocol and caching."""
    
    # Language-specific confidence thresholds for LLM routing
    CONFIDENCE_THRESHOLDS = {
        'en': 0.86,
        'de': 0.85,
        'fr': 0.84,
        'it': 0.84,
        'tr': 0.83
    }
    
    # Akkadian transliteration detection patterns
    AKKADIAN_PATTERNS = [
        re.compile(r'[šṣṭḫāēīūâêîû]'),  # Akkadian diacritics
        re.compile(r'[\u12000-\u123FF]'),  # Cuneiform Unicode block
    ]
    
    SYSTEM_PROMPT = """You are a text normalization tool. Only correct obvious OCR typos. Do not add, remove, reorder, merge, or split lines. Do not change layout, punctuation style, numbers, footnote markers, or scholarly brackets. Never invent text. If uncertain, keep the original. Output valid JSON only, following the schema."""
    
    def __init__(self, provider: str = "ollama", model: str = "mistral:latest",
                 base_url: str = "http://localhost:11434", timeout: int = 30,
                 max_workers: int = 3, cache_dir: str = "data/.cache/llm"):
        """Initialize LLM corrector.
        
        Args:
            provider: LLM provider ('ollama', 'llamacpp', or 'none')
            model: Model identifier
            base_url: Base URL for LLM API
            timeout: Request timeout in seconds
            max_workers: Maximum concurrent correction threads
            cache_dir: Directory for result caching
        """
        self.provider = provider
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.max_workers = max_workers
        self.enabled = os.getenv('LLM_ENABLED', 'true').lower() == 'true'
        
        # Initialize cache
        self.cache = LLMCache(cache_dir)
        
        # Initialize client based on provider
        self.client = None
        if self.enabled and provider != 'none':
            self._init_client()
        
        # Statistics
        self.stats = {
            'spans_processed': 0,
            'spans_sent_to_llm': 0,
            'cache_hits': 0,
            'akkadian_detected': 0,
            'low_confidence_filtered': 0,
            'total_latency_ms': 0
        }
    
    def _init_client(self):
        """Initialize the LLM client based on provider."""
        try:
            if self.provider == "ollama":
                # Import ollama client if available
                try:
                    import ollama
                    self.client = ollama.Client(host=self.base_url)
                    logger.info(f"Initialized Ollama client for {self.model}")
                except ImportError:
                    logger.warning("Ollama client not available, install with: pip install ollama")
                    self.enabled = False
                    
            elif self.provider == "llamacpp":
                # Import OpenAI-compatible client for llama.cpp
                try:
                    from openai import OpenAI
                    self.client = OpenAI(base_url=self.base_url, api_key="not-needed")
                    logger.info(f"Initialized llama.cpp client for {self.model}")
                except ImportError:
                    logger.warning("OpenAI client not available, install with: pip install openai")
                    self.enabled = False
            else:
                logger.warning(f"Unknown provider: {self.provider}")
                self.enabled = False
                
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            self.enabled = False
    
    def detect_akkadian(self, text: str) -> bool:
        """Detect if text contains Akkadian transliteration.
        
        Args:
            text: Text to analyze
            
        Returns:
            True if Akkadian content detected
        """
        if not text or len(text) < 3:
            return False
        
        # Check for Akkadian patterns
        for pattern in self.AKKADIAN_PATTERNS:
            if pattern.search(text):
                return True
        
        # Check diacritic density
        diacritics = sum(1 for c in text if ord(c) > 127)
        if len(text) > 0 and (diacritics / len(text)) > 0.03:  # >3% diacritics
            return True
        
        return False
    
    def should_correct_span(self, span: OCRSpan) -> Tuple[bool, str]:
        """Determine if a span should be sent for LLM correction.
        
        Args:
            span: OCR span to evaluate
            
        Returns:
            Tuple of (should_correct, reason)
        """
        self.stats['spans_processed'] += 1
        
        # Check if LLM is enabled
        if not self.enabled:
            return False, "LLM disabled"
        
        # Check minimum length
        if len(span.text.strip()) < 8:
            return False, "text too short"
        
        # Check for Akkadian content
        if self.detect_akkadian(span.text):
            self.stats['akkadian_detected'] += 1
            return False, "akkadian detected"
        
        # Check if mostly digits/tables
        digits_ratio = sum(1 for c in span.text if c.isdigit()) / len(span.text)
        if digits_ratio > 0.7:
            return False, "mostly digits"
        
        # Check confidence threshold
        threshold = self.CONFIDENCE_THRESHOLDS.get(span.language, 0.85)
        if span.confidence >= threshold:
            return False, f"confidence {span.confidence:.3f} >= {threshold}"
        
        self.stats['low_confidence_filtered'] += 1
        return True, "low confidence span"
    
    def build_correction_prompt(self, span: OCRSpan, context: Dict[str, str] = None) -> Dict[str, Any]:
        """Build the fix-typos-only JSON prompt for the LLM.
        
        Args:
            span: OCR span to correct
            context: Optional previous/next line context
            
        Returns:
            JSON prompt dictionary
        """
        prompt_data = {
            "schema_version": "v1",
            "language": span.language,
            "mode": "fix_typos_only",
            "span_id": span.span_id,
            "original_text": span.text,
            "context": {
                "prev_line": context.get('prev_line', '') if context else '',
                "next_line": context.get('next_line', '') if context else ''
            },
            "constraints": {
                "preserve_brackets": True,
                "preserve_footnote_markers": True,
                "max_relative_change_chars": 0.15
            },
            "return": {
                "corrected_text": "string",
                "applied_edits": [
                    {"pos": "int", "from": "string", "to": "string"}
                ],
                "notes": "string"
            }
        }
        
        return prompt_data
    
    def call_llm(self, prompt_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Call the LLM with the correction prompt.
        
        Args:
            prompt_data: JSON prompt data
            
        Returns:
            JSON response from LLM or None if failed
        """
        try:
            user_prompt = json.dumps(prompt_data, indent=2)
            
            if self.provider == "ollama" and self.client:
                response = self.client.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    options={"temperature": 0.1}  # Low temperature for consistency
                )
                return json.loads(response['message']['content'])
                
            elif self.provider == "llamacpp" and self.client:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    timeout=self.timeout
                )
                return json.loads(response.choices[0].message.content)
            
        except json.JSONDecodeError as e:
            logger.warning(f"LLM returned invalid JSON: {e}")
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
        
        return None
    
    def correct_span(self, span: OCRSpan, context: Dict[str, str] = None) -> CorrectionResult:
        """Correct a single OCR span with LLM.
        
        Args:
            span: OCR span to correct
            context: Optional context lines
            
        Returns:
            CorrectionResult with correction details
        """
        start_time = time.time()
        
        # Check if correction is needed
        should_correct, reason = self.should_correct_span(span)
        if not should_correct:
            return CorrectionResult(
                original_text=span.text,
                corrected_text=span.text,
                applied_edits=[],
                span_id=span.span_id,
                confidence_improvement=0.0,
                notes=f"skipped: {reason}",
                cached=False,
                latency_ms=0
            )
        
        # Check cache first
        constraints = {"preserve_brackets": True, "preserve_footnote_markers": True, "max_relative_change_chars": 0.15}
        cached_result = self.cache.get(self.model, span.language, span.text, constraints)
        
        if cached_result:
            self.stats['cache_hits'] += 1
            return CorrectionResult(
                original_text=span.text,
                corrected_text=cached_result.corrected_text,
                applied_edits=cached_result.applied_edits,
                span_id=span.span_id,
                confidence_improvement=0.1,  # Assume some improvement
                notes=cached_result.notes,
                cached=True,
                latency_ms=cached_result.latency_ms
            )
        
        # Call LLM for correction
        self.stats['spans_sent_to_llm'] += 1
        prompt_data = self.build_correction_prompt(span, context)
        
        llm_response = self.call_llm(prompt_data)
        latency_ms = int((time.time() - start_time) * 1000)
        self.stats['total_latency_ms'] += latency_ms
        
        if not llm_response:
            return CorrectionResult(
                original_text=span.text,
                corrected_text=span.text,
                applied_edits=[],
                span_id=span.span_id,
                confidence_improvement=0.0,
                notes="LLM call failed",
                cached=False,
                latency_ms=latency_ms
            )
        
        # Extract correction from LLM response
        corrected_text = llm_response.get('corrected_text', span.text)
        applied_edits = llm_response.get('applied_edits', [])
        notes = llm_response.get('notes', '')
        
        # Validate correction doesn't exceed constraints
        char_change_ratio = abs(len(corrected_text) - len(span.text)) / len(span.text) if span.text else 0
        if char_change_ratio > 0.15:
            logger.warning(f"LLM correction exceeds character change limit: {char_change_ratio:.2f}")
            corrected_text = span.text
            applied_edits = []
            notes = f"rejected: too many changes ({char_change_ratio:.2f})"
        
        # Cache the result
        self.cache.put(
            model_id=self.model,
            language=span.language,
            original_text=span.text,
            corrected_text=corrected_text,
            applied_edits=applied_edits,
            latency_ms=latency_ms,
            constraints=constraints,
            notes=notes
        )
        
        # Calculate confidence improvement (heuristic)
        confidence_improvement = 0.0
        if corrected_text != span.text and applied_edits:
            confidence_improvement = min(0.2, len(applied_edits) * 0.05)
        
        return CorrectionResult(
            original_text=span.text,
            corrected_text=corrected_text,
            applied_edits=applied_edits,
            span_id=span.span_id,
            confidence_improvement=confidence_improvement,
            notes=notes,
            cached=False,
            latency_ms=latency_ms
        )
    
    def correct_spans_batch(self, spans: List[OCRSpan], 
                           context_map: Dict[str, Dict[str, str]] = None) -> List[CorrectionResult]:
        """Correct multiple OCR spans with concurrent processing.
        
        Args:
            spans: List of OCR spans to correct
            context_map: Map of span_id to context lines
            
        Returns:
            List of CorrectionResults
        """
        if not spans:
            return []
        
        context_map = context_map or {}
        
        # Process spans concurrently
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for span in spans:
                context = context_map.get(span.span_id)
                future = executor.submit(self.correct_span, span, context)
                futures.append(future)
            
            results = []
            for future in futures:
                try:
                    result = future.result(timeout=self.timeout + 5)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Span correction failed: {e}")
                    # Create error result
                    error_result = CorrectionResult(
                        original_text="",
                        corrected_text="",
                        applied_edits=[],
                        span_id="error",
                        confidence_improvement=0.0,
                        notes=f"processing error: {e}",
                        cached=False,
                        latency_ms=0
                    )
                    results.append(error_result)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get correction statistics and cache info.
        
        Returns:
            Statistics dictionary
        """
        cache_stats = self.cache.get_stats()
        
        avg_latency = 0
        if self.stats['spans_sent_to_llm'] > 0:
            avg_latency = self.stats['total_latency_ms'] / self.stats['spans_sent_to_llm']
        
        return {
            'correction_stats': self.stats.copy(),
            'cache_stats': cache_stats,
            'average_latency_ms': round(avg_latency, 2),
            'cache_hit_rate': self.stats['cache_hits'] / max(1, self.stats['spans_processed']),
            'llm_enabled': self.enabled,
            'provider': self.provider,
            'model': self.model
        }