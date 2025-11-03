"""
LLM-based OCR Post-Correction with Content-Preservation Guardrails

Integrates local LLM (Ollama) for typo/diacritic restoration with strict constraints:
- Confidence-based routing (only low-confidence lines)
- Language-specific thresholds
- Transliteration protection (≤3% edits for Akkadian)
- Edit budget enforcement (≤12% for modern text)
- Bracket/structure preservation
- Vocabulary explosion detection
"""

import logging
import re
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import hashlib

from .clients.ollama_client import OllamaClient, OllamaConfig
from .prompt_schemas import build_full_prompt, validate_response_schema
from .json_schemas import (
    CorrectionRequest, CorrectionResponse, CorrectionFlags, ContextInfo
)
from akkadian_extract import is_akkadian_transliteration, estimate_transliteration_confidence

logger = logging.getLogger(__name__)


# Language-specific confidence thresholds
# Only route lines BELOW these thresholds to LLM
LANGUAGE_CONFIDENCE_THRESHOLDS = {
    'en': 0.86,
    'de': 0.85,
    'fr': 0.84,
    'it': 0.84,
    'tr': 0.83,
    'unknown': 0.80,
}

# Edit budget limits (fraction of text that can change)
EDIT_BUDGET_MODERN = 0.12  # ≤12% for modern languages
EDIT_BUDGET_TRANSLITERATION = 0.03  # ≤3% for Akkadian transliteration

# Minimum line length to process
MIN_LINE_LENGTH = 8


@dataclass
class CorrectionResult:
    """Result of LLM correction attempt."""
    original_text: str
    corrected_text: str
    applied: bool
    reason: str
    edit_ratio: float
    confidence: float
    latency_ms: int
    cached: bool
    llm_notes: Optional[str] = None


class GuardrailViolation(Exception):
    """Raised when guardrail check fails."""
    pass


class LLMCorrector:
    """
    LLM-based OCR post-corrector with content-preservation guardrails.
    """
    
    def __init__(
        self,
        ollama_config: Optional[OllamaConfig] = None,
        cache: Optional[Dict[str, Any]] = None,
        enable_telemetry: bool = True
    ):
        """
        Initialize LLM corrector.
        
        Args:
            ollama_config: Ollama client configuration (uses defaults if None)
            cache: Optional cache dict for storing corrections
            enable_telemetry: Whether to track telemetry metrics
        """
        self.client = OllamaClient(config=ollama_config or OllamaConfig())
        self.cache = cache if cache is not None else {}
        self.enable_telemetry = enable_telemetry
        
        # Telemetry counters
        self.telemetry = {
            'llm_spans_attempted': 0,
            'llm_spans_corrected': 0,
            'llm_spans_rejected': 0,
            'llm_cache_hits': 0,
            'llm_cache_misses': 0,
            'llm_latency_ms_total': 0,
            'guardrail_violations': {},
        }
    
    def should_correct_line(
        self,
        text: str,
        lang: str,
        confidence: float
    ) -> Tuple[bool, str]:
        """
        Determine if line should be sent to LLM for correction.
        
        Uses enhanced trigger heuristics (mojibake, diacritic mismatch, char-LM).
        
        Args:
            text: Line text
            lang: Detected language
            confidence: OCR confidence score
        
        Returns:
            (should_correct, reason) tuple
        """
        # Check minimum length
        if len(text.strip()) < MIN_LINE_LENGTH:
            return False, f"Line too short ({len(text.strip())} < {MIN_LINE_LENGTH} chars)"
        
        # Use new enhanced triggers
        try:
            from utils.triggers import should_trigger_llm
            
            # Build custom thresholds from config
            custom_thresholds = LANGUAGE_CONFIDENCE_THRESHOLDS.copy()
            
            should_trigger, reason = should_trigger_llm(
                text=text,
                confidence=confidence,
                language=lang,
                custom_thresholds=custom_thresholds
            )
            
            return should_trigger, reason
            
        except ImportError:
            # Fallback to legacy confidence-only logic
            threshold = LANGUAGE_CONFIDENCE_THRESHOLDS.get(lang, LANGUAGE_CONFIDENCE_THRESHOLDS['unknown'])
            
            if confidence >= threshold:
                return False, f"Confidence {confidence:.3f} >= threshold {threshold:.3f}"
            
            return True, f"Low confidence {confidence:.3f} < {threshold:.3f}"
    
    def _generate_cache_key(
        self,
        text: str,
        lang: str,
        is_transliteration: bool,
        model_id: str
    ) -> str:
        """
        Generate deterministic cache key.
        
        Args:
            text: Normalized text
            lang: Language hint
            is_transliteration: Whether text is transliteration
            model_id: LLM model ID
        
        Returns:
            SHA256 hash as hex string
        """
        # Normalize text (lowercase, collapse whitespace)
        normalized = ' '.join(text.lower().split())
        
        # Build cache key components
        components = [
            model_id,
            lang,
            'translit' if is_transliteration else 'modern',
            normalized
        ]
        
        # Hash
        key_str = '|'.join(components)
        return hashlib.sha256(key_str.encode('utf-8')).hexdigest()
    
    def _check_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Check if correction is cached."""
        if cache_key in self.cache:
            if self.enable_telemetry:
                self.telemetry['llm_cache_hits'] += 1
            return self.cache[cache_key]
        
        if self.enable_telemetry:
            self.telemetry['llm_cache_misses'] += 1
        return None
    
    def _validate_edit_budget(
        self,
        original: str,
        corrected: str,
        max_ratio: float
    ) -> None:
        """
        Validate that edit ratio doesn't exceed budget.
        
        Args:
            original: Original text
            corrected: Corrected text
            max_ratio: Maximum allowed edit ratio
        
        Raises:
            GuardrailViolation: If edit ratio exceeds budget
        """
        if len(original) == 0:
            return
        
        # Calculate character-level edit distance (simplified)
        changes = sum(1 for a, b in zip(original, corrected) if a != b)
        changes += abs(len(original) - len(corrected))
        
        edit_ratio = changes / len(original)
        
        if edit_ratio > max_ratio:
            raise GuardrailViolation(
                f"Edit ratio {edit_ratio:.2%} exceeds budget {max_ratio:.2%}"
            )
    
    def _validate_bracket_preservation(
        self,
        original: str,
        corrected: str
    ) -> None:
        """
        Validate that brackets are preserved.
        
        Args:
            original: Original text
            corrected: Corrected text
        
        Raises:
            GuardrailViolation: If brackets are not preserved
        """
        brackets = ['[]', '()', '{}']
        
        for open_br, close_br in brackets:
            orig_open = original.count(open_br)
            orig_close = original.count(close_br)
            corr_open = corrected.count(open_br)
            corr_close = corrected.count(close_br)
            
            if orig_open != corr_open or orig_close != corr_close:
                raise GuardrailViolation(
                    f"Bracket mismatch: {open_br}{close_br} "
                    f"orig({orig_open},{orig_close}) != corr({corr_open},{corr_close})"
                )
    
    def _validate_line_breaks(
        self,
        original: str,
        corrected: str
    ) -> None:
        """
        Validate that line breaks are preserved.
        
        Args:
            original: Original text
            corrected: Corrected text
        
        Raises:
            GuardrailViolation: If line breaks differ
        """
        orig_breaks = original.count('\n')
        corr_breaks = corrected.count('\n')
        
        if orig_breaks != corr_breaks:
            raise GuardrailViolation(
                f"Line break mismatch: {orig_breaks} != {corr_breaks}"
            )
    
    def _validate_vocabulary_explosion(
        self,
        original: str,
        corrected: str,
        max_increase: float = 0.15
    ) -> None:
        """
        Validate that corrected text doesn't have too many new characters.
        
        Detects cases where LLM added unwanted content.
        
        Args:
            original: Original text
            corrected: Corrected text
            max_increase: Maximum allowed increase in alphabetic chars
        
        Raises:
            GuardrailViolation: If alphabetic character count increases too much
        """
        orig_alpha = sum(1 for c in original if c.isalpha())
        corr_alpha = sum(1 for c in corrected if c.isalpha())
        
        if orig_alpha == 0:
            return
        
        increase_ratio = (corr_alpha - orig_alpha) / orig_alpha
        
        if increase_ratio > max_increase:
            raise GuardrailViolation(
                f"Vocabulary explosion: {increase_ratio:.2%} > {max_increase:.2%}"
            )
    
    def _apply_guardrails(
        self,
        original: str,
        corrected: str,
        is_transliteration: bool
    ) -> None:
        """
        Apply all content-preservation guardrails.
        
        Args:
            original: Original text
            corrected: Corrected text
            is_transliteration: Whether text is transliteration
        
        Raises:
            GuardrailViolation: If any guardrail check fails
        """
        # 1. Edit budget
        max_ratio = EDIT_BUDGET_TRANSLITERATION if is_transliteration else EDIT_BUDGET_MODERN
        self._validate_edit_budget(original, corrected, max_ratio)
        
        # 2. Bracket preservation
        self._validate_bracket_preservation(original, corrected)
        
        # 3. Line breaks
        self._validate_line_breaks(original, corrected)
        
        # 4. Vocabulary explosion
        self._validate_vocabulary_explosion(original, corrected)
    
    def correct_line(
        self,
        text: str,
        lang: str,
        confidence: float,
        prev_line: Optional[str] = None,
        next_line: Optional[str] = None,
        span_id: Optional[str] = None
    ) -> CorrectionResult:
        """
        Correct a single line using LLM with guardrails.
        
        Args:
            text: Line text to correct
            lang: Detected language
            confidence: OCR confidence score
            prev_line: Previous line for context
            next_line: Next line for context
            span_id: Unique identifier for this line
        
        Returns:
            CorrectionResult with corrected text and metadata
        """
        import time
        
        if self.enable_telemetry:
            self.telemetry['llm_spans_attempted'] += 1
        
        # Check if we should correct this line
        should_correct, reason = self.should_correct_line(text, lang, confidence)
        if not should_correct:
            return CorrectionResult(
                original_text=text,
                corrected_text=text,
                applied=False,
                reason=reason,
                edit_ratio=0.0,
                confidence=confidence,
                latency_ms=0,
                cached=False
            )
        
        # Detect transliteration
        is_translit = is_akkadian_transliteration(text)
        translit_conf = estimate_transliteration_confidence(text) if is_translit else 0.0
        
        # Generate cache key
        cache_key = self._generate_cache_key(
            text=text,
            lang=lang,
            is_transliteration=is_translit,
            model_id=self.client.config.model_id
        )
        
        # Check cache
        cached_result = self._check_cache(cache_key)
        if cached_result:
            return CorrectionResult(
                original_text=text,
                corrected_text=cached_result['corrected_text'],
                applied=cached_result['applied'],
                reason=f"Cache hit: {cached_result['reason']}",
                edit_ratio=cached_result['edit_ratio'],
                confidence=cached_result['confidence'],
                latency_ms=0,
                cached=True,
                llm_notes=cached_result.get('llm_notes')
            )
        
        # Build correction request
        mode = 'protect_transliteration' if is_translit else 'fix_typos_only'
        max_edits = EDIT_BUDGET_TRANSLITERATION if is_translit else EDIT_BUDGET_MODERN
        
        request = CorrectionRequest(
            schema_version="1.0",
            span_id=span_id or f"line_{hashlib.md5(text.encode()).hexdigest()[:8]}",
            language_hint=lang,
            original_text=text,
            context=ContextInfo(
                prev_line=prev_line or "",
                next_line=next_line or ""
            ),
            flags=CorrectionFlags(
                is_transliteration_suspected=is_translit,
                max_relative_change_chars=max_edits,
                mode=mode
            )
        )
        
        # Build prompt
        prompt = build_full_prompt(request, mode=mode)
        
        # Call LLM
        start_time = time.time()
        try:
            response_json = self.client.generate_json(
                system_message=prompt['system'],
                user_message=prompt['user']
            )
            latency_ms = int((time.time() - start_time) * 1000)
            
            if self.enable_telemetry:
                self.telemetry['llm_latency_ms_total'] += latency_ms
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            
            if self.enable_telemetry:
                self.telemetry['llm_spans_rejected'] += 1
                violation_key = f"llm_error_{type(e).__name__}"
                self.telemetry['guardrail_violations'][violation_key] = \
                    self.telemetry['guardrail_violations'].get(violation_key, 0) + 1
            
            return CorrectionResult(
                original_text=text,
                corrected_text=text,
                applied=False,
                reason=f"LLM error: {str(e)}",
                edit_ratio=0.0,
                confidence=confidence,
                latency_ms=0,
                cached=False
            )
        
        # Validate response schema
        try:
            response = validate_response_schema(response_json)
        except Exception as e:
            logger.error(f"Response validation failed: {e}")
            
            if self.enable_telemetry:
                self.telemetry['llm_spans_rejected'] += 1
                self.telemetry['guardrail_violations']['invalid_json'] = \
                    self.telemetry['guardrail_violations'].get('invalid_json', 0) + 1
            
            return CorrectionResult(
                original_text=text,
                corrected_text=text,
                applied=False,
                reason=f"Invalid JSON: {str(e)}",
                edit_ratio=0.0,
                confidence=confidence,
                latency_ms=latency_ms,
                cached=False
            )
        
        # Apply guardrails
        try:
            self._apply_guardrails(
                original=text,
                corrected=response.corrected_text,
                is_transliteration=is_translit
            )
        except GuardrailViolation as e:
            logger.warning(f"Guardrail violation: {e}")
            
            if self.enable_telemetry:
                self.telemetry['llm_spans_rejected'] += 1
                violation_key = str(e).split(':')[0]  # Extract violation type
                self.telemetry['guardrail_violations'][violation_key] = \
                    self.telemetry['guardrail_violations'].get(violation_key, 0) + 1
            
            return CorrectionResult(
                original_text=text,
                corrected_text=text,
                applied=False,
                reason=f"Guardrail: {str(e)}",
                edit_ratio=response.edit_ratio,
                confidence=response.confidence,
                latency_ms=latency_ms,
                cached=False,
                llm_notes=response.notes
            )
        
        # Success! Cache the result
        cache_entry = {
            'corrected_text': response.corrected_text,
            'applied': True,
            'reason': 'LLM correction applied',
            'edit_ratio': response.edit_ratio,
            'confidence': response.confidence,
            'llm_notes': response.notes
        }
        self.cache[cache_key] = cache_entry
        
        if self.enable_telemetry:
            self.telemetry['llm_spans_corrected'] += 1
        
        return CorrectionResult(
            original_text=text,
            corrected_text=response.corrected_text,
            applied=True,
            reason='LLM correction applied',
            edit_ratio=response.edit_ratio,
            confidence=response.confidence,
            latency_ms=latency_ms,
            cached=False,
            llm_notes=response.notes
        )
    
    def get_telemetry(self) -> Dict[str, Any]:
        """Get telemetry metrics."""
        metrics = self.telemetry.copy()
        
        # Calculate averages
        if metrics['llm_spans_attempted'] > 0:
            metrics['llm_avg_latency_ms'] = \
                metrics['llm_latency_ms_total'] / metrics['llm_spans_attempted']
            metrics['llm_correction_rate'] = \
                metrics['llm_spans_corrected'] / metrics['llm_spans_attempted']
            metrics['llm_rejection_rate'] = \
                metrics['llm_spans_rejected'] / metrics['llm_spans_attempted']
        
        total_cache_ops = metrics['llm_cache_hits'] + metrics['llm_cache_misses']
        if total_cache_ops > 0:
            metrics['llm_cache_hit_rate'] = metrics['llm_cache_hits'] / total_cache_ops
        
        return metrics


# Example usage
if __name__ == "__main__":
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test corrector
    corrector = LLMCorrector()
    
    # Test 1: German text with missing umlaut (low confidence)
    print("=" * 80)
    print("TEST 1: German text with missing diacritics")
    print("=" * 80)
    
    result1 = corrector.correct_line(
        text="Die Ubersetzung der hethitischen Texte ist schwierig.",
        lang="de",
        confidence=0.78,  # Below threshold (0.85)
        span_id="test_1"
    )
    
    print(f"Original:  {result1.original_text}")
    print(f"Corrected: {result1.corrected_text}")
    print(f"Applied:   {result1.applied}")
    print(f"Reason:    {result1.reason}")
    print(f"Edit ratio: {result1.edit_ratio:.2%}")
    print(f"Latency:   {result1.latency_ms}ms")
    
    # Test 2: Akkadian transliteration (should protect)
    print("\n" + "=" * 80)
    print("TEST 2: Akkadian transliteration (protect mode)")
    print("=" * 80)
    
    result2 = corrector.correct_line(
        text="ša-ar-ru šarrum LUGAL ᵈUTU",
        lang="unknown",
        confidence=0.75,
        prev_line="Transliteration:",
        next_line="Translation: The king",
        span_id="test_2"
    )
    
    print(f"Original:  {result2.original_text}")
    print(f"Corrected: {result2.corrected_text}")
    print(f"Applied:   {result2.applied}")
    print(f"Reason:    {result2.reason}")
    print(f"Edit ratio: {result2.edit_ratio:.2%}")
    print(f"Latency:   {result2.latency_ms}ms")
    
    # Test 3: High confidence (should skip)
    print("\n" + "=" * 80)
    print("TEST 3: High confidence text (should skip)")
    print("=" * 80)
    
    result3 = corrector.correct_line(
        text="This is a perfectly clear sentence.",
        lang="en",
        confidence=0.95,  # Above threshold (0.86)
        span_id="test_3"
    )
    
    print(f"Original:  {result3.original_text}")
    print(f"Corrected: {result3.corrected_text}")
    print(f"Applied:   {result3.applied}")
    print(f"Reason:    {result3.reason}")
    
    # Print telemetry
    print("\n" + "=" * 80)
    print("TELEMETRY")
    print("=" * 80)
    
    telemetry = corrector.get_telemetry()
    for key, value in sorted(telemetry.items()):
        if isinstance(value, dict):
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        elif isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
