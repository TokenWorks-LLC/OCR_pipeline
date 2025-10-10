#!/usr/bin/env python3
"""
LLM-in-the-loop V3 - Enhanced evaluation mode with strict typo-only corrections
Implements all requirements for eval mode v3 with span-level caching, confidence routing,
language detection, and telemetry tracking.
"""

import json
import logging
import re
import time
import hashlib
from typing import List, Dict, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

logger = logging.getLogger(__name__)

@dataclass
class SpanCorrectionResult:
    """Result of span-level LLM correction."""
    original_text: str
    corrected_text: str
    confidence: float
    language: str
    bbox: List[float]
    span_id: str
    corrections_made: List[Tuple[str, str]]
    processing_time: float
    cache_hit: bool = False

@dataclass
class LanguageThresholds:
    """Per-language confidence thresholds."""
    # Languages that need stricter thresholds due to complexity
    strict_languages: Dict[str, float] = field(default_factory=lambda: {
        'akkadian': 0.95,  # Very strict for ancient languages
        'turkish': 0.85,   # Moderate strict for Turkish
        'german': 0.80,    # Moderate for German
        'french': 0.80,    # Moderate for French
        'english': 0.75    # More lenient for English
    })

    # Default threshold for unknown languages
    default_threshold: float = 0.70

    def get_threshold(self, language: str) -> float:
        """Get confidence threshold for a language."""
        return self.strict_languages.get(language.lower(), self.default_threshold)

@dataclass
class LLMV3Config:
    """Configuration for LLM V3 system."""
    # Core settings
    llm_enabled: bool = True
    kill_switch: bool = False

    # Language settings
    akkadian_transliteration_guard: bool = True
    language_detection_per_span: bool = True

    # Filtering settings
    min_span_length: int = 3
    max_span_length: int = 1000
    filter_numeric_only: bool = True
    filter_tables: bool = True

    # Caching settings
    cache_enabled: bool = True
    cache_max_size: int = 10000
    model_id: str = "llama3.2:latest"
    prompt_version: str = "v3_strict_typo_only"

    # Performance settings
    max_workers: int = 3
    timeout: int = 30

    # Telemetry settings
    enable_telemetry: bool = True
    telemetry_log_interval: int = 100  # Log every N corrections

@dataclass
class CorrectionTelemetry:
    """Telemetry data for LLM corrections."""
    total_spans_processed: int = 0
    spans_corrected: int = 0
    spans_filtered: int = 0
    spans_cached: int = 0
    total_llm_calls: int = 0
    cache_hit_rate: float = 0.0
    avg_processing_time: float = 0.0
    total_processing_time: float = 0.0

    # Per-language stats
    language_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # Error tracking
    errors: int = 0

    def update_cache_hit_rate(self):
        """Update cache hit rate."""
        total_requests = self.total_spans_processed
        if total_requests > 0:
            self.cache_hit_rate = self.spans_cached / total_requests

    def get_avg_time_per_span(self) -> float:
        """Get average processing time per span."""
        if self.total_spans_processed > 0:
            return self.total_processing_time / self.total_spans_processed
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'total_spans_processed': self.total_spans_processed,
            'spans_corrected': self.spans_corrected,
            'spans_filtered': self.spans_filtered,
            'spans_cached': self.spans_cached,
            'total_llm_calls': self.total_llm_calls,
            'cache_hit_rate': self.cache_hit_rate,
            'avg_processing_time': self.avg_processing_time,
            'total_processing_time': self.total_processing_time,
            'language_stats': self.language_stats,
            'errors': self.errors
        }

class LLMV3Corrector:
    """Enhanced LLM corrector for evaluation mode v3 with strict typo-only corrections."""

    def __init__(self, config: LLMV3Config = None):
        """Initialize LLM V3 corrector."""
        self.config = config or LLMV3Config()
        self.language_thresholds = LanguageThresholds()
        self.telemetry = CorrectionTelemetry()

        # Initialize LLM client
        self.llm_client = None
        if self.config.llm_enabled and not self.config.kill_switch:
            self._initialize_llm_client()

        # Initialize caches and locks
        self.correction_cache = {}
        self.cache_lock = threading.Lock()

        # Akkadian patterns for transliteration guard
        self.akkadian_patterns = [
            r'[āēīū]',  # Long vowels
            r'[šṣṭḫ]',  # Special consonants
            r'[ŠṢṬḪ]',  # Uppercase special consonants
            r'\b(?:lugal|dingir|dumu|šarru|bēlu|awīlu|māru|ilu|šamû|erṣetu)\b',  # Common Akkadian words
        ]
        self.akkadian_regex = re.compile('|'.join(f'({p})' for p in self.akkadian_patterns), re.IGNORECASE)

        # Numeric and table patterns for filtering
        self.numeric_pattern = re.compile(r'^\d+[\d\s\.,\-/:]*\d*$')
        self.table_pattern = re.compile(r'^[\s\|\+\-\=\.]+$')

        logger.info(f"LLM V3 Corrector initialized (enabled={self.config.llm_enabled}, cache={self.config.cache_enabled})")

    def _initialize_llm_client(self):
        """Initialize the LLM client with improved Ollama handling."""
        if not self.config.llm_enabled:
            logger.info("LLM corrections disabled in config")
            self.llm_client = None
            return

        try:
            # First, try to start Ollama if it's not running
            self._ensure_ollama_running()

            from llm_correction import LLMCorrector
            self.llm_client = LLMCorrector(
                provider="ollama",
                model=self.config.model_id,
                timeout=self.config.timeout,
                max_workers=self.config.max_workers
            )

            # Verify the client was created successfully
            if self.llm_client and self.llm_client.provider != 'none':
                logger.info(f"LLM client initialized successfully with model: {self.config.model_id}")
            else:
                logger.error("LLM client initialization returned None or disabled provider")
                self.llm_client = None

        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            logger.error("LLM corrections will be disabled. Please ensure Ollama is running with: ollama serve")
            self.llm_client = None

    def _ensure_ollama_running(self):
        """Ensure Ollama is running and the model is available."""
        import requests
        import time
        import subprocess
        import sys

        max_retries = 3
        ollama_url = "http://localhost:11434"

        for attempt in range(max_retries):
            try:
                # Check if Ollama is responding
                response = requests.get(f"{ollama_url}/api/tags", timeout=5)
                if response.status_code == 200:
                    models_data = response.json()
                    available_models = [m['name'] for m in models_data.get('models', [])]

                    logger.info(f"Ollama is running. Available models: {available_models}")

                    # Check if our model is available
                    if self.config.model_id in available_models:
                        logger.info(f"Model {self.config.model_id} is available")
                        return True
                    else:
                        logger.warning(f"Model {self.config.model_id} not found. Available: {available_models}")
                        logger.info(f"Attempting to pull model {self.config.model_id}...")

                        # Try to pull the model
                        try:
                            result = subprocess.run(
                                [sys.executable, "-m", "ollama", "pull", self.config.model_id],
                                capture_output=True,
                                text=True,
                                timeout=300  # 5 minute timeout
                            )
                            if result.returncode == 0:
                                logger.info(f"Successfully pulled model {self.config.model_id}")
                                return True
                            else:
                                logger.error(f"Failed to pull model: {result.stderr}")
                        except subprocess.TimeoutExpired:
                            logger.error("Model pull timed out")
                        except FileNotFoundError:
                            logger.error("ollama command not found. Please install Ollama CLI")

                else:
                    logger.warning(f"Ollama responded with status {response.status_code}")

            except requests.exceptions.ConnectionError:
                logger.warning(f"Ollama not responding on {ollama_url} (attempt {attempt + 1}/{max_retries})")

                # Try to start Ollama if this is the first attempt
                if attempt == 0:
                    logger.info("Attempting to start Ollama...")
                    try:
                        # Try to start Ollama in the background
                        subprocess.Popen(
                            [sys.executable, "-m", "ollama", "serve"],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL
                        )
                        logger.info("Started Ollama serve process")
                        time.sleep(3)  # Give it time to start
                    except FileNotFoundError:
                        logger.error("ollama command not found. Please install Ollama CLI")
                        logger.error("Visit: https://ollama.com/download")
                        break
                    except Exception as e:
                        logger.error(f"Failed to start Ollama: {e}")
                        break

            except Exception as e:
                logger.error(f"Error checking Ollama status: {e}")

            # Wait before retry
            if attempt < max_retries - 1:
                time.sleep(2)

        logger.error(f"Could not establish connection to Ollama after {max_retries} attempts")
        logger.error("Please ensure Ollama is installed and running:")
        logger.error("1. Install Ollama: https://ollama.com/download")
        logger.error("2. Start Ollama: ollama serve")
        logger.error("3. Pull model: ollama pull llama3.2:latest")
        return False

    def _is_akkadian_text(self, text: str) -> bool:
        """Detect if text contains Akkadian characters or words."""
        if not text.strip():
            return False
        return bool(self.akkadian_regex.search(text))

    def _should_filter_span(self, text: str, confidence: float, bbox: List[float]) -> Tuple[bool, str]:
        """Determine if span should be filtered out from LLM correction."""
        # Length filter
        if len(text.strip()) < self.config.min_span_length:
            return True, "too_short"
        if len(text.strip()) > self.config.max_span_length:
            return True, "too_long"

        # Numeric filter
        if self.config.filter_numeric_only and self.numeric_pattern.match(text.strip()):
            return True, "numeric_only"

        # Table filter (lines that look like table separators)
        if self.config.filter_tables and self.table_pattern.match(text.strip()):
            return True, "table_separator"

        # Empty or whitespace only
        if not text.strip():
            return True, "empty"

        return False, "valid"

    def _detect_span_language(self, text: str) -> str:
        """Detect language of a span using enhanced detection."""
        if not self.config.language_detection_per_span:
            return 'unknown'

        # Check for Akkadian first
        if self._is_akkadian_text(text):
            return 'akkadian'

        # Use character-based detection for other languages
        text_lower = text.lower()

        # Turkish indicators
        turkish_chars = set('çğıöşüÇĞIİÖŞÜ')
        if any(c in turkish_chars for c in text):
            return 'turkish'

        # German indicators
        german_chars = set('äöüßÄÖÜ')
        if any(c in german_chars for c in text):
            return 'german'

        # French indicators
        french_chars = set('àâäéèêëïîôùûüÿçÀÂÄÉÈÊËÏÎÔÙÛÜŸÇ')
        if any(c in french_chars for c in text):
            return 'french'

        # Italian indicators
        italian_chars = set('àèéìíîòóùúÀÈÉÌÍÎÒÓÙÚ')
        if any(c in italian_chars for c in text):
            return 'italian'

        return 'english'

    def _should_correct_span(self, text: str, confidence: float, language: str) -> bool:
        """Determine if span should be sent to LLM based on confidence thresholds."""
        if not text.strip() or self.config.kill_switch:
            return False

        # Get language-specific threshold
        threshold = self.language_thresholds.get_threshold(language)

        # Always correct Akkadian (if enabled)
        if language == 'akkadian' and self.config.akkadian_transliteration_guard:
            return False  # Don't correct Akkadian transliteration

        # Correct if confidence is below threshold
        return confidence < threshold

    def _create_cache_key(self, text: str, language: str, bbox: List[float]) -> str:
        """Create cache key for span correction."""
        # Normalize text for consistent caching
        normalized_text = text.strip().lower()

        # Create hash of all components
        key_data = f"{self.config.model_id}:{self.config.prompt_version}:{language}:{normalized_text}:{bbox}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _create_strict_typo_prompt(self, text: str, language: str) -> str:
        """Create strict typo-only correction prompt."""
        language_context = {
            'akkadian': "Ancient Mesopotamian cuneiform transliteration. DO NOT modify transliteration.",
            'turkish': "Turkish language context. Fix OCR errors, correct diacritics (ç,ğ,ı,ö,ş,ü).",
            'german': "German language context. Fix OCR errors, correct umlauts (ä,ö,ü,ß).",
            'french': "French language context. Fix OCR errors, correct accents (à,â,ä,é,è,ê,ë,ï,î,ô,ù,û,ü,ÿ,ç).",
            'italian': "Italian language context. Fix OCR errors, correct accents (à,è,é,ì,í,î,ò,ó,ù,ú).",
            'english': "English language context. Fix OCR errors."
        }

        context = language_context.get(language.lower(), language_context['english'])

        prompt = f"""Fix ONLY clear OCR typos in this text. {context}

STRICT RULES:
- Fix ONLY obvious OCR misreadings (e.g., "sencsi" → "senesi", "kollcksiyonunun" → "kolleksiyonunun")
- Correct missing or wrong diacritics/accents ONLY if 100% certain
- Fix obvious number misreadings (e.g., "7955" → "1955" in date contexts)
- DO NOT change proper nouns, technical terms, or transliterations
- DO NOT translate, paraphrase, or reword
- DO NOT change sentence structure or meaning
- DO NOT add or remove words unless fixing clear OCR errors
- Keep original punctuation, capitalization, and formatting
- Return ONLY the corrected text with NO explanations, notes, or comments

TEXT: "{text}"

CORRECTED:"""

        return prompt

    def _correct_single_span(self, text: str, confidence: float, bbox: List[float], span_id: str) -> SpanCorrectionResult:
        """Correct a single span using LLM."""
        start_time = time.time()

        # Check cache first
        detected_language = self._detect_span_language(text)
        cache_key = self._create_cache_key(text, detected_language, bbox)

        with self.cache_lock:
            if self.config.cache_enabled and cache_key in self.correction_cache:
                cached_result = self.correction_cache[cache_key]
                self.telemetry.spans_cached += 1
                self.telemetry.total_processing_time += time.time() - start_time

                # Update language stats
                if detected_language not in self.telemetry.language_stats:
                    self.telemetry.language_stats[detected_language] = {'cached': 0, 'corrected': 0, 'filtered': 0}
                self.telemetry.language_stats[detected_language]['cached'] += 1

                return SpanCorrectionResult(
                    original_text=text,
                    corrected_text=cached_result['corrected_text'],
                    confidence=confidence,
                    language=detected_language,
                    bbox=bbox,
                    span_id=span_id,
                    corrections_made=cached_result['corrections_made'],
                    processing_time=time.time() - start_time,
                    cache_hit=True
                )

        # Check if span should be filtered
        should_filter, filter_reason = self._should_filter_span(text, confidence, bbox)
        if should_filter:
            self.telemetry.spans_filtered += 1
            if detected_language not in self.telemetry.language_stats:
                self.telemetry.language_stats[detected_language] = {'cached': 0, 'corrected': 0, 'filtered': 0}
            self.telemetry.language_stats[detected_language]['filtered'] += 1

            return SpanCorrectionResult(
                original_text=text,
                corrected_text=text,
                confidence=confidence,
                language=detected_language,
                bbox=bbox,
                span_id=span_id,
                corrections_made=[],
                processing_time=time.time() - start_time,
                cache_hit=False
            )

        # Check if span should be corrected
        if not self._should_correct_span(text, confidence, detected_language):
            return SpanCorrectionResult(
                original_text=text,
                corrected_text=text,
                confidence=confidence,
                language=detected_language,
                bbox=bbox,
                span_id=span_id,
                corrections_made=[],
                processing_time=time.time() - start_time,
                cache_hit=False
            )

        # Use LLM for correction
        if not self.llm_client:
            return SpanCorrectionResult(
                original_text=text,
                corrected_text=text,
                confidence=confidence,
                language=detected_language,
                bbox=bbox,
                span_id=span_id,
                corrections_made=[],
                processing_time=time.time() - start_time,
                cache_hit=False
            )

        try:
            # Create prompt
            prompt = self._create_strict_typo_prompt(text, detected_language)

            # Call LLM
            self.telemetry.total_llm_calls += 1
            corrected_text = self.llm_client._call_ollama(prompt)

            if not corrected_text:
                corrected_text = text  # Fallback

            # Find corrections made
            corrections = []
            if corrected_text != text:
                original_words = text.split()
                corrected_words = corrected_text.split()
                min_len = min(len(original_words), len(corrected_words))
                for i in range(min_len):
                    if original_words[i] != corrected_words[i]:
                        corrections.append((original_words[i], corrected_words[i]))

            processing_time = time.time() - start_time

            # Cache result
            if self.config.cache_enabled:
                with self.cache_lock:
                    if len(self.correction_cache) >= self.config.cache_max_size:
                        # Simple LRU: remove oldest half
                        items = list(self.correction_cache.items())
                        self.correction_cache = dict(items[len(items)//2:])

                    self.correction_cache[cache_key] = {
                        'corrected_text': corrected_text,
                        'corrections_made': corrections,
                        'language': detected_language,
                        'timestamp': time.time()
                    }

            # Update telemetry
            self.telemetry.spans_corrected += 1
            self.telemetry.total_processing_time += processing_time

            if detected_language not in self.telemetry.language_stats:
                self.telemetry.language_stats[detected_language] = {'cached': 0, 'corrected': 0, 'filtered': 0}
            self.telemetry.language_stats[detected_language]['corrected'] += 1

            # Log telemetry periodically
            self.telemetry.total_spans_processed += 1
            if self.config.enable_telemetry and self.telemetry.total_spans_processed % self.config.telemetry_log_interval == 0:
                self.telemetry.update_cache_hit_rate()
                logger.info(f"LLM V3 Telemetry: {self.telemetry.to_dict()}")

            return SpanCorrectionResult(
                original_text=text,
                corrected_text=corrected_text,
                confidence=confidence,
                language=detected_language,
                bbox=bbox,
                span_id=span_id,
                corrections_made=corrections,
                processing_time=processing_time,
                cache_hit=False
            )

        except Exception as e:
            logger.error(f"Error correcting span {span_id}: {e}")
            self.telemetry.errors += 1

            return SpanCorrectionResult(
                original_text=text,
                corrected_text=text,
                confidence=confidence,
                language=detected_language,
                bbox=bbox,
                span_id=span_id,
                corrections_made=[],
                processing_time=time.time() - start_time,
                cache_hit=False
            )

    def correct_spans(self, spans: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Correct multiple spans using LLM V3 system.

        Args:
            spans: List of span dictionaries with 'text', 'confidence', 'bbox', 'id' keys

        Returns:
            Tuple of (corrected_spans, correction_stats)
        """
        if not spans or not self.config.llm_enabled or self.config.kill_switch:
            return spans, {
                'llm_enabled': self.config.llm_enabled,
                'kill_switch': self.config.kill_switch,
                'spans_processed': 0,
                'spans_corrected': 0,
                'cache_hit_rate': 0.0,
                'total_processing_time': 0.0
            }

        start_time = time.time()
        corrected_spans = []
        total_corrected = 0

        # Process spans
        for span in spans:
            result = self._correct_single_span(
                text=span.get('text', ''),
                confidence=span.get('confidence', 0.8),
                bbox=span.get('bbox', []),
                span_id=span.get('id', f"span_{len(corrected_spans)}")
            )

            # Create corrected span
            corrected_span = span.copy()
            corrected_span['text'] = result.corrected_text
            corrected_span['original_text'] = result.original_text
            corrected_span['corrections'] = result.corrections_made
            corrected_span['llm_language'] = result.language
            corrected_span['llm_processing_time'] = result.processing_time
            corrected_span['cache_hit'] = result.cache_hit

            if result.corrected_text != result.original_text:
                total_corrected += 1

            corrected_spans.append(corrected_span)

        total_time = time.time() - start_time

        # Update telemetry
        self.telemetry.update_cache_hit_rate()
        self.telemetry.avg_processing_time = self.get_avg_time_per_span()

        return corrected_spans, {
            'llm_enabled': True,
            'kill_switch': False,
            'spans_processed': len(spans),
            'spans_corrected': total_corrected,
            'cache_hit_rate': self.telemetry.cache_hit_rate,
            'total_processing_time': total_time,
            'avg_processing_time': self.telemetry.get_avg_time_per_span(),
            'language_stats': self.telemetry.language_stats.copy(),
            'errors': self.telemetry.errors,
            'telemetry': self.telemetry.to_dict()
        }

    def get_avg_time_per_span(self) -> float:
        """Get average processing time per span."""
        return self.telemetry.get_avg_time_per_span()

    def get_telemetry_summary(self) -> Dict[str, Any]:
        """Get comprehensive telemetry summary."""
        self.telemetry.update_cache_hit_rate()
        return {
            'telemetry': self.telemetry.to_dict(),
            'cache_size': len(self.correction_cache),
            'config': {
                'llm_enabled': self.config.llm_enabled,
                'cache_enabled': self.config.cache_enabled,
                'kill_switch': self.config.kill_switch,
                'model_id': self.config.model_id,
                'prompt_version': self.config.prompt_version
            }
        }

    def reset_telemetry(self):
        """Reset telemetry counters."""
        self.telemetry = CorrectionTelemetry()
        logger.info("LLM V3 telemetry reset")

    def enable_kill_switch(self):
        """Enable kill switch to stop LLM corrections."""
        self.config.kill_switch = True
        logger.warning("LLM V3 kill switch enabled - corrections disabled")

    def disable_kill_switch(self):
        """Disable kill switch to resume LLM corrections."""
        self.config.kill_switch = False
        logger.info("LLM V3 kill switch disabled - corrections enabled")

# Global instance for evaluation mode
_llm_v3_corrector = None

def initialize_llm_v3(config: Dict[str, Any]) -> bool:
    """Initialize LLM V3 corrector for evaluation mode."""
    global _llm_v3_corrector

    llm_config = config.get('llm', {})
    v3_config = LLMV3Config(
        llm_enabled=llm_config.get('llm_enabled', True),
        kill_switch=llm_config.get('kill_switch', False),
        model_id=llm_config.get('model_id', 'llama3.2:latest'),
        prompt_version=llm_config.get('prompt_version', 'v3_strict_typo_only'),
        cache_enabled=llm_config.get('cache_enabled', True),
        max_workers=llm_config.get('max_workers', 3),
        timeout=llm_config.get('timeout', 30),
        enable_telemetry=llm_config.get('enable_telemetry', True)
    )

    try:
        _llm_v3_corrector = LLMV3Corrector(v3_config)
        logger.info("LLM V3 corrector initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize LLM V3 corrector: {e}")
        return False

def get_llm_v3_corrector():
    """Get the global LLM V3 corrector instance."""
    return _llm_v3_corrector

def cleanup_llm_v3():
    """Cleanup LLM V3 corrector."""
    global _llm_v3_corrector

    if _llm_v3_corrector:
        # Log final telemetry
        if _llm_v3_corrector.config.enable_telemetry:
            summary = _llm_v3_corrector.get_telemetry_summary()
            logger.info(f"LLM V3 final telemetry: {json.dumps(summary, indent=2)}")

        _llm_v3_corrector = None
