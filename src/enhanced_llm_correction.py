"""
Enhanced LLM-based OCR correction with strict JSON prompts, confidence-based routing,
Akkadian detection, multi-language support, and span-level caching.
"""
import json
import logging
import re
import time
import unicodedata
import hashlib
import pickle
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)

@dataclass
class BoundingBox:
    """Bounding box coordinates."""
    x: float
    y: float
    width: float
    height: float

@dataclass
class OCRSpan:
    """OCR text span with metadata."""
    text: str
    confidence: float
    bbox: Optional[BoundingBox]
    language: Optional[str] = None
    is_akkadian: bool = False
    char_density: float = 0.0

@dataclass
class CorrectionResult:
    """Enhanced result of LLM correction with detailed metadata."""
    original_text: str
    corrected_text: str
    confidence: float
    language: str
    corrections_made: List[Tuple[str, str]]  # (original_word, corrected_word) pairs
    processing_time: float
    span_count: int = 1
    avg_ocr_confidence: float = 0.0
    akkadian_detected: bool = False
    route_reason: str = ""  # Why this was sent to LLM

@dataclass
class CacheEntry:
    """Cache entry for LLM corrections with versioning."""
    input_hash: str
    corrected_text: str
    confidence: float
    language: str
    corrections_made: List[Tuple[str, str]]
    processing_time: float
    timestamp: float
    prompt_version: str
    model_version: str

class LLMCorrectionCache:
    """Span-level cache for LLM corrections with versioned prompts."""
    
    def __init__(self, cache_dir: str = "./cache", max_size: int = 10000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "llm_corrections.pkl"
        self.max_size = max_size
        self.cache: Dict[str, CacheEntry] = {}
        self.hit_count = 0
        self.miss_count = 0
        self.prompt_version = "v2.0"  # Updated with strict JSON prompts
        self.load_cache()
    
    def _hash_input(self, text: str, language: str, confidence: float) -> str:
        """Create hash for input parameters."""
        input_str = f"{text}|{language}|{confidence:.3f}|{self.prompt_version}"
        return hashlib.sha256(input_str.encode()).hexdigest()
    
    def get(self, text: str, language: str, confidence: float, model_version: str) -> Optional[CacheEntry]:
        """Get cached correction if available."""
        cache_key = self._hash_input(text, language, confidence)
        
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            # Check if model version matches
            if entry.model_version == model_version:
                self.hit_count += 1
                logger.debug(f"Cache hit for text: {text[:50]}...")
                return entry
            else:
                # Remove outdated entry
                del self.cache[cache_key]
                self.miss_count += 1
                return None
        
        self.miss_count += 1
        return None
    
    def put(self, text: str, language: str, confidence: float, model_version: str,
            corrected_text: str, corrections_made: List[Tuple[str, str]], 
            processing_time: float) -> None:
        """Store correction in cache."""
        cache_key = self._hash_input(text, language, confidence)
        
        entry = CacheEntry(
            input_hash=cache_key,
            corrected_text=corrected_text,
            confidence=confidence,
            language=language,
            corrections_made=corrections_made,
            processing_time=processing_time,
            timestamp=time.time(),
            prompt_version=self.prompt_version,
            model_version=model_version
        )
        
        self.cache[cache_key] = entry
        
        # Evict oldest entries if cache is too large
        if len(self.cache) > self.max_size:
            oldest_key = min(self.cache.keys(), 
                           key=lambda k: self.cache[k].timestamp)
            del self.cache[oldest_key]
            logger.debug(f"Evicted cache entry: {oldest_key}")
    
    def load_cache(self) -> None:
        """Load cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                logger.info(f"Loaded {len(self.cache)} cache entries")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                self.cache = {}
    
    def save_cache(self) -> None:
        """Save cache to disk."""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
            logger.debug(f"Saved {len(self.cache)} cache entries")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
        
        return {
            "cache_size": len(self.cache),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
            "total_requests": total_requests
        }
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.hit_count = 0
        self.miss_count = 0
        if self.cache_file.exists():
            self.cache_file.unlink()

class AkkadianDetector:
    """Lightweight Akkadian text detector using Unicode patterns and glyph density."""
    
    def __init__(self, min_confidence: float = 0.7, min_length: int = 5):
        self.min_confidence = min_confidence
        self.min_length = min_length
        
        # Akkadian/Cuneiform Unicode ranges
        self.cuneiform_ranges = [
            (0x12000, 0x123FF),  # Cuneiform
            (0x12400, 0x1247F),  # Cuneiform Numbers and Punctuation
        ]
        
        # Common Akkadian transliteration diacritics and patterns
        self.akkadian_diacritics = set([
            'ā', 'â', 'ē', 'ê', 'ī', 'î', 'ō', 'ô', 'ū', 'û',  # macrons and circumflexes
            'ă', 'ĕ', 'ĭ', 'ŏ', 'ŭ',  # breves
            'ṭ', 'ṣ', 'ḫ', 'š', 'ṅ', 'ṁ', 'ḷ', 'ṛ',  # dots and special chars
            'Š', 'Ṭ', 'Ṣ', 'Ḫ', 'Ṅ', 'Ṁ', 'Ḷ', 'Ṛ'
        ])
        
        # Common Akkadian words/patterns (transliteration)
        self.akkadian_patterns = [
            r'\b[šṣṭḫ][a-z]*',  # words starting with Akkadian-specific chars
            r'\b[a-z]*[āēīōūâêîôû][a-z]*',  # words with macrons/circumflexes
            r'\bša\b', r'\bina\b', r'\bana\b', r'\bištu\b',  # common Akkadian words
            r'\b[A-Z]+\b.*[0-9]+',  # artifact/tablet references
        ]
        
        # Compile patterns
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.akkadian_patterns]
    
    def calculate_akkadian_features(self, text: str) -> Dict[str, float]:
        """Calculate Akkadian-specific features from text."""
        if not text or len(text) < self.min_length:
            return {"confidence": 0.0, "diacritic_density": 0.0, "pattern_matches": 0}
        
        # Feature 1: Diacritic density
        diacritic_count = sum(1 for char in text if char in self.akkadian_diacritics)
        diacritic_density = diacritic_count / len(text)
        
        # Feature 2: Cuneiform characters
        cuneiform_count = sum(1 for char in text 
                             if any(start <= ord(char) <= end for start, end in self.cuneiform_ranges))
        cuneiform_density = cuneiform_count / len(text)
        
        # Feature 3: Pattern matches
        pattern_matches = sum(1 for pattern in self.compiled_patterns 
                             if pattern.search(text))
        pattern_score = pattern_matches / len(self.compiled_patterns)
        
        # Feature 4: Character class analysis
        total_chars = len(text)
        latin_extended = sum(1 for char in text 
                           if 0x0100 <= ord(char) <= 0x017F or 0x1E00 <= ord(char) <= 0x1EFF)
        latin_extended_density = latin_extended / total_chars if total_chars > 0 else 0
        
        # Feature 5: Word structure analysis
        words = text.split()
        akkadian_word_count = 0
        for word in words:
            # Check for typical Akkadian transliteration patterns
            if any(char in self.akkadian_diacritics for char in word):
                akkadian_word_count += 1
            elif any(pattern.match(word) for pattern in self.compiled_patterns):
                akkadian_word_count += 1
        
        akkadian_word_ratio = akkadian_word_count / len(words) if words else 0
        
        # Calculate overall confidence
        confidence = (
            diacritic_density * 0.3 +
            cuneiform_density * 0.4 +
            pattern_score * 0.2 +
            latin_extended_density * 0.05 +
            akkadian_word_ratio * 0.05
        )
        
        return {
            "confidence": min(confidence, 1.0),
            "diacritic_density": diacritic_density,
            "cuneiform_density": cuneiform_density,
            "pattern_matches": pattern_matches,
            "latin_extended_density": latin_extended_density,
            "akkadian_word_ratio": akkadian_word_ratio
        }
    
    def is_akkadian(self, text: str, min_confidence: Optional[float] = None) -> Tuple[bool, float]:
        """Detect if text contains Akkadian content."""
        confidence_threshold = min_confidence or self.min_confidence
        features = self.calculate_akkadian_features(text)
        confidence = features["confidence"]
        
        return confidence >= confidence_threshold, confidence
    
    def extract_akkadian_spans(self, text: str, spans: List[OCRSpan]) -> List[OCRSpan]:
        """Extract spans that are likely Akkadian."""
        akkadian_spans = []
        
        for span in spans:
            is_akkadian, confidence = self.is_akkadian(span.text)
            if is_akkadian:
                span.is_akkadian = True
                span.char_density = confidence
                akkadian_spans.append(span)
        
        return akkadian_spans

class LanguageDetector:
    """Enhanced language detection with multi-language support."""
    
    def __init__(self):
        # Language-specific character sets
        self.language_chars = {
            'turkish': set('çğıöşüÇĞIİÖŞÜ'),
            'german': set('äöüßÄÖÜ'),
            'french': set('àâäéèêëïîôùûüÿçÀÂÄÉÈÊËÏÎÔÙÛÜŸÇ'),
            'italian': set('àèéìíîòóùúÀÈÉÌÍÎÒÓÙÚ'),
            'spanish': set('áéíóúüñÁÉÍÓÚÜÑ'),
            'arabic': set(chr(i) for i in range(0x0600, 0x06FF)),
            'greek': set(chr(i) for i in range(0x0370, 0x03FF)),
        }
        
        # Common words for language detection
        self.language_words = {
            'english': {'the', 'and', 'of', 'to', 'in', 'is', 'for', 'with', 'on', 'by'},
            'turkish': {'ve', 'bir', 'bu', 'da', 'de', 'ile', 'için', 'olan', 'var', 'bu'},
            'german': {'der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'ist'},
            'french': {'le', 'de', 'et', 'un', 'à', 'être', 'et', 'en', 'avoir', 'que'},
            'italian': {'il', 'di', 'e', 'un', 'a', 'essere', 'per', 'in', 'con', 'su'},
            'spanish': {'el', 'de', 'y', 'un', 'en', 'ser', 'para', 'con', 'por', 'su'}
        }
    
    def detect_language(self, text: str) -> Dict[str, float]:
        """Detect language(s) in text with confidence scores."""
        if not text:
            return {'english': 1.0}
        
        text_lower = text.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))
        
        language_scores = defaultdict(float)
        
        # Character-based detection
        for lang, chars in self.language_chars.items():
            char_matches = sum(1 for c in text if c in chars)
            if char_matches > 0:
                language_scores[lang] += char_matches / len(text)
        
        # Word-based detection
        for lang, common_words in self.language_words.items():
            word_matches = len(words.intersection(common_words))
            if word_matches > 0:
                language_scores[lang] += word_matches / len(words) if words else 0
        
        # Normalize scores
        total_score = sum(language_scores.values())
        if total_score > 0:
            for lang in language_scores:
                language_scores[lang] /= total_score
        else:
            language_scores['english'] = 1.0
        
        return dict(language_scores)
    
    def get_primary_language(self, text: str) -> str:
        """Get the most likely primary language."""
        scores = self.detect_language(text)
        return max(scores.items(), key=lambda x: x[1])[0]

class EnhancedLLMCorrector:
    """Enhanced LLM corrector with confidence-based routing and strict JSON prompts."""
    
    def __init__(self, 
                 provider: str = "ollama",
                 model: str = "llama3.1:8b",
                 base_url: str = "http://localhost:11434",
                 timeout: int = 30,
                 max_workers: int = 3,
                 confidence_threshold: float = 0.8,
                 min_text_length: int = 3):
        """
        Initialize enhanced LLM corrector.
        
        Args:
            provider: 'ollama' or 'llamacpp' or 'none'
            model: Model name to use
            base_url: Base URL for Ollama API
            timeout: Request timeout in seconds
            max_workers: Maximum concurrent correction threads
            confidence_threshold: Only correct spans below this confidence
            min_text_length: Minimum text length to consider for correction
        """
        self.provider = provider
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.max_workers = max_workers
        self.confidence_threshold = confidence_threshold
        self.min_text_length = min_text_length
        
        self.client = None
        self.correction_cache = {}
        
        # Initialize cache system
        self.cache = LLMCorrectionCache()
        
        # Initialize detectors
        self.akkadian_detector = AkkadianDetector()
        self.language_detector = LanguageDetector()
        
        # Telemetry and statistics
        self.stats = {
            'total_corrections': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_processing_time': 0.0,
            'corrections_by_language': defaultdict(int),
            'corrections_by_confidence': defaultdict(int)
        }
        
        if provider != 'none':
            self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the LLM client."""
        if self.provider == 'ollama':
            self._initialize_ollama()
        elif self.provider == 'llamacpp':
            self._initialize_llamacpp()
        else:
            logger.warning(f"Unknown provider {self.provider}, corrections disabled")
            self.provider = 'none'
    
    def _initialize_ollama(self):
        """Initialize Ollama client."""
        try:
            import requests
            self.requests = requests
            
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                if self.model in model_names:
                    logger.info(f"Connected to Ollama, using model: {self.model}")
                else:
                    logger.warning(f"Model {self.model} not found. Available: {model_names}")
            else:
                logger.error(f"Failed to connect to Ollama at {self.base_url}")
                self.provider = 'none'
                
        except ImportError:
            logger.error("requests library not available for Ollama")
            self.provider = 'none'
        except Exception as e:
            logger.error(f"Failed to initialize Ollama: {e}")
            self.provider = 'none'
    
    def _initialize_llamacpp(self):
        """Initialize llama-cpp-python client."""
        try:
            from llama_cpp import Llama
            logger.info("llama-cpp-python support not yet implemented")
            self.provider = 'none'
        except ImportError:
            logger.error("llama-cpp-python not available")
            self.provider = 'none'
    
    def _create_strict_json_prompt(self, text: str, language: str, context: str = "") -> str:
        """Create strict JSON prompt for typo-only fixes."""
        
        language_instructions = {
            'turkish': "Turkish context: Fix diacritics (ç,ğ,ı,ö,ş,ü) and common Turkish OCR errors",
            'german': "German context: Fix umlauts (ä,ö,ü,ß) and common German OCR errors", 
            'french': "French context: Fix accents (à,â,ä,é,è,ê,ë,ï,î,ô,ù,û,ü,ÿ,ç) and French OCR errors",
            'italian': "Italian context: Fix accents (à,è,é,ì,í,î,ò,ó,ù,ú) and Italian OCR errors",
            'spanish': "Spanish context: Fix accents (á,é,í,ó,ú,ü,ñ) and Spanish OCR errors",
            'arabic': "Arabic context: Fix Arabic script OCR errors and diacritics",
            'english': "English context: Fix common English OCR errors"
        }
        
        instruction = language_instructions.get(language, language_instructions['english'])
        context_note = f" Context: {context}" if context else ""
        
        prompt = f"""You are an OCR error correction specialist. Fix ONLY clear typos and OCR errors.{context_note}

STRICT RULES:
- Fix ONLY obvious OCR misreadings (e.g., "sencsi" → "senesi", "1ife" → "life")
- Correct missing/wrong diacritics ONLY if certain (based on {instruction})
- Fix clear number substitutions (e.g., "7" → "1" in dates like "7955" → "1955")
- Do NOT change proper nouns unless clearly wrong OCR
- Do NOT translate, paraphrase, or rewrite
- Keep ALL punctuation, spacing, and formatting exactly
- Return ONLY valid JSON with no additional text

INPUT TEXT: "{text}"

Respond with valid JSON only:
{{
  "corrected_text": "exact corrected text here",
  "corrections": [
    {{"original": "wrong_word", "corrected": "right_word", "reason": "brief_ocr_error_type"}}
  ],
  "confidence": 0.95
}}"""

        return prompt
    
    def _should_route_to_llm(self, span: OCRSpan) -> Tuple[bool, str]:
        """Determine if span should be routed to LLM based on confidence and characteristics."""
        # Skip if text too short
        if len(span.text.strip()) < self.min_text_length:
            return False, "text_too_short"
        
        # Skip if confidence is high
        if span.confidence >= self.confidence_threshold:
            return False, "confidence_too_high"
        
        # Skip if Akkadian (route to specialized Akkadian processor instead)
        if span.is_akkadian:
            return False, "akkadian_detected"
        
        # Check for obvious OCR error patterns
        text = span.text.lower()
        
        # Common OCR error indicators
        ocr_error_patterns = [
            r'[0-9]+[a-z]',  # numbers mixed with letters
            r'[a-z]+[0-9]+[a-z]',  # letters-numbers-letters
            r'[il1|]+',  # common OCR confusion chars
            r'[rn]+m',  # rn -> m confusion
            r'[vv]+w',  # vv -> w confusion
        ]
        
        has_likely_errors = any(re.search(pattern, text) for pattern in ocr_error_patterns)
        
        if has_likely_errors:
            return True, "likely_ocr_errors_detected"
        
        # Route low confidence spans
        if span.confidence < self.confidence_threshold * 0.7:  # Very low confidence
            return True, f"low_confidence_{span.confidence:.2f}"
        
        return False, "no_correction_needed"
    
    def _call_ollama_json(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Make API call to Ollama expecting JSON response."""
        if self.provider != 'ollama':
            return None
        
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "format": "json",  # Request JSON format
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "max_tokens": 512,
                }
            }
            
            response = self.requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '').strip()
                
                # Parse JSON response
                try:
                    json_response = json.loads(response_text)
                    return json_response
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response: {e}")
                    logger.error(f"Raw response: {response_text[:200]}")
                    return None
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Ollama API call failed: {e}")
            return None
    
    def correct_span(self, span: OCRSpan, context: str = "") -> CorrectionResult:
        """Correct a single OCR span with caching."""
        start_time = time.time()
        
        # Check if we should route to LLM
        should_correct, reason = self._should_route_to_llm(span)
        
        if not should_correct or self.provider == 'none':
            return CorrectionResult(
                original_text=span.text,
                corrected_text=span.text,
                confidence=span.confidence,
                language='unknown',
                corrections_made=[],
                processing_time=time.time() - start_time,
                span_count=1,
                avg_ocr_confidence=span.confidence,
                akkadian_detected=span.is_akkadian,
                route_reason=reason
            )
        
        # Detect language if not provided
        language = span.language or self.language_detector.get_primary_language(span.text)
        
        # Check cache first
        cached_entry = self.cache.get(span.text, language, span.confidence, self.model)
        if cached_entry:
            self.stats['cache_hits'] += 1
            return CorrectionResult(
                original_text=span.text,
                corrected_text=cached_entry.corrected_text,
                confidence=cached_entry.confidence,
                language=cached_entry.language,
                corrections_made=cached_entry.corrections_made,
                processing_time=time.time() - start_time,  # Just cache lookup time
                span_count=1,
                avg_ocr_confidence=span.confidence,
                akkadian_detected=span.is_akkadian,
                route_reason=f"{reason}_cached"
            )
        
        self.stats['cache_misses'] += 1
        
        # Create JSON prompt
        prompt = self._create_strict_json_prompt(span.text, language, context)
        
        # Call LLM
        json_response = self._call_ollama_json(prompt)
        
        if json_response and 'corrected_text' in json_response:
            corrected_text = json_response['corrected_text']
            corrections = json_response.get('corrections', [])
            llm_confidence = json_response.get('confidence', 0.8)
            
            # Convert corrections format
            corrections_made = [(c.get('original', ''), c.get('corrected', '')) 
                              for c in corrections if 'original' in c and 'corrected' in c]
        else:
            corrected_text = span.text
            corrections_made = []
            llm_confidence = 0.0
        
        processing_time = time.time() - start_time
        
        # Store in cache
        self.cache.put(
            text=span.text,
            language=language,
            confidence=span.confidence,
            model_version=self.model,
            corrected_text=corrected_text,
            corrections_made=corrections_made,
            processing_time=processing_time
        )
        
        # Update statistics
        self.stats['total_corrections'] += 1
        self.stats['total_processing_time'] += processing_time
        self.stats['corrections_by_language'][language] += 1
        self.stats['corrections_by_confidence'][f"{span.confidence:.1f}"] += 1
        
        result = CorrectionResult(
            original_text=span.text,
            corrected_text=corrected_text,
            confidence=llm_confidence,
            language=language,
            corrections_made=corrections_made,
            processing_time=processing_time,
            span_count=1,
            avg_ocr_confidence=span.confidence,
            akkadian_detected=span.is_akkadian,
            route_reason=reason
        )
        
        return result
    
    def correct_spans(self, spans: List[OCRSpan], context: str = "") -> List[CorrectionResult]:
        """Correct multiple OCR spans with confidence-based routing."""
        if self.provider == 'none' or not spans:
            return [CorrectionResult(
                original_text=span.text,
                corrected_text=span.text,
                confidence=span.confidence,
                language='unknown',
                corrections_made=[],
                processing_time=0.0,
                span_count=1,
                avg_ocr_confidence=span.confidence,
                akkadian_detected=span.is_akkadian,
                route_reason="llm_disabled"
            ) for span in spans]
        
        # Detect Akkadian spans first
        for span in spans:
            is_akkadian, confidence = self.akkadian_detector.is_akkadian(span.text)
            span.is_akkadian = is_akkadian
            span.char_density = confidence
        
        # Detect languages for all spans
        for span in spans:
            if not span.language:
                span.language = self.language_detector.get_primary_language(span.text)
        
        results = [None] * len(spans)
        
        # Process spans concurrently
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {
                executor.submit(self.correct_span, span, context): i
                for i, span in enumerate(spans)
            }
            
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    logger.error(f"Error correcting span {index}: {e}")
                    # Fallback result
                    results[index] = CorrectionResult(
                        original_text=spans[index].text,
                        corrected_text=spans[index].text,
                        confidence=0.0,
                        language='unknown',
                        corrections_made=[],
                        processing_time=0.0,
                        span_count=1,
                        avg_ocr_confidence=spans[index].confidence,
                        akkadian_detected=spans[index].is_akkadian,
                        route_reason="processing_error"
                    )
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive correction statistics including cache performance."""
        cache_stats = self.cache.get_stats()
        
        # Get basic correction stats
        stats = {
            'total_corrections': self.stats['total_corrections'],
            'total_processing_time': self.stats['total_processing_time'],
            'avg_processing_time': (self.stats['total_processing_time'] / 
                                  max(1, self.stats['total_corrections'])),
            'corrections_by_language': dict(self.stats['corrections_by_language']),
            'corrections_by_confidence': dict(self.stats['corrections_by_confidence']),
            'cache': cache_stats
        }
        
        return stats
    
    def get_correction_stats(self) -> Dict[str, Any]:
        """Alias for get_stats for backward compatibility."""
        return self.get_stats()
    
    def save_cache(self) -> None:
        """Save cache to disk."""
        self.cache.save_cache()
    
    def clear_cache(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()


def create_enhanced_llm_corrector(config: Dict[str, Any] = None) -> EnhancedLLMCorrector:
    """Create enhanced LLM corrector with configuration."""
    if config is None:
        config = {}
    
    return EnhancedLLMCorrector(
        provider=config.get('provider', 'ollama'),
        model=config.get('model', 'llama3.1:8b'),
        base_url=config.get('base_url', 'http://localhost:11434'),
        timeout=config.get('timeout', 30),
        max_workers=config.get('max_workers', 3),
        confidence_threshold=config.get('confidence_threshold', 0.8),
        min_text_length=config.get('min_text_length', 3)
    )