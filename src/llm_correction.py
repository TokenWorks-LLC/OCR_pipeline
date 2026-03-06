"""
LLM-based multilingual spelling correction integration for OCR pipeline.
"""
import json
import logging
import re
import time
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CorrectionResult:
    """Result of LLM correction with metadata."""
    original_text: str
    corrected_text: str
    confidence: float
    language: str
    corrections_made: List[Tuple[str, str]]  # (original_word, corrected_word) pairs
    processing_time: float

class LLMCorrector:
    """LLM-based multilingual OCR correction with Ollama/llama.cpp support."""
    
    def __init__(self, 
                 provider: str = "ollama",
                 model: str = "llama3.2:latest",
                 base_url: str = "http://localhost:11434",
                 timeout: int = 30,
                 max_workers: int = 3):
        """
        Initialize LLM corrector.
        
        Args:
            provider: 'ollama' or 'llamacpp' or 'none'
            model: Model name to use
            base_url: Base URL for Ollama API
            timeout: Request timeout in seconds
            max_workers: Maximum concurrent correction threads
        """
        self.provider = provider
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.max_workers = max_workers
        self.client = None
        self.correction_cache = {}
        
        if provider != 'none':
            self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the LLM client based on provider."""
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
            
            # Test connection
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
            
            # This would need model path configuration
            logger.info("llama-cpp-python support not yet implemented")
            self.provider = 'none'
            
        except ImportError:
            logger.error("llama-cpp-python not available")
            self.provider = 'none'
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection based on character patterns."""
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
    
    def _create_correction_prompt(self, text: str, language: str) -> str:
        """Create language-specific correction prompt."""
        
        language_instructions = {
            'turkish': "Turkish language context. Fix OCR errors, correct diacritics (ç,ğ,ı,ö,ş,ü), and common Turkish words.",
            'german': "German language context. Fix OCR errors, correct umlauts (ä,ö,ü,ß), and common German words.",
            'french': "French language context. Fix OCR errors, correct accents (à,â,ä,é,è,ê,ë,ï,î,ô,ù,û,ü,ÿ,ç), and common French words.",
            'italian': "Italian language context. Fix OCR errors, correct accents (à,è,é,ì,í,î,ò,ó,ù,ú), and common Italian words.",
            'english': "English language context. Fix OCR errors and common English words."
        }
        
        instruction = language_instructions.get(language, language_instructions['english'])
        
        prompt = f"""Fix OCR errors in this text. {instruction}

RULES:
- Fix clear OCR misreadings (e.g., "sencsi" → "senesi", "kollcksiyonunun" → "kolleksiyonunun")
- Correct missing or wrong diacritics/accents only if certain
- Fix obvious number misreadings (e.g., "7955" → "1955" in date contexts)
- Do NOT change proper nouns unless clearly wrong OCR
- Do NOT translate or paraphrase
- Keep original punctuation and spacing
- Return ONLY the corrected text with NO explanations, notes, or comments

TEXT: "{text}"

CORRECTED:"""

        return prompt
    
    def _call_ollama(self, prompt: str) -> Optional[str]:
        """Make API call to Ollama."""
        if self.provider != 'ollama':
            return None
        
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temperature for consistent corrections
                    "top_p": 0.9,
                    "max_tokens": 512,
                    "stop": ["\n\n", "CORRECTED:", "TEXT:", "RULES:", "(No", "(no", "Note:"]
                }
            }
            
            response = self.requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                corrected = result.get('response', '').strip()
                
                # Clean up response more aggressively
                corrected = re.sub(r'^["\']|["\']$', '', corrected)  # Remove quotes
                corrected = corrected.replace('CORRECTED TEXT:', '').replace('CORRECTED:', '').strip()
                
                # Remove any explanation notes or comments
                corrected = re.sub(r'\(No.*?\)', '', corrected, flags=re.IGNORECASE)
                corrected = re.sub(r'\(no.*?\)', '', corrected, flags=re.IGNORECASE)  
                corrected = re.sub(r'\(assuming.*?\)', '', corrected, flags=re.IGNORECASE)
                corrected = re.sub(r'Note:.*', '', corrected, flags=re.IGNORECASE)
                corrected = re.sub(r'There are no.*', '', corrected, flags=re.IGNORECASE)
                
                # Clean up extra whitespace
                corrected = ' '.join(corrected.split())
                
                return corrected if corrected else None
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Ollama API call failed: {e}")
            return None
    
    def _find_corrections(self, original: str, corrected: str) -> List[Tuple[str, str]]:
        """Find individual word corrections between original and corrected text."""
        original_words = original.split()
        corrected_words = corrected.split()
        
        corrections = []
        
        # Simple word-by-word comparison
        min_len = min(len(original_words), len(corrected_words))
        for i in range(min_len):
            if original_words[i] != corrected_words[i]:
                corrections.append((original_words[i], corrected_words[i]))
        
        return corrections
    
    def correct_text(self, text: str, language_hint: Optional[str] = None) -> CorrectionResult:
        """
        Correct OCR errors in text using LLM.
        
        Args:
            text: Text to correct
            language_hint: Language hint (optional)
        
        Returns:
            CorrectionResult with original text, corrections, and metadata
        """
        start_time = time.time()
        
        if self.provider == 'none' or not text or len(text.strip()) < 3:
            return CorrectionResult(
                original_text=text,
                corrected_text=text,
                confidence=0.0,
                language='unknown',
                corrections_made=[],
                processing_time=time.time() - start_time
            )
        
        # Check cache
        cache_key = (text, language_hint)
        if cache_key in self.correction_cache:
            cached = self.correction_cache[cache_key]
            cached.processing_time = time.time() - start_time  # Update timing
            return cached
        
        # Detect language
        detected_language = language_hint or self._detect_language(text)
        
        # Create prompt
        prompt = self._create_correction_prompt(text, detected_language)
        
        # Call LLM
        corrected_text = self._call_ollama(prompt)
        
        if not corrected_text:
            corrected_text = text  # Fallback to original
        
        # Find corrections made
        corrections = self._find_corrections(text, corrected_text)
        
        # Calculate confidence based on changes made
        confidence = 1.0 if corrected_text == text else 0.8
        if corrections:
            confidence = min(1.0, 0.6 + (len(corrections) * 0.1))
        
        result = CorrectionResult(
            original_text=text,
            corrected_text=corrected_text,
            confidence=confidence,
            language=detected_language,
            corrections_made=corrections,
            processing_time=time.time() - start_time
        )
        
        # Cache result
        self.correction_cache[cache_key] = result
        
        return result
    
    def correct_multiple_texts(self, texts: List[str], 
                             language_hint: Optional[str] = None) -> List[CorrectionResult]:
        """
        Correct multiple texts concurrently.
        
        Args:
            texts: List of texts to correct
            language_hint: Language hint for all texts
        
        Returns:
            List of CorrectionResult objects
        """
        if self.provider == 'none' or not texts:
            return [CorrectionResult(
                original_text=text,
                corrected_text=text,
                confidence=0.0,
                language='unknown',
                corrections_made=[],
                processing_time=0.0
            ) for text in texts]
        
        results = [None] * len(texts)
        
        # Process texts concurrently
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {
                executor.submit(self.correct_text, text, language_hint): i
                for i, text in enumerate(texts)
            }
            
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    logger.error(f"Error correcting text {index}: {e}")
                    # Fallback result
                    results[index] = CorrectionResult(
                        original_text=texts[index],
                        corrected_text=texts[index],
                        confidence=0.0,
                        language='unknown',
                        corrections_made=[],
                        processing_time=0.0
                    )
        
        return results
    
    def get_correction_stats(self) -> Dict[str, Any]:
        """Get correction statistics."""
        if not self.correction_cache:
            return {"total_corrections": 0}
        
        total = len(self.correction_cache)
        corrections_made = sum(1 for result in self.correction_cache.values() 
                              if result.corrected_text != result.original_text)
        
        avg_time = sum(result.processing_time for result in self.correction_cache.values()) / total
        
        languages = {}
        for result in self.correction_cache.values():
            lang = result.language
            languages[lang] = languages.get(lang, 0) + 1
        
        return {
            "total_corrections": total,
            "texts_changed": corrections_made,
            "change_rate": corrections_made / total if total > 0 else 0,
            "avg_processing_time": avg_time,
            "languages_processed": languages,
            "cache_size": total
        }


# Convenience function for easy integration
def create_llm_corrector(config: Dict[str, Any] = None) -> LLMCorrector:
    """Create LLM corrector with configuration."""
    if config is None:
        config = {}
    
    return LLMCorrector(
        provider=config.get('provider', 'ollama'),
        model=config.get('model', 'llama3.2:latest'),
        base_url=config.get('base_url', 'http://localhost:11434'),
        timeout=config.get('timeout', 30),
        max_workers=config.get('max_workers', 3)
    )


def correct_ocr_lines(lines, language_hint=None):
    """
    Correct OCR lines using LLM.
    
    Args:
        lines: List of Line objects from OCR
        language_hint: Optional language hint
        
    Returns:
        Tuple of (corrected_lines, correction_stats)
    """
    if not lines:
        return lines, {'available': False, 'lines_processed': 0, 'lines_changed': 0}
    
    # Create LLM corrector
    corrector = LLMCorrector()
    
    if corrector.provider == 'none':
        return lines, {'available': False, 'lines_processed': 0, 'lines_changed': 0}
    
    # Extract texts for correction
    texts = [line.text for line in lines]
    
    # Correct texts
    correction_results = corrector.correct_multiple_texts(texts, language_hint)
    
    # Apply corrections to lines
    corrected_lines = []
    lines_changed = 0
    
    for i, (line, correction) in enumerate(zip(lines, correction_results)):
        if correction.corrected_text != correction.original_text:
            # Create new line with corrected text
            corrected_line = type(line)(
                text=correction.corrected_text,
                conf=line.conf,
                bbox=line.bbox,
                engine=line.engine
            )
            corrected_lines.append(corrected_line)
            lines_changed += 1
        else:
            corrected_lines.append(line)
    
    correction_stats = {
        'available': True,
        'lines_processed': len(lines),
        'lines_changed': lines_changed
    }
    
    return corrected_lines, correction_stats
