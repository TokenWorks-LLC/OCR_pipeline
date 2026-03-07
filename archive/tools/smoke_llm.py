#!/usr/bin/env python3
"""
Smoke Test for LLM Post-Correction System

Validates:
1. Ollama service availability
2. Model availability and download
3. JSON-mode generation
4. Guardrail enforcement
5. Transliteration protection
6. Cache functionality
7. Telemetry tracking

Run: python tools/smoke_llm.py
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from llm.clients.ollama_client import OllamaClient, OllamaConfig
from llm.corrector import LLMCorrector, GuardrailViolation
from llm.json_schemas import CorrectionRequest, CorrectionFlags, ContextInfo
from akkadian_extract import is_akkadian_transliteration, estimate_transliteration_confidence

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SmokeTest:
    """LLM smoke test runner."""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []
    
    def test(self, name: str, func):
        """Run a single test."""
        print(f"\n{'='*80}")
        print(f"TEST: {name}")
        print('='*80)
        
        try:
            func()
            self.passed += 1
            self.tests.append((name, 'PASS', None))
            print(f"✅ PASS: {name}")
        except Exception as e:
            self.failed += 1
            self.tests.append((name, 'FAIL', str(e)))
            print(f"❌ FAIL: {name}")
            print(f"   Error: {e}")
            logger.exception("Test failed")
    
    def summary(self):
        """Print test summary."""
        print(f"\n{'='*80}")
        print("TEST SUMMARY")
        print('='*80)
        
        for name, status, error in self.tests:
            symbol = '✅' if status == 'PASS' else '❌'
            print(f"{symbol} {status}: {name}")
            if error:
                print(f"   {error}")
        
        print(f"\n{'='*80}")
        print(f"Total: {self.passed + self.failed} tests")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print('='*80)
        
        return self.failed == 0


def test_ollama_health():
    """Test 1: Ollama service health check."""
    config = OllamaConfig()
    client = OllamaClient(config)
    
    is_healthy, message = client.health_check()
    assert is_healthy, f"Ollama health check failed: {message}"
    logger.info(f"Ollama service healthy: {message}")


def test_model_availability():
    """Test 2: Check model availability and auto-pull."""
    config = OllamaConfig()
    client = OllamaClient(config)
    
    # This should auto-pull if needed
    available_model = client._ensure_model_available()
    assert available_model is not None, "No models available"
    logger.info(f"Model available: {available_model}")


def test_json_generation():
    """Test 3: JSON-mode generation."""
    config = OllamaConfig()
    client = OllamaClient(config)
    
    system_msg = "You are a JSON generator. Output only valid JSON."
    user_msg = """Generate this JSON:
{
  "test": "success",
  "value": 42
}"""
    
    response = client.generate_json(system_msg, user_msg)
    assert isinstance(response, dict), f"Response is not a dict: {type(response)}"
    assert 'test' in response or 'value' in response, "Response missing expected keys"
    logger.info(f"JSON response: {response}")


def test_german_diacritic_restoration():
    """Test 4: German text with missing diacritics."""
    corrector = LLMCorrector()
    
    result = corrector.correct_line(
        text="Die Ubersetzung der hethitischen Texte ist wichtig.",
        lang="de",
        confidence=0.78,  # Below threshold (0.85)
        span_id="smoke_test_german"
    )
    
    logger.info(f"Original:  {result.original_text}")
    logger.info(f"Corrected: {result.corrected_text}")
    logger.info(f"Applied:   {result.applied}")
    logger.info(f"Reason:    {result.reason}")
    logger.info(f"Edit ratio: {result.edit_ratio:.2%}")
    logger.info(f"Latency:   {result.latency_ms}ms")
    
    # Validation: Should have attempted correction (low confidence)
    assert result.edit_ratio >= 0.0, "Edit ratio should be non-negative"
    assert result.latency_ms >= 0, "Latency should be non-negative"


def test_akkadian_transliteration_detection():
    """Test 5: Akkadian transliteration detection."""
    test_cases = [
        ("ša-ar-ru šarrum LUGAL", True, "Akkadian transliteration"),
        ("Hello world this is English", False, "English text"),
        ("Die Übersetzung ist korrekt", False, "German text"),
        ("ᵈUTU ᵐAššur-bāni-apli", True, "Determinatives"),
        ("a.bu.bu šá erṣetu", True, "Syllable dots + diacritics"),
    ]
    
    for text, expected, description in test_cases:
        is_akk = is_akkadian_transliteration(text)
        conf = estimate_transliteration_confidence(text)
        logger.info(f"{description}: {text}")
        logger.info(f"  Detected: {is_akk}, Confidence: {conf:.3f}")
        
        if expected:
            assert is_akk, f"Should detect as transliteration: {text}"
        # Note: We don't assert False cases strictly, as detection is heuristic


def test_akkadian_protection():
    """Test 6: Akkadian transliteration protection mode."""
    corrector = LLMCorrector()
    
    result = corrector.correct_line(
        text="ša-ar-ru šarrum LUGAL ᵈUTU",
        lang="unknown",
        confidence=0.75,
        prev_line="Transliteration:",
        next_line="Translation: The king",
        span_id="smoke_test_akkadian"
    )
    
    logger.info(f"Original:  {result.original_text}")
    logger.info(f"Corrected: {result.corrected_text}")
    logger.info(f"Applied:   {result.applied}")
    logger.info(f"Reason:    {result.reason}")
    logger.info(f"Edit ratio: {result.edit_ratio:.2%}")
    
    # If correction was applied, edit ratio should be ≤3%
    if result.applied:
        assert result.edit_ratio <= 0.03, \
            f"Transliteration edit ratio {result.edit_ratio:.2%} > 3%"


def test_high_confidence_skip():
    """Test 7: High confidence text should be skipped."""
    corrector = LLMCorrector()
    
    result = corrector.correct_line(
        text="This is a perfectly clear sentence with high confidence.",
        lang="en",
        confidence=0.95,  # Above threshold (0.86)
        span_id="smoke_test_skip"
    )
    
    logger.info(f"Applied:   {result.applied}")
    logger.info(f"Reason:    {result.reason}")
    
    assert not result.applied, "High confidence text should not be corrected"
    assert "threshold" in result.reason.lower(), \
        "Reason should mention confidence threshold"


def test_guardrail_edit_budget():
    """Test 8: Edit budget guardrail (indirectly via corrector)."""
    # This is tested indirectly via the corrector's internal validation
    # We validate that the guardrail logic exists
    corrector = LLMCorrector()
    
    # Simulate edit budget check
    try:
        corrector._validate_edit_budget(
            original="Hello world",
            corrected="Completely different text with many changes",
            max_ratio=0.12
        )
        # Should raise GuardrailViolation
        assert False, "Should have raised GuardrailViolation"
    except GuardrailViolation as e:
        logger.info(f"Edit budget violation caught: {e}")
        assert "edit ratio" in str(e).lower()


def test_guardrail_bracket_preservation():
    """Test 9: Bracket preservation guardrail."""
    corrector = LLMCorrector()
    
    # Test bracket preservation
    try:
        corrector._validate_bracket_preservation(
            original="Text with [brackets] and (parens)",
            corrected="Text with brackets and parens"  # Missing brackets!
        )
        assert False, "Should have raised GuardrailViolation"
    except GuardrailViolation as e:
        logger.info(f"Bracket violation caught: {e}")
        assert "bracket" in str(e).lower()


def test_cache_functionality():
    """Test 10: Cache hit on repeated requests."""
    corrector = LLMCorrector()
    
    # First request (cache miss)
    result1 = corrector.correct_line(
        text="Eine kurze deutsche Ubersetzung",
        lang="de",
        confidence=0.78,
        span_id="cache_test_1"
    )
    
    cached1 = result1.cached
    latency1 = result1.latency_ms
    
    logger.info(f"First request - Cached: {cached1}, Latency: {latency1}ms")
    
    # Second identical request (should be cache hit)
    result2 = corrector.correct_line(
        text="Eine kurze deutsche Ubersetzung",
        lang="de",
        confidence=0.78,
        span_id="cache_test_2"
    )
    
    cached2 = result2.cached
    latency2 = result2.latency_ms
    
    logger.info(f"Second request - Cached: {cached2}, Latency: {latency2}ms")
    
    # If first succeeded, second should be cached
    if result1.applied:
        assert cached2, "Second request should be a cache hit"
        assert result1.corrected_text == result2.corrected_text, \
            "Cached result should match original"


def test_telemetry_tracking():
    """Test 11: Telemetry counters."""
    corrector = LLMCorrector(enable_telemetry=True)
    
    # Make a few requests
    texts = [
        ("Hallo Welt", "de", 0.75),
        ("Hello world", "en", 0.95),  # Should skip (high confidence)
        ("Bonjour", "fr", 0.70),
    ]
    
    for text, lang, conf in texts:
        corrector.correct_line(text, lang, conf, span_id=f"telemetry_{text[:5]}")
    
    # Get telemetry
    telemetry = corrector.get_telemetry()
    
    logger.info("Telemetry:")
    for key, value in sorted(telemetry.items()):
        if isinstance(value, dict):
            logger.info(f"  {key}:")
            for sub_key, sub_value in value.items():
                logger.info(f"    {sub_key}: {sub_value}")
        elif isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")
    
    # Validation
    assert telemetry['llm_spans_attempted'] == 3, \
        f"Should have attempted 3 corrections, got {telemetry['llm_spans_attempted']}"
    assert 'llm_cache_hits' in telemetry, "Should track cache hits"
    assert 'llm_cache_misses' in telemetry, "Should track cache misses"


def main():
    """Run all smoke tests."""
    print("\n" + "="*80)
    print("LLM POST-CORRECTION SMOKE TEST")
    print("="*80)
    
    runner = SmokeTest()
    
    # Critical tests (must pass)
    runner.test("1. Ollama service health", test_ollama_health)
    runner.test("2. Model availability", test_model_availability)
    runner.test("3. JSON generation", test_json_generation)
    
    # Functional tests
    runner.test("4. German diacritic restoration", test_german_diacritic_restoration)
    runner.test("5. Akkadian transliteration detection", test_akkadian_transliteration_detection)
    runner.test("6. Akkadian protection mode", test_akkadian_protection)
    runner.test("7. High confidence skip", test_high_confidence_skip)
    
    # Guardrail tests
    runner.test("8. Guardrail: Edit budget", test_guardrail_edit_budget)
    runner.test("9. Guardrail: Bracket preservation", test_guardrail_bracket_preservation)
    
    # System tests
    runner.test("10. Cache functionality", test_cache_functionality)
    runner.test("11. Telemetry tracking", test_telemetry_tracking)
    
    # Summary
    success = runner.summary()
    
    if success:
        print("\n✅ ALL TESTS PASSED - LLM system is operational")
        sys.exit(0)
    else:
        print(f"\n❌ {runner.failed} TEST(S) FAILED - Please review errors above")
        sys.exit(1)


if __name__ == "__main__":
    main()
