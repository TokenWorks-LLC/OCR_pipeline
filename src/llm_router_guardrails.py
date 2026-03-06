#!/usr/bin/env python3
"""
Guardrailed LLM router for Akkadian-safe OCR correction.

Key features:
- Routes only non-Akkadian low-confidence blocks to LLM
- Batches by language (never mixes Akkadian with modern text)
- Validates edit budgets: 3% for Akkadian, 12% for modern languages
- Preserves diacritics, determinatives, brackets, numerals, superscripts
- Retry with lower temperature on validation failure
- Content-addressed caching
"""

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from grapheme_metrics import compute_cer_wer

from blockification import TextBlock, TextLine
from llm_cache import LLMCache

logger = logging.getLogger(__name__)


# Akkadian-specific characters that must be preserved
AKKADIAN_DIACRITICS = {'š', 'ṣ', 'ṭ', 'ḫ', 'ā', 'ē', 'ī', 'ū', 'Š', 'Ṣ', 'Ṭ', 'Ḫ', 'Ā', 'Ē', 'Ī', 'Ū'}
DETERMINATIVES = {'ᵈ', 'ᵐ', 'ᶠ'}  # Superscript determinatives
SPECIAL_MARKERS = {'[', ']', '(', ')', '⟨', '⟩', '«', '»'}  # Scholarly brackets


@dataclass
class CorrectionResult:
    """Result of LLM correction for a block."""
    block_id: str
    original_text: str
    corrected_text: str
    edit_ratio: float
    validation_passed: bool
    validation_errors: List[str]
    cached: bool
    latency_ms: int
    temperature: float


class GuardrailValidator:
    """
    Validates LLM corrections against Akkadian preservation rules.
    
    Checks:
    - Edit budget caps (3% Akkadian, 12% modern)
    - Line count preservation
    - Line order preservation
    - Diacritic preservation
    - Determinative preservation
    - Bracket/superscript preservation
    """
    
    def __init__(
        self,
        edit_budget_akkadian: float = 0.03,
        edit_budget_non_akk: float = 0.12
    ):
        """
        Initialize validator.
        
        Args:
            edit_budget_akkadian: Max edit ratio for Akkadian (default 3%)
            edit_budget_non_akk: Max edit ratio for non-Akkadian (default 12%)
        """
        self.edit_budget_akkadian = edit_budget_akkadian
        self.edit_budget_non_akk = edit_budget_non_akk
    
    def validate(
        self,
        original: str,
        corrected: str,
        is_akkadian: bool
    ) -> Tuple[bool, List[str]]:
        """
        Validate corrected text against guardrails.
        
        Args:
            original: Original text
            corrected: LLM-corrected text
            is_akkadian: Whether text is Akkadian transliteration
            
        Returns:
            (is_valid, list_of_errors)
        """
        errors = []
        
        # 1. Check line count
        orig_lines = original.split('\n')
        corr_lines = corrected.split('\n')
        
        if len(orig_lines) != len(corr_lines):
            errors.append(
                f"Line count mismatch: {len(orig_lines)} → {len(corr_lines)}"
            )
        
        # 2. Check edit budget per line
        budget = self.edit_budget_akkadian if is_akkadian else self.edit_budget_non_akk
        
        for i, (orig_line, corr_line) in enumerate(zip(orig_lines, corr_lines)):
            if not orig_line:  # Skip empty lines
                continue
            
            metrics = compute_cer_wer(orig_line, corr_line)
            cer = metrics['cer']
            
            if cer > budget:
                errors.append(
                    f"Line {i+1} exceeds edit budget: "
                    f"CER={cer:.3f} > {budget:.3f} "
                    f"('{orig_line[:30]}...' → '{corr_line[:30]}...')"
                )
        
        # 3. Check diacritic preservation (critical for Akkadian)
        if is_akkadian or self._has_akkadian_chars(original):
            diacritic_errors = self._check_diacritics(original, corrected)
            errors.extend(diacritic_errors)
        
        # 4. Check determinative preservation
        det_errors = self._check_determinatives(original, corrected)
        errors.extend(det_errors)
        
        # 5. Check bracket/parenthesis preservation
        bracket_errors = self._check_brackets(original, corrected)
        errors.extend(bracket_errors)
        
        # 6. Check numeral preservation
        numeral_errors = self._check_numerals(original, corrected)
        errors.extend(numeral_errors)
        
        # 7. Check superscript preservation
        super_errors = self._check_superscripts(original, corrected)
        errors.extend(super_errors)
        
        is_valid = len(errors) == 0
        
        return is_valid, errors
    
    def _has_akkadian_chars(self, text: str) -> bool:
        """Check if text contains Akkadian diacritics."""
        return any(c in text for c in AKKADIAN_DIACRITICS)
    
    def _check_diacritics(self, original: str, corrected: str) -> List[str]:
        """Check that Akkadian diacritics are preserved."""
        errors = []
        
        # Count diacritics in original
        orig_diacritics = {c: original.count(c) for c in AKKADIAN_DIACRITICS if c in original}
        
        # Count diacritics in corrected
        corr_diacritics = {c: corrected.count(c) for c in AKKADIAN_DIACRITICS if c in corrected}
        
        # Check for missing or extra diacritics
        for char, orig_count in orig_diacritics.items():
            corr_count = corr_diacritics.get(char, 0)
            if corr_count != orig_count:
                errors.append(
                    f"Diacritic '{char}' count changed: {orig_count} → {corr_count}"
                )
        
        # Check for introduced diacritics
        for char, corr_count in corr_diacritics.items():
            if char not in orig_diacritics:
                errors.append(
                    f"Diacritic '{char}' introduced ({corr_count} occurrences)"
                )
        
        return errors
    
    def _check_determinatives(self, original: str, corrected: str) -> List[str]:
        """Check that determinatives are preserved."""
        errors = []
        
        for det in DETERMINATIVES:
            orig_count = original.count(det)
            corr_count = corrected.count(det)
            
            if orig_count != corr_count:
                errors.append(
                    f"Determinative '{det}' count changed: {orig_count} → {corr_count}"
                )
        
        return errors
    
    def _check_brackets(self, original: str, corrected: str) -> List[str]:
        """Check that brackets and parentheses are preserved."""
        errors = []
        
        for bracket in SPECIAL_MARKERS:
            orig_count = original.count(bracket)
            corr_count = corrected.count(bracket)
            
            if orig_count != corr_count:
                errors.append(
                    f"Bracket '{bracket}' count changed: {orig_count} → {corr_count}"
                )
        
        return errors
    
    def _check_numerals(self, original: str, corrected: str) -> List[str]:
        """Check that numerals are preserved."""
        errors = []
        
        # Extract all digit sequences
        orig_numbers = re.findall(r'\d+', original)
        corr_numbers = re.findall(r'\d+', corrected)
        
        if len(orig_numbers) != len(corr_numbers):
            errors.append(
                f"Numeral count changed: {len(orig_numbers)} → {len(corr_numbers)}"
            )
        else:
            # Check if numbers match
            for i, (orig_num, corr_num) in enumerate(zip(orig_numbers, corr_numbers)):
                if orig_num != corr_num:
                    errors.append(
                        f"Numeral changed: '{orig_num}' → '{corr_num}'"
                    )
        
        return errors
    
    def _check_superscripts(self, original: str, corrected: str) -> List[str]:
        """Check that superscripts are preserved."""
        errors = []
        
        # Superscript pattern (Unicode superscript block)
        superscript_pattern = r'[\u00B0-\u00B9\u00B2\u00B3\u00B9\u2070-\u209F\u1D2C-\u1D61]'
        
        orig_supers = re.findall(superscript_pattern, original)
        corr_supers = re.findall(superscript_pattern, corrected)
        
        if len(orig_supers) != len(corr_supers):
            errors.append(
                f"Superscript count changed: {len(orig_supers)} → {len(corr_supers)}"
            )
        
        return errors


class LLMRouter:
    """
    Routes text blocks to LLM for correction with Akkadian-aware guardrails.
    
    Workflow:
    1. Filter blocks: only non-Akkadian AND low-confidence
    2. Batch by language (never mix Akkadian with modern text)
    3. Send to LLM with strict schema
    4. Validate corrections with guardrails
    5. Retry once with lower temperature on failure
    6. Cache all results
    """
    
    def __init__(
        self,
        model: str = "qwen2.5:7b-instruct",
        base_url: str = "http://localhost:11434/api/generate",
        confidence_threshold: float = 0.55,
        edit_budget_akkadian: float = 0.03,
        edit_budget_non_akk: float = 0.12,
        temperature: float = 0.1,
        cache_dir: str = "cache/llm",
        enabled: bool = True
    ):
        """
        Initialize LLM router.
        
        Args:
            model: LLM model name
            base_url: LLM API base URL
            confidence_threshold: Only correct blocks below this confidence
            edit_budget_akkadian: Max edit ratio for Akkadian (default 3%)
            edit_budget_non_akk: Max edit ratio for non-Akkadian (default 12%)
            temperature: LLM temperature (default 0.1 for conservative edits)
            cache_dir: Cache directory
            enabled: Whether LLM correction is enabled
        """
        self.model = model
        self.base_url = base_url
        self.confidence_threshold = confidence_threshold
        self.temperature = temperature
        self.enabled = enabled
        
        # Initialize validator
        self.validator = GuardrailValidator(
            edit_budget_akkadian=edit_budget_akkadian,
            edit_budget_non_akk=edit_budget_non_akk
        )
        
        # Initialize cache
        self.cache = LLMCache(cache_dir=cache_dir)
        
        # Statistics
        self.stats = {
            'blocks_total': 0,
            'blocks_akkadian_skipped': 0,
            'blocks_high_conf_skipped': 0,
            'blocks_sent_to_llm': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'validation_passes': 0,
            'validation_failures': 0,
            'retries': 0
        }
        
        logger.info(f"LLMRouter initialized (model={model}, enabled={enabled})")
    
    def route_blocks(
        self,
        blocks: List[TextBlock]
    ) -> List[Tuple[TextBlock, Optional[CorrectionResult]]]:
        """
        Route blocks through LLM correction pipeline.
        
        Args:
            blocks: List of text blocks from blockification
            
        Returns:
            List of (block, correction_result) tuples
            correction_result is None if block was skipped
        """
        if not self.enabled:
            logger.info("LLM correction disabled, skipping all blocks")
            return [(block, None) for block in blocks]
        
        results = []
        
        # Group blocks by language for batching
        lang_batches = self._group_by_language(blocks)
        
        for lang, lang_blocks in lang_batches.items():
            logger.info(f"Processing {len(lang_blocks)} blocks for language: {lang}")
            
            for block in lang_blocks:
                self.stats['blocks_total'] += 1
                
                # Filter: skip Akkadian blocks
                if block.is_akk:
                    self.stats['blocks_akkadian_skipped'] += 1
                    logger.debug(f"Skipping Akkadian block: {block.block_id}")
                    results.append((block, None))
                    continue
                
                # Filter: skip high-confidence blocks
                if block.mean_conf >= self.confidence_threshold:
                    self.stats['blocks_high_conf_skipped'] += 1
                    logger.debug(
                        f"Skipping high-confidence block: {block.block_id} "
                        f"(conf={block.mean_conf:.2f})"
                    )
                    results.append((block, None))
                    continue
                
                # Send to LLM
                correction = self._correct_block(block)
                results.append((block, correction))
        
        return results
    
    def _group_by_language(self, blocks: List[TextBlock]) -> Dict[str, List[TextBlock]]:
        """Group blocks by detected language."""
        lang_groups = {}
        
        for block in blocks:
            lang = block.lang
            if lang not in lang_groups:
                lang_groups[lang] = []
            lang_groups[lang].append(block)
        
        return lang_groups
    
    def _correct_block(self, block: TextBlock) -> CorrectionResult:
        """
        Correct a single block with LLM.
        
        Workflow:
        1. Check cache
        2. If miss, send to LLM
        3. Validate response
        4. If validation fails, retry with lower temperature
        5. Cache result
        
        Args:
            block: Text block to correct
            
        Returns:
            CorrectionResult
        """
        self.stats['blocks_sent_to_llm'] += 1
        
        # Check cache
        cache_key = self._generate_cache_key(block)
        cached = self.cache.get(cache_key)
        
        if cached:
            self.stats['cache_hits'] += 1
            logger.debug(f"Cache hit for block: {block.block_id}")
            
            return CorrectionResult(
                block_id=block.block_id,
                original_text=block.text,
                corrected_text=cached['corrected_text'],
                edit_ratio=cached['edit_ratio'],
                validation_passed=True,
                validation_errors=[],
                cached=True,
                latency_ms=0,
                temperature=cached.get('temperature', self.temperature)
            )
        
        self.stats['cache_misses'] += 1
        
        # Try correction with current temperature
        result = self._llm_correct(block, self.temperature)
        
        # If validation failed, retry with lower temperature
        if not result.validation_passed:
            logger.warning(
                f"Validation failed for {block.block_id}, retrying with lower temperature"
            )
            self.stats['retries'] += 1
            result = self._llm_correct(block, self.temperature * 0.5)
        
        # Cache result (even if validation failed - we won't retry again)
        if result.validation_passed:
            self.stats['validation_passes'] += 1
            self.cache.put(cache_key, {
                'corrected_text': result.corrected_text,
                'edit_ratio': result.edit_ratio,
                'temperature': result.temperature,
                'model': self.model
            })
        else:
            self.stats['validation_failures'] += 1
            logger.error(
                f"Validation failed after retry for {block.block_id}: "
                f"{result.validation_errors}"
            )
        
        return result
    
    def _llm_correct(self, block: TextBlock, temperature: float) -> CorrectionResult:
        """
        Call LLM to correct text.
        
        Args:
            block: Text block to correct
            temperature: LLM temperature
            
        Returns:
            CorrectionResult
        """
        start_time = time.time()
        
        # Mock LLM call for now (in production, this would call actual LLM API)
        # For testing purposes, return minimal edits
        corrected_text = block.text  # No changes in mock mode
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Calculate edit ratio
        metrics = compute_cer_wer(block.text, corrected_text)
        edit_ratio = metrics['cer']
        
        # Validate
        is_valid, errors = self.validator.validate(
            original=block.text,
            corrected=corrected_text,
            is_akkadian=block.is_akk
        )
        
        return CorrectionResult(
            block_id=block.block_id,
            original_text=block.text,
            corrected_text=corrected_text,
            edit_ratio=edit_ratio,
            validation_passed=is_valid,
            validation_errors=errors,
            cached=False,
            latency_ms=latency_ms,
            temperature=temperature
        )
    
    def _generate_cache_key(self, block: TextBlock) -> str:
        """
        Generate content-addressed cache key.
        
        Key: sha1(model|prompt_template|normalized_text|language)
        
        Args:
            block: Text block
            
        Returns:
            SHA-1 hash as cache key
        """
        # Normalize text
        normalized_text = block.text.strip().lower()
        
        # Create key components
        key_data = {
            'model': self.model,
            'template_version': 'fix-typos-v1',
            'text': normalized_text,
            'language': block.lang,
            'temperature': self.temperature
        }
        
        # Generate SHA-1 hash
        key_str = json.dumps(key_data, sort_keys=True, separators=(',', ':'))
        cache_key = hashlib.sha1(key_str.encode('utf-8')).hexdigest()
        
        return cache_key
    
    def get_statistics(self) -> Dict:
        """Get routing statistics."""
        stats = dict(self.stats)
        
        # Calculate rates
        if stats['blocks_total'] > 0:
            stats['akkadian_skip_rate'] = stats['blocks_akkadian_skipped'] / stats['blocks_total']
            stats['high_conf_skip_rate'] = stats['blocks_high_conf_skipped'] / stats['blocks_total']
            stats['llm_send_rate'] = stats['blocks_sent_to_llm'] / stats['blocks_total']
        
        cache_ops = stats['cache_hits'] + stats['cache_misses']
        if cache_ops > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / cache_ops
        
        if stats['blocks_sent_to_llm'] > 0:
            stats['validation_pass_rate'] = stats['validation_passes'] / stats['blocks_sent_to_llm']
        
        return stats


if __name__ == '__main__':
    # Test guardrails
    validator = GuardrailValidator()
    
    # Test 1: Diacritic preservation
    original = "šarru ina māti erṣetu"
    corrupted = "sarru ina mati erṣetu"  # Missing diacritics
    
    is_valid, errors = validator.validate(original, corrupted, is_akkadian=True)
    print(f"\nTest 1 (diacritic corruption):")
    print(f"  Valid: {is_valid}")
    print(f"  Errors: {errors}")
    
    # Test 2: Edit budget
    original = "The king ruled the land"
    heavily_edited = "A monarch governed this territory"  # >12% edits
    
    is_valid, errors = validator.validate(original, heavily_edited, is_akkadian=False)
    print(f"\nTest 2 (excessive edits):")
    print(f"  Valid: {is_valid}")
    print(f"  Errors: {errors}")
    
    # Test 3: Valid minor correction
    original = "The kinng ruled the land"
    corrected = "The king ruled the land"  # Fixed typo
    
    is_valid, errors = validator.validate(original, corrected, is_akkadian=False)
    print(f"\nTest 3 (valid typo fix):")
    print(f"  Valid: {is_valid}")
    print(f"  Errors: {errors}")
