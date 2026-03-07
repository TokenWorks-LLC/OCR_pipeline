#!/usr/bin/env python3
"""
LLM Correction Smoke Test Tool (Prompt 3)

Validates Akkadian-safe LLM correction against acceptance criteria:
1. Akkadian corruption rate < 1% (diacritics/determinatives preserved)
2. Non-Akkadian WER improvement ≥ 10% (LLM on vs off)
3. LLM cache hit rate > 80% on rerun

Usage:
    # Basic smoke test with sample data
    python tools/run_llm_smoke.py --sample-pages 5

    # Production test with gold data
    python tools/run_llm_smoke.py \\
        --manifest data/gold_validation.txt \\
        --output results/llm_smoke_results.csv
        
    # Test cache effectiveness
    python tools/run_llm_smoke.py --sample-pages 10 --run-twice

Exit codes:
    0 - All acceptance criteria met
    1 - One or more criteria failed
"""

import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Tuple
import re

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from blockification import TextBlockifier, TextBlock
from llm_router_guardrails import LLMRouter, GuardrailValidator
from grapheme_metrics import compute_cer_wer


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Akkadian special characters for corruption detection
AKKADIAN_DIACRITICS = {'š', 'ṣ', 'ṭ', 'ḫ', 'ā', 'ē', 'ī', 'ū'}
DETERMINATIVES = {'ᵈ', 'ᵐ', 'ᶠ'}
ALL_AKKADIAN_CHARS = AKKADIAN_DIACRITICS | DETERMINATIVES


@dataclass
class PageMetrics:
    """Metrics for a single page."""
    page_id: str
    is_akkadian_page: bool
    
    # Corruption metrics (Akkadian pages only)
    special_char_count_original: int
    special_char_count_corrected: int
    special_chars_lost: int
    corruption_rate: float
    
    # WER metrics (non-Akkadian pages only)
    wer_before_llm: float
    wer_after_llm: float
    wer_improvement: float
    
    # Cache metrics
    blocks_processed: int
    blocks_from_cache: int
    cache_hit_rate: float
    
    # Validation
    guardrail_violations: int


@dataclass
class SmokeTestResults:
    """Overall smoke test results."""
    # Summary
    total_pages: int
    akkadian_pages: int
    non_akkadian_pages: int
    
    # Acceptance criteria results
    avg_corruption_rate: float
    corruption_pass: bool  # < 1%
    
    avg_wer_improvement: float
    wer_improvement_pass: bool  # ≥ 10%
    
    cache_hit_rate: float
    cache_pass: bool  # > 80%
    
    # Details
    page_metrics: List[PageMetrics]
    
    # Overall pass/fail
    all_criteria_met: bool


class LLMSmokeTest:
    """
    Smoke test for LLM correction system.
    """
    
    def __init__(
        self,
        model: str = "qwen2.5:7b-instruct",
        temperature: float = 0.3,
        enabled: bool = True
    ):
        """Initialize smoke test."""
        self.model = model
        self.temperature = temperature
        self.enabled = enabled
        
        # Initialize components
        self.router = LLMRouter(
            model=model,
            temperature=temperature,
            edit_budget_akkadian=0.03,
            edit_budget_non_akk=0.12,
            enabled=enabled
        )
        
        self.blockifier = TextBlockifier()
    
    def run_smoke_test(
        self,
        pages: List[Tuple[str, str, str]],  # (page_id, fusion_text, gold_text)
        run_twice: bool = False
    ) -> SmokeTestResults:
        """
        Run smoke test on pages.
        
        Args:
            pages: List of (page_id, fusion_text, gold_text) tuples
            run_twice: Run twice to test cache effectiveness
            
        Returns:
            SmokeTestResults with metrics and pass/fail status
        """
        logger.info(f"Starting smoke test on {len(pages)} pages...")
        
        # First pass
        page_metrics_pass1 = self._process_pages(pages, pass_num=1)
        
        # Second pass (if requested)
        page_metrics_pass2 = None
        if run_twice:
            logger.info("Running second pass to test cache...")
            page_metrics_pass2 = self._process_pages(pages, pass_num=2)
        
        # Use second pass metrics if available, otherwise first pass
        page_metrics = page_metrics_pass2 if page_metrics_pass2 else page_metrics_pass1
        
        # Compute aggregate metrics
        results = self._compute_results(page_metrics)
        
        return results
    
    def _process_pages(
        self,
        pages: List[Tuple[str, str, str]],
        pass_num: int
    ) -> List[PageMetrics]:
        """Process all pages and return metrics."""
        metrics = []
        
        for page_id, fusion_text, gold_text in pages:
            page_metric = self._process_page(page_id, fusion_text, gold_text)
            metrics.append(page_metric)
            
            logger.info(
                f"[Pass {pass_num}] Page {page_id}: "
                f"akk={page_metric.is_akkadian_page}, "
                f"corr={page_metric.corruption_rate:.3f}, "
                f"Δwer={page_metric.wer_improvement:.3f}, "
                f"cache={page_metric.cache_hit_rate:.1%}"
            )
        
        return metrics
    
    def _process_page(
        self,
        page_id: str,
        fusion_text: str,
        gold_text: str
    ) -> PageMetrics:
        """Process a single page and compute metrics."""
        
        # Create blocks from fusion text
        # Simplified: treat each line as a block with mock bbox
        lines_fusion = fusion_text.strip().split('\n')
        lines_gold = gold_text.strip().split('\n')
        
        # Mock blocks (in real pipeline, blockifier would create these)
        blocks = self._create_mock_blocks(lines_fusion, page_id)
        
        # Detect if page is primarily Akkadian
        is_akkadian_page = self._is_akkadian_page(blocks)
        
        # Get router statistics before
        stats_before = self.router.get_statistics().copy()
        
        # Run LLM correction
        results = self.router.route_blocks(blocks)
        
        # Extract corrected blocks (use corrected text if available, else original)
        corrected_blocks = []
        for block, correction in results:
            if correction and correction.validation_passed:
                # Create updated block with corrected text
                corrected_block = TextBlock(
                    block_id=block.block_id,
                    page=block.page,
                    bbox=block.bbox,
                    text=correction.corrected_text,
                    mean_conf=block.mean_conf,
                    lines=block.lines,
                    lang=block.lang,
                    is_akk=block.is_akk,
                    akk_conf=block.akk_conf
                )
                corrected_blocks.append(corrected_block)
            else:
                corrected_blocks.append(block)
        
        # Get router statistics after
        stats_after = self.router.get_statistics().copy()
        
        # Compute metrics
        if is_akkadian_page:
            # Measure corruption
            corruption_rate = self._compute_corruption_rate(blocks, corrected_blocks)
            wer_before = wer_after = wer_improvement = 0.0
        else:
            # Measure WER improvement
            wer_before = self._compute_wer_vs_gold(blocks, lines_gold)
            wer_after = self._compute_wer_vs_gold(corrected_blocks, lines_gold)
            wer_improvement = wer_before - wer_after  # Positive = improvement
            corruption_rate = 0.0
        
        # Compute cache metrics for this page
        blocks_total_delta = stats_after['blocks_total'] - stats_before['blocks_total']
        cache_hits_delta = stats_after['cache_hits'] - stats_before['cache_hits']
        cache_hit_rate = (
            cache_hits_delta / blocks_total_delta if blocks_total_delta > 0 else 0.0
        )
        
        # Count special char corruption
        special_orig, special_corr, special_lost = self._count_special_chars(
            blocks, corrected_blocks
        )
        
        # Count violations
        violations = stats_after['validation_failures'] - stats_before['validation_failures']
        
        return PageMetrics(
            page_id=page_id,
            is_akkadian_page=is_akkadian_page,
            special_char_count_original=special_orig,
            special_char_count_corrected=special_corr,
            special_chars_lost=special_lost,
            corruption_rate=corruption_rate,
            wer_before_llm=wer_before,
            wer_after_llm=wer_after,
            wer_improvement=wer_improvement,
            blocks_processed=blocks_total_delta,
            blocks_from_cache=cache_hits_delta,
            cache_hit_rate=cache_hit_rate,
            guardrail_violations=violations
        )
    
    def _create_mock_blocks(
        self,
        lines: List[str],
        page_id: str
    ) -> List[TextBlock]:
        """Create mock blocks from lines (simplified for smoke test)."""
        from blockification import TextLine, TextBlock
        
        blocks = []
        for i, line_text in enumerate(lines):
            if not line_text.strip():
                continue
            
            text_line = TextLine(
                text=line_text,
                bbox=(0, i*20, 500, i*20+15),
                confidence=0.65,  # Low confidence to trigger LLM
                line_id=f"{page_id}_line_{i}"
            )
            
            # Detect language (simplified)
            is_akk = any(c in line_text for c in AKKADIAN_DIACRITICS | DETERMINATIVES)
            lang = "tr" if is_akk else "en"
            
            block = TextBlock(
                block_id=f"{page_id}_block_{i}",
                page=1,
                bbox=(0, i*20, 500, i*20+15),
                text=line_text,
                mean_conf=0.65,
                lines=[text_line],
                lang=lang,
                is_akk=is_akk,
                akk_conf=0.9 if is_akk else 0.1
            )
            blocks.append(block)
        
        return blocks
    
    def _is_akkadian_page(self, blocks: List[TextBlock]) -> bool:
        """Determine if page is primarily Akkadian."""
        akk_blocks = sum(1 for b in blocks if b.is_akk)
        return akk_blocks > len(blocks) / 2
    
    def _compute_corruption_rate(
        self,
        original_blocks: List[TextBlock],
        corrected_blocks: List[TextBlock]
    ) -> float:
        """
        Compute corruption rate: special chars lost / total special chars.
        """
        special_orig, special_corr, special_lost = self._count_special_chars(
            original_blocks, corrected_blocks
        )
        
        if special_orig == 0:
            return 0.0
        
        return special_lost / special_orig
    
    def _count_special_chars(
        self,
        original_blocks: List[TextBlock],
        corrected_blocks: List[TextBlock]
    ) -> Tuple[int, int, int]:
        """
        Count special characters (diacritics, determinatives).
        
        Returns:
            (original_count, corrected_count, chars_lost)
        """
        orig_text = "\n".join(b.text for b in original_blocks)
        corr_text = "\n".join(b.text for b in corrected_blocks)
        
        orig_counts = {char: orig_text.count(char) for char in ALL_AKKADIAN_CHARS}
        corr_counts = {char: corr_text.count(char) for char in ALL_AKKADIAN_CHARS}
        
        total_orig = sum(orig_counts.values())
        total_corr = sum(corr_counts.values())
        chars_lost = sum(max(0, orig_counts[c] - corr_counts[c]) for c in ALL_AKKADIAN_CHARS)
        
        return total_orig, total_corr, chars_lost
    
    def _compute_wer_vs_gold(
        self,
        blocks: List[TextBlock],
        gold_lines: List[str]
    ) -> float:
        """Compute WER against gold standard."""
        hypothesis = "\n".join(b.text for b in blocks)
        reference = "\n".join(gold_lines)
        
        metrics = compute_cer_wer(reference, hypothesis)
        return metrics['wer']
    
    def _compute_results(self, page_metrics: List[PageMetrics]) -> SmokeTestResults:
        """Compute aggregate results and check acceptance criteria."""
        
        akk_pages = [p for p in page_metrics if p.is_akkadian_page]
        non_akk_pages = [p for p in page_metrics if not p.is_akkadian_page]
        
        # 1. Corruption rate (Akkadian pages)
        if akk_pages:
            avg_corruption = sum(p.corruption_rate for p in akk_pages) / len(akk_pages)
        else:
            avg_corruption = 0.0
        corruption_pass = avg_corruption < 0.01  # < 1%
        
        # 2. WER improvement (non-Akkadian pages)
        if non_akk_pages:
            avg_wer_improvement = sum(p.wer_improvement for p in non_akk_pages) / len(non_akk_pages)
        else:
            avg_wer_improvement = 0.0
        wer_improvement_pass = avg_wer_improvement >= 0.10  # ≥ 10%
        
        # 3. Cache hit rate (all pages)
        total_blocks = sum(p.blocks_processed for p in page_metrics)
        total_cached = sum(p.blocks_from_cache for p in page_metrics)
        cache_hit_rate = total_cached / total_blocks if total_blocks > 0 else 0.0
        cache_pass = cache_hit_rate > 0.80  # > 80%
        
        # Overall pass
        all_pass = corruption_pass and wer_improvement_pass and cache_pass
        
        return SmokeTestResults(
            total_pages=len(page_metrics),
            akkadian_pages=len(akk_pages),
            non_akkadian_pages=len(non_akk_pages),
            avg_corruption_rate=avg_corruption,
            corruption_pass=corruption_pass,
            avg_wer_improvement=avg_wer_improvement,
            wer_improvement_pass=wer_improvement_pass,
            cache_hit_rate=cache_hit_rate,
            cache_pass=cache_pass,
            page_metrics=page_metrics,
            all_criteria_met=all_pass
        )


def generate_sample_data(num_pages: int = 5) -> List[Tuple[str, str, str]]:
    """Generate sample test data."""
    pages = []
    
    # Akkadian pages
    for i in range(num_pages // 2):
        page_id = f"akk_page_{i+1}"
        fusion = f"ᵈUTU šarru ērēbu\nᵐPuzur-Aššur māru\nḫarrānu ṭuppu"
        gold = f"ᵈUTU šarru ērēbu\nᵐPuzur-Aššur māru\nḫarrānu ṭuppu"
        pages.append((page_id, fusion, gold))
    
    # Non-Akkadian pages
    for i in range(num_pages - num_pages // 2):
        page_id = f"modern_page_{i+1}"
        fusion = "The quick brown fox jumps\nover teh lazy dog"  # Typo: "teh"
        gold = "The quick brown fox jumps\nover the lazy dog"
        pages.append((page_id, fusion, gold))
    
    return pages


def print_results(results: SmokeTestResults):
    """Print formatted results."""
    print("\n" + "="*80)
    print("LLM CORRECTION SMOKE TEST RESULTS (Prompt 3)")
    print("="*80)
    print(f"Total pages: {results.total_pages}")
    print(f"  Akkadian pages: {results.akkadian_pages}")
    print(f"  Non-Akkadian pages: {results.non_akkadian_pages}")
    print()
    
    # Acceptance criteria
    print("ACCEPTANCE CRITERIA")
    print("-"*80)
    
    # 1. Corruption
    print(f"1. Akkadian corruption rate: {results.avg_corruption_rate:.2%}")
    print(f"   Threshold: < 1.0%")
    print(f"   Status: {'✅ PASS' if results.corruption_pass else '❌ FAIL'}")
    print()
    
    # 2. WER improvement
    print(f"2. Non-Akkadian WER improvement: {results.avg_wer_improvement:.2%}")
    print(f"   Threshold: ≥ 10.0%")
    print(f"   Status: {'✅ PASS' if results.wer_improvement_pass else '❌ FAIL'}")
    print()
    
    # 3. Cache hit rate
    print(f"3. LLM cache hit rate: {results.cache_hit_rate:.1%}")
    print(f"   Threshold: > 80.0%")
    print(f"   Status: {'✅ PASS' if results.cache_pass else '❌ FAIL'}")
    print()
    
    # Overall
    print("="*80)
    if results.all_criteria_met:
        print("🎉 ALL ACCEPTANCE CRITERIA MET ✅")
    else:
        print("⚠️  SOME ACCEPTANCE CRITERIA FAILED ❌")
    print("="*80)


def save_csv(results: SmokeTestResults, output_path: Path):
    """Save detailed metrics to CSV."""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'page_id', 'is_akkadian', 'corruption_rate', 'wer_before', 
            'wer_after', 'wer_improvement', 'cache_hit_rate', 'violations'
        ])
        writer.writeheader()
        
        for pm in results.page_metrics:
            writer.writerow({
                'page_id': pm.page_id,
                'is_akkadian': pm.is_akkadian_page,
                'corruption_rate': f"{pm.corruption_rate:.4f}",
                'wer_before': f"{pm.wer_before_llm:.4f}",
                'wer_after': f"{pm.wer_after_llm:.4f}",
                'wer_improvement': f"{pm.wer_improvement:.4f}",
                'cache_hit_rate': f"{pm.cache_hit_rate:.4f}",
                'violations': pm.guardrail_violations
            })
    
    logger.info(f"Saved detailed metrics to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="LLM correction smoke test (Prompt 3)"
    )
    parser.add_argument(
        '--sample-pages',
        type=int,
        default=0,
        help="Generate N sample pages for quick test (default: use manifest)"
    )
    parser.add_argument(
        '--manifest',
        type=Path,
        help="Path to test manifest (page_id, fusion_text, gold_text per line)"
    )
    parser.add_argument(
        '--output',
        type=Path,
        help="Output CSV path for detailed metrics"
    )
    parser.add_argument(
        '--run-twice',
        action='store_true',
        help="Run twice to test cache effectiveness"
    )
    parser.add_argument(
        '--model',
        default='qwen2.5:7b-instruct',
        help="LLM model name (default: qwen2.5:7b-instruct)"
    )
    
    args = parser.parse_args()
    
    # Get test data
    if args.sample_pages > 0:
        logger.info(f"Generating {args.sample_pages} sample pages...")
        pages = generate_sample_data(args.sample_pages)
    elif args.manifest:
        logger.info(f"Loading manifest from {args.manifest}...")
        pages = load_manifest(args.manifest)
    else:
        logger.error("Must specify either --sample-pages or --manifest")
        sys.exit(1)
    
    # Run smoke test
    tester = LLMSmokeTest(model=args.model)
    results = tester.run_smoke_test(pages, run_twice=args.run_twice)
    
    # Print results
    print_results(results)
    
    # Save CSV if requested
    if args.output:
        save_csv(results, args.output)
    
    # Exit code
    sys.exit(0 if results.all_criteria_met else 1)


def load_manifest(manifest_path: Path) -> List[Tuple[str, str, str]]:
    """Load test data from manifest file."""
    pages = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                pages.append((parts[0], parts[1], parts[2]))
    return pages


if __name__ == '__main__':
    main()
