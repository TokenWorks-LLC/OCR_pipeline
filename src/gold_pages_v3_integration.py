#!/usr/bin/env python3
"""
Gold Pages V3 Integration - CER/WER delta reporting for LLM-in-the-loop evaluation
Calculates accuracy deltas between LLM-corrected and no-LLM baseline results.
"""

import json
import logging
import time
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import csv
import re

logger = logging.getLogger(__name__)

@dataclass
class AccuracyDelta:
    """CER/WER delta metrics between LLM and no-LLM results."""
    page_id: str
    language: str
    llm_cer: float
    baseline_cer: float
    cer_delta: float
    cer_improvement: float  # Positive = improvement, negative = degradation

    llm_wer: float
    baseline_wer: float
    wer_delta: float
    wer_improvement: float

    llm_processing_time: float
    baseline_processing_time: float
    time_delta: float

    spans_corrected: int
    total_spans: int
    correction_rate: float

@dataclass
class V3EvaluationMetrics:
    """Comprehensive V3 evaluation metrics."""
    total_pages: int
    total_spans: int
    avg_cer_improvement: float
    avg_wer_improvement: float
    avg_time_per_page: float
    avg_correction_rate: float

    # Language-specific metrics
    language_metrics: Dict[str, Dict[str, float]]

    # Error analysis
    errors: int
    failed_pages: List[str]

    # Telemetry summary
    cache_hit_rate: float
    total_llm_calls: int

class GoldPagesV3Integrator:
    """Integrates LLM V3 with gold pages for CER/WER delta analysis."""

    def __init__(self, gold_pages_dir: str, output_dir: str):
        """Initialize V3 gold pages integrator."""
        self.gold_pages_dir = Path(gold_pages_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load gold pages data
        self.gold_pages_data = self._load_gold_pages()
        self.ground_truth = self._extract_ground_truth()

        logger.info(f"Gold Pages V3 integrator initialized with {len(self.gold_pages_data)} gold pages")

    def _load_gold_pages(self) -> Dict[str, Any]:
        """Load gold pages data."""
        gold_pages_file = self.gold_pages_dir / "gold_pages_restructured.json"
        if not gold_pages_file.exists():
            logger.warning(f"Gold pages file not found: {gold_pages_file}")
            return {}

        try:
            with open(gold_pages_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load gold pages: {e}")
            return {}

    def _extract_ground_truth(self) -> Dict[str, Dict[str, str]]:
        """Extract ground truth text for each page."""
        ground_truth = {}

        if not self.gold_pages_data:
            return ground_truth

        for page_id, page_data in self.gold_pages_data.items():
            if 'ground_truth' in page_data:
                ground_truth[page_id] = {
                    'text': page_data['ground_truth'].get('text', ''),
                    'language': page_data['ground_truth'].get('language', 'unknown')
                }
            elif 'text' in page_data:
                # Fallback for older format
                ground_truth[page_id] = {
                    'text': page_data['text'],
                    'language': page_data.get('language', 'unknown')
                }

        return ground_truth

    def calculate_cer_wer(self, ocr_text: str, ground_truth: str) -> Tuple[float, float]:
        """Calculate Character Error Rate and Word Error Rate."""
        def normalize_text(text: str) -> str:
            """Normalize text for comparison."""
            # Remove extra whitespace and normalize
            text = re.sub(r'\s+', ' ', text.strip())
            return text.lower()

        def character_error_rate(ocr: str, gt: str) -> float:
            """Calculate Character Error Rate."""
            ocr_norm = normalize_text(ocr)
            gt_norm = normalize_text(gt)

            if not gt_norm:
                return 0.0 if not ocr_norm else 1.0

            # Simple Levenshtein distance for CER
            return self._levenshtein_distance(ocr_norm, gt_norm) / len(gt_norm)

        def word_error_rate(ocr: str, gt: str) -> float:
            """Calculate Word Error Rate."""
            ocr_words = normalize_text(ocr).split()
            gt_words = normalize_text(gt).split()

            if not gt_words:
                return 0.0 if not ocr_words else 1.0

            # Word-level Levenshtein distance
            return self._word_error_distance(ocr_words, gt_words) / len(gt_words)

        return character_error_rate(ocr_text, ground_truth), word_error_rate(ocr_text, ground_truth)

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def _word_error_distance(self, words1: List[str], words2: List[str]) -> int:
        """Calculate word-level error distance."""
        if not words2:
            return len(words1)

        # Simple approach: count words that are different
        errors = 0
        for w1, w2 in zip(words1, words2):
            if w1 != w2:
                errors += 1

        # Account for different lengths
        errors += abs(len(words1) - len(words2))

        return errors

    def calculate_page_deltas(self, page_id: str, llm_result: Dict[str, Any],
                            baseline_result: Dict[str, Any]) -> Optional[AccuracyDelta]:
        """Calculate CER/WER deltas for a single page."""
        if page_id not in self.ground_truth:
            logger.warning(f"No ground truth found for page {page_id}")
            return None

        ground_truth = self.ground_truth[page_id]
        gt_text = ground_truth['text']
        language = ground_truth['language']

        # Extract text from results
        llm_text = self._extract_text_from_result(llm_result)
        baseline_text = self._extract_text_from_result(baseline_result)

        if not llm_text or not baseline_text:
            logger.warning(f"Missing text data for page {page_id}")
            return None

        # Calculate CER/WER for both versions
        llm_cer, llm_wer = self.calculate_cer_wer(llm_text, gt_text)
        baseline_cer, baseline_wer = self.calculate_cer_wer(baseline_text, gt_text)

        # Calculate deltas (negative = improvement)
        cer_delta = baseline_cer - llm_cer
        wer_delta = baseline_wer - llm_wer

        cer_improvement = max(0, cer_delta)  # Only count improvements
        wer_improvement = max(0, wer_delta)  # Only count improvements

        # Extract processing times
        llm_time = llm_result.get('processing_time', 0.0)
        baseline_time = baseline_result.get('processing_time', 0.0)
        time_delta = llm_time - baseline_time

        # Extract correction stats
        correction_stats = llm_result.get('correction_stats', {})
        spans_corrected = correction_stats.get('spans_corrected', 0)
        total_spans = correction_stats.get('spans_processed', 0)
        correction_rate = spans_corrected / total_spans if total_spans > 0 else 0.0

        return AccuracyDelta(
            page_id=page_id,
            language=language,
            llm_cer=llm_cer,
            baseline_cer=baseline_cer,
            cer_delta=cer_delta,
            cer_improvement=cer_improvement,
            llm_wer=llm_wer,
            baseline_wer=baseline_wer,
            wer_delta=wer_delta,
            wer_improvement=wer_improvement,
            llm_processing_time=llm_time,
            baseline_processing_time=baseline_time,
            time_delta=time_delta,
            spans_corrected=spans_corrected,
            total_spans=total_spans,
            correction_rate=correction_rate
        )

    def _extract_text_from_result(self, result: Dict[str, Any]) -> str:
        """Extract text content from OCR result."""
        if 'text' in result:
            return result['text']

        # Try to reconstruct from elements
        elements = result.get('text_elements', [])
        if elements:
            return ' '.join(elem.get('text', '') for elem in elements)

        return ''

    def calculate_v3_metrics(self, llm_results: Dict[str, Any],
                           baseline_results: Dict[str, Any]) -> V3EvaluationMetrics:
        """Calculate comprehensive V3 evaluation metrics."""
        deltas = []
        errors = 0
        failed_pages = []

        # Calculate deltas for each page
        for page_id in llm_results.keys():
            if page_id in baseline_results:
                try:
                    delta = self.calculate_page_deltas(page_id, llm_results[page_id], baseline_results[page_id])
                    if delta:
                        deltas.append(delta)
                    else:
                        failed_pages.append(page_id)
                except Exception as e:
                    logger.error(f"Error calculating delta for page {page_id}: {e}")
                    errors += 1
                    failed_pages.append(page_id)

        if not deltas:
            logger.warning("No valid deltas calculated")
            return V3EvaluationMetrics(
                total_pages=0,
                total_spans=0,
                avg_cer_improvement=0.0,
                avg_wer_improvement=0.0,
                avg_time_per_page=0.0,
                avg_correction_rate=0.0,
                language_metrics={},
                errors=errors,
                failed_pages=failed_pages,
                cache_hit_rate=0.0,
                total_llm_calls=0
            )

        # Calculate aggregate metrics
        total_pages = len(deltas)
        total_spans = sum(delta.total_spans for delta in deltas)

        avg_cer_improvement = sum(delta.cer_improvement for delta in deltas) / total_pages
        avg_wer_improvement = sum(delta.wer_improvement for delta in deltas) / total_pages
        avg_time_per_page = sum(delta.llm_processing_time for delta in deltas) / total_pages
        avg_correction_rate = sum(delta.correction_rate for delta in deltas) / total_pages

        # Language-specific metrics
        language_metrics = {}
        for delta in deltas:
            lang = delta.language
            if lang not in language_metrics:
                language_metrics[lang] = {
                    'pages': 0,
                    'avg_cer_improvement': 0.0,
                    'avg_wer_improvement': 0.0,
                    'avg_correction_rate': 0.0
                }

            metrics = language_metrics[lang]
            metrics['pages'] += 1
            metrics['avg_cer_improvement'] = (
                (metrics['avg_cer_improvement'] * (metrics['pages'] - 1) + delta.cer_improvement)
                / metrics['pages']
            )
            metrics['avg_wer_improvement'] = (
                (metrics['avg_wer_improvement'] * (metrics['pages'] - 1) + delta.wer_improvement)
                / metrics['pages']
            )
            metrics['avg_correction_rate'] = (
                (metrics['avg_correction_rate'] * (metrics['pages'] - 1) + delta.correction_rate)
                / metrics['pages']
            )

        # Extract telemetry from first LLM result (assuming consistent across pages)
        cache_hit_rate = 0.0
        total_llm_calls = 0
        if llm_results:
            first_result = next(iter(llm_results.values()))
            correction_stats = first_result.get('correction_stats', {})
            cache_hit_rate = correction_stats.get('cache_hit_rate', 0.0)
            total_llm_calls = correction_stats.get('total_llm_calls', 0)

        return V3EvaluationMetrics(
            total_pages=total_pages,
            total_spans=total_spans,
            avg_cer_improvement=avg_cer_improvement,
            avg_wer_improvement=avg_wer_improvement,
            avg_time_per_page=avg_time_per_page,
            avg_correction_rate=avg_correction_rate,
            language_metrics=language_metrics,
            errors=errors,
            failed_pages=failed_pages,
            cache_hit_rate=cache_hit_rate,
            total_llm_calls=total_llm_calls
        )

    def save_deltas_report(self, deltas: List[AccuracyDelta], metrics: V3EvaluationMetrics,
                          output_file: str = None):
        """Save detailed deltas report to CSV and JSON."""
        if not output_file:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"llm_v3_deltas_{timestamp}"

        # Save detailed CSV report
        csv_file = f"{output_file}.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'Page ID', 'Language', 'LLM CER', 'Baseline CER', 'CER Delta', 'CER Improvement',
                'LLM WER', 'Baseline WER', 'WER Delta', 'WER Improvement',
                'LLM Time', 'Baseline Time', 'Time Delta', 'Spans Corrected', 'Total Spans', 'Correction Rate'
            ])

            # Data rows
            for delta in deltas:
                writer.writerow([
                    delta.page_id, delta.language,
                    f"{delta.llm_cer:.4f}", f"{delta.baseline_cer:.4f}", f"{delta.cer_delta:.4f}", f"{delta.cer_improvement:.4f}",
                    f"{delta.llm_wer:.4f}", f"{delta.baseline_wer:.4f}", f"{delta.wer_delta:.4f}", f"{delta.wer_improvement:.4f}",
                    f"{delta.llm_processing_time:.3f}", f"{delta.baseline_processing_time:.3f}", f"{delta.time_delta:.3f}",
                    delta.spans_corrected, delta.total_spans, f"{delta.correction_rate:.3f}"
                ])

        # Save summary JSON report
        summary_data = {
            'evaluation_summary': {
                'total_pages': metrics.total_pages,
                'total_spans': metrics.total_spans,
                'avg_cer_improvement': metrics.avg_cer_improvement,
                'avg_wer_improvement': metrics.avg_wer_improvement,
                'avg_time_per_page': metrics.avg_time_per_page,
                'avg_correction_rate': metrics.avg_correction_rate,
                'cache_hit_rate': metrics.cache_hit_rate,
                'total_llm_calls': metrics.total_llm_calls,
                'errors': metrics.errors,
                'failed_pages': metrics.failed_pages
            },
            'language_metrics': metrics.language_metrics,
            'deltas_count': len(deltas)
        }

        json_file = f"{output_file}_summary.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2)

        logger.info(f"V3 deltas report saved to {csv_file} and {json_file}")

    def print_evaluation_summary(self, metrics: V3EvaluationMetrics):
        """Print evaluation summary to console."""
        print("\n" + "="*80)
        print("LLM V3 EVALUATION SUMMARY - CER/WER DELTAS")
        print("="*80)
        print(f"Total Pages Evaluated: {metrics.total_pages}")
        print(f"Total Spans Processed: {metrics.total_spans}")
        print(f"Average CER Improvement: {metrics.avg_cer_improvement:.4f}")
        print(f"Average WER Improvement: {metrics.avg_wer_improvement:.4f}")
        print(f"Average Time per Page: {metrics.avg_time_per_page:.3f}s")
        print(f"Average Correction Rate: {metrics.avg_correction_rate:.3f}")
        print(f"Cache Hit Rate: {metrics.cache_hit_rate:.3f}")
        print(f"Total LLM Calls: {metrics.total_llm_calls}")
        print(f"Errors: {metrics.errors}")
        print(f"Failed Pages: {len(metrics.failed_pages)}")

        if metrics.language_metrics:
            print("\nLanguage-specific Metrics:")
            for lang, lang_metrics in metrics.language_metrics.items():
                print(f"  {lang.upper()}:")
                print(f"    Pages: {lang_metrics['pages']}")
                print(f"    Avg CER Improvement: {lang_metrics['avg_cer_improvement']:.4f}")
                print(f"    Avg WER Improvement: {lang_metrics['avg_wer_improvement']:.4f}")
                print(f"    Avg Correction Rate: {lang_metrics['avg_correction_rate']:.3f}")

        print("="*80)

def run_v3_evaluation(llm_results_file: str, baseline_results_file: str,
                     gold_pages_dir: str, output_dir: str):
    """Run V3 evaluation with CER/WER delta analysis."""
    integrator = GoldPagesV3Integrator(gold_pages_dir, output_dir)

    # Load results
    with open(llm_results_file, 'r', encoding='utf-8') as f:
        llm_results = json.load(f)

    with open(baseline_results_file, 'r', encoding='utf-8') as f:
        baseline_results = json.load(f)

    # Calculate metrics
    metrics = integrator.calculate_v3_metrics(llm_results, baseline_results)

    # Save reports
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"llm_v3_evaluation_{timestamp}"

    # Get deltas for detailed report
    deltas = []
    for page_id in llm_results.keys():
        if page_id in baseline_results:
            delta = integrator.calculate_page_deltas(page_id, llm_results[page_id], baseline_results[page_id])
            if delta:
                deltas.append(delta)

    integrator.save_deltas_report(deltas, metrics, str(output_file))
    integrator.print_evaluation_summary(metrics)

    return metrics
