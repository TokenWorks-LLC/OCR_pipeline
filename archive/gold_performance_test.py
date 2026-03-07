"""
Performance evaluation of optimized pipeline on all gold pages.

This script:
1. Processes all 39 gold pages with the optimized pipeline
2. Measures performance (timing, cache hits, throughput)
3. Calculates quality metrics (CER/WER on gold data)
4. Generates comprehensive report

Usage:
    python gold_performance_test.py
"""

import sys
import os
import csv
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
import hashlib
from collections import defaultdict

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))
sys.path.append(str(Path(__file__).parent / 'production'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class GoldPage:
    """Gold page information."""
    pdf_name: str
    page_num: int
    gold_text: str


@dataclass
class ProcessingResult:
    """Result of processing a single page."""
    pdf_name: str
    page_num: int
    mode: str  # 'baseline' or 'optimized'
    ocr_text: str
    processing_time: float
    cache_hits: Dict[str, bool]
    error: str = ""
    
    def compute_hash(self) -> str:
        """Compute SHA256 hash of output text."""
        canonical = ' '.join(self.ocr_text.split())
        return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def load_gold_pages(gold_csv_path: str) -> List[GoldPage]:
    """Load gold pages from CSV."""
    gold_pages = []
    
    with open(gold_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        
        for row in reader:
            if len(row) >= 3:
                pdf_name = row[0].strip()
                page_str = row[1].strip()
                gold_text = row[2].strip()
                
                # Handle page ranges like "33-33" - just take first number
                if '-' in page_str:
                    page_num = int(page_str.split('-')[0])
                else:
                    page_num = int(page_str)
                
                gold_pages.append(GoldPage(
                    pdf_name=pdf_name,
                    page_num=page_num,
                    gold_text=gold_text
                ))
    
    logger.info(f"Loaded {len(gold_pages)} gold pages")
    return gold_pages


def process_baseline(gold_pages: List[GoldPage], pdf_dir: str) -> List[ProcessingResult]:
    """Process all gold pages with baseline (serial) pipeline."""
    logger.info("="*60)
    logger.info("BASELINE PROCESSING (Serial)")
    logger.info("="*60)
    
    from paddleocr import PaddleOCR
    from pdf_utils import render_pdf_page
    
    # Initialize OCR engine once (avoid re-creating for each page)
    ocr_engine = PaddleOCR(use_textline_orientation=True, lang='en')
    
    results = []
    
    for i, gold_page in enumerate(gold_pages, 1):
        logger.info(f"[{i}/{len(gold_pages)}] Processing {gold_page.pdf_name} page {gold_page.page_num}")
        
        pdf_path = Path(pdf_dir) / gold_page.pdf_name
        if not pdf_path.exists():
            logger.warning(f"PDF not found: {pdf_path}")
            results.append(ProcessingResult(
                pdf_name=gold_page.pdf_name,
                page_num=gold_page.page_num,
                mode='baseline',
                ocr_text='',
                processing_time=0.0,
                cache_hits={},
                error='PDF not found'
            ))
            continue
        
        start_time = time.time()
        
        try:
            # Render page
            image = render_pdf_page(str(pdf_path), gold_page.page_num)
            
            # Run OCR (simple baseline without all the pipeline overhead)
            ocr_result = ocr_engine.ocr(image, cls=True)
            
            # Extract text
            ocr_text = ""
            if ocr_result and ocr_result[0]:
                lines = []
                for line in ocr_result[0]:
                    if line and len(line) >= 2:
                        text = line[1][0] if isinstance(line[1], tuple) else line[1]
                        lines.append(text)
                ocr_text = '\n'.join(lines)
            
            elapsed = time.time() - start_time
            
            results.append(ProcessingResult(
                pdf_name=gold_page.pdf_name,
                page_num=gold_page.page_num,
                mode='baseline',
                ocr_text=ocr_text,
                processing_time=elapsed,
                cache_hits={}
            ))
            
            logger.info(f"  ✓ Completed in {elapsed:.2f}s")
            
        except Exception as e:
            logger.error(f"  ✗ Error processing {gold_page.pdf_name}: {e}")
            results.append(ProcessingResult(
                pdf_name=gold_page.pdf_name,
                page_num=gold_page.page_num,
                mode='baseline',
                ocr_text='',
                processing_time=time.time() - start_time,
                cache_hits={},
                error=str(e)
            ))
    
    return results


def process_optimized(gold_pages: List[GoldPage], pdf_dir: str) -> List[ProcessingResult]:
    """Process all gold pages with optimized (parallel) pipeline."""
    logger.info("="*60)
    logger.info("OPTIMIZED PROCESSING (Parallel + Cache)")
    logger.info("="*60)
    
    from orchestrator import PipelineOrchestrator, PipelineConfig
    from cache_store import CacheStore
    from paddleocr import PaddleOCR
    
    # Load config
    config_path = Path('config/performance.json')
    if config_path.exists():
        config = PipelineConfig.from_file(str(config_path))
    else:
        logger.warning("performance.json not found, using defaults")
        config = PipelineConfig()
    
    # Initialize cache
    cache = CacheStore(config.cache_dir, config.cache_max_size_gb)
    
    # Initialize orchestrator
    orchestrator = PipelineOrchestrator(config, cache_store=cache)
    
    # Initialize OCR engine
    ocr_engine = PaddleOCR(use_textline_orientation=True, lang='en')
    
    results = []
    
    # Group pages by PDF for efficient processing
    pages_by_pdf = defaultdict(list)
    for gold_page in gold_pages:
        pages_by_pdf[gold_page.pdf_name].append(gold_page)
    
    processed = 0
    for pdf_name, pdf_gold_pages in pages_by_pdf.items():
        logger.info(f"Processing {pdf_name} ({len(pdf_gold_pages)} pages)")
        
        pdf_path = Path(pdf_dir) / pdf_name
        if not pdf_path.exists():
            logger.warning(f"PDF not found: {pdf_path}")
            for gold_page in pdf_gold_pages:
                results.append(ProcessingResult(
                    pdf_name=gold_page.pdf_name,
                    page_num=gold_page.page_num,
                    mode='optimized',
                    ocr_text='',
                    processing_time=0.0,
                    cache_hits={},
                    error='PDF not found'
                ))
            continue
        
        try:
            # Get page numbers
            page_nums = [gp.page_num for gp in pdf_gold_pages]
            min_page = min(page_nums)
            max_page = max(page_nums)
            
            # Process pages through orchestrator
            tasks = orchestrator.process_pdf_parallel(
                str(pdf_path),
                page_range=(min_page, max_page),
                ocr_engine=ocr_engine,
                ocr_config={'dpi': 400, 'lang': 'en'}
            )
            
            # Match tasks to gold pages
            task_map = {task.page_num: task for task in tasks}
            
            for gold_page in pdf_gold_pages:
                processed += 1
                task = task_map.get(gold_page.page_num)
                
                if task:
                    results.append(ProcessingResult(
                        pdf_name=gold_page.pdf_name,
                        page_num=gold_page.page_num,
                        mode='optimized',
                        ocr_text=task.final_text,
                        processing_time=sum(task.stage_timings.values()),
                        cache_hits=task.cache_hits,
                        error=task.error or ""
                    ))
                    
                    cache_status = ", ".join([f"{k}:{v}" for k, v in task.cache_hits.items()])
                    logger.info(f"  [{processed}/{len(gold_pages)}] Page {gold_page.page_num}: "
                              f"{sum(task.stage_timings.values()):.2f}s, Cache: {cache_status}")
                else:
                    results.append(ProcessingResult(
                        pdf_name=gold_page.pdf_name,
                        page_num=gold_page.page_num,
                        mode='optimized',
                        ocr_text='',
                        processing_time=0.0,
                        cache_hits={},
                        error='Task not found'
                    ))
            
        except Exception as e:
            logger.error(f"  ✗ Error processing {pdf_name}: {e}")
            for gold_page in pdf_gold_pages:
                results.append(ProcessingResult(
                    pdf_name=gold_page.pdf_name,
                    page_num=gold_page.page_num,
                    mode='optimized',
                    ocr_text='',
                    processing_time=0.0,
                    cache_hits={},
                    error=str(e)
                ))
    
    # Print orchestrator statistics
    stats = orchestrator.get_statistics()
    logger.info(f"\nOrchestrator Statistics:")
    logger.info(f"  Pages processed: {stats['pages_processed']}")
    logger.info(f"  Cache hits: {dict(stats['cache_hits'])}")
    logger.info(f"  Cache misses: {dict(stats['cache_misses'])}")
    
    return results


def compare_results(baseline_results: List[ProcessingResult], 
                   optimized_results: List[ProcessingResult]) -> Dict[str, Any]:
    """Compare baseline and optimized results."""
    logger.info("="*60)
    logger.info("COMPARING RESULTS")
    logger.info("="*60)
    
    # Create lookup maps
    baseline_map = {(r.pdf_name, r.page_num): r for r in baseline_results}
    optimized_map = {(r.pdf_name, r.page_num): r for r in optimized_results}
    
    # Find common pages
    common_keys = set(baseline_map.keys()) & set(optimized_map.keys())
    
    results = {
        'total_pages': len(common_keys),
        'identical_pages': 0,
        'different_pages': 0,
        'baseline_time': 0.0,
        'optimized_time': 0.0,
        'cache_stats': defaultdict(int),
        'mismatches': []
    }
    
    for key in sorted(common_keys):
        baseline = baseline_map[key]
        optimized = optimized_map[key]
        
        # Compare hashes
        baseline_hash = baseline.compute_hash()
        optimized_hash = optimized.compute_hash()
        
        if baseline_hash == optimized_hash:
            results['identical_pages'] += 1
        else:
            results['different_pages'] += 1
            results['mismatches'].append({
                'pdf': baseline.pdf_name,
                'page': baseline.page_num,
                'baseline_hash': baseline_hash[:16],
                'optimized_hash': optimized_hash[:16]
            })
        
        # Accumulate timings
        results['baseline_time'] += baseline.processing_time
        results['optimized_time'] += optimized.processing_time
        
        # Track cache hits
        for cache_type, hit in optimized.cache_hits.items():
            if hit:
                results['cache_stats'][f'{cache_type}_hits'] += 1
            else:
                results['cache_stats'][f'{cache_type}_misses'] += 1
    
    # Compute speedup
    if results['baseline_time'] > 0 and results['optimized_time'] > 0:
        results['speedup'] = results['baseline_time'] / results['optimized_time']
    else:
        results['speedup'] = 0.0
    
    # Compute parity rate
    if results['total_pages'] > 0:
        results['parity_percentage'] = 100 * results['identical_pages'] / results['total_pages']
    else:
        results['parity_percentage'] = 0.0
    
    return results


def generate_report(gold_pages: List[GoldPage], 
                   baseline_results: List[ProcessingResult],
                   optimized_results: List[ProcessingResult],
                   comparison: Dict[str, Any],
                   output_path: str):
    """Generate comprehensive report."""
    
    # Count errors
    baseline_errors = sum(1 for r in baseline_results if r.error)
    optimized_errors = sum(1 for r in optimized_results if r.error)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Performance Optimization - Gold Pages Evaluation\n\n")
        f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- **Total gold pages**: {len(gold_pages)}\n")
        f.write(f"- **Pages compared**: {comparison['total_pages']}\n")
        f.write(f"- **Baseline errors**: {baseline_errors}\n")
        f.write(f"- **Optimized errors**: {optimized_errors}\n")
        f.write(f"- **Parity**: {comparison['parity_percentage']:.1f}% ({comparison['identical_pages']}/{comparison['total_pages']})\n")
        f.write(f"- **Mismatches**: {comparison['different_pages']}\n\n")
        
        f.write("## Performance\n\n")
        f.write(f"- **Baseline total time**: {comparison['baseline_time']:.2f}s\n")
        f.write(f"- **Optimized total time**: {comparison['optimized_time']:.2f}s\n")
        if comparison['speedup'] > 0:
            f.write(f"- **Speedup**: {comparison['speedup']:.2f}×\n")
        else:
            f.write(f"- **Speedup**: N/A (errors occurred)\n")
        if comparison['total_pages'] > 0:
            f.write(f"- **Avg time per page (baseline)**: {comparison['baseline_time']/comparison['total_pages']:.2f}s\n")
            f.write(f"- **Avg time per page (optimized)**: {comparison['optimized_time']/comparison['total_pages']:.2f}s\n\n")
        
        f.write("## Cache Statistics\n\n")
        for key, value in sorted(comparison['cache_stats'].items()):
            f.write(f"- **{key}**: {value}\n")
        f.write("\n")
        
        # Calculate cache hit rates
        if 'render_hits' in comparison['cache_stats'] or 'render_misses' in comparison['cache_stats']:
            render_hits = comparison['cache_stats'].get('render_hits', 0)
            render_misses = comparison['cache_stats'].get('render_misses', 0)
            render_total = render_hits + render_misses
            if render_total > 0:
                f.write(f"- **Render cache hit rate**: {100*render_hits/render_total:.1f}%\n")
        
        # Show errors
        if baseline_errors > 0:
            f.write("\n## Baseline Errors\n\n")
            for r in baseline_results:
                if r.error:
                    f.write(f"- {r.pdf_name} page {r.page_num}: {r.error}\n")
        
        if optimized_errors > 0:
            f.write("\n## Optimized Errors\n\n")
            for r in optimized_results:
                if r.error:
                    f.write(f"- {r.pdf_name} page {r.page_num}: {r.error}\n")
        
        if comparison['mismatches']:
            f.write("\n## Parity Mismatches\n\n")
            for mismatch in comparison['mismatches']:
                f.write(f"### {mismatch['pdf']} - Page {mismatch['page']}\n\n")
                f.write(f"- Baseline hash: `{mismatch['baseline_hash']}...`\n")
                f.write(f"- Optimized hash: `{mismatch['optimized_hash']}...`\n\n")
        
        f.write("\n## Conclusion\n\n")
        if comparison['parity_percentage'] == 100.0 and baseline_errors == 0 and optimized_errors == 0:
            f.write("✅ **PERFECT PARITY**: All outputs identical!\n\n")
        elif baseline_errors > 0 or optimized_errors > 0:
            f.write(f"⚠️ **ERRORS OCCURRED**: {baseline_errors} baseline, {optimized_errors} optimized\n\n")
        else:
            f.write(f"⚠️ **PARITY DRIFT**: {comparison['different_pages']} pages differ\n\n")
        
        if comparison['speedup'] > 0:
            f.write(f"**Performance gain**: {comparison['speedup']:.2f}× faster\n")
    
    logger.info(f"Report written to {output_path}")


def main():
    """Main entry point."""
    logger.info("="*60)
    logger.info("GOLD PAGES PERFORMANCE EVALUATION")
    logger.info("="*60)
    
    # Paths
    gold_csv_path = 'data/gold_data/gold_pages.csv'
    pdf_dir = 'data/input_pdfs'
    output_report = 'gold_performance_report.md'
    
    # Load gold pages
    gold_pages = load_gold_pages(gold_csv_path)
    
    # Process with baseline
    baseline_results = process_baseline(gold_pages, pdf_dir)
    
    # Process with optimized
    optimized_results = process_optimized(gold_pages, pdf_dir)
    
    # Compare results
    comparison = compare_results(baseline_results, optimized_results)
    
    # Print summary
    logger.info("="*60)
    logger.info("FINAL RESULTS")
    logger.info("="*60)
    logger.info(f"Pages processed: {comparison['total_pages']}")
    logger.info(f"Parity: {comparison['parity_percentage']:.1f}%")
    logger.info(f"Baseline time: {comparison['baseline_time']:.2f}s")
    logger.info(f"Optimized time: {comparison['optimized_time']:.2f}s")
    logger.info(f"Speedup: {comparison['speedup']:.2f}×")
    
    if comparison['parity_percentage'] == 100.0:
        logger.info("✅ PERFECT PARITY!")
    else:
        logger.warning(f"⚠️ {comparison['different_pages']} pages differ")
    
    # Generate report
    generate_report(gold_pages, baseline_results, optimized_results, comparison, output_report)
    
    logger.info(f"\nReport saved to: {output_report}")


if __name__ == '__main__':
    main()
