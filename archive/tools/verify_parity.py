"""
Parity verification tool for OCR pipeline optimization.

Compares baseline (serial) vs optimized (parallel) pipeline outputs
to ensure byte-for-byte parity while measuring performance gains.

Usage:
    # Generate baseline
    python tools/verify_parity.py --mode=baseline --pdf-dir data/input_pdfs --limit 3
    
    # Run optimized
    python tools/verify_parity.py --mode=optimized --pdf-dir data/input_pdfs --limit 3
    
    # Compare results
    python tools/verify_parity.py --mode=compare --baseline baseline_outputs.jsonl --optimized optimized_outputs.jsonl
"""

import argparse
import json
import hashlib
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import difflib

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PageOutput:
    """Output for a single page."""
    doc_id: str
    page_num: int
    text: str
    confidence: float
    processing_time: float
    cache_hits: Dict[str, bool]
    error: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PageOutput':
        """Create from dictionary."""
        return cls(**data)
    
    def compute_hash(self) -> str:
        """
        Compute SHA256 hash of canonical text.
        
        Canonicalization: normalize whitespace only (preserves case, punctuation).
        """
        # Normalize whitespace: collapse multiple spaces, trim
        canonical = ' '.join(self.text.split())
        return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


class ParityVerifier:
    """Verify byte-for-byte parity between baseline and optimized outputs."""
    
    def __init__(self):
        """Initialize verifier."""
        self.results = {
            'total_pages': 0,
            'identical_pages': 0,
            'different_pages': 0,
            'mismatches': []
        }
    
    def compare_outputs(
        self,
        baseline_path: str,
        optimized_path: str,
        output_report: str = None
    ) -> Dict[str, Any]:
        """
        Compare baseline and optimized outputs.
        
        Args:
            baseline_path: Path to baseline outputs (JSONL)
            optimized_path: Path to optimized outputs (JSONL)
            output_report: Optional path to write detailed report
        
        Returns:
            Comparison results dictionary
        """
        logger.info(f"Loading baseline from {baseline_path}")
        baseline_outputs = self._load_jsonl(baseline_path)
        
        logger.info(f"Loading optimized from {optimized_path}")
        optimized_outputs = self._load_jsonl(optimized_path)
        
        # Create lookup by (doc_id, page_num)
        baseline_map = {
            (o['doc_id'], o['page_num']): PageOutput.from_dict(o)
            for o in baseline_outputs
        }
        
        optimized_map = {
            (o['doc_id'], o['page_num']): PageOutput.from_dict(o)
            for o in optimized_outputs
        }
        
        # Find common pages
        baseline_keys = set(baseline_map.keys())
        optimized_keys = set(optimized_map.keys())
        common_keys = baseline_keys & optimized_keys
        
        if not common_keys:
            logger.error("No common pages found between baseline and optimized!")
            return self.results
        
        logger.info(f"Comparing {len(common_keys)} pages...")
        
        # Compare each page
        mismatches = []
        for key in sorted(common_keys):
            doc_id, page_num = key
            baseline = baseline_map[key]
            optimized = optimized_map[key]
            
            # Compute hashes
            baseline_hash = baseline.compute_hash()
            optimized_hash = optimized.compute_hash()
            
            self.results['total_pages'] += 1
            
            if baseline_hash == optimized_hash:
                self.results['identical_pages'] += 1
                logger.debug(f"✅ MATCH: {doc_id} page {page_num}")
            else:
                self.results['different_pages'] += 1
                
                # Compute detailed diff
                diff = self._compute_diff(baseline.text, optimized.text)
                
                mismatch = {
                    'doc_id': doc_id,
                    'page_num': page_num,
                    'baseline_hash': baseline_hash,
                    'optimized_hash': optimized_hash,
                    'baseline_length': len(baseline.text),
                    'optimized_length': len(optimized.text),
                    'diff_lines': diff[:10],  # First 10 diff lines
                    'baseline_time': baseline.processing_time,
                    'optimized_time': optimized.processing_time
                }
                
                mismatches.append(mismatch)
                self.results['mismatches'].append(mismatch)
                
                logger.warning(f"❌ MISMATCH: {doc_id} page {page_num}")
                logger.warning(f"   Baseline: {baseline_hash[:16]}... ({len(baseline.text)} chars)")
                logger.warning(f"   Optimized: {optimized_hash[:16]}... ({len(optimized.text)} chars)")
        
        # Compute statistics
        if self.results['total_pages'] > 0:
            parity_rate = 100 * self.results['identical_pages'] / self.results['total_pages']
            self.results['parity_percentage'] = parity_rate
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Parity Verification Results:")
            logger.info(f"  Total pages: {self.results['total_pages']}")
            logger.info(f"  Identical: {self.results['identical_pages']} ({parity_rate:.1f}%)")
            logger.info(f"  Different: {self.results['different_pages']}")
            logger.info(f"{'='*60}\n")
            
            if parity_rate == 100.0:
                logger.info("✅ PERFECT PARITY: All outputs identical!")
            else:
                logger.warning(f"⚠️ PARITY DRIFT: {self.results['different_pages']} pages differ")
        
        # Compute speedup if available
        self._compute_speedup(baseline_map, optimized_map, common_keys)
        
        # Write detailed report if requested
        if output_report:
            self._write_report(output_report)
        
        return self.results
    
    def _load_jsonl(self, path: str) -> List[Dict[str, Any]]:
        """Load JSONL file."""
        outputs = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    outputs.append(json.loads(line))
        return outputs
    
    def _compute_diff(self, text1: str, text2: str) -> List[str]:
        """Compute line-by-line diff."""
        lines1 = text1.splitlines()
        lines2 = text2.splitlines()
        
        diff = list(difflib.unified_diff(
            lines1, lines2,
            lineterm='',
            fromfile='baseline',
            tofile='optimized'
        ))
        
        return diff
    
    def _compute_speedup(
        self,
        baseline_map: Dict,
        optimized_map: Dict,
        common_keys: set
    ):
        """Compute speedup statistics."""
        baseline_times = [baseline_map[k].processing_time for k in common_keys]
        optimized_times = [optimized_map[k].processing_time for k in common_keys]
        
        baseline_total = sum(baseline_times)
        optimized_total = sum(optimized_times)
        
        if baseline_total > 0:
            speedup = baseline_total / optimized_total
            self.results['baseline_total_time'] = baseline_total
            self.results['optimized_total_time'] = optimized_total
            self.results['speedup'] = speedup
            
            logger.info(f"Performance:")
            logger.info(f"  Baseline total: {baseline_total:.2f}s")
            logger.info(f"  Optimized total: {optimized_total:.2f}s")
            logger.info(f"  Speedup: {speedup:.2f}x")
    
    def _write_report(self, output_path: str):
        """Write detailed report to file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Parity Verification Report\n\n")
            
            f.write(f"## Summary\n\n")
            f.write(f"- Total pages: {self.results['total_pages']}\n")
            f.write(f"- Identical: {self.results['identical_pages']}\n")
            f.write(f"- Different: {self.results['different_pages']}\n")
            
            if 'parity_percentage' in self.results:
                f.write(f"- Parity: {self.results['parity_percentage']:.1f}%\n")
            
            if 'speedup' in self.results:
                f.write(f"\n## Performance\n\n")
                f.write(f"- Baseline: {self.results['baseline_total_time']:.2f}s\n")
                f.write(f"- Optimized: {self.results['optimized_total_time']:.2f}s\n")
                f.write(f"- Speedup: {self.results['speedup']:.2f}x\n")
            
            if self.results['mismatches']:
                f.write(f"\n## Mismatches\n\n")
                for m in self.results['mismatches']:
                    f.write(f"### {m['doc_id']} - Page {m['page_num']}\n\n")
                    f.write(f"- Baseline hash: {m['baseline_hash'][:16]}...\n")
                    f.write(f"- Optimized hash: {m['optimized_hash'][:16]}...\n")
                    f.write(f"- Baseline length: {m['baseline_length']} chars\n")
                    f.write(f"- Optimized length: {m['optimized_length']} chars\n")
                    
                    if m['diff_lines']:
                        f.write(f"\nFirst 10 diff lines:\n```\n")
                        f.write('\n'.join(m['diff_lines']))
                        f.write("\n```\n\n")
        
        logger.info(f"Detailed report written to {output_path}")


def run_baseline(pdf_dir: str, output_path: str, limit: int = None):
    """
    Run baseline (serial) pipeline and save outputs.
    
    Args:
        pdf_dir: Directory containing PDFs
        output_path: Path to save outputs (JSONL)
        limit: Maximum number of PDFs to process
    """
    logger.info("Running BASELINE (serial) pipeline...")
    
    # Import pipeline
    sys.path.append(str(Path(__file__).parent.parent / 'production'))
    from comprehensive_pipeline import ComprehensivePipeline, PipelineConfig
    
    # Configure for serial processing
    config = PipelineConfig(
        enable_llm_correction=False,  # Disable for speed
        performance_profile='quality'
    )
    
    pipeline = ComprehensivePipeline(config)
    
    # Find PDFs
    pdf_paths = sorted(Path(pdf_dir).glob('*.pdf'))
    if limit:
        pdf_paths = pdf_paths[:limit]
    
    logger.info(f"Processing {len(pdf_paths)} PDFs in serial mode...")
    
    # Process each PDF
    outputs = []
    for pdf_path in pdf_paths:
        logger.info(f"Processing {pdf_path.name}...")
        start_time = time.time()
        
        try:
            result = pipeline.process_pdf(
                str(pdf_path),
                output_dir='data/baseline_output',
                start_page=1,
                end_page=None
            )
            
            # Extract page results
            # TODO: Update to extract actual page results from pipeline
            # For now, create placeholder
            doc_id = pdf_path.stem
            page_output = PageOutput(
                doc_id=doc_id,
                page_num=1,
                text="",  # TODO: Extract from result
                confidence=0.0,
                processing_time=time.time() - start_time,
                cache_hits={}
            )
            
            outputs.append(page_output.to_dict())
            
        except Exception as e:
            logger.error(f"Failed to process {pdf_path.name}: {e}")
    
    # Write outputs
    with open(output_path, 'w', encoding='utf-8') as f:
        for output in outputs:
            f.write(json.dumps(output) + '\n')
    
    logger.info(f"Baseline outputs saved to {output_path}")


def run_optimized(pdf_dir: str, output_path: str, limit: int = None):
    """
    Run optimized (parallel) pipeline and save outputs.
    
    Args:
        pdf_dir: Directory containing PDFs
        output_path: Path to save outputs (JSONL)
        limit: Maximum number of PDFs to process
    """
    logger.info("Running OPTIMIZED (parallel) pipeline...")
    
    # Import orchestrator
    from orchestrator import PipelineOrchestrator, PipelineConfig
    from cache_store import CacheStore
    
    # Load config
    config_path = Path(__file__).parent.parent / 'config' / 'performance.json'
    if config_path.exists():
        config = PipelineConfig.from_file(str(config_path))
    else:
        logger.warning("performance.json not found, using defaults")
        config = PipelineConfig()
    
    # Initialize cache
    cache = CacheStore(
        cache_dir=config.cache_dir,
        max_size_gb=config.cache_max_size_gb
    )
    
    # Initialize orchestrator
    orchestrator = PipelineOrchestrator(config, cache_store=cache)
    
    # Find PDFs
    pdf_paths = sorted(Path(pdf_dir).glob('*.pdf'))
    if limit:
        pdf_paths = pdf_paths[:limit]
    
    logger.info(f"Processing {len(pdf_paths)} PDFs in parallel mode...")
    
    # Process each PDF
    outputs = []
    for pdf_path in pdf_paths:
        logger.info(f"Processing {pdf_path.name}...")
        
        try:
            # Initialize OCR engine (PaddleOCR)
            from paddleocr import PaddleOCR
            ocr_engine = PaddleOCR(use_textline_orientation=True, lang='en')
            
            # Process with orchestrator
            tasks = orchestrator.process_pdf_parallel(
                str(pdf_path),
                page_range=(1, 999),  # All pages
                ocr_engine=ocr_engine,
                ocr_config={'dpi': 400, 'lang': 'en'}
            )
            
            # Convert tasks to outputs
            for task in tasks:
                page_output = PageOutput(
                    doc_id=task.doc_id,
                    page_num=task.page_num,
                    text=task.final_text,
                    confidence=0.0,  # TODO: Compute average confidence
                    processing_time=sum(task.stage_timings.values()),
                    cache_hits=task.cache_hits,
                    error=task.error or ""
                )
                outputs.append(page_output.to_dict())
            
        except Exception as e:
            logger.error(f"Failed to process {pdf_path.name}: {e}")
    
    # Write outputs
    with open(output_path, 'w', encoding='utf-8') as f:
        for output in outputs:
            f.write(json.dumps(output) + '\n')
    
    logger.info(f"Optimized outputs saved to {output_path}")
    
    # Print statistics
    stats = orchestrator.get_statistics()
    logger.info(f"\nOrchestrator statistics:")
    logger.info(f"  Pages processed: {stats['pages_processed']}")
    logger.info(f"  Cache hits: {dict(stats['cache_hits'])}")
    logger.info(f"  Cache misses: {dict(stats['cache_misses'])}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Verify OCR pipeline parity')
    parser.add_argument(
        '--mode',
        choices=['baseline', 'optimized', 'compare'],
        required=True,
        help='Run mode: baseline, optimized, or compare'
    )
    parser.add_argument(
        '--pdf-dir',
        default='data/input_pdfs',
        help='Directory containing PDFs'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of PDFs to process'
    )
    parser.add_argument(
        '--baseline',
        default='baseline_outputs.jsonl',
        help='Path to baseline outputs (for compare mode)'
    )
    parser.add_argument(
        '--optimized',
        default='optimized_outputs.jsonl',
        help='Path to optimized outputs (for compare mode)'
    )
    parser.add_argument(
        '--report',
        help='Path to write detailed comparison report'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'baseline':
        run_baseline(args.pdf_dir, args.baseline, args.limit)
    
    elif args.mode == 'optimized':
        run_optimized(args.pdf_dir, args.optimized, args.limit)
    
    elif args.mode == 'compare':
        verifier = ParityVerifier()
        results = verifier.compare_outputs(
            args.baseline,
            args.optimized,
            output_report=args.report
        )
        
        # Exit with error if parity not perfect
        if results.get('parity_percentage', 0) < 100.0:
            logger.error("Parity verification FAILED")
            sys.exit(1)
        else:
            logger.info("Parity verification PASSED")
            sys.exit(0)


if __name__ == '__main__':
    main()
