#!/usr/bin/env python3
"""
Final evaluation script that uses best_config.json for comprehensive OCR evaluation.
Processes all PDFs in input_pdfs that have gold pages, computes CER/WER metrics.
"""
import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Add required paths
sys.path.append(str(Path(__file__).parent.parent / 'src'))
sys.path.append(str(Path(__file__).parent.parent / 'production'))

try:
    from comprehensive_pipeline import ComprehensivePipeline, PipelineConfig
    from telemetry import PageTiming, get_telemetry
    from grapheme_metrics import compute_cer_wer, compute_grapheme_cer_wer
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

logger = logging.getLogger(__name__)

class MetricsCalculator:
    """Calculate CER and WER metrics for OCR evaluation."""
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text for fair comparison."""
        text = re.sub(r'\s+', ' ', text.strip())
        text = text.lower()
        text = re.sub(r'[""''`´]', '"', text)
        text = re.sub(r'[–—]', '-', text)
        return text
    
    @staticmethod
    def calculate_cer(reference: str, hypothesis: str) -> float:
        """Calculate Character Error Rate using grapheme metrics."""
        ref_norm = MetricsCalculator.normalize_text(reference)
        hyp_norm = MetricsCalculator.normalize_text(hypothesis)
        
        metrics = compute_cer_wer(ref_norm, hyp_norm)
        return metrics['cer']
    
    @staticmethod
    def calculate_wer(reference: str, hypothesis: str) -> float:
        """Calculate Word Error Rate using grapheme metrics."""
        ref_norm = MetricsCalculator.normalize_text(reference)
        hyp_norm = MetricsCalculator.normalize_text(hypothesis)
        
        metrics = compute_cer_wer(ref_norm, hyp_norm)
        return metrics['wer']

class FinalEvaluator:
    """Final comprehensive OCR evaluation using best_config.json."""
    
    def __init__(self, config_path: str = "reports/best_config.json"):
        """Initialize evaluator."""
        self.config_path = Path(config_path)
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(f"reports/final_eval_{self.run_timestamp}")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self.config_dict = json.load(f)
            print(f"✅ Loaded configuration from {self.config_path}")
        else:
            # Fallback to quality defaults
            self.config_dict = self._get_quality_defaults()
            print(f"⚠️  {self.config_path} not found, using quality defaults")
        
        # Ensure LLM is disabled
        self.config_dict["llm"]["enabled"] = False
        
        # Results storage
        self.all_results = []
        self.pdf_results = {}
        self.lang_results = {}
        
        # Configure logging
        log_file = self.run_dir / "evaluation.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def _get_quality_defaults(self) -> Dict[str, Any]:
        """Get quality configuration defaults."""
        return {
            "profile": "quality",
            "render": {"dpi": 400, "min_dpi": 300},
            "ocr": {
                "engine": "router",
                "dpi": 400,
                "languages": ["en", "tr", "de", "fr", "it"],
                "enable_text_correction": True,
                "enable_reading_order": True,
                "confidence_threshold": 0.5
            },
            "llm": {"enabled": False},
            "orientation": {
                "enable": True,
                "coarse_full360": True,
                "fine_deg": 2.0,
                "fine_step_deg": 0.1
            },
            "detector": {
                "engine": "mmocr_dbnetpp",
                "wbf_union": True,
                "det_db_box_thresh": 0.35,
                "det_db_unclip_ratio": 2.0
            },
            "router": {
                "primary": "abinet",
                "fallback": "parseq", 
                "ensemble": ["abinet", "parseq", "doctr_sar"],
                "thresholds": {"en": 0.92, "de": 0.90, "fr": 0.90, "it": 0.90, "tr": 0.88},
                "beam_size": 5
            }
        }
    
    def load_gold_pages(self, gold_csv: str) -> List[Tuple[str, str, str]]:
        """Load gold standard pages from CSV."""
        gold_pages = []
        with open(gold_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pdf_name = row['PDF LINK'].strip()
                page_spec = row['PAGE'].strip()
                handtyped = row['HANDTYPED'].strip()
                
                # Normalize page specifications
                pages = self._parse_page_spec(page_spec)
                for page_num in pages:
                    gold_pages.append((pdf_name, str(page_num), handtyped))
        
        logger.info(f"Loaded {len(gold_pages)} gold pages from {gold_csv}")
        return gold_pages
    
    def _parse_page_spec(self, page_spec: str) -> List[int]:
        """Parse page specification (supports single pages and ranges)."""
        pages = []
        specs = page_spec.split(',')
        
        for spec in specs:
            spec = spec.strip()
            if '–' in spec or '-' in spec:
                # Range specification
                if '–' in spec:
                    start, end = spec.split('–', 1)
                else:
                    start, end = spec.split('-', 1)
                try:
                    start_page = int(start.strip())
                    end_page = int(end.strip())
                    pages.extend(range(start_page, end_page + 1))
                except ValueError:
                    logger.warning(f"Could not parse page range: {spec}")
            else:
                # Single page
                try:
                    pages.append(int(spec))
                except ValueError:
                    logger.warning(f"Could not parse page number: {spec}")
        
        return pages
    
    def filter_available_pdfs(self, gold_pages: List[Tuple[str, str, str]], 
                            input_dir: str) -> List[Tuple[str, str, str]]:
        """Filter gold pages to only include PDFs that exist."""
        input_path = Path(input_dir)
        available_pdfs = set()
        
        for pdf_file in input_path.glob("*.pdf"):
            available_pdfs.add(pdf_file.name)
        for pdf_file in input_path.glob("*.PDF"):
            available_pdfs.add(pdf_file.name)
        
        filtered_pages = []
        for pdf_name, page_num, handtyped in gold_pages:
            if pdf_name in available_pdfs:
                filtered_pages.append((pdf_name, page_num, handtyped))
            else:
                logger.warning(f"PDF not found: {pdf_name}")
        
        logger.info(f"Filtered to {len(filtered_pages)} pages from available PDFs")
        return filtered_pages
    
    def run_evaluation(self, gold_csv: str, input_dir: str) -> Dict[str, Any]:
        """Run comprehensive evaluation."""
        logger.info("Starting final OCR evaluation")
        start_time = time.time()
        
        # Load and filter gold pages
        gold_pages = self.load_gold_pages(gold_csv)
        filtered_pages = self.filter_available_pdfs(gold_pages, input_dir)
        
        if not filtered_pages:
            logger.error("No valid PDF pages found for evaluation")
            return {"error": "No valid pages found"}
        
        # Convert config to PipelineConfig
        pipeline_config = self._config_dict_to_pipeline_config(self.config_dict)
        pipeline = ComprehensivePipeline(pipeline_config)
        
        # Process each page
        total_pages = len(filtered_pages)
        processed_pages = 0
        
        for i, (pdf_name, page_num, gold_text) in enumerate(filtered_pages, 1):
            logger.info(f"Processing {i}/{total_pages}: {pdf_name} page {page_num}")
            
            try:
                pdf_path = Path(input_dir) / pdf_name
                page_num_int = int(page_num)
                
                # Run OCR on the page
                results = pipeline.process_pdf(str(pdf_path), 
                                             target_pages=[page_num_int],
                                             output_dir=str(self.run_dir / "temp"))
                
                if results and len(results) > 0:
                    ocr_text = results[0].get('corrected_text', 
                                            results[0].get('text', ''))
                    
                    # Calculate metrics
                    cer = MetricsCalculator.calculate_cer(gold_text, ocr_text)
                    wer = MetricsCalculator.calculate_wer(gold_text, ocr_text)
                    
                    # Guess language (simple heuristic)
                    lang_guess = self._guess_language(ocr_text)
                    
                    # Store result
                    result = {
                        'run_id': self.run_timestamp,
                        'pdf': pdf_name,
                        'page': page_num,
                        'lang_guess': lang_guess,
                        'cer': cer,
                        'wer': wer,
                        'chars_ref': len(gold_text),
                        'chars_hyp': len(ocr_text),
                        'words_ref': len(gold_text.split()),
                        'words_hyp': len(ocr_text.split()),
                        'engine': 'router',
                        'profile': 'quality',
                        'dpi': self.config_dict.get('render', {}).get('dpi', 400),
                        'orientation_angle': 0,  # TODO: extract from results
                        'method': 'quality_enhanced',
                        'engines_used': ','.join(self.config_dict.get('router', {}).get('ensemble', []))
                    }
                    
                    self.all_results.append(result)
                    processed_pages += 1
                    
                    # Update aggregates
                    self._update_pdf_aggregate(pdf_name, result)
                    self._update_lang_aggregate(lang_guess, result)
                    
                else:
                    logger.warning(f"No OCR results for {pdf_name} page {page_num}")
                    
            except Exception as e:
                logger.error(f"Error processing {pdf_name} page {page_num}: {e}")
                continue
        
        # Calculate overall metrics
        if self.all_results:
            overall_cer = sum(r['cer'] for r in self.all_results) / len(self.all_results)
            overall_wer = sum(r['wer'] for r in self.all_results) / len(self.all_results)
        else:
            overall_cer = overall_wer = 1.0
        
        end_time = time.time()
        duration = end_time - start_time
        
        summary = {
            'timestamp': self.run_timestamp,
            'processed_pages': processed_pages,
            'total_pages': total_pages,
            'overall_cer': overall_cer,
            'overall_wer': overall_wer,
            'duration_seconds': duration,
            'config_used': 'best_config.json' if self.config_path.exists() else 'quality_defaults'
        }
        
        logger.info(f"Evaluation complete: {processed_pages}/{total_pages} pages processed")
        logger.info(f"Overall CER: {overall_cer:.4f}, WER: {overall_wer:.4f}")
        
        return summary
    
    def _config_dict_to_pipeline_config(self, config_dict: Dict[str, Any]) -> PipelineConfig:
        """Convert config dictionary to PipelineConfig object."""
        return PipelineConfig(
            llm_provider="none",  # Force disable LLM
            dpi=config_dict.get('render', {}).get('dpi', 400),
            paddle_use_gpu=True,
            enable_reading_order=config_dict.get('ocr', {}).get('enable_reading_order', True),
            enable_llm_correction=False,  # Disable LLM correction
            enable_akkadian_extraction=config_dict.get('akkadian', {}).get('enable_extraction', True),
            create_html_overlay=False,  # Speed up evaluation
            create_visualization=False,  # Speed up evaluation
            performance_profile="quality"
        )
    
    def _guess_language(self, text: str) -> str:
        """Simple language guessing based on character patterns."""
        text_lower = text.lower()
        
        # Turkish indicators
        if any(char in text_lower for char in ['ç', 'ğ', 'ı', 'ş', 'ü', 'ö']):
            return 'tr'
        
        # German indicators  
        if any(word in text_lower for word in ['der', 'die', 'das', 'und', 'von', 'mit']):
            return 'de'
        
        # French indicators
        if any(word in text_lower for word in ['le', 'la', 'les', 'de', 'du', 'des', 'et']):
            return 'fr'
        
        # Italian indicators
        if any(word in text_lower for word in ['il', 'la', 'le', 'di', 'del', 'della', 'e']):
            return 'it'
        
        # Default to English
        return 'en'
    
    def _update_pdf_aggregate(self, pdf_name: str, result: Dict[str, Any]):
        """Update per-PDF aggregates."""
        if pdf_name not in self.pdf_results:
            self.pdf_results[pdf_name] = {
                'pages': 0, 'total_cer': 0.0, 'total_wer': 0.0,
                'chars_ref': 0, 'chars_hyp': 0, 'words_ref': 0, 'words_hyp': 0
            }
        
        agg = self.pdf_results[pdf_name]
        agg['pages'] += 1
        agg['total_cer'] += result['cer']
        agg['total_wer'] += result['wer']
        agg['chars_ref'] += result['chars_ref']
        agg['chars_hyp'] += result['chars_hyp']
        agg['words_ref'] += result['words_ref']
        agg['words_hyp'] += result['words_hyp']
    
    def _update_lang_aggregate(self, lang: str, result: Dict[str, Any]):
        """Update per-language aggregates."""
        if lang not in self.lang_results:
            self.lang_results[lang] = {
                'pages': 0, 'total_cer': 0.0, 'total_wer': 0.0,
                'chars_ref': 0, 'chars_hyp': 0, 'words_ref': 0, 'words_hyp': 0
            }
        
        agg = self.lang_results[lang]
        agg['pages'] += 1
        agg['total_cer'] += result['cer']
        agg['total_wer'] += result['wer']
        agg['chars_ref'] += result['chars_ref']
        agg['chars_hyp'] += result['chars_hyp']
        agg['words_ref'] += result['words_ref']
        agg['words_hyp'] += result['words_hyp']
    
    def save_results(self):
        """Save all results to files."""
        metrics_dir = self.run_dir / "metrics"
        metrics_dir.mkdir(exist_ok=True)
        
        # Save detailed metrics
        with open(metrics_dir / "metrics.csv", 'w', newline='', encoding='utf-8') as f:
            if self.all_results:
                fieldnames = self.all_results[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.all_results)
        
        # Save PDF aggregates
        with open(metrics_dir / "metrics_by_pdf.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['pdf', 'pages', 'avg_cer', 'avg_wer', 'chars_ref', 'chars_hyp', 'words_ref', 'words_hyp'])
            for pdf, agg in self.pdf_results.items():
                avg_cer = agg['total_cer'] / agg['pages'] if agg['pages'] > 0 else 0.0
                avg_wer = agg['total_wer'] / agg['pages'] if agg['pages'] > 0 else 0.0
                writer.writerow([pdf, agg['pages'], avg_cer, avg_wer, 
                               agg['chars_ref'], agg['chars_hyp'], 
                               agg['words_ref'], agg['words_hyp']])
        
        # Save language aggregates
        with open(metrics_dir / "metrics_by_lang.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['language', 'pages', 'avg_cer', 'avg_wer', 'chars_ref', 'chars_hyp', 'words_ref', 'words_hyp'])
            for lang, agg in self.lang_results.items():
                avg_cer = agg['total_cer'] / agg['pages'] if agg['pages'] > 0 else 0.0
                avg_wer = agg['total_wer'] / agg['pages'] if agg['pages'] > 0 else 0.0
                writer.writerow([lang, agg['pages'], avg_cer, avg_wer,
                               agg['chars_ref'], agg['chars_hyp'],
                               agg['words_ref'], agg['words_hyp']])
        
        # Generate summary report
        self._generate_summary_report()
        
        logger.info(f"Results saved to {self.run_dir}")
    
    def _generate_summary_report(self):
        """Generate final summary markdown report."""
        summary_path = self.run_dir / "summary.md"
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"# Final OCR Evaluation Report\n\n")
            f.write(f"**Timestamp:** {self.run_timestamp}\n\n")
            f.write(f"**Configuration:** {self.config_path}\n\n")
            f.write(f"**LLM Enabled:** False\n\n")
            
            # Overall metrics
            if self.all_results:
                overall_cer = sum(r['cer'] for r in self.all_results) / len(self.all_results)
                overall_wer = sum(r['wer'] for r in self.all_results) / len(self.all_results)
                
                f.write(f"## Overall Performance\n\n")
                f.write(f"- **Overall CER:** {overall_cer:.4f} ({overall_cer*100:.2f}%)\n")
                f.write(f"- **Overall WER:** {overall_wer:.4f} ({overall_wer*100:.2f}%)\n")
                f.write(f"- **Pages Processed:** {len(self.all_results)}\n")
                f.write(f"- **Target CER ≤ 0.10:** {'✅ ACHIEVED' if overall_cer <= 0.10 else '❌ NOT ACHIEVED'}\n")
                f.write(f"- **Target WER ≤ 0.10:** {'✅ ACHIEVED' if overall_wer <= 0.10 else '❌ NOT ACHIEVED'}\n\n")
            
            # Per-PDF table
            f.write(f"## Per-PDF Results\n\n")
            f.write(f"| PDF | Pages | Avg CER | Avg WER | Status |\n")
            f.write(f"|-----|-------|---------|---------|--------|\n")
            for pdf, agg in sorted(self.pdf_results.items()):
                avg_cer = agg['total_cer'] / agg['pages'] if agg['pages'] > 0 else 0.0
                avg_wer = agg['total_wer'] / agg['pages'] if agg['pages'] > 0 else 0.0
                status = "✅" if avg_cer <= 0.10 and avg_wer <= 0.10 else "❌"
                f.write(f"| {pdf} | {agg['pages']} | {avg_cer:.3f} | {avg_wer:.3f} | {status} |\n")
            
            # Per-language table
            f.write(f"\n## Per-Language Results\n\n")
            f.write(f"| Language | Pages | Avg CER | Avg WER | Status |\n")
            f.write(f"|----------|-------|---------|---------|--------|\n")
            for lang, agg in sorted(self.lang_results.items()):
                avg_cer = agg['total_cer'] / agg['pages'] if agg['pages'] > 0 else 0.0
                avg_wer = agg['total_wer'] / agg['pages'] if agg['pages'] > 0 else 0.0
                status = "✅" if avg_cer <= 0.10 and avg_wer <= 0.10 else "❌"
                f.write(f"| {lang} | {agg['pages']} | {avg_cer:.3f} | {avg_wer:.3f} | {status} |\n")
            
            f.write(f"\n## File Paths\n\n")
            f.write(f"- **Detailed metrics:** `{self.run_dir}/metrics/metrics.csv`\n")
            f.write(f"- **PDF aggregates:** `{self.run_dir}/metrics/metrics_by_pdf.csv`\n")
            f.write(f"- **Language aggregates:** `{self.run_dir}/metrics/metrics_by_lang.csv`\n")
            f.write(f"- **Summary report:** `{summary_path}`\n")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Final OCR evaluation using best_config.json")
    parser.add_argument('--config', default='reports/best_config.json',
                       help="Path to configuration JSON file")
    parser.add_argument('--gold-csv', default='data/gold_data/gold_pages.csv',
                       help="Path to gold pages CSV")
    parser.add_argument('--input-dir', default='data/input_pdfs',
                       help="Directory containing PDF files")
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = FinalEvaluator(config_path=args.config)
    
    # Run evaluation
    summary = evaluator.run_evaluation(args.gold_csv, args.input_dir)
    
    if 'error' not in summary:
        # Save results
        evaluator.save_results()
        
        # Print final summary
        print(f"\n{'='*60}")
        print(f"FINAL OCR EVALUATION COMPLETE")
        print(f"{'='*60}")
        print(f"Overall CER: {summary['overall_cer']:.4f} ({summary['overall_cer']*100:.2f}%)")
        print(f"Overall WER: {summary['overall_wer']:.4f} ({summary['overall_wer']*100:.2f}%)")
        print(f"Pages processed: {summary['processed_pages']}/{summary['total_pages']}")
        print(f"Duration: {summary['duration_seconds']:.1f} seconds")
        print(f"")
        print(f"Target CER ≤ 0.10: {'✅ ACHIEVED' if summary['overall_cer'] <= 0.10 else '❌ NOT ACHIEVED'}")
        print(f"Target WER ≤ 0.10: {'✅ ACHIEVED' if summary['overall_wer'] <= 0.10 else '❌ NOT ACHIEVED'}")
        print(f"")
        print(f"📊 Summary report: {evaluator.run_dir}/summary.md")
        print(f"📈 Detailed metrics: {evaluator.run_dir}/metrics/")
    else:
        print(f"❌ Evaluation failed: {summary['error']}")
        sys.exit(1)

if __name__ == "__main__":
    main()