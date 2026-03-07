#!/usr/bin/env python3
"""
Gold standard evaluation runner for OCR pipeline.
Reads CSV with gold pages, runs OCR, computes CER/WER metrics.
"""
import argparse
import csv
import logging
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import re

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

try:
    from grapheme_metrics import compute_cer_wer, compute_grapheme_cer_wer
except ImportError as e:
    print(f"Error importing grapheme_metrics: {e}")
    sys.exit(1)

from pipeline_profiles import profile_manager

logger = logging.getLogger(__name__)

class MetricsCalculator:
    """Calculate CER and WER metrics for OCR evaluation."""
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text for fair comparison."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        # Convert to lowercase for case-insensitive comparison
        text = text.lower()
        # Remove common OCR artifacts
        text = re.sub(r'[""''`´]', '"', text)  # Normalize quotes
        text = re.sub(r'[–—]', '-', text)  # Normalize dashes
        return text
    
    @staticmethod
    def calculate_cer(reference: str, hypothesis: str) -> float:
        """Calculate Character Error Rate using grapheme metrics."""
        ref_norm = MetricsCalculator.normalize_text(reference)
        hyp_norm = MetricsCalculator.normalize_text(hypothesis)
        
        metrics = compute_cer_wer(ref_norm, hyp_norm)
        return min(1.0, metrics['cer'])  # Cap at 100%
    
    @staticmethod
    def calculate_wer(reference: str, hypothesis: str) -> float:
        """Calculate Word Error Rate using grapheme metrics."""
        ref_norm = MetricsCalculator.normalize_text(reference)
        hyp_norm = MetricsCalculator.normalize_text(hypothesis)
        
        metrics = compute_cer_wer(ref_norm, hyp_norm)
        return min(1.0, metrics['wer'])  # Cap at 100%
    
    @staticmethod
    def _edit_distance(seq1, seq2) -> int:
        """Calculate edit distance using dynamic programming (fallback)."""
        len1, len2 = len(seq1), len(seq2)
        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        
        # Initialize base cases
        for i in range(len1 + 1):
            dp[i][0] = i
        for j in range(len2 + 1):
            dp[0][j] = j
        
        # Fill the DP table
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j],    # deletion
                                       dp[i][j-1],    # insertion
                                       dp[i-1][j-1])  # substitution
        
        return dp[len1][len2]

class GoldPageEvaluator:
    """Evaluator for gold standard OCR pages."""
    
    def __init__(self, profile_name: str = "quality", input_dir: str = "data/input_pdfs",
                 gold_csv: str = "data/gold_data/gold_pages.csv",
                 output_dir: str = "reports"):
        """Initialize evaluator.
        
        Args:
            profile_name: Pipeline profile to use ('fast' or 'quality')
            input_dir: Directory containing PDF files
            gold_csv: Path to gold pages CSV file
            output_dir: Output directory for results
        """
        self.profile_name = profile_name
        self.input_dir = Path(input_dir)
        self.gold_csv = Path(gold_csv)
        self.output_dir = Path(output_dir)
        
        # Load profile
        self.profile = profile_manager.get_profile(profile_name)
        if not self.profile:
            raise ValueError(f"Unknown profile: {profile_name}")
        
        # Create run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"gold_eval_{profile_name}_{timestamp}"
        self.run_dir = self.output_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize results storage
        self.results = []
        self.summary_stats = {
            'total_pages': 0,
            'processed_pages': 0,
            'failed_pages': 0,
            'by_language': {},
            'by_pdf': {},
            'overall_cer': 0.0,
            'overall_wer': 0.0
        }
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Set up logging for the evaluation run."""
        log_file = self.run_dir / "evaluation.log"
        
        # Configure logger
        logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        logger.info(f"Starting gold evaluation with profile: {self.profile_name}")
        logger.info(f"Run ID: {self.run_id}")
        logger.info(f"Output directory: {self.run_dir}")
    
    def load_gold_pages(self) -> List[Dict[str, str]]:
        """Load gold pages from CSV file.
        
        Returns:
            List of gold page records
        """
        if not self.gold_csv.exists():
            raise FileNotFoundError(f"Gold CSV not found: {self.gold_csv}")
        
        gold_pages = []
        
        try:
            with open(self.gold_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Validate required columns
                    if 'PDF LINK' not in row or 'PAGE' not in row or 'HANDTYPED' not in row:
                        logger.warning(f"Skipping row with missing columns: {row}")
                        continue
                    
                    gold_pages.append({
                        'pdf_filename': row['PDF LINK'].strip(),
                        'page_spec': row['PAGE'].strip(),
                        'ground_truth': row['HANDTYPED'].strip(),
                        'language': row.get('LANGUAGE', 'unknown').strip(),
                        'notes': row.get('NOTES', '').strip()
                    })
            
            logger.info(f"Loaded {len(gold_pages)} gold pages from {self.gold_csv}")
            return gold_pages
            
        except Exception as e:
            logger.error(f"Failed to load gold pages: {e}")
            raise
    
    def parse_page_spec(self, page_spec: str) -> List[int]:
        """Parse page specification into list of page numbers.
        
        Args:
            page_spec: Page specification (e.g., "7", "6-7", "6–7")
            
        Returns:
            List of page numbers (1-indexed)
        """
        page_spec = page_spec.strip()
        
        # Single page
        if page_spec.isdigit():
            return [int(page_spec)]
        
        # Page range
        if '-' in page_spec or '–' in page_spec:
            # Handle both hyphen and en-dash
            separator = '-' if '-' in page_spec else '–'
            parts = page_spec.split(separator)
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                start_page = int(parts[0])
                end_page = int(parts[1])
                return list(range(start_page, end_page + 1))
        
        logger.warning(f"Invalid page spec: {page_spec}")
        return []
    
    def run_ocr_on_page(self, pdf_path: Path, page_num: int) -> Tuple[str, Dict[str, Any]]:
        """Run OCR on a specific page of a PDF.
        
        Args:
            pdf_path: Path to PDF file
            page_num: Page number (1-indexed)
            
        Returns:
            Tuple of (extracted_text, metadata)
        """
        try:
            # Import OCR pipeline components
            from comprehensive_pipeline import ComprehensivePipeline, PipelineConfig
            from pdf_utils import extract_images_from_pdf
            
            # Convert profile to pipeline config
            config = PipelineConfig(
                llm_provider=self.profile.llm.provider,
                llm_model=self.profile.llm.model,
                llm_base_url=self.profile.llm.base_url,
                llm_timeout=self.profile.llm.timeout,
                dpi=self.profile.dpi,
                paddle_use_gpu=self.profile.enable_gpu,
                ocr_languages=self.profile.ocr_languages,
                enable_reading_order=self.profile.preprocessing.enable_reading_order,
                enable_llm_correction=self.profile.llm.enable_llm_correction,
                max_concurrent_corrections=self.profile.llm.max_concurrent_corrections
            )
            
            # Initialize pipeline
            pipeline = ComprehensivePipeline(config)
            
            # Extract specific page
            images = extract_images_from_pdf(str(pdf_path), dpi=self.profile.dpi, 
                                           page_numbers=[page_num])
            
            if not images:
                raise ValueError(f"No image extracted for page {page_num}")
            
            # Process the page
            page_image = images[0]
            start_time = time.time()
            
            # Run OCR
            result = pipeline.process_page(page_image, page_num, str(pdf_path))
            
            processing_time = time.time() - start_time
            
            # Extract text from result
            extracted_text = ""
            if hasattr(result, 'final_text'):
                extracted_text = result.final_text
            elif hasattr(result, 'corrected_text'):
                extracted_text = result.corrected_text
            elif hasattr(result, 'text'):
                extracted_text = result.text
            else:
                extracted_text = str(result)
            
            # Collect metadata
            metadata = {
                'processing_time_ms': int(processing_time * 1000),
                'profile_used': self.profile_name,
                'ocr_languages': self.profile.ocr_languages,
                'llm_enabled': self.profile.llm.enable_llm_correction,
                'preprocessing_enabled': {
                    'deskew': self.profile.preprocessing.enable_deskew,
                    'denoise': self.profile.preprocessing.enable_denoise,
                    'column_detection': self.profile.preprocessing.enable_column_detection
                }
            }
            
            return extracted_text, metadata
            
        except Exception as e:
            logger.error(f"OCR failed for {pdf_path} page {page_num}: {e}")
            return "", {'error': str(e), 'processing_time_ms': 0}
    
    def evaluate_page(self, gold_page: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Evaluate a single gold page.
        
        Args:
            gold_page: Gold page record
            
        Returns:
            Evaluation result dictionary or None if failed
        """
        pdf_filename = gold_page['pdf_filename']
        page_spec = gold_page['page_spec']
        ground_truth = gold_page['ground_truth']
        language = gold_page['language']
        
        # Find PDF file
        pdf_path = self.input_dir / pdf_filename
        if not pdf_path.exists():
            logger.warning(f"PDF not found: {pdf_path}")
            return None
        
        # Parse page specification
        page_numbers = self.parse_page_spec(page_spec)
        if not page_numbers:
            logger.warning(f"Invalid page spec: {page_spec}")
            return None
        
        # Process each page and concatenate results
        all_extracted_text = []
        all_metadata = []
        
        for page_num in page_numbers:
            logger.info(f"Processing {pdf_filename} page {page_num}")
            extracted_text, metadata = self.run_ocr_on_page(pdf_path, page_num)
            all_extracted_text.append(extracted_text)
            all_metadata.append(metadata)
        
        # Combine extracted text
        combined_text = ' '.join(all_extracted_text).strip()
        
        # Calculate metrics
        cer = MetricsCalculator.calculate_cer(ground_truth, combined_text)
        wer = MetricsCalculator.calculate_wer(ground_truth, combined_text)
        
        # Create result record
        result = {
            'pdf_filename': pdf_filename,
            'page_spec': page_spec,
            'page_numbers': page_numbers,
            'language': language,
            'ground_truth': ground_truth,
            'extracted_text': combined_text,
            'cer': cer,
            'wer': wer,
            'character_count': len(ground_truth),
            'word_count': len(ground_truth.split()),
            'metadata': all_metadata,
            'evaluation_time': datetime.now().isoformat()
        }
        
        logger.info(f"Evaluated {pdf_filename} {page_spec}: CER={cer:.3f}, WER={wer:.3f}")
        
        return result
    
    def run_evaluation(self) -> Dict[str, Any]:
        """Run complete gold standard evaluation.
        
        Returns:
            Summary results dictionary
        """
        logger.info("Starting gold standard evaluation")
        
        # Load gold pages
        gold_pages = self.load_gold_pages()
        self.summary_stats['total_pages'] = len(gold_pages)
        
        # Filter to available PDFs
        available_gold_pages = []
        for gold_page in gold_pages:
            pdf_path = self.input_dir / gold_page['pdf_filename']
            if pdf_path.exists():
                available_gold_pages.append(gold_page)
            else:
                logger.warning(f"Skipping {gold_page['pdf_filename']} - file not found")
        
        logger.info(f"Found {len(available_gold_pages)} PDFs available for evaluation")
        
        # Process each gold page
        all_cers = []
        all_wers = []
        
        for gold_page in available_gold_pages:
            try:
                result = self.evaluate_page(gold_page)
                if result:
                    self.results.append(result)
                    all_cers.append(result['cer'])
                    all_wers.append(result['wer'])
                    
                    # Update by-language stats
                    lang = result['language']
                    if lang not in self.summary_stats['by_language']:
                        self.summary_stats['by_language'][lang] = {'count': 0, 'cer_sum': 0, 'wer_sum': 0}
                    self.summary_stats['by_language'][lang]['count'] += 1
                    self.summary_stats['by_language'][lang]['cer_sum'] += result['cer']
                    self.summary_stats['by_language'][lang]['wer_sum'] += result['wer']
                    
                    # Update by-PDF stats
                    pdf = result['pdf_filename']
                    if pdf not in self.summary_stats['by_pdf']:
                        self.summary_stats['by_pdf'][pdf] = {'count': 0, 'cer_sum': 0, 'wer_sum': 0}
                    self.summary_stats['by_pdf'][pdf]['count'] += 1
                    self.summary_stats['by_pdf'][pdf]['cer_sum'] += result['cer']
                    self.summary_stats['by_pdf'][pdf]['wer_sum'] += result['wer']
                    
                    self.summary_stats['processed_pages'] += 1
                else:
                    self.summary_stats['failed_pages'] += 1
                    
            except Exception as e:
                logger.error(f"Failed to evaluate {gold_page['pdf_filename']}: {e}")
                self.summary_stats['failed_pages'] += 1
        
        # Calculate overall metrics
        if all_cers:
            self.summary_stats['overall_cer'] = sum(all_cers) / len(all_cers)
        if all_wers:
            self.summary_stats['overall_wer'] = sum(all_wers) / len(all_wers)
        
        # Calculate per-language averages
        for lang_stats in self.summary_stats['by_language'].values():
            if lang_stats['count'] > 0:
                lang_stats['avg_cer'] = lang_stats['cer_sum'] / lang_stats['count']
                lang_stats['avg_wer'] = lang_stats['wer_sum'] / lang_stats['count']
        
        # Calculate per-PDF averages
        for pdf_stats in self.summary_stats['by_pdf'].values():
            if pdf_stats['count'] > 0:
                pdf_stats['avg_cer'] = pdf_stats['cer_sum'] / pdf_stats['count']
                pdf_stats['avg_wer'] = pdf_stats['wer_sum'] / pdf_stats['count']
        
        logger.info(f"Evaluation complete: {self.summary_stats['processed_pages']} pages processed")
        logger.info(f"Overall CER: {self.summary_stats['overall_cer']:.3f}")
        logger.info(f"Overall WER: {self.summary_stats['overall_wer']:.3f}")
        
        return self.summary_stats
    
    def save_results(self):
        """Save evaluation results to files."""
        # Save detailed results
        results_file = self.run_dir / "detailed_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        # Save summary stats
        summary_file = self.run_dir / "summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(self.summary_stats, f, indent=2, ensure_ascii=False)
        
        # Save metrics CSV
        metrics_file = self.run_dir / "metrics.csv"
        with open(metrics_file, 'w', newline='', encoding='utf-8') as f:
            if self.results:
                fieldnames = ['pdf_filename', 'page_spec', 'language', 'cer', 'wer', 
                             'character_count', 'word_count']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for result in self.results:
                    writer.writerow({k: result[k] for k in fieldnames})
        
        # Generate summary markdown
        self._generate_summary_markdown()
        
        logger.info(f"Results saved to {self.run_dir}")
    
    def _generate_summary_markdown(self):
        """Generate summary markdown report."""
        summary_md = self.run_dir / "summary.md"
        
        with open(summary_md, 'w', encoding='utf-8') as f:
            f.write(f"# Gold Standard Evaluation Summary\n\n")
            f.write(f"**Profile:** {self.profile_name}\n")
            f.write(f"**Run ID:** {self.run_id}\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overall metrics
            f.write("## Overall Results\n\n")
            f.write(f"- **Pages Processed:** {self.summary_stats['processed_pages']}\n")
            f.write(f"- **Pages Failed:** {self.summary_stats['failed_pages']}\n")
            f.write(f"- **Overall CER:** {self.summary_stats['overall_cer']:.3f}\n")
            f.write(f"- **Overall WER:** {self.summary_stats['overall_wer']:.3f}\n\n")
            
            # Per-language results
            if self.summary_stats['by_language']:
                f.write("## Results by Language\n\n")
                f.write("| Language | Pages | Average CER | Average WER |\n")
                f.write("|----------|-------|-------------|-------------|\n")
                for lang, stats in self.summary_stats['by_language'].items():
                    f.write(f"| {lang} | {stats['count']} | {stats.get('avg_cer', 0):.3f} | {stats.get('avg_wer', 0):.3f} |\n")
                f.write("\n")
            
            # Per-PDF results (top 10)
            if self.summary_stats['by_pdf']:
                f.write("## Results by PDF (Top 10)\n\n")
                f.write("| PDF | Pages | Average CER | Average WER |\n")
                f.write("|-----|-------|-------------|-------------|\n")
                sorted_pdfs = sorted(self.summary_stats['by_pdf'].items(), 
                                   key=lambda x: x[1].get('avg_cer', 1), reverse=True)
                for pdf, stats in sorted_pdfs[:10]:
                    f.write(f"| {pdf} | {stats['count']} | {stats.get('avg_cer', 0):.3f} | {stats.get('avg_wer', 0):.3f} |\n")
                f.write("\n")
            
            # Profile information
            f.write(f"## Profile Configuration: {self.profile_name}\n\n")
            profile_info = profile_manager.get_profile_info(self.profile_name)
            if profile_info:
                f.write(f"**Description:** {profile_info['description']}\n\n")
                f.write(f"**Languages:** {', '.join(profile_info['languages'])}\n\n")
                f.write(f"**LLM Enabled:** {profile_info['llm_enabled']}\n\n")
                f.write("**Features:**\n")
                for feature, enabled in profile_info['preprocessing_features'].items():
                    f.write(f"- {feature}: {enabled}\n")

def main():
    """Main entry point for gold evaluation."""
    parser = argparse.ArgumentParser(description="Run gold standard OCR evaluation")
    parser.add_argument('--profile', choices=['fast', 'balanced', 'quality'], 
                       default='quality', help="Pipeline profile to use")
    parser.add_argument('--input-dir', default='data/input_pdfs',
                       help="Directory containing PDF files")
    parser.add_argument('--gold-csv', default='data/gold_data/gold_pages.csv',
                       help="Path to gold pages CSV")
    parser.add_argument('--output-dir', default='reports',
                       help="Output directory for results")
    parser.add_argument('--check-targets', action='store_true',
                       help="Check if results meet target thresholds")
    
    args = parser.parse_args()
    
    try:
        # Run evaluation
        evaluator = GoldPageEvaluator(
            profile_name=args.profile,
            input_dir=args.input_dir,
            gold_csv=args.gold_csv,
            output_dir=args.output_dir
        )
        
        summary = evaluator.run_evaluation()
        evaluator.save_results()
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"Gold Standard Evaluation Complete")
        print(f"{'='*60}")
        print(f"Profile: {args.profile}")
        print(f"Pages processed: {summary['processed_pages']}")
        print(f"Overall CER: {summary['overall_cer']:.3f}")
        print(f"Overall WER: {summary['overall_wer']:.3f}")
        print(f"Results saved to: {evaluator.run_dir}")
        
        # Check targets if requested
        if args.check_targets:
            # Simple target checking (can be enhanced with targets.yaml)
            target_cer = 0.1  # 10% target
            target_wer = 0.2  # 20% target
            
            cer_pass = summary['overall_cer'] <= target_cer
            wer_pass = summary['overall_wer'] <= target_wer
            
            print(f"\nTarget Check:")
            print(f"CER target ({target_cer:.1%}): {'PASS' if cer_pass else 'FAIL'}")
            print(f"WER target ({target_wer:.1%}): {'PASS' if wer_pass else 'FAIL'}")
            
            if not (cer_pass and wer_pass):
                print("WARNING: Some targets not met")
                sys.exit(1)
            else:
                print("All targets met!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()