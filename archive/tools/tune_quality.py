#!/usr/bin/env python3
"""
Quality-focused parameter tuning for OCR pipeline.
Grid search optimization to achieve CER ≤ 0.10 and WER ≤ 0.10 on gold pages.
"""
import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import hashlib

import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "production"))

from src.csv_writer import calculate_cer_wer
from production.comprehensive_pipeline import ComprehensivePipeline, PipelineConfig

def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        # Return basic default config
        return {
            "profile": "quality",
            "ocr": {"engine": "router", "dpi": 300},
            "llm": {"enabled": False}
        }

logger = logging.getLogger(__name__)


class QualityTuner:
    """Grid search tuner for quality-focused OCR optimization."""
    
    def __init__(self, config_base: Dict[str, Any], gold_csv_path: str, 
                 input_dir: str, limit_pdfs: int = 2, seed: int = 17):
        """
        Initialize the quality tuner.
        
        Args:
            config_base: Base configuration dict
            gold_csv_path: Path to gold standard CSV
            input_dir: Directory containing input PDFs
            limit_pdfs: Number of PDFs to use for tuning
            seed: Random seed for reproducibility
        """
        self.config_base = config_base.copy()
        self.gold_csv_path = Path(gold_csv_path)
        self.input_dir = Path(input_dir)
        self.limit_pdfs = limit_pdfs
        self.seed = seed
        
        # Setup output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path("reports") / f"tuning_quality_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Target metrics
        self.target_cer = 0.10
        self.target_wer = 0.10
        
        # Best configuration tracking
        self.best_config = None
        self.best_metrics = {'cer': float('inf'), 'wer': float('inf')}
        self.best_combo_id = None
        
        # Load and prepare gold data
        self.gold_data = self._load_gold_data()
        self.selected_pdfs = self._select_tuning_pdfs()
        
        logger.info(f"QualityTuner initialized: {len(self.selected_pdfs)} PDFs, output: {self.output_dir}")
    
    def _load_gold_data(self) -> pd.DataFrame:
        """Load and validate gold standard data."""
        try:
            gold_df = pd.read_csv(self.gold_csv_path)
            
            # Handle the actual CSV format: "PDF LINK,PAGE,HANDTYPED"
            if 'PDF LINK' in gold_df.columns:
                gold_df = gold_df.rename(columns={
                    'PDF LINK': 'pdf_file',
                    'PAGE': 'page_num', 
                    'HANDTYPED': 'gold_text'
                })
            
            required_cols = ['pdf_file', 'page_num', 'gold_text']
            for col in required_cols:
                if col not in gold_df.columns:
                    raise ValueError(f"Missing required column: {col}")
            
            # Clean up the data
            gold_df = gold_df.dropna(subset=['pdf_file', 'page_num', 'gold_text'])
            gold_df['page_num'] = pd.to_numeric(gold_df['page_num'], errors='coerce')
            gold_df = gold_df.dropna(subset=['page_num'])
            gold_df['page_num'] = gold_df['page_num'].astype(int)
            
            # Remove quotes from PDF filenames if present
            gold_df['pdf_file'] = gold_df['pdf_file'].str.strip('"').str.strip("'")
            
            # Filter to PDFs that exist in input directory
            available_pdfs = set()
            for pdf_path in self.input_dir.glob("*.pdf"):
                available_pdfs.add(pdf_path.name)
            for pdf_path in self.input_dir.glob("*.PDF"):
                available_pdfs.add(pdf_path.name)
            
            # Case-insensitive matching
            available_pdfs_lower = {pdf.lower(): pdf for pdf in available_pdfs}
            
            def find_matching_pdf(pdf_name):
                # Direct match
                if pdf_name in available_pdfs:
                    return pdf_name
                # Case-insensitive match
                if pdf_name.lower() in available_pdfs_lower:
                    return available_pdfs_lower[pdf_name.lower()]
                return None
            
            gold_df['matched_pdf'] = gold_df['pdf_file'].apply(find_matching_pdf)
            gold_df = gold_df[gold_df['matched_pdf'].notna()]
            gold_df['pdf_file'] = gold_df['matched_pdf']
            gold_df = gold_df.drop(columns=['matched_pdf'])
            
            logger.info(f"Loaded gold data: {len(gold_df)} pages from {gold_df['pdf_file'].nunique()} PDFs")
            logger.info(f"Available PDFs: {sorted(gold_df['pdf_file'].unique())}")
            return gold_df
            
        except Exception as e:
            logger.error(f"Failed to load gold data: {e}")
            raise
    
    def _select_tuning_pdfs(self) -> List[str]:
        """Select PDFs with most gold rows for tuning."""
        # Count pages per PDF
        pdf_counts = self.gold_data['pdf_file'].value_counts()
        
        # Select top PDFs with most pages
        top_pdfs = pdf_counts.head(self.limit_pdfs).index.tolist()
        
        # Ensure deterministic selection with seed
        np.random.seed(self.seed)
        if len(top_pdfs) > self.limit_pdfs:
            top_pdfs = np.random.choice(top_pdfs, self.limit_pdfs, replace=False).tolist()
        
        logger.info(f"Selected PDFs for tuning: {top_pdfs}")
        return top_pdfs
    
    def _generate_parameter_grid(self) -> List[Dict[str, Any]]:
        """Generate parameter combinations for grid search."""
        # Define parameter ranges (from most impactful to least)
        # Start with smaller grid for testing
        param_grid = {
            'render_dpi': [300, 400],  # Reduced from [300, 360, 400]
            'det_db_box_thresh': [0.35],  # Reduced from [0.25, 0.35, 0.45]
            'det_db_unclip_ratio': [2.0],  # Reduced from [1.5, 2.0, 2.5]
            'router_threshold_tr': [0.86, 0.88],  # Reduced from [0.86, 0.88, 0.90]
            'beam_size': [5],  # Reduced from [5, 10]
            'wbf_union': [False]  # Reduced from [True, False]
        }
        
        # Generate all combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        combinations = []
        for combo in product(*param_values):
            param_dict = dict(zip(param_names, combo))
            combinations.append(param_dict)
        
        logger.info(f"Generated {len(combinations)} parameter combinations")
        return combinations
    
    def _create_config_for_combo(self, combo: Dict[str, Any]) -> Dict[str, Any]:
        """Create full configuration for a parameter combination."""
        config = self.config_base.copy()
        
        # Always use quality profile
        config['profile'] = 'quality'
        
        # Render settings
        config['render'] = config.get('render', {})
        config['render']['dpi'] = combo['render_dpi']
        config['render']['min_dpi'] = 300
        
        # OCR engine and settings
        config['ocr']['engine'] = 'router'
        config['ocr']['dpi'] = combo['render_dpi']
        
        # LLM disabled
        config['llm']['enabled'] = False
        config['llm']['enable_correction'] = False
        
        # Orientation (always full 360°)
        config['orientation'] = {
            'enable': True,
            'coarse_full360': True,
            'fine_deg': 2.0,
            'fine_step_deg': 0.1
        }
        
        # Detector settings
        config['detector'] = {
            'engine': 'mmocr_dbnetpp',
            'wbf_union': combo['wbf_union'],
            'det_db_box_thresh': combo['det_db_box_thresh'],
            'det_db_unclip_ratio': combo['det_db_unclip_ratio'],
            'min_box_size_px': 12,
            'scales': [1.0, 1.5],
            'wbf_iou_threshold': 0.55
        }
        
        # Router settings
        config['router'] = {
            'primary': 'abinet',
            'fallback': 'parseq',
            'ensemble': ['abinet', 'parseq', 'doctr_sar'],
            'thresholds': {
                'en': 0.92,
                'de': 0.90,
                'fr': 0.90,
                'it': 0.90,
                'tr': combo['router_threshold_tr']
            },
            'delta_disagree': 0.04,
            'beam_size': combo['beam_size']
        }
        
        # Preprocessing settings
        config['preprocessing'] = {
            'clahe_clip_limit': 3.5,
            'bilateral_filter': True,
            'bilateral_d': 7,
            'bilateral_sigma_color': 50,
            'bilateral_sigma_space': 50,
            'sauvola_k': 0.3,
            'crop_pad_percent': 0.025
        }
        
        return config
    
    def _run_ocr_on_combo(self, combo_id: int, combo: Dict[str, Any]) -> Dict[str, Any]:
        """Run OCR pipeline on a parameter combination."""
        logger.info(f"Running combo {combo_id}: {combo}")
        
        # Create configuration
        config = self._create_config_for_combo(combo)
        
        # Create output directory for this combo
        combo_dir = self.output_dir / f"combo_{combo_id:03d}"
        combo_dir.mkdir(exist_ok=True)
        
        # Save configuration
        config_file = combo_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Initialize pipeline
        try:
            # Convert config dict to PipelineConfig
            pipeline_config = PipelineConfig(
                llm_provider="none" if not config.get('llm', {}).get('enabled', False) else "ollama",
                dpi=config.get('render', {}).get('dpi', 300),
                enable_llm_correction=config.get('llm', {}).get('enabled', False),
                performance_profile=config.get('profile', 'quality')
            )
            
            pipeline = ComprehensivePipeline(pipeline_config)
            
            all_results = []
            total_pages = 0
            start_time = time.time()
            
            # Process each selected PDF
            for pdf_name in self.selected_pdfs:
                pdf_path = self.input_dir / pdf_name
                if not pdf_path.exists():
                    logger.warning(f"PDF not found: {pdf_path}")
                    continue
                
                # Get gold pages for this PDF
                pdf_gold = self.gold_data[self.gold_data['pdf_file'] == pdf_name]
                
                logger.info(f"Processing {pdf_name} ({len(pdf_gold)} gold pages)")
                
                # Process PDF
                try:
                    pdf_results = pipeline.process_pdf(str(pdf_path))
                    
                    # Match results with gold data
                    for _, gold_row in pdf_gold.iterrows():
                        page_num = gold_row['page_num']
                        gold_text = gold_row['gold_text']
                        
                        # Find corresponding OCR result
                        ocr_text = ""
                        confidence = 0.0
                        
                        if page_num < len(pdf_results['pages']):
                            page_result = pdf_results['pages'][page_num]
                            ocr_text = page_result.get('text', '')
                            confidence = page_result.get('avg_confidence', 0.0)
                        
                        all_results.append({
                            'pdf_file': pdf_name,
                            'page_num': page_num,
                            'gold_text': gold_text,
                            'ocr_text': ocr_text,
                            'confidence': confidence
                        })
                        total_pages += 1
                
                except Exception as e:
                    logger.error(f"Failed to process {pdf_name}: {e}")
                    continue
            
            elapsed = time.time() - start_time
            
            if not all_results:
                logger.error(f"No results for combo {combo_id}")
                return {
                    'combo_id': combo_id,
                    'combo': combo,
                    'success': False,
                    'error': 'No results generated'
                }
            
            # Calculate metrics
            cer_values = []
            wer_values = []
            
            for result in all_results:
                cer, wer = calculate_cer_wer(result['gold_text'], result['ocr_text'])
                cer_values.append(cer)
                wer_values.append(wer)
                result['cer'] = cer
                result['wer'] = wer
            
            # Overall metrics
            overall_cer = np.mean(cer_values)
            overall_wer = np.mean(wer_values)
            time_per_page = elapsed / max(total_pages, 1)
            
            # Save detailed results
            results_csv = combo_dir / "metrics.csv"
            pd.DataFrame(all_results).to_csv(results_csv, index=False)
            
            # Create summary
            summary = {
                'combo_id': combo_id,
                'parameters': combo,
                'metrics': {
                    'cer': overall_cer,
                    'wer': overall_wer,
                    'pages_processed': total_pages,
                    'time_per_page': time_per_page,
                    'total_time': elapsed
                },
                'success': True,
                'target_met': overall_cer <= self.target_cer and overall_wer <= self.target_wer
            }
            
            # Save summary
            summary_file = combo_dir / "summary.md"
            self._write_summary_markdown(summary_file, summary, all_results)
            
            logger.info(f"Combo {combo_id}: CER={overall_cer:.4f}, WER={overall_wer:.4f}, "
                       f"target_met={summary['target_met']}")
            
            return summary
            
        except Exception as e:
            logger.error(f"Combo {combo_id} failed: {e}")
            return {
                'combo_id': combo_id,
                'combo': combo,
                'success': False,
                'error': str(e)
            }
    
    def _write_summary_markdown(self, summary_file: Path, summary: Dict[str, Any], 
                               results: List[Dict[str, Any]]):
        """Write summary markdown report for a combination."""
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"# Combo {summary['combo_id']} Summary\n\n")
            
            # Parameters
            f.write("## Parameters\n\n")
            for key, value in summary['parameters'].items():
                f.write(f"- **{key}**: {value}\n")
            f.write("\n")
            
            # Metrics
            metrics = summary['metrics']
            f.write("## Results\n\n")
            f.write(f"- **CER**: {metrics['cer']:.4f}\n")
            f.write(f"- **WER**: {metrics['wer']:.4f}\n")
            f.write(f"- **Pages Processed**: {metrics['pages_processed']}\n")
            f.write(f"- **Time per Page**: {metrics['time_per_page']:.2f}s\n")
            f.write(f"- **Total Time**: {metrics['total_time']:.2f}s\n")
            f.write(f"- **Target Met**: {'✅ YES' if summary['target_met'] else '❌ NO'}\n\n")
            
            # Target comparison
            f.write("## Target Comparison\n\n")
            f.write(f"- **Target CER**: ≤ {self.target_cer:.2f}\n")
            f.write(f"- **Actual CER**: {metrics['cer']:.4f} "
                   f"({'✅' if metrics['cer'] <= self.target_cer else '❌'})\n")
            f.write(f"- **Target WER**: ≤ {self.target_wer:.2f}\n")
            f.write(f"- **Actual WER**: {metrics['wer']:.4f} "
                   f"({'✅' if metrics['wer'] <= self.target_wer else '❌'})\n\n")
            
            # Per-page breakdown
            if results:
                f.write("## Per-Page Results\n\n")
                f.write("| PDF | Page | CER | WER | Confidence |\n")
                f.write("|-----|------|-----|-----|------------|\n")
                
                for result in results[:10]:  # Limit to first 10 for readability
                    f.write(f"| {result['pdf_file']} | {result['page_num']} | "
                           f"{result['cer']:.3f} | {result['wer']:.3f} | "
                           f"{result['confidence']:.3f} |\n")
                
                if len(results) > 10:
                    f.write(f"\n... and {len(results) - 10} more pages\n")
    
    def _update_best_config(self, summary: Dict[str, Any]):
        """Update best configuration if this one is better."""
        if not summary['success']:
            return
        
        metrics = summary['metrics']
        cer = metrics['cer']
        wer = metrics['wer']
        
        # Check if this beats current best (prioritize CER, then WER, then time)
        is_better = False
        
        if cer < self.best_metrics['cer']:
            is_better = True
        elif cer == self.best_metrics['cer'] and wer < self.best_metrics['wer']:
            is_better = True
        elif (cer == self.best_metrics['cer'] and wer == self.best_metrics['wer'] and
              metrics['time_per_page'] < self.best_metrics.get('time_per_page', float('inf'))):
            is_better = True
        
        if is_better:
            self.best_metrics = {
                'cer': cer,
                'wer': wer,
                'time_per_page': metrics['time_per_page'],
                'pages_processed': metrics['pages_processed']
            }
            self.best_config = self._create_config_for_combo(summary['parameters'])
            self.best_combo_id = summary['combo_id']
            
            logger.info(f"New best config: Combo {self.best_combo_id} "
                       f"(CER={cer:.4f}, WER={wer:.4f})")
    
    def run_tuning(self) -> Dict[str, Any]:
        """Run the complete tuning process."""
        logger.info("Starting quality tuning process")
        
        # Generate parameter combinations
        combinations = self._generate_parameter_grid()
        
        # Track all results
        all_summaries = []
        leaderboard_data = []
        
        # Early stopping flag
        target_met = False
        
        for i, combo in enumerate(combinations):
            logger.info(f"Processing combination {i+1}/{len(combinations)}")
            
            # Run OCR on this combination
            summary = self._run_ocr_on_combo(i, combo)
            all_summaries.append(summary)
            
            if summary['success']:
                # Update best configuration
                self._update_best_config(summary)
                
                # Add to leaderboard
                metrics = summary['metrics']
                leaderboard_data.append({
                    'combo_id': i,
                    'cer': metrics['cer'],
                    'wer': metrics['wer'],
                    'time_per_page': metrics['time_per_page'],
                    'target_met': summary['target_met'],
                    **combo  # Add all parameters
                })
                
                # Check for early stopping
                if summary['target_met'] and not target_met:
                    target_met = True
                    logger.info(f"🎯 TARGET ACHIEVED! Combo {i} reached CER ≤ {self.target_cer} "
                               f"and WER ≤ {self.target_wer}")
                    
                    # Save best config immediately
                    best_config_file = self.output_dir / "best_config.json"
                    with open(best_config_file, 'w') as f:
                        json.dump(self.best_config, f, indent=2)
                    
                    # Continue with remaining combinations for completeness
                    # but we could break here for early stopping
        
        # Create leaderboard
        leaderboard_df = pd.DataFrame(leaderboard_data)
        if not leaderboard_df.empty:
            # Sort by CER first, then WER, then time
            leaderboard_df = leaderboard_df.sort_values(
                ['cer', 'wer', 'time_per_page'], 
                ascending=[True, True, True]
            )
        
        leaderboard_file = self.output_dir / "leaderboard.csv"
        leaderboard_df.to_csv(leaderboard_file, index=False)
        
        # Save final best config
        if self.best_config:
            best_config_file = self.output_dir / "best_config.json"
            with open(best_config_file, 'w') as f:
                json.dump(self.best_config, f, indent=2)
        
        # Generate final report
        final_report = self._generate_final_report(all_summaries, leaderboard_df)
        
        return {
            'success': True,
            'target_achieved': target_met,
            'best_config_path': str(self.output_dir / "best_config.json") if self.best_config else None,
            'leaderboard_path': str(leaderboard_file),
            'output_directory': str(self.output_dir),
            'best_metrics': self.best_metrics,
            'combinations_tested': len(combinations),
            'successful_combinations': len(leaderboard_data)
        }
    
    def _generate_final_report(self, summaries: List[Dict[str, Any]], 
                              leaderboard_df: pd.DataFrame) -> str:
        """Generate final tuning report."""
        report_file = self.output_dir / "tuning_report.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# OCR Quality Tuning Report\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overview
            f.write("## Overview\n\n")
            f.write(f"- **Target CER**: ≤ {self.target_cer:.2f}\n")
            f.write(f"- **Target WER**: ≤ {self.target_wer:.2f}\n")
            f.write(f"- **PDFs Used**: {len(self.selected_pdfs)}\n")
            f.write(f"- **Combinations Tested**: {len(summaries)}\n")
            f.write(f"- **Successful Runs**: {len(leaderboard_df)}\n\n")
            
            # Best result
            if self.best_config:
                f.write("## Best Configuration\n\n")
                f.write(f"- **Combo ID**: {self.best_combo_id}\n")
                f.write(f"- **CER**: {self.best_metrics['cer']:.4f}\n")
                f.write(f"- **WER**: {self.best_metrics['wer']:.4f}\n")
                f.write(f"- **Time per Page**: {self.best_metrics['time_per_page']:.2f}s\n")
                target_met = (self.best_metrics['cer'] <= self.target_cer and 
                             self.best_metrics['wer'] <= self.target_wer)
                f.write(f"- **Target Met**: {'✅ YES' if target_met else '❌ NO'}\n\n")
                
                # Reproduce command
                f.write("### Reproduce Best Result\n\n")
                f.write("```bash\n")
                f.write(f"python tools/run_baseline_eval.py \\\n")
                f.write(f"  --config {self.output_dir}/best_config.json \\\n")
                f.write(f"  --gold-csv {self.gold_csv_path} \\\n")
                f.write(f"  --input-dir {self.input_dir} \\\n")
                f.write(f"  --limit-pdfs {self.limit_pdfs} \\\n")
                f.write(f"  --report-md\n")
                f.write("```\n\n")
            
            # Top 5 results
            if not leaderboard_df.empty:
                f.write("## Top 5 Results\n\n")
                f.write("| Rank | Combo | CER | WER | Time/Page | Target Met |\n")
                f.write("|------|-------|-----|-----|-----------|------------|\n")
                
                for i, (_, row) in enumerate(leaderboard_df.head(5).iterrows()):
                    target_met_icon = "✅" if row['target_met'] else "❌"
                    f.write(f"| {i+1} | {row['combo_id']} | {row['cer']:.4f} | "
                           f"{row['wer']:.4f} | {row['time_per_page']:.2f}s | {target_met_icon} |\n")
                f.write("\n")
            
            # Parameter impact analysis
            if not leaderboard_df.empty:
                f.write("## Parameter Impact Analysis\n\n")
                
                # Analyze each parameter's effect on CER
                for param in ['render_dpi', 'det_db_box_thresh', 'det_db_unclip_ratio', 
                             'router_threshold_tr', 'beam_size', 'wbf_union']:
                    if param in leaderboard_df.columns:
                        param_analysis = leaderboard_df.groupby(param)['cer'].agg(['mean', 'std', 'count'])
                        f.write(f"### {param}\n\n")
                        f.write("| Value | Avg CER | Std CER | Count |\n")
                        f.write("|-------|---------|---------|-------|\n")
                        
                        for value, stats in param_analysis.iterrows():
                            f.write(f"| {value} | {stats['mean']:.4f} | "
                                   f"{stats['std']:.4f} | {stats['count']} |\n")
                        f.write("\n")
        
        logger.info(f"Final report saved: {report_file}")
        return str(report_file)


def main():
    """Main entry point for quality tuning."""
    parser = argparse.ArgumentParser(description="OCR Quality Tuning System")
    parser.add_argument('--gold-csv', required=True, help='Path to gold standard CSV')
    parser.add_argument('--input-dir', required=True, help='Directory containing input PDFs')
    parser.add_argument('--limit-pdfs', type=int, default=2, help='Number of PDFs to use')
    parser.add_argument('--seed', type=int, default=17, help='Random seed')
    parser.add_argument('--config', help='Base configuration file')
    parser.add_argument('--report-md', action='store_true', help='Generate markdown reports')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load base configuration
    if args.config:
        with open(args.config, 'r') as f:
            config_base = json.load(f)
    else:
        config_base = load_config()
    
    try:
        # Initialize tuner
        tuner = QualityTuner(
            config_base=config_base,
            gold_csv_path=args.gold_csv,
            input_dir=args.input_dir,
            limit_pdfs=args.limit_pdfs,
            seed=args.seed
        )
        
        # Run tuning
        results = tuner.run_tuning()
        
        # Print final results
        print("\n" + "="*60)
        print("🎯 QUALITY TUNING COMPLETED")
        print("="*60)
        
        if results['best_config_path']:
            print(f"📄 Best Config: {results['best_config_path']}")
            print(f"📊 Best CER: {tuner.best_metrics['cer']:.4f}")
            print(f"📊 Best WER: {tuner.best_metrics['wer']:.4f}")
            
            target_met = (tuner.best_metrics['cer'] <= tuner.target_cer and 
                         tuner.best_metrics['wer'] <= tuner.target_wer)
            print(f"🎯 Target Achieved: {'✅ YES' if target_met else '❌ NO'}")
        
        print(f"📈 Leaderboard: {results['leaderboard_path']}")
        print(f"📁 Full Results: {results['output_directory']}")
        
        # Reproduce command
        if results['best_config_path']:
            print(f"\n🔄 Reproduce Best Result:")
            print(f"python tools/run_baseline_eval.py \\")
            print(f"  --config {results['best_config_path']} \\")
            print(f"  --gold-csv {args.gold_csv} \\")
            print(f"  --input-dir {args.input_dir} \\")
            print(f"  --limit-pdfs {args.limit_pdfs} \\")
            print(f"  --report-md")
        
        return 0 if results['success'] else 1
        
    except Exception as e:
        logger.error(f"Tuning failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())