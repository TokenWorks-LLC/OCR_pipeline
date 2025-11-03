"""
A/B Gold Standard Evaluation Tool
Compares OCR pipeline performance with LLM correction OFF vs ON.
Generates Excel report with metrics, deltas, and gating recommendations.
"""

import sys
import csv
import time
import argparse
import statistics
from pathlib import Path
from typing import Dict, List, Tuple
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from production.comprehensive_pipeline import ComprehensivePipeline, PipelineConfig
from cer_evaluation import calculate_cer, calculate_wer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_gold_data(gold_csv_path: str) -> List[Dict]:
    """Load gold standard data with page range support."""
    gold_pages = []
    with open(gold_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            page_str = row['PAGE'].strip()
            gold_text = row['HANDTYPED']
            pdf_name = row['PDF LINK']
            
            # Handle page ranges like "33-33" or "6-7"
            if '-' in page_str:
                parts = page_str.split('-')
                start_page = int(parts[0].strip())
                end_page = int(parts[1].strip())
                
                gold_pages.append({
                    'pdf_name': pdf_name,
                    'page_number': start_page,
                    'end_page': end_page,
                    'gold_text': gold_text,
                    'page_span': page_str,
                    'partial_span': start_page != end_page
                })
            else:
                page_num = int(page_str)
                gold_pages.append({
                    'pdf_name': pdf_name,
                    'page_number': page_num,
                    'end_page': page_num,
                    'gold_text': gold_text,
                    'page_span': page_str,
                    'partial_span': False
                })
    return gold_pages


def run_evaluation_pass(
    gold_pages: List[Dict],
    input_pdfs_dir: Path,
    output_dir: Path,
    llm_enabled: bool,
    llm_model: str = "qwen2.5:7b-instruct"
) -> List[Dict]:
    """
    Run single evaluation pass with LLM enabled or disabled.
    
    Args:
        gold_pages: Gold standard data
        input_pdfs_dir: Directory with PDF files
        output_dir: Output directory for this pass
        llm_enabled: Whether to enable LLM correction
        llm_model: LLM model to use if enabled
    
    Returns:
        List of result dictionaries
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create pipeline configuration
    config = PipelineConfig(
        enable_new_llm_correction=llm_enabled,
        llm_correction_model=llm_model,
        llm_correction_edit_budget=0.12,
        llm_correction_cache_enabled=True,
        enable_reading_order=True,
        dpi=300
    )
    
    pipeline = ComprehensivePipeline(config)
    status = "ENABLED" if llm_enabled else "DISABLED"
    logger.info(f"✅ Pipeline initialized - LLM correction: {status}")
    
    results = []
    start_time = time.time()
    
    # Process each gold page
    for idx, gold_page in enumerate(gold_pages, 1):
        pdf_name = gold_page['pdf_name']
        page_num = gold_page['page_number']
        end_page = gold_page.get('end_page', page_num)
        gold_text = gold_page['gold_text']
        page_span = gold_page.get('page_span', str(page_num))
        is_multipage = gold_page.get('partial_span', False)
        
        logger.info(f"[{idx}/{len(gold_pages)}] Processing: {pdf_name}, page {page_span}")
        
        # Find PDF file
        pdf_path = input_pdfs_dir / pdf_name
        if not pdf_path.exists():
            for ext in ['.pdf', '.PDF']:
                alt_path = input_pdfs_dir / f"{pdf_name}{ext}"
                if alt_path.exists():
                    pdf_path = alt_path
                    break
        
        if not pdf_path.exists():
            logger.warning(f"⚠️  PDF not found: {pdf_path}")
            continue
        
        try:
            # Use safe_path for output directory
            try:
                from utils.shortpath import safe_path, ensure_dir
                page_output_dir = Path(safe_path(
                    str(output_dir),
                    f"{pdf_path.stem}_page_{page_span.replace('-', '_')}"
                ))
                ensure_dir(str(page_output_dir))
            except ImportError:
                # Fallback if shortpath not available
                page_output_dir = output_dir / f"{pdf_path.stem}_page_{page_span.replace('-', '_')}"
                page_output_dir.mkdir(exist_ok=True)
            
            # Process page range
            pipeline.process_pdf(
                pdf_path=str(pdf_path),
                output_dir=str(page_output_dir),
                start_page=page_num,
                end_page=end_page
            )
            
            # Read generated CSV
            csv_path = page_output_dir / "comprehensive_results.csv"
            if not csv_path.exists():
                logger.warning(f"⚠️  Output CSV not found: {csv_path}")
                continue
            
            # Extract OCR text - concatenate all rows if multi-page span
            ocr_text_parts = []
            correction_stats = None
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    page_text = row.get('page_text', '').strip()
                    if page_text:
                        ocr_text_parts.append(page_text)
                    if 'correction_stats' in row and not correction_stats:
                        import json
                        try:
                            correction_stats = json.loads(row['correction_stats'])
                        except:
                            pass
            
            # Concatenate with newline if multi-page
            ocr_text = '\n'.join(ocr_text_parts) if is_multipage else (ocr_text_parts[0] if ocr_text_parts else "")
            
            # Calculate metrics
            cer = calculate_cer(gold_text, ocr_text)
            wer = calculate_wer(gold_text, ocr_text)
            
            # Check for mojibake
            try:
                from utils.encoding import has_mojibake
                has_mojibake_flag = has_mojibake(ocr_text)
            except ImportError:
                has_mojibake_flag = False
            
            # Outlier detection
            is_outlier = cer > 10.0
            
            # Check if Akkadian
            is_akkadian = any(c in gold_text for c in ['š', 'ṣ', 'ṭ', 'ḫ', 'ā', 'ē', 'ī', 'ū'])
            
            result = {
                'pdf_name': pdf_name,
                'page_span': page_span,
                'cer': cer,
                'wer': wer,
                'is_akkadian': is_akkadian,
                'llm_corrections': correction_stats.get('corrections_applied', 0) if correction_stats else 0,
                'ref_len': len(gold_text),
                'ocr_len': len(ocr_text),
                'len_ratio': len(ocr_text) / len(gold_text) if len(gold_text) > 0 else 0.0,
                'is_outlier': is_outlier,
                'is_multipage': is_multipage,
                'has_mojibake': has_mojibake_flag
            }
            results.append(result)
            
            logger.info(f"  CER: {cer:.2%}, WER: {wer:.2%}, LLM corrections: {result['llm_corrections']}")
            
        except Exception as e:
            logger.error(f"❌ Error processing {pdf_name} page {page_span}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    elapsed = time.time() - start_time
    logger.info(f"Pass complete: {len(results)}/{len(gold_pages)} pages, {elapsed:.1f}s")
    
    return results


def save_results_csv(results: List[Dict], output_path: Path, encoding: str = 'utf-8'):
    """Save results to CSV with proper encoding."""
    if not results:
        logger.warning(f"No results to save to {output_path}")
        return
    
    with open(output_path, 'w', newline='', encoding=encoding) as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    logger.info(f"✅ Saved {len(results)} results to {output_path}")


def calculate_aggregate_metrics(results: List[Dict]) -> Dict:
    """Calculate aggregate statistics from results."""
    if not results:
        return {}
    
    # Filter out outliers for cleaner stats
    clean_results = [r for r in results if not r.get('is_outlier', False)]
    
    cer_values = [r['cer'] for r in clean_results]
    wer_values = [r['wer'] for r in clean_results]
    
    metrics = {
        'total_pages': len(results),
        'clean_pages': len(clean_results),
        'outliers': len(results) - len(clean_results),
        'median_cer': statistics.median(cer_values) if cer_values else 0.0,
        'mean_cer': statistics.mean(cer_values) if cer_values else 0.0,
        'median_wer': statistics.median(wer_values) if wer_values else 0.0,
        'mean_wer': statistics.mean(wer_values) if wer_values else 0.0,
        'llm_corrections_total': sum(r.get('llm_corrections', 0) for r in results),
        'pages_with_llm': sum(1 for r in results if r.get('llm_corrections', 0) > 0),
        'mojibake_pages': sum(1 for r in results if r.get('has_mojibake', False)),
        'akkadian_pages': sum(1 for r in results if r.get('is_akkadian', False))
    }
    
    return metrics


def generate_ab_summary(
    metrics_off: Dict,
    metrics_on: Dict,
    output_path: Path
):
    """Generate markdown summary of A/B comparison."""
    
    # Calculate deltas
    cer_delta = metrics_on['median_cer'] - metrics_off['median_cer']
    wer_delta = metrics_on['median_wer'] - metrics_off['median_wer']
    
    # Gating decision
    cer_improvement = -cer_delta  # Negative delta = improvement
    wer_improvement = -wer_delta
    
    enable_llm = (cer_improvement >= 0.03) or (wer_improvement >= 0.05)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# A/B Evaluation Summary: LLM Correction OFF vs ON\n\n")
        f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Model**: qwen2.5:7b-instruct\n\n")
        f.write("---\n\n")
        
        f.write("## Overall Metrics\n\n")
        f.write("| Metric | LLM OFF | LLM ON | Delta | Change |\n")
        f.write("|--------|---------|--------|-------|--------|\n")
        f.write(f"| **Median CER** | {metrics_off['median_cer']:.2%} | {metrics_on['median_cer']:.2%} | "
                f"{cer_delta:+.2%} | {cer_improvement:+.2%} |\n")
        f.write(f"| **Mean CER** | {metrics_off['mean_cer']:.2%} | {metrics_on['mean_cer']:.2%} | "
                f"{metrics_on['mean_cer'] - metrics_off['mean_cer']:+.2%} | |\n")
        f.write(f"| **Median WER** | {metrics_off['median_wer']:.2%} | {metrics_on['median_wer']:.2%} | "
                f"{wer_delta:+.2%} | {wer_improvement:+.2%} |\n")
        f.write(f"| **Mean WER** | {metrics_off['mean_wer']:.2%} | {metrics_on['mean_wer']:.2%} | "
                f"{metrics_on['mean_wer'] - metrics_off['mean_wer']:+.2%} | |\n")
        f.write(f"| **Pages Processed** | {metrics_off['total_pages']} | {metrics_on['total_pages']} | | |\n")
        f.write(f"| **Outliers** | {metrics_off['outliers']} | {metrics_on['outliers']} | | |\n")
        f.write("\n")
        
        f.write("## LLM Correction Activity\n\n")
        f.write(f"- **Total LLM Corrections**: {metrics_on['llm_corrections_total']}\n")
        f.write(f"- **Pages with LLM Corrections**: {metrics_on['pages_with_llm']}/{metrics_on['total_pages']}\n")
        f.write(f"- **Mojibake Pages (LLM OFF)**: {metrics_off['mojibake_pages']}\n")
        f.write(f"- **Mojibake Pages (LLM ON)**: {metrics_on['mojibake_pages']}\n")
        f.write(f"- **Akkadian Pages**: {metrics_on['akkadian_pages']}\n\n")
        
        f.write("---\n\n")
        f.write("## Gating Decision\n\n")
        f.write("**Criteria**:\n")
        f.write("- Enable LLM if CER improves by ≥3% **OR** WER improves by ≥5%\n\n")
        f.write("**Results**:\n")
        f.write(f"- CER improvement: {cer_improvement:+.2%} {'✅ PASS' if cer_improvement >= 0.03 else '❌ FAIL'} (threshold: ≥3%)\n")
        f.write(f"- WER improvement: {wer_improvement:+.2%} {'✅ PASS' if wer_improvement >= 0.05 else '❌ FAIL'} (threshold: ≥5%)\n\n")
        
        if enable_llm:
            f.write("### ✅ **RECOMMENDATION: ENABLE LLM CORRECTION**\n\n")
            f.write(f"LLM correction shows measurable benefit:\n")
            if cer_improvement >= 0.03:
                f.write(f"- CER improved by {cer_improvement:.2%} (exceeds 3% threshold)\n")
            if wer_improvement >= 0.05:
                f.write(f"- WER improved by {wer_improvement:.2%} (exceeds 5% threshold)\n")
        else:
            f.write("### ❌ **RECOMMENDATION: SUPPRESS LLM CORRECTION**\n\n")
            f.write("LLM correction does not provide sufficient benefit:\n")
            f.write(f"- CER improvement: {cer_improvement:+.2%} (below 3% threshold)\n")
            f.write(f"- WER improvement: {wer_improvement:+.2%} (below 5% threshold)\n\n")
            f.write("**Possible reasons**:\n")
            f.write("- OCR quality already high (median CER < 10%)\n")
            f.write("- LLM making harmful edits\n")
            f.write("- Insufficient trigger sensitivity\n")
        
        f.write("\n---\n\n")
        f.write("## Files Generated\n\n")
        f.write("- `metrics_llm_off.csv` - Results with LLM disabled\n")
        f.write("- `metrics_llm_on.csv` - Results with LLM enabled\n")
        f.write("- `metrics.xlsx` - Excel workbook with all metrics\n")
        f.write("- `summary_llm_ab.md` - This summary report\n")
    
    logger.info(f"✅ Summary report saved to {output_path}")


def generate_excel_report(
    results_off: List[Dict],
    results_on: List[Dict],
    metrics_off: Dict,
    metrics_on: Dict,
    output_path: Path
):
    """Generate Excel workbook with multiple sheets."""
    try:
        import pandas as pd
    except ImportError:
        logger.warning("pandas not available, skipping Excel generation")
        return
    
    try:
        # Create Excel writer
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Sheet 1: Summary
            summary_data = {
                'Metric': ['Median CER', 'Mean CER', 'Median WER', 'Mean WER', 
                          'Pages Processed', 'Outliers', 'LLM Corrections', 
                          'Pages with LLM', 'Mojibake Pages'],
                'LLM OFF': [
                    f"{metrics_off['median_cer']:.4f}",
                    f"{metrics_off['mean_cer']:.4f}",
                    f"{metrics_off['median_wer']:.4f}",
                    f"{metrics_off['mean_wer']:.4f}",
                    metrics_off['total_pages'],
                    metrics_off['outliers'],
                    0,
                    0,
                    metrics_off['mojibake_pages']
                ],
                'LLM ON': [
                    f"{metrics_on['median_cer']:.4f}",
                    f"{metrics_on['mean_cer']:.4f}",
                    f"{metrics_on['median_wer']:.4f}",
                    f"{metrics_on['mean_wer']:.4f}",
                    metrics_on['total_pages'],
                    metrics_on['outliers'],
                    metrics_on['llm_corrections_total'],
                    metrics_on['pages_with_llm'],
                    metrics_on['mojibake_pages']
                ],
                'Delta': [
                    f"{metrics_on['median_cer'] - metrics_off['median_cer']:+.4f}",
                    f"{metrics_on['mean_cer'] - metrics_off['mean_cer']:+.4f}",
                    f"{metrics_on['median_wer'] - metrics_off['median_wer']:+.4f}",
                    f"{metrics_on['mean_wer'] - metrics_off['mean_wer']:+.4f}",
                    '',
                    '',
                    metrics_on['llm_corrections_total'],
                    metrics_on['pages_with_llm'],
                    metrics_on['mojibake_pages'] - metrics_off['mojibake_pages']
                ]
            }
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
            
            # Sheet 2: Detailed Results (LLM OFF)
            df_off = pd.DataFrame(results_off)
            df_off.to_excel(writer, sheet_name='LLM_OFF', index=False)
            
            # Sheet 3: Detailed Results (LLM ON)
            df_on = pd.DataFrame(results_on)
            df_on.to_excel(writer, sheet_name='LLM_ON', index=False)
            
            # Sheet 4: Per-Page Comparison
            if len(results_off) == len(results_on):
                comparison_data = []
                for off, on in zip(results_off, results_on):
                    comparison_data.append({
                        'pdf_name': off['pdf_name'],
                        'page_span': off['page_span'],
                        'cer_off': off['cer'],
                        'cer_on': on['cer'],
                        'cer_delta': on['cer'] - off['cer'],
                        'wer_off': off['wer'],
                        'wer_on': on['wer'],
                        'wer_delta': on['wer'] - off['wer'],
                        'llm_corrections': on.get('llm_corrections', 0)
                    })
                df_comparison = pd.DataFrame(comparison_data)
                df_comparison.to_excel(writer, sheet_name='Comparison', index=False)
        
        logger.info(f"✅ Excel report saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to generate Excel: {e}")


def main():
    parser = argparse.ArgumentParser(description='A/B Gold Standard Evaluation')
    parser.add_argument('--gold-csv', type=str, default='data/gold_data/gold_pages.csv',
                       help='Path to gold standard CSV')
    parser.add_argument('--input-pdfs', type=str, default='data/input_pdfs',
                       help='Directory with input PDFs')
    parser.add_argument('--output-dir', type=str, default='ab_evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--ab', action='store_true',
                       help='Run A/B test (LLM off, then LLM on)')
    parser.add_argument('--llm-enabled', type=str, choices=['true', 'false'], default='true',
                       help='Enable LLM correction (ignored if --ab is set)')
    parser.add_argument('--llm-model', type=str, default='qwen2.5:7b-instruct',
                       help='LLM model to use')
    parser.add_argument('--report-md', action='store_true',
                       help='Generate markdown summary report')
    parser.add_argument('--excel-out', type=str,
                       help='Output Excel file path (e.g., metrics.xlsx)')
    
    args = parser.parse_args()
    
    # Setup paths
    gold_csv = Path(args.gold_csv)
    input_pdfs_dir = Path(args.input_pdfs)
    output_base = Path(args.output_dir)
    
    if not gold_csv.exists():
        logger.error(f"Gold CSV not found: {gold_csv}")
        return 1
    
    if not input_pdfs_dir.exists():
        logger.error(f"Input PDFs directory not found: {input_pdfs_dir}")
        return 1
    
    # Load gold data
    logger.info(f"Loading gold data from {gold_csv}")
    gold_pages = load_gold_data(str(gold_csv))
    logger.info(f"Loaded {len(gold_pages)} gold pages")
    
    if args.ab:
        logger.info("=" * 60)
        logger.info("A/B EVALUATION MODE: Testing LLM OFF vs ON")
        logger.info("=" * 60)
        
        # Pass 1: LLM OFF
        logger.info("\n🔴 PASS 1: LLM CORRECTION DISABLED\n")
        output_off = output_base / "llm_off"
        results_off = run_evaluation_pass(
            gold_pages=gold_pages,
            input_pdfs_dir=input_pdfs_dir,
            output_dir=output_off,
            llm_enabled=False
        )
        
        # Save results
        csv_off = output_base / "metrics_llm_off.csv"
        save_results_csv(results_off, csv_off, encoding='utf-8')
        save_results_csv(results_off, output_base / "metrics_llm_off_bom.csv", encoding='utf-8-sig')
        
        # Pass 2: LLM ON
        logger.info("\n🟢 PASS 2: LLM CORRECTION ENABLED\n")
        output_on = output_base / "llm_on"
        results_on = run_evaluation_pass(
            gold_pages=gold_pages,
            input_pdfs_dir=input_pdfs_dir,
            output_dir=output_on,
            llm_enabled=True,
            llm_model=args.llm_model
        )
        
        # Save results
        csv_on = output_base / "metrics_llm_on.csv"
        save_results_csv(results_on, csv_on, encoding='utf-8')
        save_results_csv(results_on, output_base / "metrics_llm_on_bom.csv", encoding='utf-8-sig')
        
        # Calculate aggregate metrics
        metrics_off = calculate_aggregate_metrics(results_off)
        metrics_on = calculate_aggregate_metrics(results_on)
        
        # Generate reports
        if args.report_md:
            summary_md = output_base / "summary_llm_ab.md"
            generate_ab_summary(metrics_off, metrics_on, summary_md)
        
        if args.excel_out:
            excel_path = Path(args.excel_out)
            if not excel_path.is_absolute():
                excel_path = output_base / excel_path
            generate_excel_report(results_off, results_on, metrics_off, metrics_on, excel_path)
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("A/B EVALUATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"LLM OFF - Median CER: {metrics_off['median_cer']:.2%}, Median WER: {metrics_off['median_wer']:.2%}")
        logger.info(f"LLM ON  - Median CER: {metrics_on['median_cer']:.2%}, Median WER: {metrics_on['median_wer']:.2%}")
        cer_delta = metrics_on['median_cer'] - metrics_off['median_cer']
        wer_delta = metrics_on['median_wer'] - metrics_off['median_wer']
        logger.info(f"Delta   - CER: {cer_delta:+.2%}, WER: {wer_delta:+.2%}")
        logger.info(f"LLM Corrections Applied: {metrics_on['llm_corrections_total']}")
        logger.info("=" * 60)
        
    else:
        # Single pass mode
        llm_enabled = args.llm_enabled == 'true'
        logger.info(f"Running single pass with LLM {'ENABLED' if llm_enabled else 'DISABLED'}")
        
        results = run_evaluation_pass(
            gold_pages=gold_pages,
            input_pdfs_dir=input_pdfs_dir,
            output_dir=output_base,
            llm_enabled=llm_enabled,
            llm_model=args.llm_model
        )
        
        # Save results
        csv_path = output_base / "metrics.csv"
        save_results_csv(results, csv_path, encoding='utf-8')
        save_results_csv(results, output_base / "metrics_bom.csv", encoding='utf-8-sig')
        
        # Calculate metrics
        metrics = calculate_aggregate_metrics(results)
        logger.info(f"Median CER: {metrics['median_cer']:.2%}, Median WER: {metrics['median_wer']:.2%}")
    
    logger.info("\n✅ Evaluation complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
