#!/usr/bin/env python3
"""
Enhanced OCR evaluation tool with multi-engine support.
Evaluates different OCR engines (Paddle, docTR, MMOCR, Kraken) on gold standard data.
"""
import argparse
import csv
import json
import logging
import os
import re
import subprocess
import sys
import time
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_git_commit_sha() -> str:
    """Get current git commit SHA."""
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                              capture_output=True, text=True, check=True)
        return result.stdout.strip()[:8]
    except:
        return "unknown"

def detect_device() -> str:
    """Detect if running on GPU or CPU."""
    try:
        # Check if we're in Docker with GPU
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            return "gpu"
    except:
        pass
    return "cpu"

def normalize_text(text: str) -> str:
    """Normalize text for fair comparison."""
    if not text:
        return ""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    # Convert to lowercase for case-insensitive comparison
    text = text.lower()
    # Normalize quotes and dashes
    text = re.sub(r'[""''`´]', '"', text)
    text = re.sub(r'[–—]', '-', text)
    return text

def calculate_edit_distance(s1: str, s2: str) -> int:
    """Calculate edit distance using dynamic programming."""
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)
    
    len1, len2 = len(s1), len(s2)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    
    # Initialize base cases
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j
    
    # Fill the DP table
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[len1][len2]

def calculate_cer(reference: str, hypothesis: str) -> float:
    """Calculate Character Error Rate."""
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)
    
    if not ref_norm:
        return 0.0 if not hyp_norm else 1.0
    
    distance = calculate_edit_distance(ref_norm, hyp_norm)
    cer = distance / len(ref_norm)
    return min(1.0, cer)  # Cap at 100%

def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate."""
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()
    
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    
    distance = calculate_edit_distance(ref_words, hyp_words)
    wer = distance / len(ref_words)
    return min(1.0, wer)  # Cap at 100%

def parse_page_spec(page_spec: str) -> List[int]:
    """Parse page specification into list of page numbers."""
    page_spec = page_spec.strip()
    
    # Single page
    if page_spec.isdigit():
        return [int(page_spec)]
    
    # Page range with - or –
    for separator in ['-', '–']:
        if separator in page_spec:
            parts = page_spec.split(separator)
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                start_page = int(parts[0])
                end_page = int(parts[1])
                return list(range(start_page, end_page + 1))
    
    logger.warning(f"Invalid page spec: {page_spec}")
    return []

def find_gold_csv(primary_path: str, fallback_path: str = None) -> Optional[str]:
    """Find gold CSV with primary/fallback paths and user typo handling."""
    if os.path.exists(primary_path):
        if fallback_path and os.path.exists(fallback_path):
            logger.warning(f"Both CSV paths exist, preferring {primary_path}")
        return primary_path
    
    if fallback_path and os.path.exists(fallback_path):
        logger.info(f"Primary path not found, using fallback: {fallback_path}")
        return fallback_path
    
    return None

def load_gold_pages(csv_path: str) -> List[Dict[str, str]]:
    """Load gold pages from CSV file."""
    gold_pages = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row_num, row in enumerate(reader, 1):
            # Check required columns
            if 'PDF LINK' not in row or 'PAGE' not in row or 'HANDTYPED' not in row:
                logger.warning(f"Row {row_num}: missing required columns")
                continue
            
            pdf_link = row['PDF LINK'].strip()
            page_spec = row['PAGE'].strip()
            handtyped = row['HANDTYPED'].strip()
            
            # Skip empty/NaN handtyped
            if not handtyped or handtyped.lower() in ['nan', 'null', '']:
                logger.debug(f"Row {row_num}: skipping empty HANDTYPED")
                continue
            
            gold_pages.append({
                'pdf_filename': pdf_link,
                'page_spec': page_spec,
                'ground_truth': handtyped,
                'row_num': row_num
            })
    
    logger.info(f"Loaded {len(gold_pages)} valid gold pages from {csv_path}")
    return gold_pages

def select_pdfs(gold_pages: List[Dict[str, str]], input_dir: str, 
                limit_pdfs: int, seed: int) -> List[str]:
    """Select PDFs based on availability and gold row count."""
    input_path = Path(input_dir)
    
    # Count gold rows per PDF and check existence
    pdf_counts = {}
    for page in gold_pages:
        pdf_name = page['pdf_filename'].strip()
        
        # Skip empty/invalid PDF names
        if not pdf_name or pdf_name.lower() in ['nan', 'null', '']:
            logger.debug(f"Skipping empty/invalid PDF name")
            continue
            
        pdf_path = input_path / pdf_name
        
        if pdf_path.exists():
            pdf_counts[pdf_name] = pdf_counts.get(pdf_name, 0) + 1
        else:
            logger.debug(f"PDF not found: {pdf_path}")
    
    if not pdf_counts:
        logger.error("No PDFs found in input directory matching gold CSV")
        return []
    
    # Sort by count (desc), then by filename for deterministic order
    sorted_pdfs = sorted(pdf_counts.items(), key=lambda x: (-x[1], x[0]))
    
    # Select up to limit_pdfs with deterministic seeding
    random.seed(seed)
    selected = [pdf for pdf, count in sorted_pdfs[:limit_pdfs]]
    
    logger.info(f"Selected {len(selected)} PDFs: {selected}")
    logger.info(f"Gold page counts: {dict(sorted_pdfs[:limit_pdfs])}")
    
    return selected

def get_available_engines() -> List[str]:
    """Get list of available OCR engines."""
    available = ['paddle']  # Always available in our setup
    
    try:
        # Check if new engines are available
        sys.path.append("/app/src")
        from engines import get_available_engines
        available = get_available_engines()
    except ImportError:
        logger.debug("New engine system not available, using legacy engines")
    
    return available

def run_ocr_on_page(pdf_path: str, page_num: int, engine: str, profile: str, device: str) -> str:
    """Run OCR on a specific page using the specified engine."""
    try:
        # Import the pipeline
        sys.path.append("/app/src")
        from pdf_utils import extract_images_from_pdf
        from pipeline import process_image
        
        # Extract image
        dpi = 300 if profile in ["quality", "balanced"] else 200
        all_images = extract_images_from_pdf(pdf_path, dpi=dpi)
        
        if not all_images:
            logger.error(f"No images extracted from {pdf_path}")
            return ""
        
        page_index = page_num - 1
        if page_index >= len(all_images):
            logger.error(f"Page {page_num} not found in {pdf_path}")
            return ""
        
        image_array, page_id = all_images[page_index]
        
        # Run OCR through pipeline with specified engine
        results = process_image(
            image_array, 
            extract_all_text=True,
            ocr_engine=engine,
            ocr_profile=profile,
            ocr_device=device
        )
        
        # Extract text from results
        if results:
            text_lines = []
            for result in results:
                if 'text' in result:
                    text_lines.append(result['text'])
                elif 'original_text' in result:  # Fallback
                    text_lines.append(result['original_text'])
            
            text = ' '.join(text_lines).strip()
            if text:
                logger.info(f"OCR with {engine} extracted {len(text)} characters")
                return text
        
        logger.warning(f"No text extracted by {engine} for {pdf_path} page {page_num}")
        return ""
        
    except Exception as e:
        logger.error(f"OCR error with {engine} for {pdf_path} page {page_num}: {e}")
        return ""

def create_diff_excerpt(reference: str, hypothesis: str, max_chars: int = 200) -> str:
    """Create a short diff excerpt showing typical errors."""
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)
    
    if len(ref_norm) > max_chars:
        ref_norm = ref_norm[:max_chars] + "..."
    if len(hyp_norm) > max_chars:
        hyp_norm = hyp_norm[:max_chars] + "..."
    
    return f"**Reference:** {ref_norm}\n**Hypothesis:** {hyp_norm}"

def get_engine_info(engine: str) -> Dict[str, Any]:
    """Get information about the specified engine."""
    try:
        sys.path.append("/app/src")
        from engines import create_engine
        engine_instance = create_engine(engine, 'balanced', 'auto')
        return engine_instance.get_engine_info()
    except Exception as e:
        logger.warning(f"Could not get info for engine {engine}: {e}")
        return {"name": engine, "version": "unknown", "device": "unknown"}

def main():
    parser = argparse.ArgumentParser(description="Enhanced OCR evaluation with multi-engine support")
    parser.add_argument('--engine', choices=['paddle', 'doctr', 'mmocr', 'kraken'], 
                       default='paddle', help="OCR engine to use")
    parser.add_argument('--profile', choices=['fast', 'balanced', 'quality'], 
                       default='balanced', help="Performance profile")
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], 
                       default='auto', help="Device to use")
    parser.add_argument('--gold-csv', required=True, help="Path to gold CSV file")
    parser.add_argument('--input-dir', default='data/input_pdfs', help="Input PDFs directory")
    parser.add_argument('--limit-pdfs', type=int, default=2, help="Max PDFs to process")
    parser.add_argument('--report-md', action='store_true', help="Generate markdown report")
    parser.add_argument('--seed', type=int, default=17, help="Random seed for deterministic selection")
    
    args = parser.parse_args()
    
    # Check if engine is available
    available_engines = get_available_engines()
    if args.engine not in available_engines:
        logger.error(f"Engine '{args.engine}' not available. Available engines: {available_engines}")
        sys.exit(1)
    
    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{args.engine}_{args.profile}_{timestamp}"
    reports_dir = Path("reports") / run_id
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_dir = reports_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    
    logger.info(f"Starting {args.engine} evaluation - Run ID: {run_id}")
    
    # Find gold CSV with fallback
    fallback_csv = args.gold_csv.replace("data/gold_data", "daata/gold_data")
    csv_path = find_gold_csv(args.gold_csv, fallback_csv)
    
    if not csv_path:
        logger.error(f"Gold CSV not found: {args.gold_csv}")
        sys.exit(1)
    
    # Load gold pages
    gold_pages = load_gold_pages(csv_path)
    if not gold_pages:
        logger.error("No valid gold pages loaded")
        sys.exit(1)
    
    # Select PDFs
    selected_pdfs = select_pdfs(gold_pages, args.input_dir, args.limit_pdfs, args.seed)
    if not selected_pdfs:
        logger.error("No PDFs selected for evaluation")
        sys.exit(1)
    
    # Filter gold pages to selected PDFs
    filtered_pages = []
    for p in gold_pages:
        pdf_name = p['pdf_filename'].strip()
        if pdf_name and pdf_name in selected_pdfs:
            filtered_pages.append(p)
    logger.info(f"Processing {len(filtered_pages)} pages across {len(selected_pdfs)} PDFs")
    
    # Get engine information
    engine_info = get_engine_info(args.engine)
    device = detect_device()
    
    # Process pages and collect results
    results = []
    all_cers = []
    all_wers = []
    diff_examples = []
    
    for page_data in filtered_pages:
        pdf_name = page_data['pdf_filename']
        page_spec = page_data['page_spec']
        ground_truth = page_data['ground_truth']
        
        page_numbers = parse_page_spec(page_spec)
        if not page_numbers:
            logger.warning(f"Skipping invalid page spec: {page_spec}")
            continue
        
        # Process each page in the spec
        for page_num in page_numbers:
            pdf_path = f"/app/{args.input_dir}/{pdf_name}"
            
            logger.info(f"Processing {pdf_name} page {page_num} with {args.engine}")
            
            # Run OCR
            hypothesis = run_ocr_on_page(pdf_path, page_num, args.engine, args.profile, args.device)
            
            if not hypothesis:
                logger.warning(f"No OCR result for {pdf_name} page {page_num}")
                continue
            
            # Calculate metrics
            cer = calculate_cer(ground_truth, hypothesis)
            wer = calculate_wer(ground_truth, hypothesis)
            
            # Guess language (simple heuristic)
            lang_guess = "unknown"
            if any(char in hypothesis for char in 'çğıöşüÇĞIİÖŞÜ'):
                lang_guess = "tr"
            elif any(char in hypothesis for char in 'äöüßÄÖÜ'):
                lang_guess = "de"
            
            result = {
                'run_id': run_id,
                'engine': args.engine,
                'profile': args.profile,
                'device': device,
                'pdf': pdf_name,
                'page': page_num,
                'lang_guess': lang_guess,
                'cer': cer,
                'wer': wer,
                'chars_ref': len(ground_truth),
                'chars_hyp': len(hypothesis),
                'words_ref': len(ground_truth.split()),
                'words_hyp': len(hypothesis.split()),
                'ground_truth': ground_truth,
                'hypothesis': hypothesis
            }
            
            results.append(result)
            all_cers.append(cer)
            all_wers.append(wer)
            
            # Collect diff examples (up to 3)
            if len(diff_examples) < 3 and cer > 0.05:  # Only if there are errors
                diff_examples.append({
                    'pdf': pdf_name,
                    'page': page_num,
                    'diff': create_diff_excerpt(ground_truth, hypothesis)
                })
            
            logger.info(f"  CER: {cer:.3f}, WER: {wer:.3f}")
    
    if not results:
        logger.error("No results generated")
        sys.exit(1)
    
    # Save metrics CSV
    metrics_csv = metrics_dir / "metrics.csv"
    with open(metrics_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['run_id', 'engine', 'profile', 'device', 'pdf', 'page', 'lang_guess', 
                     'cer', 'wer', 'chars_ref', 'chars_hyp', 'words_ref', 'words_hyp']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({k: result[k] for k in fieldnames})
    
    # Calculate overall metrics
    overall_cer = sum(all_cers) / len(all_cers) if all_cers else 0.0
    overall_wer = sum(all_wers) / len(all_wers) if all_wers else 0.0
    
    # Generate markdown report
    if args.report_md:
        summary_md = reports_dir / "summary.md"
        commit_sha = get_git_commit_sha()
        
        with open(summary_md, 'w', encoding='utf-8') as f:
            f.write(f"# {args.engine.upper()} OCR Evaluation Report\n\n")
            f.write(f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Commit SHA:** {commit_sha}\n")
            f.write(f"**Engine:** {args.engine}\n")
            f.write(f"**Profile:** {args.profile}\n")
            f.write(f"**Device:** {device}\n")
            f.write(f"**Run ID:** {run_id}\n\n")
            
            # Engine information
            f.write("## Engine Information\n\n")
            f.write(f"- **Engine:** {engine_info.get('name', args.engine)}\n")
            f.write(f"- **Version:** {engine_info.get('version', 'unknown')}\n")
            f.write(f"- **Device:** {engine_info.get('device', 'unknown')}\n")
            if 'models' in engine_info:
                f.write(f"- **Models:** {engine_info['models']}\n")
            f.write("\n")
            
            # Overall averages
            f.write("## Overall Results\n\n")
            f.write(f"- **Pages Processed:** {len(results)}\n")
            f.write(f"- **PDFs Processed:** {len(selected_pdfs)}\n")
            f.write(f"- **Overall CER:** {overall_cer:.3f}\n")
            f.write(f"- **Overall WER:** {overall_wer:.3f}\n\n")
            
            # Per-PDF results
            f.write("## Results by PDF\n\n")
            f.write("| PDF | Page | CER | WER | Language |\n")
            f.write("|-----|------|-----|-----|----------|\n")
            
            for result in results:
                f.write(f"| {result['pdf']} | {result['page']} | {result['cer']:.3f} | {result['wer']:.3f} | {result['lang_guess']} |\n")
            
            f.write("\n")
            
            # Example diffs
            if diff_examples:
                f.write("## Example OCR Errors\n\n")
                for i, example in enumerate(diff_examples[:3], 1):
                    f.write(f"### Example {i}: {example['pdf']} Page {example['page']}\n\n")
                    f.write(f"{example['diff']}\n\n")
        
        # Print absolute path to summary
        summary_abs_path = summary_md.resolve()
        print(f"\nEvaluation complete!")
        print(f"Report saved to: {summary_abs_path}")
        
        # Also print key metrics
        print(f"Engine: {args.engine} ({args.profile} profile)")
        print(f"Overall CER: {overall_cer:.3f}")
        print(f"Overall WER: {overall_wer:.3f}")
        print(f"Pages processed: {len(results)}")
    
    logger.info(f"{args.engine} evaluation complete - Run ID: {run_id}")

if __name__ == "__main__":
    main()