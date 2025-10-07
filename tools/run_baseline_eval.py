#!/usr/bin/env python3
"""
Lightweight baseline evaluation runner for OCR pipeline.
Runs evaluation on selected PDFs from gold CSV with basic CER/WER metrics.
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
import io
import numpy as np
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

def run_ocr_on_page(pdf_path: str, page_num: int, profile: str, engine: str = "paddle", use_router: bool = False) -> str:
    """
    Enhanced OCR with orientation correction, preprocessing, and optional routing.
    
    Args:
        pdf_path: Path to PDF file
        page_num: Page number (1-indexed)
        profile: Pipeline profile (fast/quality)
        engine: OCR engine name or "router" for routing system
        use_router: Whether to use the recognition router
    """
    try:
        import sys
        sys.path.append("src")
        sys.path.append("production")
        
        # Use local PDF extraction
        import fitz  # PyMuPDF
        import numpy as np
        from PIL import Image
        
        # Import our enhanced modules
        from orientation import correct_page_orientation
        from preprocess import enhanced_clahe_preprocessing, filter_short_lines
        from lang_and_akkadian import analyze_text_line
        
        # Extract page as image
        doc = fitz.open(pdf_path)
        if page_num > len(doc):
            logger.error(f"Page {page_num} not found in {pdf_path} (only {len(doc)} pages)")
            doc.close()
            return ""
        
        # Get page and convert to image
        page = doc[page_num - 1]  # 0-indexed
        dpi = 300 if profile in ["quality", "balanced"] else 200
        mat = fitz.Matrix(dpi/72, dpi/72)  # Scale factor for DPI
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("ppm")
        image = Image.open(io.BytesIO(img_data))
        image_array = np.array(image)
        doc.close()
        
        # Step 1: Orientation correction
        orientation_config = {
            'cache_enabled': True,
            'max_side': 1200,
            'fine_deskew_range': 10
        }
        
        corrected_image, orientation_metadata = correct_page_orientation(
            image_array, orientation_config
        )
        
        # Step 2: Enhanced preprocessing
        preprocess_config = {
            'bilateral_filter': profile in ["quality", "balanced"],
            'bilateral_d': 5 if profile == "fast" else 9,
            'clahe': True,
            'clahe_clip_limit': 2.5 if profile == "fast" else 3.0,
            'gamma_correction': True,
            'gamma': 1.2
        }
        
        processed_image = enhanced_clahe_preprocessing(corrected_image, preprocess_config)
        
        # Step 3: OCR with routing (if enabled) or direct engine
        text = ""
        ocr_metadata = {}
        
        if use_router and engine == "router":
            # Use recognition router
            try:
                from recognition_router import route_line_recognition
                
                # Quick language detection from a center crop
                h, w = processed_image.shape[:2]
                center_crop = processed_image[h//3:2*h//3, w//4:3*w//4]
                
                # Run quick OCR for language detection
                quick_text = _run_quick_ocr(center_crop)
                lang_analysis = analyze_text_line(quick_text)
                
                router_config = {
                    'router': {
                        'primary': 'paddle',
                        'fallback': 'doctr',
                        'ensemble': ['paddle', 'doctr', 'easyocr'],
                        'thresholds': {
                            'en': 0.90, 'de': 0.88, 'fr': 0.88, 'it': 0.88, 'tr': 0.86
                        },
                        'delta_disagree': 0.04
                    },
                    'cache_enabled': True
                }
                
                # Route recognition for the full page
                result = route_line_recognition(
                    processed_image,
                    language=lang_analysis['language'],
                    is_akkadian=lang_analysis['is_akkadian'],
                    config=router_config
                )
                
                text = result.text
                ocr_metadata = {
                    'engine': result.engine,
                    'method': result.method,
                    'confidence': result.confidence,
                    'engines_used': result.engines_used,
                    'language': result.language,
                    'execution_time': result.execution_time
                }
                
                logger.info(f"ROUTER used {result.method} with {result.engines_used}, conf={result.confidence:.3f}")
                
            except Exception as e:
                logger.warning(f"Router failed: {e}, falling back to direct engine")
                use_router = False
                engine = "paddle"  # Fallback
        
        if not use_router:
            # Direct engine usage (existing logic with enhancements)
            if engine.lower() == "paddle":
                # PaddleOCR
                from paddleocr import PaddleOCR
                ocr = PaddleOCR(lang='en')
                result = ocr.ocr(processed_image)
                
                if result and result[0]:
                    text_lines = []
                    for line in result[0]:
                        if line and len(line) >= 2:
                            text_lines.append(line[1][0])
                    
                    # Apply line filtering
                    filtered_lines = filter_short_lines(text_lines, min_length=3)
                    text = ' '.join(filtered_lines).strip()
                    
            elif engine.lower() == "doctr":
                # docTR
                try:
                    import tempfile
                    import os
                    from doctr.io import DocumentFile
                    from doctr.models import ocr_predictor
                    
                    # Save image to temporary file for docTR
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                        pil_img = Image.fromarray(processed_image)
                        pil_img.save(tmp.name, 'PNG')
                        
                        # Load with docTR
                        doc_file = DocumentFile.from_images([tmp.name])
                        model = ocr_predictor(pretrained=True)
                        result = model(doc_file)
                        
                        # Extract text from result
                        text_lines = []
                        for page_result in result.pages:
                            for block in page_result.blocks:
                                for line in block.lines:
                                    for word in line.words:
                                        text_lines.append(word.value)
                        
                        # Apply line filtering
                        filtered_lines = filter_short_lines(text_lines, min_length=2)
                        text = ' '.join(filtered_lines).strip()
                        
                        # Clean up temp file
                        try:
                            os.unlink(tmp.name)
                        except:
                            pass
                    
                except Exception as e:
                    logger.warning(f"docTR failed: {e}, falling back to PaddleOCR")
                    text = _fallback_paddle_ocr(processed_image)
            
            elif engine.lower() == "easyocr":
                # EasyOCR
                try:
                    import easyocr
                    reader = easyocr.Reader(['en'], verbose=False)
                    result = reader.readtext(processed_image)
                    
                    # Extract text from result
                    text_lines = []
                    for detection in result:
                        if len(detection) >= 2:
                            text_lines.append(detection[1])
                    
                    # Apply line filtering
                    filtered_lines = filter_short_lines(text_lines, min_length=2)
                    text = ' '.join(filtered_lines).strip()
                    
                except Exception as e:
                    logger.warning(f"EasyOCR failed: {e}, falling back to PaddleOCR")
                    text = _fallback_paddle_ocr(processed_image)
            
            elif engine.lower() == "tesseract":
                # Tesseract
                try:
                    import pytesseract
                    from PIL import Image
                    
                    pil_img = Image.fromarray(processed_image)
                    
                    # Use PyTesseract to extract text
                    raw_text = pytesseract.image_to_string(pil_img, lang='eng')
                    
                    # Split into lines and filter
                    text_lines = raw_text.split('\n')
                    filtered_lines = filter_short_lines(text_lines, min_length=3)
                    text = ' '.join(filtered_lines).strip()
                    
                except Exception as e:
                    logger.warning(f"Tesseract failed: {e}, falling back to PaddleOCR")
                    text = _fallback_paddle_ocr(processed_image)
            
            elif engine.lower() == "trocr":
                # TrOCR
                try:
                    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
                    from PIL import Image
                    
                    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
                    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
                    
                    pil_img = Image.fromarray(processed_image)
                    pixel_values = processor(pil_img, return_tensors="pt").pixel_values
                    generated_ids = model.generate(pixel_values, max_length=512)
                    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    
                except Exception as e:
                    logger.warning(f"TrOCR failed: {e}, falling back to PaddleOCR")
                    text = _fallback_paddle_ocr(processed_image)
            
            else:
                # Fallback to PaddleOCR
                logger.warning(f"Engine {engine} not supported, using PaddleOCR")
                text = _fallback_paddle_ocr(processed_image)
        
        # Log results with metadata
        if text:
            char_count = len(text)
            total_angle = orientation_metadata.get('total_rotation', 0)
            coarse_angle = orientation_metadata.get('coarse_rotation', 0)
            fine_skew = orientation_metadata.get('fine_skew_deg', 0)
            
            if use_router and ocr_metadata:
                logger.info(f"{engine.upper()}: {char_count} chars, angle={total_angle:.1f}° ({coarse_angle}°+{fine_skew:.1f}°), method={ocr_metadata.get('method', 'direct')}")
            else:
                logger.info(f"{engine.upper()}: {char_count} chars, angle={total_angle:.1f}° ({coarse_angle}°+{fine_skew:.1f}°)")
            
            return text
        else:
            logger.warning(f"No text extracted by {engine} for {pdf_path} page {page_num}")
            return ""
            
    except Exception as e:
        logger.error(f"Enhanced OCR error with {engine} for {pdf_path} page {page_num}: {e}")
        return ""


def _run_quick_ocr(image_crop: np.ndarray) -> str:
    """Quick OCR for language detection."""
    try:
        import easyocr
        reader = easyocr.Reader(['en'], verbose=False)
        results = reader.readtext(image_crop)
        
        text_parts = []
        for (bbox, text, confidence) in results:
            if confidence > 0.5:  # Only high-confidence text
                text_parts.append(text)
        
        return ' '.join(text_parts).strip()
    except:
        return ""


def _fallback_paddle_ocr(image: np.ndarray) -> str:
    """Fallback PaddleOCR implementation."""
    try:
        from paddleocr import PaddleOCR
        from preprocess import filter_short_lines
        
        ocr = PaddleOCR(lang='en')
        result = ocr.ocr(image)
        
        if result and result[0]:
            text_lines = []
            for line in result[0]:
                if line and len(line) >= 2:
                    text_lines.append(line[1][0])
            
            filtered_lines = filter_short_lines(text_lines)
            return ' '.join(filtered_lines).strip()
    except Exception as e:
        logger.error(f"Fallback PaddleOCR failed: {e}")
    
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

def main():
    parser = argparse.ArgumentParser(description="Lightweight baseline OCR evaluation")
    parser.add_argument('--gold-csv', required=True, help="Path to gold CSV file")
    parser.add_argument('--input-dir', default='data/input_pdfs', help="Input PDFs directory")
    parser.add_argument('--limit-pdfs', type=int, default=2, help="Max PDFs to process")
    parser.add_argument('--profile', default='fast', help="Pipeline profile")
    parser.add_argument('--engine', default='paddle', choices=['paddle', 'doctr', 'easyocr', 'trocr', 'tesseract', 'router'], help="OCR engine to use or 'router' for intelligent routing")
    parser.add_argument('--use-router', action='store_true', help="Enable recognition router system")
    parser.add_argument('--report-md', action='store_true', help="Generate markdown report")
    parser.add_argument('--seed', type=int, default=17, help="Random seed for deterministic selection")
    
    args = parser.parse_args()
    
    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"baseline_{args.engine}_{args.profile}_{timestamp}"
    reports_dir = Path("reports") / run_id
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_dir = reports_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    
    logger.info(f"Starting baseline evaluation - Run ID: {run_id}")
    
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
    
    # Filter gold pages to selected PDFs and skip empty PDF names
    filtered_pages = []
    for p in gold_pages:
        pdf_name = p['pdf_filename'].strip()
        if pdf_name and pdf_name in selected_pdfs:
            filtered_pages.append(p)
    logger.info(f"Processing {len(filtered_pages)} pages across {len(selected_pdfs)} PDFs")
    
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
            pdf_path = f"{args.input_dir}/{pdf_name}"
            
            logger.info(f"Processing {pdf_name} page {page_num}")
            
            # Run OCR
            use_router = args.use_router or args.engine == "router"
            hypothesis = run_ocr_on_page(pdf_path, page_num, args.profile, args.engine, use_router)
            
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
        fieldnames = ['run_id', 'pdf', 'page', 'lang_guess', 'cer', 'wer', 
                     'chars_ref', 'chars_hyp', 'words_ref', 'words_hyp']
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
        device = detect_device()
        
        with open(summary_md, 'w', encoding='utf-8') as f:
            f.write(f"# Baseline OCR Evaluation Report\n\n")
            f.write(f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Commit SHA:** {commit_sha}\n")
            f.write(f"**Profile:** {args.profile}\n")
            f.write(f"**Device:** {device}\n")
            f.write(f"**Run ID:** {run_id}\n\n")
            
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
        print(f"Overall CER: {overall_cer:.3f}")
        print(f"Overall WER: {overall_wer:.3f}")
        print(f"Pages processed: {len(results)}")
    
    logger.info(f"Baseline evaluation complete - Run ID: {run_id}")

if __name__ == "__main__":
    main()