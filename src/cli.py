#!/usr/bin/env python3
"""
Command-line interface for the OCR translation pipeline.
"""
import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Optional

import cv2
from tqdm import tqdm

from pipeline import process_image
from csv_writer import write_csv
from pdf_utils import is_pdf_file, extract_images_from_pdf, get_pdf_info
from healthcheck import run_health_check, print_health_report
from preflight import run_preflight, print_preflight_report
import os


def setup_logging(level: str = 'INFO') -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def get_image_files(input_path: Path) -> List[Path]:
    """Get list of image and PDF files from input path."""
    supported_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp', '.pdf'}
    
    if input_path.is_file():
        if input_path.suffix.lower() in supported_extensions:
            return [input_path]
        else:
            raise ValueError(f"Unsupported file type: {input_path.suffix}")
    
    elif input_path.is_dir():
        files = []
        for ext in supported_extensions:
            files.extend(input_path.glob(f"*{ext}"))
            files.extend(input_path.glob(f"*{ext.upper()}"))
        
        if not files:
            raise ValueError(f"No supported files found in directory: {input_path}")
        
        return sorted(files)
    
    else:
        raise ValueError(f"Input path does not exist: {input_path}")


def process_single_file(args: tuple) -> Optional[str]:
    """Process a single image or PDF file. Used for multiprocessing."""
    file_path, output_dir, dpi, extract_all = args
    
    try:
        if is_pdf_file(file_path):
            # Handle PDF file - extract images first
            logging.info(f"📄 Processing PDF: {Path(file_path).name}")
            
            # Get PDF info
            pdf_info = get_pdf_info(file_path)
            total_pages = pdf_info.get('pages', 0)
            logging.info(f"📖 PDF has {total_pages} pages")
            
            # Extract images from PDF
            logging.info(f"🔍 Extracting images from PDF at {dpi} DPI...")
            images = extract_images_from_pdf(file_path, dpi=dpi)
            
            if not images:
                logging.error(f"❌ No images extracted from PDF: {Path(file_path).name}")
                return None
            
            logging.info(f"✓ Extracted {len(images)} images from PDF")
            # Optionally limit pages via environment variable for testing
            try:
                max_pages_env = os.getenv('MAX_PAGES')
                if max_pages_env:
                    max_pages = int(max_pages_env)
                    if max_pages > 0:
                        images = images[:max_pages]
                        logging.info(f"Limiting to first {max_pages} pages from PDF (MAX_PAGES env)")
            except Exception:
                pass
            
            # Process each page
            all_translations = []
            successful_pages = 0
            
            # Create progress bar for pages
            with tqdm(total=len(images), desc=f"Processing {Path(file_path).name}", unit="page") as pbar:
                for img, page_id in images:
                    try:
                        # Update progress bar description with current page
                        pbar.set_description(f"Processing {Path(file_path).name} - {page_id}")
                        
                        # Process image through pipeline
                        text_results = process_image(img, extract_all_text=extract_all)
                        
                        if text_results:
                            # Write CSV for this page
                            output_path = output_dir / f"{page_id}.csv"
                            write_csv(output_path, text_results, page_id)
                            all_translations.extend(text_results)
                            successful_pages += 1
                            if extract_all:
                                logging.info(f"✓ Completed {page_id} ({len(text_results)} text lines extracted)")
                            else:
                                logging.info(f"✓ Completed {page_id} ({len(text_results)} translations found)")
                        else:
                            if extract_all:
                                logging.warning(f"⚠ {page_id}: No text detected by OCR")
                            else:
                                logging.warning(f"⚠ {page_id}: No translations found (text was OCR'd but no language patterns detected)")
                            
                        # Update progress bar
                        pbar.update(1)
                        
                    except Exception as e:
                        logging.error(f"✗ Error processing {page_id}: {e}")
                        pbar.update(1)  # Still update progress even on error
                        continue
            
            if successful_pages > 0:
                logging.info(f"Successfully processed {successful_pages}/{len(images)} pages from {Path(file_path).name}")
                if extract_all:
                    logging.info(f"📊 Total text lines extracted: {len(all_translations)}")
                    return f"{successful_pages} pages processed, {len(all_translations)} text lines"
                else:
                    logging.info(f"📊 Total translations extracted: {len(all_translations)}")
                    return f"{successful_pages} pages processed, {len(all_translations)} translations"
            else:
                logging.warning(f"❌ No pages successfully processed from {Path(file_path).name}")
                return None
        
        else:
            # Handle regular image file
            logging.info(f"🖼️ Processing image: {Path(file_path).name}")
            
            img = cv2.imread(str(file_path))
            if img is None:
                logging.error(f"❌ Failed to load image: {Path(file_path).name}")
                return None
            
            # Process image through pipeline
            logging.info(f"🔍 Running OCR and text extraction...")
            text_results = process_image(img, extract_all_text=extract_all)
            
            if not text_results:
                if extract_all:
                    logging.warning(f"⚠ {Path(file_path).name}: No text detected by OCR")
                else:
                    logging.warning(f"⚠ {Path(file_path).name}: No translations found (text was OCR'd but no language patterns detected)")
                return None
            
            # Write CSV output
            page_id = file_path.stem
            output_path = output_dir / f"{page_id}.csv"
            write_csv(output_path, text_results, page_id)
            
            if extract_all:
                logging.info(f"✓ Completed {Path(file_path).name} ({len(text_results)} text lines extracted)")
                return f"1 image processed, {len(text_results)} text lines"
            else:
                logging.info(f"✓ Completed {Path(file_path).name} ({len(text_results)} translations found)")
                return f"1 image processed, {len(text_results)} translations"
        
    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")
        return None


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Extract translations from textbook images and PDFs to CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py --input page001.jpg --output ./results
  python cli.py --input textbook.pdf --output ./results --dpi 300
  python cli.py --input ./files/ --output ./results --jobs 4
        """
    )
    
    parser.add_argument(
        '--input',
        type=Path,
        help='Input image/PDF file or directory containing images/PDFs'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        help='Output directory for CSV files'
    )
    
    parser.add_argument(
        '--jobs',
        type=int,
        default=1,
        help='Number of parallel processing jobs (default: 1)'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for PDF extraction (default: 300)'
    )

    parser.add_argument(
        '--max-pages',
        type=int,
        default=None,
        help='Maximum number of pages to process from each PDF (for testing)'
    )
    
    parser.add_argument(
        '--extract-all',
        action='store_true',
        help='Extract all OCR text instead of just translation patterns'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--healthcheck',
        action='store_true',
        help='Run comprehensive health check and exit'
    )
    
    parser.add_argument(
        '--preflight',
        action='store_true',
        help='Run preflight validation and exit'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Handle special modes
    if args.healthcheck:
        logger.info("Running comprehensive health check...")
        health_results = run_health_check()
        print_health_report(health_results)
        
        # Exit with appropriate code
        summary = health_results.get('summary', {})
        status = summary.get('overall_status', 'unknown')
        if status in ['healthy', 'healthy_with_warnings']:
            return 0
        elif status == 'degraded':
            return 1
        else:  # unhealthy
            return 2
    
    if args.preflight:
        logger.info("Running preflight validation...")
        passed, preflight_results = run_preflight()
        print_preflight_report(preflight_results)
        return 0 if passed else 1
    
    # Standard processing mode - validate required arguments
    if not args.input or not args.output:
        parser.error("--input and --output are required for processing mode")
    
    try:
        # Validate input and get files
        files = get_image_files(args.input)
        logger.info(f"Found {len(files)} files to process")
        
        # Count PDFs vs images
        pdf_count = sum(1 for f in files if is_pdf_file(f))
        img_count = len(files) - pdf_count
        if pdf_count > 0:
            logger.info(f"Files: {img_count} images, {pdf_count} PDFs")
        
        # Create output directory
        args.output.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {args.output}")
        
        # Prepare arguments for processing
        process_args = [(file_path, args.output, args.dpi, args.extract_all) for file_path in files]
        
        # Process files
        if args.jobs == 1:
            # Sequential processing
            results = []
            for arg in tqdm(process_args, desc="Processing files"):
                result = process_single_file(arg)
                results.append(result)
        else:
            # Parallel processing
            with ProcessPoolExecutor(max_workers=args.jobs) as executor:
                results = list(tqdm(
                    executor.map(process_single_file, process_args),
                    total=len(process_args),
                    desc="Processing files"
                ))
        
        # Report results
        successful = sum(1 for r in results if r is not None)
        failed = len(results) - successful
        
        logger.info(f"Processing complete: {successful} successful, {failed} failed")
        
        if failed > 0:
            logger.warning(f"{failed} files failed to process")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
