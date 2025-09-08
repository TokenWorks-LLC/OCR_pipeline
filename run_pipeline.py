#!/usr/bin/env python3
"""
OCR Pipeline - Unified Main Entry Point
Comprehensive document processing pipeline with configuration-based execution
"""

import json
import sys
import os
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Any
import glob
import time
from datetime import datetime

# Add necessary directories to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent / 'production'))

def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"❌ Config file not found: {config_path}")
        print("Please ensure config.json exists in the project root directory.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in config file: {e}")
        sys.exit(1)

def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration parameters."""
    required_sections = ['input', 'output', 'ocr']
    
    for section in required_sections:
        if section not in config:
            print(f"❌ Missing required config section: {section}")
            return False
    
    # Validate input directory
    input_dir = config['input']['input_directory']
    if not config['input']['process_all_files']:
        # Check specific file
        specific_file = config['input']['specific_file']
        if not specific_file:
            print("❌ When process_all_files is false, specific_file must be provided")
            return False
        if not os.path.exists(specific_file):
            print(f"❌ Specific file not found: {specific_file}")
            return False
    else:
        # Check input directory
        if not os.path.exists(input_dir):
            print(f"❌ Input directory not found: {input_dir}")
            return False
    
    return True

def get_files_to_process(config: Dict[str, Any]) -> List[str]:
    """Get list of files to process based on configuration."""
    files_to_process = []
    
    if not config['input']['process_all_files']:
        # Process single file
        specific_file = config['input']['specific_file']
        if os.path.exists(specific_file):
            files_to_process.append(specific_file)
    else:
        # Process all files in directory
        input_dir = config['input']['input_directory']
        supported_formats = config['input']['supported_formats']
        recursive = config['input'].get('recursive_search', False)
        
        search_pattern = "**/*" if recursive else "*"
        
        for format_ext in supported_formats:
            pattern = os.path.join(input_dir, f"{search_pattern}{format_ext}")
            if recursive:
                files_to_process.extend(glob.glob(pattern, recursive=True))
            else:
                files_to_process.extend(glob.glob(pattern))
    
    return sorted(list(set(files_to_process)))  # Remove duplicates and sort

def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """Setup logging configuration."""
    log_level = logging.INFO if config.get('processing', {}).get('verbose', True) else logging.WARNING
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('pipeline.log', encoding='utf-8')
        ]
    )
    
    return logging.getLogger(__name__)

def create_pipeline_config(config: Dict[str, Any]):
    """Create PipelineConfig object from loaded config."""
    try:
        from comprehensive_pipeline import PipelineConfig
        
        # Map configuration to PipelineConfig
        pipeline_config = PipelineConfig(
            # LLM settings
            llm_provider=config.get('llm', {}).get('provider', 'ollama'),
            llm_model=config.get('llm', {}).get('model', 'mistral:latest'),
            llm_base_url=config.get('llm', {}).get('base_url', 'http://localhost:11434'),
            llm_timeout=config.get('llm', {}).get('timeout', 30),
            
            # OCR settings
            dpi=config.get('ocr', {}).get('dpi', 300),
            paddle_use_gpu=False,  # Can be made configurable if needed
            
            # Processing settings
            enable_reading_order=config.get('ocr', {}).get('enable_reading_order', True),
            enable_llm_correction=config.get('llm', {}).get('enable_correction', True),
            max_concurrent_corrections=config.get('llm', {}).get('max_concurrent_corrections', 3),
            
            # Akkadian settings
            enable_akkadian_extraction=config.get('akkadian', {}).get('enable_extraction', True),
            generate_translations_pdf=config.get('akkadian', {}).get('generate_pdf_report', True),
            akkadian_confidence_threshold=config.get('akkadian', {}).get('confidence_threshold', 0.8),
            
            # Output settings
            output_csv_metadata=config.get('csv_output', {}).get('include_metadata', True),
            create_html_overlay=config.get('processing', {}).get('create_html_overlay', True),
            create_visualization=config.get('processing', {}).get('create_visualizations', True)
        )
        
        return pipeline_config
        
    except ImportError:
        print("❌ Unable to import comprehensive_pipeline module")
        print("Please ensure all required modules are available")
        sys.exit(1)

def process_single_file(file_path: str, config: Dict[str, Any], pipeline_config, pipeline, logger) -> Dict[str, Any]:
    """Process a single file."""
    logger.info(f"📄 Processing: {file_path}")
    
    # Determine output directory
    output_dir = config['output']['output_directory']
    
    if config['output'].get('create_subdirectories', True):
        # Create subdirectory based on input filename
        file_stem = Path(file_path).stem
        output_dir = os.path.join(output_dir, file_stem)
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Process the file
        if file_path.lower().endswith('.pdf'):
            result = pipeline.process_pdf(
                pdf_path=file_path,
                output_dir=output_dir
            )
        else:
            # For image files, we might need a different method
            # This would depend on your pipeline implementation
            result = pipeline.process_image(
                image_path=file_path,
                output_dir=output_dir
            )
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Failed to process {file_path}: {str(e)}")
        return {'error': str(e), 'file': file_path}

def print_processing_summary(results: List[Dict[str, Any]], start_time: float, logger):
    """Print processing summary."""
    end_time = time.time()
    total_time = end_time - start_time
    
    successful = [r for r in results if 'error' not in r]
    failed = [r for r in results if 'error' in r]
    
    print("\n" + "="*60)
    print("📊 PROCESSING SUMMARY")
    print("="*60)
    print(f"⏱️  Total processing time: {total_time:.2f} seconds")
    print(f"📁 Files processed: {len(results)}")
    print(f"✅ Successful: {len(successful)}")
    print(f"❌ Failed: {len(failed)}")
    
    if successful:
        print(f"\n✅ Successfully processed files:")
        for result in successful:
            if 'output_csv' in result:
                print(f"   📄 {result.get('file', 'Unknown')} → {result['output_csv']}")
            
            # Show Akkadian results if available
            akkadian_count = result.get('akkadian_translations_found', 0)
            if akkadian_count > 0:
                print(f"      🏛️  Akkadian translations found: {akkadian_count}")
                if result.get('akkadian_translations_pdf'):
                    print(f"      📋 PDF report: {result['akkadian_translations_pdf']}")
    
    if failed:
        print(f"\n❌ Failed files:")
        for result in failed:
            print(f"   📄 {result.get('file', 'Unknown')}: {result.get('error', 'Unknown error')}")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="OCR Pipeline - Unified Document Processor")
    parser.add_argument("-c", "--config", default="config.json", 
                       help="Path to configuration file (default: config.json)")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Show what would be processed without actually processing")
    parser.add_argument("--validate-only", action="store_true", 
                       help="Only validate configuration and exit")
    
    args = parser.parse_args()
    
    print("🚀 OCR Pipeline - Unified Document Processor")
    print("=" * 50)
    
    # Load configuration
    print(f"📄 Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Validate configuration
    print("🔍 Validating configuration...")
    if not validate_config(config):
        sys.exit(1)
    print("✅ Configuration valid")
    
    if args.validate_only:
        print("✅ Configuration validation completed successfully")
        return 0
    
    # Setup logging
    logger = setup_logging(config)
    
    # Get files to process
    files_to_process = get_files_to_process(config)
    
    if not files_to_process:
        print("❌ No files found to process")
        print("Please check your input directory and file formats configuration")
        return 1
    
    print(f"\n📁 Found {len(files_to_process)} file(s) to process:")
    for i, file_path in enumerate(files_to_process, 1):
        print(f"   {i}. {file_path}")
    
    if args.dry_run:
        print("\n🏃 Dry run completed - no files were processed")
        return 0
    
    # Create pipeline configuration and initialize pipeline
    print("\n🔧 Initializing OCR pipeline...")
    try:
        from comprehensive_pipeline import ComprehensivePipeline
        
        pipeline_config = create_pipeline_config(config)
        pipeline = ComprehensivePipeline(pipeline_config)
        
        print("✅ Pipeline initialized successfully")
        
        # Display configuration summary
        print(f"\n⚙️  Configuration Summary:")
        print(f"   🔧 OCR Engine: {config.get('ocr', {}).get('engine', 'paddleocr')}")
        print(f"   🎯 DPI: {config.get('ocr', {}).get('dpi', 300)}")
        print(f"   🧠 LLM Correction: {'Enabled' if config.get('llm', {}).get('enable_correction', True) else 'Disabled'}")
        print(f"   🏛️  Akkadian Extraction: {'Enabled' if config.get('akkadian', {}).get('enable_extraction', True) else 'Disabled'}")
        
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Process files
    print(f"\n🚀 Starting processing of {len(files_to_process)} file(s)...")
    start_time = time.time()
    results = []
    
    for i, file_path in enumerate(files_to_process, 1):
        print(f"\n📄 Processing file {i}/{len(files_to_process)}: {os.path.basename(file_path)}")
        
        # Check if we should skip existing files
        if config.get('processing', {}).get('skip_existing', True):
            output_dir = config['output']['output_directory']
            if config['output'].get('create_subdirectories', True):
                output_dir = os.path.join(output_dir, Path(file_path).stem)
            
            expected_csv = os.path.join(output_dir, 'comprehensive_results.csv')
            if os.path.exists(expected_csv):
                print(f"   ⏭️  Skipping (output already exists): {expected_csv}")
                continue
        
        result = process_single_file(file_path, config, pipeline_config, pipeline, logger)
        result['file'] = file_path  # Add file path to result for tracking
        results.append(result)
    
    # Print summary
    print_processing_summary(results, start_time, logger)
    
    # Save processing report
    output_dir = config['output']['output_directory']
    report_path = os.path.join(output_dir, f"processing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump({
                'processing_time': time.time() - start_time,
                'files_processed': len(results),
                'successful': len([r for r in results if 'error' not in r]),
                'failed': len([r for r in results if 'error' in r]),
                'config': config,
                'results': results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n📋 Processing report saved: {report_path}")
        
    except Exception as e:
        logger.warning(f"Could not save processing report: {e}")
    
    # Return exit code based on results
    failed_count = len([r for r in results if 'error' in r])
    if failed_count == 0:
        print("\n🎉 All files processed successfully!")
        return 0
    elif failed_count < len(results):
        print(f"\n⚠️  Processing completed with {failed_count} failures")
        return 0
    else:
        print(f"\n❌ All files failed to process")
        return 1

if __name__ == "__main__":
    exit(main())
