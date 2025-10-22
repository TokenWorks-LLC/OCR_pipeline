#!/usr/bin/env python3
"""
DeepSeek-OCR Evaluation Runner
Evaluates DeepSeek-OCR performance using Gold Pages ground truth integration.
Supports basic, advanced, and akkadian evaluation modes.
"""

import json
import sys
import os
import logging
import argparse
import time
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime

# Add necessary directories to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent / 'production'))

# Import our Gold Pages modules
from gold_pages_manager import GoldPagesManager, create_gold_pages_manager
from accuracy_measurement import AccuracyMeasurer, create_accuracy_measurer

# Import V3 language detection
from language_detection_v3 import initialize_language_detection_v3, cleanup_language_detection_v3, print_language_detection_summary
from language_metrics_patch import patch_comprehensive_pipeline_metrics, unpatch_comprehensive_pipeline_metrics

# Import Gold Pages analysis functions (copied from run_eval_gold_pages_v2.py)
def run_gold_pages_analysis(gold_pages_manager: GoldPagesManager,
                           accuracy_measurer: AccuracyMeasurer,
                           output_dir: str,
                           mode_name: str) -> bool:
    """Run Gold Pages analysis on evaluation results."""
    try:
        # Create analysis output directory
        analysis_dir = Path(output_dir) / "gold_pages_analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)

        # Get accuracy measurements
        accuracy_stats = accuracy_measurer.get_summary_statistics()

        # Create analysis report
        analysis_report = {
            'metadata': {
                'mode_name': mode_name,
                'analysis_date': datetime.now().isoformat(),
                'gold_pages_version': 'v3',
                'evaluation_type': 'deepseek_ocr'
            },
            'gold_pages_metrics': {
                'total_gold_pages': len(gold_pages_manager.gold_pages),
                'gold_pages_loaded': gold_pages_manager.get_loaded_count(),
                'gold_pages_with_matches': accuracy_measurer.get_pages_with_matches()
            },
            'accuracy_measurements': accuracy_stats,
            'success_thresholds': {
                'min_accuracy': 0.80,
                'target_accuracy': 0.90
            }
        }

        # Save analysis report
        report_file = analysis_dir / "gold_pages_analysis_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_report, f, indent=2, ensure_ascii=False)

        # Save detailed measurements
        measurements_file = analysis_dir / "accuracy_measurements.json"
        with open(measurements_file, 'w', encoding='utf-8') as f:
            json.dump(accuracy_measurer.get_detailed_measurements(), f, indent=2, ensure_ascii=False)

        # Generate summary report
        summary_report = generate_gold_pages_summary(analysis_report)
        summary_file = analysis_dir / "gold_pages_summary.md"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary_report)

        print(f"[STATS] Gold Pages analysis saved to: {analysis_dir}")
        return True

    except Exception as e:
        print(f"[ERROR] Gold Pages analysis failed: {e}")
        return False

def generate_gold_pages_summary(analysis_report: Dict[str, Any]) -> str:
    """Generate a markdown summary of Gold Pages analysis."""
    gold_metrics = analysis_report['gold_pages_metrics']
    accuracy_stats = analysis_report['accuracy_measurements']
    thresholds = analysis_report['success_thresholds']

    summary = f"""# Gold Pages Analysis Summary

## Overview
- **Mode**: {analysis_report['metadata']['mode_name']}
- **Analysis Date**: {analysis_report['metadata']['analysis_date']}
- **Gold Pages Version**: {analysis_report['metadata']['gold_pages_version']}
- **OCR Engine**: DeepSeek-OCR

## Gold Pages Metrics
- **Total Gold Pages**: {gold_metrics['total_gold_pages']}
- **Gold Pages Loaded**: {gold_metrics['gold_pages_loaded']}
- **Pages with Matches**: {gold_metrics['gold_pages_with_matches']}

## Accuracy Measurements
"""

    if accuracy_stats:
        summary += f"- **Average CER**: {accuracy_stats.get('avg_cer', 'N/A')}\n"
        summary += f"- **Average WER**: {accuracy_stats.get('avg_wer', 'N/A')}\n"
        summary += f"- **Success Rate**: {accuracy_stats.get('success_rate', 'N/A')}\n"
        summary += f"- **High Accuracy Pages**: {accuracy_stats.get('high_accuracy_pages', 0)}\n"
    else:
        summary += "- No accuracy measurements available\n"

    # Add recommendations
    summary += "\n## Recommendations\n"

    success_rate = accuracy_stats.get('success_rate', 0) if accuracy_stats else 0
    if success_rate >= thresholds['target_accuracy']:
        summary += "- [SUCCESS] Excellent accuracy - DeepSeek-OCR performing well\n"
    elif success_rate >= thresholds['min_accuracy']:
        summary += "- [OK] Good accuracy - DeepSeek-OCR acceptable\n"
    else:
        summary += "- [WARNING] Low accuracy - DeepSeek-OCR may need optimization\n"

    if gold_metrics['gold_pages_with_matches'] == 0:
        summary += "- [WARNING] No Gold Pages matches - ensure proper file naming\n"
    else:
        summary += "- [OK] Gold Pages integration working\n"

    return summary

# Import DeepSeek components
from deepseek_ocr import is_deepseek_available, DeepSeekOCRError

def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"[ERROR] Config file not found: {config_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON in config file: {e}")
        sys.exit(1)

def validate_deepseek_config(config: Dict[str, Any]) -> bool:
    """Validate DeepSeek evaluation configuration."""
    # Check that OCR engine is set to deepseek
    ocr_config = config.get('ocr', {})
    if ocr_config.get('engine') != 'deepseek':
        print("[ERROR] OCR engine must be set to 'deepseek' for DeepSeek evaluation")
        return False

    # Check DeepSeek configuration
    deepseek_config = config.get('deepseek', {})
    required_deepseek_fields = ['model', 'device', 'prompt', 'base_size', 'image_size']
    for field in required_deepseek_fields:
        if field not in deepseek_config:
            print(f"[ERROR] Missing required DeepSeek config field: {field}")
            return False

    # Check that DeepSeek is available
    if not is_deepseek_available():
        print("[ERROR] DeepSeek-OCR is not available. Check installation and dependencies.")
        return False

    print("[OK] DeepSeek-OCR configuration validated")
    return True

def create_deepseek_pipeline_config(config: Dict[str, Any]) -> Any:
    """Create PipelineConfig object from loaded config for DeepSeek evaluation."""
    try:
        from comprehensive_pipeline import PipelineConfig

        # Map configuration to PipelineConfig
        pipeline_config = PipelineConfig(
            # OCR settings - force DeepSeek
            ocr_engine="deepseek",

            # LLM settings (for correction)
            llm_provider=config.get('llm', {}).get('provider', 'none'),
            llm_model=config.get('llm', {}).get('model', ''),
            llm_base_url=config.get('llm', {}).get('base_url', 'http://localhost:11434'),
            llm_timeout=config.get('llm', {}).get('timeout', 30),

            # Processing settings
            dpi=config.get('ocr', {}).get('dpi', 300),
            enable_reading_order=config.get('ocr', {}).get('enable_reading_order', True),
            enable_llm_correction=config.get('llm', {}).get('enable_correction', False),
            enable_llm_v3=config.get('llm', {}).get('enable_llm_v3', False),
            max_concurrent_corrections=config.get('llm', {}).get('max_concurrent_corrections', 1),

            # Akkadian settings
            enable_akkadian_extraction=config.get('akkadian', {}).get('enable_extraction', False),
            akkadian_confidence_threshold=config.get('akkadian', {}).get('confidence_threshold', 0.8),

            # Output settings
            output_csv_metadata=config.get('csv_output', {}).get('include_metadata', True),
            create_html_overlay=config.get('processing', {}).get('create_html_overlay', False),
            create_visualization=config.get('processing', {}).get('create_visualizations', False)
        )

        return pipeline_config

    except ImportError:
        print("[ERROR] Unable to import comprehensive_pipeline module")
        print("Please ensure all required modules are available")
        sys.exit(1)

def process_single_file_deepseek(file_path: str, config: Dict[str, Any], pipeline_config, pipeline, gold_pages_manager, accuracy_measurer, logger) -> Dict[str, Any]:
    """Process a single file with DeepSeek-OCR evaluation."""
    logger.info(f"[SEARCH] Processing with DeepSeek-OCR: {file_path}")

    # Determine output directory
    output_dir = config['output']['output_directory']

    if config['output'].get('create_subdirectories', True):
        # Create subdirectory based on input filename
        file_stem = Path(file_path).stem
        output_dir = os.path.join(output_dir, file_stem)

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    try:
        # Get gold standard for this file
        gold_standard = gold_pages_manager.get_gold_standard_for_file(file_path)
        if not gold_standard:
            logger.warning(f"No gold standard found for {file_path}")
            return {'error': 'No gold standard available', 'file': file_path}

        # Process the file with DeepSeek-OCR
        start_time = time.time()

        if file_path.lower().endswith('.pdf'):
            result = pipeline.process_pdf(
                pdf_path=file_path,
                output_dir=output_dir
            )
        else:
            # For image files
            result = pipeline.process_image(
                image_path=file_path,
                output_dir=output_dir
            )

        processing_time = time.time() - start_time

        # Extract OCR text for accuracy measurement
        ocr_text = ""
        if 'page_results' in result:
            for page_result in result['page_results']:
                if 'ocr_text' in page_result:
                    ocr_text += page_result['ocr_text'] + "\n"

        # Measure accuracy against gold standard
        accuracy_metrics = accuracy_measurer.measure_accuracy(
            ocr_text=ocr_text,
            gold_standard=gold_standard,
            file_path=file_path
        )

        # Add DeepSeek-specific metadata
        result['deepseek_metadata'] = {
            'model': config.get('deepseek', {}).get('model'),
            'device': config.get('deepseek', {}).get('device'),
            'processing_time': processing_time,
            'accuracy_metrics': accuracy_metrics
        }

        result['file'] = file_path
        return result

    except DeepSeekOCRError as e:
        logger.error(f"DeepSeek-OCR failed for {file_path}: {e}")
        return {'error': f'DeepSeek-OCR error: {e}', 'file': file_path}
    except Exception as e:
        logger.error(f"Failed to process {file_path}: {e}")
        return {'error': str(e), 'file': file_path}

def print_deepseek_summary(results: List[Dict[str, Any]], config: Dict[str, Any], logger):
    """Print DeepSeek evaluation summary."""
    print("\n" + "="*80)
    print("[TARGET] DEEPSEEK-OCR EVALUATION SUMMARY")
    print("="*80)

    successful = [r for r in results if 'error' not in r]
    failed = [r for r in results if 'error' in r]

    print(f"[STATS] Files processed: {len(results)}")
    print(f"[OK] Successful: {len(successful)}")
    print(f"[ERROR] Failed: {len(failed)}")

    # Calculate aggregate metrics
    total_accuracy = 0.0
    total_files_with_accuracy = 0
    total_processing_time = 0.0

    for result in successful:
        metadata = result.get('deepseek_metadata', {})
        accuracy = metadata.get('accuracy_metrics', {}).get('overall_accuracy', 0)
        if accuracy > 0:
            total_accuracy += accuracy
            total_files_with_accuracy += 1

        processing_time = metadata.get('processing_time', 0)
        total_processing_time += processing_time

    if total_files_with_accuracy > 0:
        avg_accuracy = total_accuracy / total_files_with_accuracy
        print(f"Average accuracy: {avg_accuracy:.1f}%")
    else:
        print("Accuracy metrics: No accuracy data available")

    if successful:
        avg_processing_time = total_processing_time / len(successful)
        print(f"Average processing time: {avg_processing_time:.1f}s")

    # Show DeepSeek configuration
    deepseek_config = config.get('deepseek', {})
    print("\nDeepSeek Configuration:")
    print(f"   Model: {deepseek_config.get('model', 'unknown')}")
    print(f"   Device: {deepseek_config.get('device', 'unknown')}")
    print(f"   Image Size: {deepseek_config.get('image_size', 'unknown')}")

    if failed:
        print("\nFailed files:")
        for result in failed:
            print(f"   File: {result.get('file', 'Unknown')}: {result.get('error', 'Unknown error')}")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="DeepSeek-OCR Evaluation Runner")
    parser.add_argument("-c", "--config", required=True,
                       help="Path to DeepSeek evaluation config file")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be processed without actually processing")

    args = parser.parse_args()

    print("[START] DeepSeek-OCR Evaluation Runner")
    print("=" * 50)

    # Load configuration
    print(f"[FILE] Loading configuration from: {args.config}")
    config = load_config(args.config)

    # Validate configuration
    print("[SEARCH] Validating DeepSeek configuration...")
    if not validate_deepseek_config(config):
        sys.exit(1)
    print("[OK] Configuration valid")

    if args.dry_run:
        print("\n[DRY RUN] Dry run completed")
        return 0

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('eval_deepseek.log', encoding='utf-8')
        ]
    )
    logger = logging.getLogger(__name__)

    # Initialize Gold Pages and accuracy measurement
    try:
        gold_pages_manager = create_gold_pages_manager()
        accuracy_measurer = create_accuracy_measurer()

        # Initialize V3 language detection if needed
        if config.get('language_detection', {}).get('enable_enhanced_detection', False):
            if initialize_language_detection_v3(config):
                patch_comprehensive_pipeline_metrics()
            else:
                print("[ERROR] Failed to initialize language detection V3")
                return 1

    except Exception as e:
        print(f"[ERROR] Failed to initialize evaluation components: {e}")
        return 1

    try:
        # Create pipeline configuration
        print("[INIT] Initializing DeepSeek evaluation pipeline...")
        from comprehensive_pipeline import ComprehensivePipeline

        pipeline_config = create_deepseek_pipeline_config(config)
        pipeline = ComprehensivePipeline(pipeline_config)

        print("[OK] DeepSeek evaluation pipeline initialized")

        # Get files to process
        files_to_process = []
        input_config = config['input']

        if not input_config['process_all_files']:
            # Process single file
            specific_file = input_config['specific_file']
            if os.path.exists(specific_file):
                files_to_process.append(specific_file)
        else:
            # Process all files in directory
            input_dir = input_config['input_directory']
            supported_formats = input_config['supported_formats']
            recursive = input_config.get('recursive_search', False)

            for format_ext in supported_formats:
                import glob
                pattern = os.path.join(input_dir, f"**/*{format_ext}" if recursive else f"*{format_ext}")
                if recursive:
                    files_to_process.extend(glob.glob(pattern, recursive=True))
                else:
                    files_to_process.extend(glob.glob(pattern))

        files_to_process = sorted(list(set(files_to_process)))  # Remove duplicates

        if not files_to_process:
            print("[ERROR] No files found to process")
            return 1

        print(f"\n[FILES] Found {len(files_to_process)} file(s) to evaluate with DeepSeek-OCR:")
        for i, file_path in enumerate(files_to_process, 1):
            print(f"   {i}. {file_path}")

        # Process files
        print(f"\n[START] Starting DeepSeek-OCR evaluation of {len(files_to_process)} file(s)...")
        start_time = time.time()
        results = []

        for i, file_path in enumerate(files_to_process, 1):
            print(f"\n[SEARCH] Evaluating file {i}/{len(files_to_process)}: {os.path.basename(file_path)}")

            result = process_single_file_deepseek(
                file_path, config, pipeline_config, pipeline,
                gold_pages_manager, accuracy_measurer, logger
            )
            results.append(result)

        # Print summary
        print_deepseek_summary(results, config, logger)

        # Post-process with Gold Pages analysis (same as gold pages eval)
        print("\n[STATS] Running Gold Pages analysis...")
        try:
            # Get evaluation results
            eval_config = config['evaluation']
            mode_name = eval_config.get('mode_name', 'deepseek_basic')
            output_dir = config['output']['output_directory']

            # Run Gold Pages analysis
            gold_pages_analysis = run_gold_pages_analysis(
                gold_pages_manager,
                accuracy_measurer,
                output_dir,
                mode_name
            )

            if gold_pages_analysis:
                print("[OK] Gold Pages analysis completed successfully")
                print(f"[REPORT] Analysis results saved to: {output_dir}")
            else:
                print("[WARNING] Gold Pages analysis completed with warnings")

        except Exception as e:
            print(f"[ERROR] Gold Pages analysis failed: {e}")

        # Print V3 Language Detection summary if enabled
        try:
            if config.get('language_detection', {}).get('enable_enhanced_detection', False):
                print_language_detection_summary()
        except Exception as e:
            print(f"[WARNING] Language detection summary failed: {e}")

        # Save evaluation report
        output_dir = config['output']['output_directory']
        report_path = os.path.join(output_dir, f"eval_deepseek_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

        try:
            os.makedirs(output_dir, exist_ok=True)
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'evaluation_mode': 'deepseek_ocr',
                    'config': config,
                    'results': results,
                    'summary': {
                        'total_files': len(results),
                        'successful': len([r for r in results if 'error' not in r]),
                        'failed': len([r for r in results if 'error' in r]),
                        'total_time': time.time() - start_time
                    }
                }, f, indent=2, ensure_ascii=False)

            print(f"\n[REPORT] Evaluation report saved: {report_path}")

        except Exception as e:
            logger.warning(f"Could not save evaluation report: {e}")

        # Return exit code
        failed_count = len([r for r in results if 'error' in r])
        if failed_count == 0:
            print("\n[SUCCESS] DeepSeek-OCR evaluation completed successfully!")
            return 0
        elif failed_count < len(results):
            print(f"\n[WARNING]  Evaluation completed with {failed_count} failures")
            return 0
        else:
            print(f"\n[ERROR] All evaluations failed")
            return 1

    finally:
        # Cleanup
        try:
            cleanup_language_detection_v3()
            unpatch_comprehensive_pipeline_metrics()
        except:
            pass

if __name__ == "__main__":
    exit(main())
