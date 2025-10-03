#!/usr/bin/env python3
"""
Gold Pages Evaluation Runner - Enhanced evaluation with ground truth integration
Extends existing evaluation modes with Gold Pages support for accuracy measurement and LLM improvement tracking.
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

def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in config file: {e}")
        sys.exit(1)

def validate_gold_pages_config(config: Dict[str, Any]) -> bool:
    """Validate Gold Pages configuration."""
    if 'gold_pages' not in config:
        print("❌ Gold Pages configuration not found")
        return False
    
    gold_config = config['gold_pages']
    
    # Check required fields
    required_fields = ['enable_gold_pages', 'gold_pages_directory', 'enable_accuracy_measurement']
    for field in required_fields:
        if field not in gold_config:
            print(f"❌ Missing required Gold Pages field: {field}")
            return False
    
    # Check if Gold Pages is enabled
    if not gold_config.get('enable_gold_pages', False):
        print("❌ Gold Pages is disabled in configuration")
        return False
    
    return True

def setup_gold_pages_system(config: Dict[str, Any]) -> tuple[GoldPagesManager, AccuracyMeasurer]:
    """Setup Gold Pages system components."""
    gold_config = config['gold_pages']
    
    # Initialize Gold Pages Manager
    gold_pages_dir = gold_config.get('gold_pages_directory', './data/gold_pages')
    gold_pages_manager = create_gold_pages_manager(gold_pages_dir)
    
    # Initialize Accuracy Measurer
    accuracy_measurer = create_accuracy_measurer(gold_pages_manager)
    
    # Set success thresholds
    thresholds = gold_config.get('success_thresholds', {})
    accuracy_measurer.min_improvement_threshold = thresholds.get('min_improvement', 0.05)
    accuracy_measurer.max_processing_time_increase = thresholds.get('max_time_increase', 0.5)
    accuracy_measurer.cost_per_accuracy_point = thresholds.get('cost_per_accuracy_point', 0.10)
    
    return gold_pages_manager, accuracy_measurer

def patch_smart_llm_correction():
    """Patch the comprehensive pipeline to use smart LLM correction."""
    try:
        # Import the comprehensive pipeline
        from comprehensive_pipeline import ComprehensivePipeline
        
        # Store original method
        original_process_single_page = ComprehensivePipeline.process_single_page
        
        def smart_process_single_page(self, pdf_path: str, page_num: int, output_dir: str, 
                                    base_filename: str) -> tuple:
            """Enhanced process_single_page with smart LLM correction."""
            import time
            from smart_llm_correction import correct_ocr_lines_smart
            
            # Get the original method result
            result, stats = original_process_single_page(self, pdf_path, page_num, output_dir, base_filename)
            
            if result and hasattr(result, 'text_elements') and result.text_elements:
                # Apply smart LLM correction to text elements
                llm_config = self.config.__dict__ if hasattr(self.config, '__dict__') else {}
                
                # Check if smart triggering is enabled
                if llm_config.get('smart_triggering', False):
                    print(f"Applying smart LLM correction to page {page_num}...")
                    
                    # Convert text elements to Line objects for smart correction
                    from ocr_utils import Line
                    lines = []
                    for elem in result.text_elements:
                        if 'text' in elem and 'bbox' in elem and 'conf' in elem:
                            lines.append(Line(
                                text=elem['text'],
                                conf=elem.get('conf', 0.0),
                                bbox=elem['bbox'],
                                engine=elem.get('engine', 'unknown')
                            ))
                    
                    # Apply smart correction
                    modern_lang_max_conf = llm_config.get('modern_lang_max_confidence', 0.8)
                    akkadian_always = llm_config.get('akkadian_always_correct', True)
                    
                    corrected_lines, correction_stats = correct_ocr_lines_smart(
                        lines, 
                        language_hint=None,
                        modern_lang_max_confidence=modern_lang_max_conf,
                        akkadian_always_correct=akkadian_always
                    )
                    
                    # Update text elements with corrected text
                    for i, (elem, corrected_line) in enumerate(zip(result.text_elements, corrected_lines)):
                        if i < len(corrected_lines):
                            elem['text'] = corrected_line.text
                            elem['original_text'] = lines[i].text if lines[i].text != corrected_line.text else None
                            elem['llm_corrected'] = lines[i].text != corrected_line.text
                    
                    # Update stats
                    stats['smart_llm_stats'] = correction_stats
                    stats['llm_lines_processed'] = correction_stats.get('lines_processed', 0)
                    stats['llm_lines_changed'] = correction_stats.get('lines_changed', 0)
                    stats['llm_lines_skipped'] = correction_stats.get('lines_skipped', 0)
                    stats['llm_akkadian_lines'] = correction_stats.get('lines_akkadian', 0)
                    stats['llm_low_conf_lines'] = correction_stats.get('lines_low_conf', 0)
                    
                    print(f"   Smart LLM: {correction_stats.get('lines_changed', 0)} changed, "
                          f"{correction_stats.get('lines_skipped', 0)} skipped")
            
            return result, stats
        
        # Patch the method
        ComprehensivePipeline.process_single_page = smart_process_single_page
        print("Smart LLM correction patched into Gold Pages evaluation pipeline")
        
    except Exception as e:
        print(f"Failed to patch smart LLM correction: {e}")
        raise

def run_gold_pages_evaluation(config_path: str, validate_only: bool = False) -> int:
    """Run Gold Pages evaluation with enhanced accuracy measurement."""
    
    print("Gold Pages Evaluation Runner")
    print("=" * 60)
    
    # Load configuration
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)
    
    # Check if Smart LLM is enabled
    llm_config = config.get('llm', {})
    smart_llm_enabled = llm_config.get('smart_triggering', False)
    
    if smart_llm_enabled:
        print("Smart LLM triggering enabled")
        print(f"   Modern language max confidence: {llm_config.get('modern_lang_max_confidence', 0.8)}")
        print(f"   Akkadian always correct: {llm_config.get('akkadian_always_correct', True)}")
    else:
        print("Standard LLM correction (no smart triggering)")
    
    # Validate Gold Pages configuration
    print("Validating Gold Pages configuration...")
    if not validate_gold_pages_config(config):
        print("ERROR: Gold Pages configuration validation failed")
        return 1
    print("OK: Gold Pages configuration valid")
    
    if validate_only:
        print("OK: Gold Pages configuration validation completed successfully")
        return 0
    
    # Patch Smart LLM correction if enabled
    if smart_llm_enabled:
        print("\nPatching Smart LLM correction...")
        try:
            patch_smart_llm_correction()
        except Exception as e:
            print(f"Failed to patch smart LLM correction: {e}")
            return 1
    
    # Setup Gold Pages system
    print("🔧 Setting up Gold Pages system...")
    try:
        gold_pages_manager, accuracy_measurer = setup_gold_pages_system(config)
        print(f"✅ Gold Pages Manager initialized: {len(gold_pages_manager.get_all_gold_pages())} entries")
        print(f"✅ Accuracy Measurer initialized with thresholds:")
        print(f"   - Min improvement: {accuracy_measurer.min_improvement_threshold:.1%}")
        print(f"   - Max time increase: {accuracy_measurer.max_processing_time_increase:.1%}")
        print(f"   - Cost per accuracy point: ${accuracy_measurer.cost_per_accuracy_point:.2f}")
    except Exception as e:
        print(f"❌ Failed to setup Gold Pages system: {e}")
        return 1
    
    # Check if Gold Pages data exists
    gold_pages_count = len(gold_pages_manager.get_all_gold_pages())
    if gold_pages_count == 0:
        print("⚠️  No Gold Pages data found!")
        print("   This evaluation will run without ground truth comparison.")
        print("   To add Gold Pages data, use the Gold Pages Manager.")
    else:
        print(f"📊 Found {gold_pages_count} Gold Pages entries for reference")
    
    # Setup V3 Language Detection if enabled
    is_v3_config = 'v3' in config_path.lower() or 'language_detection' in config
    if is_v3_config:
        print("\n🔧 Setting up V3 Language Detection...")
        try:
            if initialize_language_detection_v3(config):
                print("✅ V3 Language Detection initialized successfully")
                # Patch metrics collection
                if patch_comprehensive_pipeline_metrics():
                    print("✅ Language detection metrics patching applied")
                else:
                    print("⚠️  Language detection metrics patching failed")
            else:
                print("⚠️  V3 Language Detection initialization failed, continuing with standard detection")
        except Exception as e:
            print(f"⚠️  V3 Language Detection setup failed: {e}")
            print("   Continuing with standard language detection")
    else:
        print("\n📝 Using standard language detection (V2 mode)")
    
    # Run the standard evaluation with Gold Pages enhancement
    print("\n🔄 Running enhanced evaluation...")
    
    # Import and run the standard evaluation
    try:
        from run_eval_incremental import main as run_standard_eval
        
        # Override sys.argv to pass our config
        original_argv = sys.argv
        sys.argv = ['run_eval_incremental.py', '-c', config_path]
        
        if validate_only:
            sys.argv.append('--validate-only')
        
        # Run standard evaluation
        result = run_standard_eval()
        
        # Restore original argv
        sys.argv = original_argv
        
        if result != 0:
            print(f"❌ Standard evaluation failed with code: {result}")
            return result
        
        print("✅ Standard evaluation completed successfully")
        
    except Exception as e:
        print(f"❌ Failed to run standard evaluation: {e}")
        return 1
    
    # Post-process with Gold Pages analysis
    print("\n📊 Running Gold Pages analysis...")
    try:
        # Get evaluation results
        eval_config = config['evaluation']
        mode_name = eval_config.get('mode_name', 'unknown')
        output_dir = config['output']['output_directory']
        
        # Run Gold Pages analysis
        gold_pages_analysis = run_gold_pages_analysis(
            gold_pages_manager, 
            accuracy_measurer, 
            output_dir, 
            mode_name
        )
        
        if gold_pages_analysis:
            print("✅ Gold Pages analysis completed successfully")
            print(f"📈 Analysis results saved to: {output_dir}")
        else:
            print("⚠️  Gold Pages analysis completed with warnings")
        
    except Exception as e:
        print(f"❌ Gold Pages analysis failed: {e}")
        return 1
    
    # Print V3 Language Detection summary if enabled
    if is_v3_config:
        print("\n📊 V3 Language Detection Summary:")
        print_language_detection_summary()
    
    # Cleanup V3 Language Detection
    if is_v3_config:
        print("\n🧹 Cleaning up V3 Language Detection...")
        cleanup_language_detection_v3()
        unpatch_comprehensive_pipeline_metrics()
        print("✅ V3 Language Detection cleanup completed")
    
    print("\n🎉 Gold Pages evaluation completed successfully!")
    return 0

def run_gold_pages_analysis(gold_pages_manager: GoldPagesManager, 
                           accuracy_measurer: AccuracyMeasurer,
                           output_dir: str, 
                           mode_name: str) -> bool:
    """Run Gold Pages analysis on evaluation results."""
    try:
        # Create analysis output directory
        analysis_dir = Path(output_dir) / "gold_pages_analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Get Gold Pages metrics
        gold_metrics = gold_pages_manager.get_metrics()
        
        # Get accuracy measurements
        accuracy_stats = accuracy_measurer.get_summary_statistics()
        
        # Create analysis report
        analysis_report = {
            'metadata': {
                'mode_name': mode_name,
                'analysis_date': datetime.now().isoformat(),
                'gold_pages_version': '1.0'
            },
            'gold_pages_metrics': {
                'total_entries': gold_metrics.total_entries,
                'verified_entries': gold_metrics.verified_entries,
                'average_confidence': gold_metrics.average_confidence,
                'document_coverage': gold_metrics.document_coverage,
                'language_distribution': gold_metrics.language_distribution
            },
            'accuracy_measurements': accuracy_stats,
            'success_thresholds': {
                'min_improvement': accuracy_measurer.min_improvement_threshold,
                'max_time_increase': accuracy_measurer.max_processing_time_increase,
                'cost_per_accuracy_point': accuracy_measurer.cost_per_accuracy_point
            }
        }
        
        # Save analysis report
        report_file = analysis_dir / "gold_pages_analysis_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_report, f, indent=2, ensure_ascii=False)
        
        # Export accuracy measurements
        measurements_file = analysis_dir / "accuracy_measurements.json"
        accuracy_measurer.export_measurements(str(measurements_file))
        
        # Generate summary report
        summary_report = generate_gold_pages_summary(analysis_report)
        summary_file = analysis_dir / "gold_pages_summary.md"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary_report)
        
        print(f"📊 Gold Pages Analysis Summary:")
        print(f"   - Gold Pages entries: {gold_metrics.total_entries}")
        print(f"   - Accuracy measurements: {accuracy_stats.get('total_measurements', 0)}")
        print(f"   - Success rate: {accuracy_stats.get('success_rate', 0):.1%}")
        print(f"   - Average improvement: {accuracy_stats.get('average_improvement', 0):.1%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Gold Pages analysis failed: {e}")
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

## Gold Pages Metrics
- **Total Entries**: {gold_metrics['total_entries']}
- **Verified Entries**: {gold_metrics['verified_entries']}
- **Average Confidence**: {gold_metrics['average_confidence']:.3f}
- **Document Coverage**: {len(gold_metrics['document_coverage'])} documents

## Accuracy Measurements
- **Total Measurements**: {accuracy_stats.get('total_measurements', 0)}
- **Success Rate**: {accuracy_stats.get('success_rate', 0):.1%}
- **Average Improvement**: {accuracy_stats.get('average_improvement', 0):.1%}
- **Meets Threshold**: {accuracy_stats.get('meets_threshold_count', 0)} measurements

## Success Thresholds
- **Minimum Improvement**: {thresholds['min_improvement']:.1%}
- **Maximum Time Increase**: {thresholds['max_time_increase']:.1%}
- **Cost per Accuracy Point**: ${thresholds['cost_per_accuracy_point']:.2f}

## Recommendations
"""
    
    # Add recommendations based on results
    if accuracy_stats.get('total_measurements', 0) == 0:
        summary += "- ⚠️ No accuracy measurements available - ensure Gold Pages data is properly configured\n"
    elif accuracy_stats.get('success_rate', 0) < 0.5:
        summary += "- ⚠️ Low success rate - consider adjusting LLM correction parameters\n"
    else:
        summary += "- ✅ Good success rate - LLM correction is providing value\n"
    
    if gold_metrics['total_entries'] == 0:
        summary += "- ⚠️ No Gold Pages data - add ground truth data for better accuracy measurement\n"
    else:
        summary += "- ✅ Gold Pages data available - accuracy measurement is active\n"
    
    return summary

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Gold Pages Evaluation Runner")
    parser.add_argument("-c", "--config", default="config_eval_gold_pages_basic.json", 
                       help="Path to Gold Pages evaluation config file")
    parser.add_argument("--validate-only", action="store_true", 
                       help="Only validate configuration and exit")
    
    args = parser.parse_args()
    
    # Run Gold Pages evaluation
    result = run_gold_pages_evaluation(args.config, args.validate_only)
    
    if result == 0:
        print("\n🎉 Gold Pages evaluation completed successfully!")
    else:
        print(f"\n❌ Gold Pages evaluation failed with code: {result}")
    
    return result

if __name__ == "__main__":
    sys.exit(main())
