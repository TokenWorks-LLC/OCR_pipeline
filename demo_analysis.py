#!/usr/bin/env python3
"""
Demo script for OCR Pipeline Analysis System
Shows how to use the summary analysis and baseline accuracy features
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def demo_baseline_accuracy():
    """Demonstrate baseline accuracy calculation."""
    print("🔍 DEMO: Baseline OCR Accuracy Calculation")
    print("=" * 50)
    
    try:
        from baseline_accuracy import BaselineAccuracyCalculator
        
        calculator = BaselineAccuracyCalculator()
        
        # Example OCR results with some errors
        ocr_text = "This is a test document with OCR err0rs and mispellings."
        ground_truth = "This is a test document with OCR errors and misspellings."
        
        print(f"OCR Text:     {ocr_text}")
        print(f"Ground Truth: {ground_truth}")
        print()
        
        # Calculate accuracy metrics
        metrics = calculator.calculate_overall_accuracy(ocr_text, ground_truth)
        
        print("📊 ACCURACY METRICS:")
        print(f"Overall Score: {metrics['overall_score']:.3f}")
        print(f"Character Accuracy: {metrics['character_accuracy']['accuracy']:.1%}")
        print(f"Word Accuracy: {metrics['word_accuracy']['accuracy']:.1%}")
        print(f"Line Accuracy: {metrics['line_accuracy']['accuracy']:.1%}")
        print()
        
        # Show detailed character metrics
        char_metrics = metrics['character_accuracy']
        print("📝 CHARACTER-LEVEL DETAILS:")
        print(f"Characters compared: {char_metrics['characters_compared']}")
        print(f"Characters matched: {char_metrics['characters_matched']}")
        print(f"Characters missed: {char_metrics['characters_missed']}")
        print(f"Error rate: {char_metrics['error_rate']:.1%}")
        
    except ImportError as e:
        print(f"❌ Could not import baseline accuracy module: {e}")
        print("Please ensure the module is in the correct location.")

def demo_summary_analysis():
    """Demonstrate simplified analysis functionality."""
    print("\n📊 DEMO: Simplified Analysis System")
    print("=" * 50)
    
    # Check if evaluation data exists
    eval_dirs = [
        "./data/eval_output/eval_research_results_basic",
        "./data/eval_output/eval_research_results_akkadian"
    ]
    
    available_dirs = [d for d in eval_dirs if Path(d).exists()]
    
    if not available_dirs:
        print("❌ No evaluation data found!")
        print("Please run evaluations first using:")
        print("   python run_pipeline.py -c config_eval_basic.json --eval-mode")
        print("   python run_pipeline.py -c config_eval_akkadian.json --eval-mode")
        return
    
    print(f"✅ Found evaluation data in {len(available_dirs)} directory(ies)")
    for dir_path in available_dirs:
        print(f"   - {dir_path}")
    
    print("\n🚀 To run simplified analysis:")
    print("   python run_analysis_menu.py")
    print("\n📋 Available analysis options:")
    print("   1. Compare Basic vs Akkadian modes")
    print("   2. Generate cost of compute analysis")
    print("   3. Create performance visualizations")
    print("   4. Generate baseline accuracy assessment")
    print("\n🆕 New Features:")
    print("   • Word counting for modern languages")
    print("   • Token counting (text elements for Akkadian)")
    print("   • Time per word/token metrics")
    print("   • CPU and memory usage monitoring")

def demo_menu_integration():
    """Demonstrate menu integration."""
    print("\n🎛️ DEMO: Menu Integration")
    print("=" * 50)
    
    print("The analysis system is integrated into existing menus:")
    print()
    print("📱 Quick Start Menu (quick_start.py):")
    print("   - Option 5: Run Summary Analysis")
    print()
    print("🖥️  Windows Batch Menu (run_ocr.bat):")
    print("   - Option 6: Run Summary Analysis")
    print()
    print("🔧 Direct Command Line:")
    print("   python run_analysis_menu.py")
    print("   python src/summary_analysis.py <eval_dir1> <eval_dir2>")
    print()
    print("📊 Analysis Features:")
    print("   - Interactive mode selection")
    print("   - Comprehensive performance comparison")
    print("   - Visual charts and graphs")
    print("   - Detailed markdown reports")
    print("   - JSON summaries for programmatic access")
    print("   - Baseline accuracy assessment guidelines")

def main():
    """Run all demos."""
    print("🚀 OCR PIPELINE ANALYSIS SYSTEM DEMO")
    print("=" * 60)
    
    demo_baseline_accuracy()
    demo_summary_analysis()
    demo_menu_integration()
    
    print("\n" + "=" * 60)
    print("✅ DEMO COMPLETE")
    print("=" * 60)
    print("Next steps:")
    print("1. Install analysis dependencies: pip install -r requirements_analysis.txt")
    print("2. Run evaluations to generate data")
    print("3. Use the analysis menu to compare results")
    print("4. Implement ground truth data for baseline accuracy")

if __name__ == "__main__":
    main()
