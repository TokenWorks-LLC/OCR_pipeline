#!/usr/bin/env python3
"""
OCR Pipeline - Analysis Menu Integration
Provides menu-driven access to summary analysis functionality
"""

import json
import sys
import os
import subprocess
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def find_evaluation_directories() -> List[str]:
    """Find all available evaluation output directories."""
    eval_base = Path("./data/eval_output")
    if not eval_base.exists():
        return []
    
    eval_dirs = []
    for item in eval_base.iterdir():
        if item.is_dir() and item.name.startswith("eval_research_results_"):
            eval_dirs.append(str(item))
    
    return sorted(eval_dirs)

def list_available_evaluations() -> Dict[str, List[str]]:
    """List all available evaluation results with details."""
    eval_dirs = find_evaluation_directories()
    evaluations = {}
    
    for eval_dir in eval_dirs:
        mode_name = Path(eval_dir).name.replace("eval_research_results_", "")
        evaluations[mode_name] = []
        
        # Find all comprehensive reports in this directory
        for report_file in Path(eval_dir).rglob("comprehensive_report.json"):
            try:
                with open(report_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                doc_name = report_file.parent.name
                evaluations[mode_name].append({
                    'name': doc_name,
                    'path': str(report_file.parent),
                    'pages': data.get('total_pages', 0),
                    'success_rate': data.get('successful_pages', 0) / data.get('total_pages', 1) if data.get('total_pages', 0) > 0 else 0,
                    'confidence': data.get('page_statistics', [{}])[0].get('avg_confidence', 0) if data.get('page_statistics') else 0
                })
            except Exception as e:
                print(f"⚠️  Error reading {report_file}: {e}")
    
    return evaluations

def display_evaluation_summary(evaluations: Dict[str, List[Dict]]) -> None:
    """Display a summary of available evaluations."""
    print("\n📊 AVAILABLE EVALUATION RESULTS")
    print("=" * 60)
    
    total_docs = 0
    for mode, docs in evaluations.items():
        if docs:
            print(f"\n🔧 {mode.upper()} Mode:")
            print(f"   Documents: {len(docs)}")
            total_docs += len(docs)
            
            # Show top 3 documents by success rate
            sorted_docs = sorted(docs, key=lambda x: x['success_rate'], reverse=True)
            for i, doc in enumerate(sorted_docs[:3]):
                print(f"   {i+1}. {doc['name'][:50]}...")
                print(f"      Pages: {doc['pages']}, Success: {doc['success_rate']:.1%}, Confidence: {doc['confidence']:.3f}")
            
            if len(docs) > 3:
                print(f"   ... and {len(docs) - 3} more documents")
    
    print(f"\n📈 Total Documents Available: {total_docs}")
    print("=" * 60)

def select_evaluation_modes(evaluations: Dict[str, List[Dict]]) -> List[str]:
    """Interactive selection of evaluation modes to compare."""
    available_modes = [mode for mode, docs in evaluations.items() if docs]
    
    if not available_modes:
        print("❌ No evaluation results found!")
        return []
    
    if len(available_modes) == 1:
        print(f"📊 Only one mode available: {available_modes[0]}")
        return [f"./data/eval_output/eval_research_results_{available_modes[0]}"]
    
    print("\n🔍 SELECT MODES TO COMPARE")
    print("=" * 40)
    print("Available modes:")
    for i, mode in enumerate(available_modes, 1):
        doc_count = len(evaluations[mode])
        print(f"   {i}. {mode.upper()} ({doc_count} documents)")
    
    print(f"   {len(available_modes) + 1}. Compare ALL modes")
    print(f"   {len(available_modes) + 2}. Custom selection")
    
    while True:
        try:
            choice = input(f"\nSelect option (1-{len(available_modes) + 2}): ").strip()
            
            if choice == str(len(available_modes) + 1):
                # Compare all modes
                return [f"./data/eval_output/eval_research_results_{mode}" for mode in available_modes]
            
            elif choice == str(len(available_modes) + 2):
                # Custom selection
                return custom_mode_selection(available_modes)
            
            else:
                mode_idx = int(choice) - 1
                if 0 <= mode_idx < len(available_modes):
                    selected_mode = available_modes[mode_idx]
                    return [f"./data/eval_output/eval_research_results_{selected_mode}"]
                else:
                    print("❌ Invalid selection. Please try again.")
        
        except ValueError:
            print("❌ Please enter a valid number.")
        except KeyboardInterrupt:
            print("\n👋 Cancelled by user.")
            return []

def custom_mode_selection(available_modes: List[str]) -> List[str]:
    """Allow user to select multiple modes for comparison."""
    print("\n📋 CUSTOM MODE SELECTION")
    print("Enter mode numbers separated by commas (e.g., 1,3,4)")
    
    while True:
        try:
            selection = input("Modes to compare: ").strip()
            if not selection:
                return []
            
            indices = [int(x.strip()) - 1 for x in selection.split(',')]
            selected_modes = []
            
            for idx in indices:
                if 0 <= idx < len(available_modes):
                    selected_modes.append(available_modes[idx])
                else:
                    print(f"❌ Invalid index: {idx + 1}")
                    break
            else:
                return [f"./data/eval_output/eval_research_results_{mode}" for mode in selected_modes]
        
        except ValueError:
            print("❌ Please enter valid numbers separated by commas.")
        except KeyboardInterrupt:
            print("\n👋 Cancelled by user.")
            return []

def run_simple_analysis(eval_dirs: List[str], output_dir: str = "./data/analysis_output") -> bool:
    """Run the simple analysis."""
    try:
        from simple_analysis import SimpleAnalyzer
        
        print(f"\n🚀 Starting simple analysis of {len(eval_dirs)} evaluation mode(s)...")
        print("=" * 60)
        
        analyzer = SimpleAnalyzer(output_dir)
        results = analyzer.run_analysis(eval_dirs)
        
        print("\n✅ ANALYSIS COMPLETE!")
        print("=" * 60)
        print(f"📄 Report: {results['markdown_report']}")
        print(f"📊 JSON Summary: {results['json_summary']}")
        print(f"📈 Charts: {len(results['charts'])} generated")
        
        for chart in results['charts']:
            print(f"   📊 {Path(chart).name}")
        
        print(f"\n📁 All results saved to: {output_dir}")
        print("=" * 60)
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import analysis module: {e}")
        print("Please ensure required dependencies are installed:")
        print("   pip install matplotlib pandas")
        return False
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return False

def run_comprehensive_analysis(eval_dirs: List[str], output_dir: str = "./data/analysis_output") -> bool:
    """Run the comprehensive analysis."""
    try:
        from summary_analysis import SummaryAnalyzer
        
        print(f"\n🚀 Starting comprehensive analysis of {len(eval_dirs)} evaluation mode(s)...")
        print("=" * 60)
        
        analyzer = SummaryAnalyzer(output_dir)
        results = analyzer.run_analysis(eval_dirs)
        
        print("\n✅ ANALYSIS COMPLETE!")
        print("=" * 60)
        print(f"📄 Report: {results['markdown_report']}")
        print(f"📊 JSON Summary: {results['json_summary']}")
        print(f"📈 Charts: {len(results['charts'])} generated")
        
        for chart in results['charts']:
            print(f"   📊 {Path(chart).name}")
        
        print(f"\n📁 All results saved to: {output_dir}")
        print("=" * 60)
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import analysis module: {e}")
        print("Please ensure required dependencies are installed:")
        print("   pip install matplotlib pandas seaborn")
        return False
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return False

def show_analysis_menu() -> None:
    """Display the main analysis menu."""
    print("\n" + "="*60)
    print("📊 OCR PIPELINE - EVALUATION ANALYSIS")
    print("="*60)
    print("1. 📈 Run Summary Analysis (Simple)")
    print("2. 🎯 Run Comprehensive Analysis (Advanced)")
    print("3. 🔍 View Available Evaluations")
    print("4. 📊 Compare Specific Modes")
    print("5. 📋 Generate Baseline Report")
    print("6. 🚀 Quick Analysis (All Modes)")
    print("7. ❓ Help")
    print("8. 🚪 Exit")
    print("="*60)

def generate_baseline_report() -> None:
    """Generate a baseline OCR accuracy assessment report."""
    print("\n📋 GENERATING BASELINE OCR ACCURACY REPORT")
    print("=" * 50)
    
    baseline_content = """# OCR Pipeline Baseline Accuracy Assessment

## Overview
This document outlines the methodology for establishing OCR accuracy baselines for the OCR pipeline evaluation system.

## Baseline Metrics

### 1. Character-Level Accuracy
- **Definition**: Percentage of correctly recognized characters
- **Formula**: (Correct Characters / Total Characters) × 100
- **Target**: >95% for high-quality documents

### 2. Word-Level Accuracy
- **Definition**: Percentage of correctly recognized words
- **Formula**: (Correct Words / Total Words) × 100
- **Target**: >90% for academic documents

### 3. Line-Level Accuracy
- **Definition**: Percentage of correctly recognized text lines
- **Formula**: (Correct Lines / Total Lines) × 100
- **Target**: >85% for complex layouts

## Implementation Strategy

### Phase 1: Ground Truth Collection
1. Select representative documents from each evaluation mode
2. Manually transcribe 10-20 pages per mode
3. Create ground truth datasets with character-level annotations

### Phase 2: Baseline Calculation
```python
def calculate_ocr_baseline(ocr_results, ground_truth):
    char_accuracy = calculate_character_accuracy(ocr_results, ground_truth)
    word_accuracy = calculate_word_accuracy(ocr_results, ground_truth)
    line_accuracy = calculate_line_accuracy(ocr_results, ground_truth)
    
    return {
        'character_accuracy': char_accuracy,
        'word_accuracy': word_accuracy,
        'line_accuracy': line_accuracy,
        'overall_score': (char_accuracy + word_accuracy + line_accuracy) / 3
    }
```

### Phase 3: Mode-Specific Baselines
- **Basic Mode**: Focus on general text recognition
- **Akkadian Mode**: Focus on specialized academic text
- **Advanced Mode**: Focus on complex layouts and corrections

## Quality Thresholds

| Mode | Character Accuracy | Word Accuracy | Line Accuracy |
|------|-------------------|---------------|---------------|
| Basic | >95% | >90% | >85% |
| Akkadian | >92% | >87% | >80% |
| Advanced | >97% | >93% | >88% |

## Next Steps
1. Implement baseline calculation functions
2. Create ground truth datasets
3. Integrate baseline comparison into evaluation pipeline
4. Set up automated baseline monitoring
"""
    
    baseline_path = Path("./data/analysis_output/baseline_assessment.md")
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(baseline_path, 'w', encoding='utf-8') as f:
        f.write(baseline_content)
    
    print(f"✅ Baseline report generated: {baseline_path}")
    print("\n📋 This report provides:")
    print("   - Baseline accuracy metrics definitions")
    print("   - Implementation strategy")
    print("   - Quality thresholds by mode")
    print("   - Next steps for implementation")

def main():
    """Main function for the analysis menu."""
    print("🚀 OCR Pipeline - Evaluation Analysis System")
    
    while True:
        show_analysis_menu()
        
        try:
            choice = input("\nSelect option (1-8): ").strip()
            
            if choice == "1":
                # Run Summary Analysis (Simple)
                evaluations = list_available_evaluations()
                if not evaluations:
                    print("❌ No evaluation results found!")
                    print("Please run evaluations first using the main pipeline.")
                    continue
                
                eval_dirs = select_evaluation_modes(evaluations)
                if eval_dirs:
                    output_dir = input("Output directory (default: ./data/analysis_output): ").strip()
                    if not output_dir:
                        output_dir = "./data/analysis_output"
                    
                    run_simple_analysis(eval_dirs, output_dir)
            
            elif choice == "2":
                # Run Comprehensive Analysis (Advanced)
                evaluations = list_available_evaluations()
                if not evaluations:
                    print("❌ No evaluation results found!")
                    print("Please run evaluations first using the main pipeline.")
                    continue
                
                eval_dirs = select_evaluation_modes(evaluations)
                if eval_dirs:
                    output_dir = input("Output directory (default: ./data/analysis_output): ").strip()
                    if not output_dir:
                        output_dir = "./data/analysis_output"
                    
                    run_comprehensive_analysis(eval_dirs, output_dir)
            
            elif choice == "3":
                # View Available Evaluations
                evaluations = list_available_evaluations()
                display_evaluation_summary(evaluations)
            
            elif choice == "4":
                # Compare Specific Modes
                evaluations = list_available_evaluations()
                if not evaluations:
                    print("❌ No evaluation results found!")
                    continue
                
                eval_dirs = select_evaluation_modes(evaluations)
                if eval_dirs:
                    run_simple_analysis(eval_dirs)
            
            elif choice == "5":
                # Generate Baseline Report
                generate_baseline_report()
            
            elif choice == "6":
                # Quick Analysis (All Modes)
                evaluations = list_available_evaluations()
                if not evaluations:
                    print("❌ No evaluation results found!")
                    continue
                
                eval_dirs = [f"./data/eval_output/eval_research_results_{mode}" 
                            for mode, docs in evaluations.items() if docs]
                
                if eval_dirs:
                    print(f"🚀 Running quick analysis on {len(eval_dirs)} modes...")
                    run_simple_analysis(eval_dirs)
                else:
                    print("❌ No evaluation data found!")
            
            elif choice == "7":
                # Help
                print("\n📚 HELP - SIMPLIFIED EVALUATION ANALYSIS")
                print("=" * 50)
                print("This system analyzes OCR pipeline evaluation results and provides:")
                print("• Performance comparisons between modes")
                print("• Cost of compute analysis (time per word/token)")
                print("• Resource usage monitoring (CPU/memory)")
                print("• Simple visual charts and reports")
                print("• Baseline accuracy assessment guidelines")
                print("\nRequirements:")
                print("• Evaluation results in ./data/eval_output/")
                print("• Python packages: matplotlib, pandas (optional)")
                print("\nUsage:")
                print("• Option 1: Interactive mode selection")
                print("• Option 5: Quick analysis of all available modes")
                print("• Option 4: Generate baseline assessment guidelines")
                print("\nNew Metrics:")
                print("• Word count (modern languages only)")
                print("• Token count (text elements for Akkadian)")
                print("• Time per word/token (cost efficiency)")
                print("• CPU and memory usage during processing")
            
            elif choice == "8":
                # Exit
                print("👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice. Please select 1-8.")
        
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
