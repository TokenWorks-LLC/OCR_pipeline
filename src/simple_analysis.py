#!/usr/bin/env python3
"""
Simplified Analysis for OCR Pipeline Evaluation Results
Clean, elegant analysis with minimal dependencies (matplotlib + pandas only)
"""

import json
import os
import sys
import logging
import argparse
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Optional imports with graceful fallback
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

class SimpleAnalyzer:
    """Simplified analyzer for OCR pipeline evaluation results."""
    
    def __init__(self, output_dir: str = "./data/analysis_output"):
        """Initialize the analyzer with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(self.output_dir / 'analysis.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Analysis results storage
        self.analysis_results = {}
        
    def load_evaluation_data(self, eval_dirs: List[str]) -> Dict[str, List[Dict]]:
        """Load comprehensive reports from evaluation directories."""
        all_data = {}
        
        for eval_dir in eval_dirs:
            eval_path = Path(eval_dir)
            if not eval_path.exists():
                self.logger.warning(f"Evaluation directory not found: {eval_dir}")
                continue
                
            mode_name = eval_path.name
            all_data[mode_name] = []
            
            # Find all comprehensive_report.json files
            for report_file in eval_path.rglob("comprehensive_report.json"):
                try:
                    with open(report_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    data['report_path'] = str(report_file)
                    data['mode'] = mode_name
                    all_data[mode_name].append(data)
                    self.logger.info(f"Loaded: {str(report_file)}")
                except Exception as e:
                    self.logger.error(f"Failed to load {report_file}: {e}")
        
        return all_data
    
    def calculate_metrics(self, data: List[Dict]) -> Dict[str, Any]:
        """Calculate comprehensive metrics for a mode."""
        if not data:
            return {}
        
        # Basic metrics
        total_documents = len(data)
        total_pages = sum(d.get('pages_processed', 0) for d in data)
        successful_pages = sum(d.get('successful_pages', 0) for d in data)
        total_processing_time = sum(d.get('total_processing_time', 0) for d in data)
        
        # New cost of compute metrics
        total_text_elements = sum(d.get('total_text_elements', 0) for d in data)
        total_word_count = sum(d.get('total_word_count', 0) for d in data)
        total_token_count = sum(d.get('total_token_count', 0) for d in data)
        
        # Resource usage metrics
        resource_data = []
        for doc in data:
            for page_stat in doc.get('page_statistics', []):
                if 'resource_usage' in page_stat:
                    resource_data.append(page_stat['resource_usage'])
        
        # Calculate confidence scores
        confidence_scores = []
        for doc in data:
            for page_stat in doc.get('page_statistics', []):
                if 'avg_confidence' in page_stat:
                    confidence_scores.append(page_stat['avg_confidence'])
        
        # Calculate corrections and Akkadian translations
        total_corrections = sum(d.get('total_corrections', 0) for d in data)
        total_akkadian = sum(d.get('akkadian_translations_found', 0) for d in data)
        
        # Calculate Smart LLM metrics
        total_llm_lines_processed = 0
        total_llm_lines_changed = 0
        total_llm_lines_skipped = 0
        total_llm_akkadian_lines = 0
        total_llm_low_conf_lines = 0
        
        for doc in data:
            for page_stat in doc.get('page_statistics', []):
                if 'smart_llm_stats' in page_stat:
                    smart_stats = page_stat['smart_llm_stats']
                    total_llm_lines_processed += smart_stats.get('lines_processed', 0)
                    total_llm_lines_changed += smart_stats.get('lines_changed', 0)
                    total_llm_lines_skipped += smart_stats.get('lines_skipped', 0)
                    total_llm_akkadian_lines += smart_stats.get('lines_akkadian', 0)
                    total_llm_low_conf_lines += smart_stats.get('lines_low_conf', 0)
        
        # Calculate Language Detection V3 metrics
        language_detection_metrics = {}
        total_lang_pages = 0
        total_advanced_detections = 0
        total_character_based_detections = 0
        total_fallback_detections = 0
        language_distribution = {}
        total_fallback_percentage = 0.0
        threshold_exceeded_count = 0
        
        for doc in data:
            lang_metrics = doc.get('language_detection_v3', {})
            if lang_metrics:
                total_lang_pages += lang_metrics.get('total_pages', 0)
                total_advanced_detections += lang_metrics.get('advanced_detections', 0)
                total_character_based_detections += lang_metrics.get('character_based_detections', 0)
                total_fallback_detections += lang_metrics.get('fallback_detections', 0)
                total_fallback_percentage += lang_metrics.get('fallback_percentage', 0.0)
                if lang_metrics.get('threshold_exceeded', False):
                    threshold_exceeded_count += 1
                
                # Aggregate language distribution
                doc_dist = lang_metrics.get('language_distribution', {})
                for lang, count in doc_dist.items():
                    language_distribution[lang] = language_distribution.get(lang, 0) + count
        
        if total_lang_pages > 0:
            language_detection_metrics = {
                'total_pages_analyzed': total_lang_pages,
                'advanced_detections': total_advanced_detections,
                'character_based_detections': total_character_based_detections,
                'fallback_detections': total_fallback_detections,
                'advanced_success_rate': total_advanced_detections / total_lang_pages,
                'character_based_success_rate': total_character_based_detections / total_lang_pages,
                'fallback_rate': total_fallback_detections / total_lang_pages,
                'avg_fallback_percentage': total_fallback_percentage / total_documents if total_documents > 0 else 0,
                'threshold_exceeded_documents': threshold_exceeded_count,
                'language_distribution': language_distribution
            }
        
        # Calculate averages
        avg_confidence = statistics.mean(confidence_scores) if confidence_scores else 0
        avg_cpu_percent = 0.0
        avg_memory_mb = 0.0
        
        if resource_data:
            cpu_values = [r.get('cpu_percent', 0) for r in resource_data if r.get('available', False)]
            memory_values = [r.get('memory_mb', 0) for r in resource_data if r.get('available', False)]
            avg_cpu_percent = statistics.mean(cpu_values) if cpu_values else 0.0
            avg_memory_mb = statistics.mean(memory_values) if memory_values else 0.0
        
        return {
            'total_documents': total_documents,
            'total_pages': total_pages,
            'successful_pages': successful_pages,
            'success_rate': successful_pages / total_pages if total_pages > 0 else 0,
            'total_processing_time': total_processing_time,
            'avg_processing_time_per_page': total_processing_time / total_pages if total_pages > 0 else 0,
            'total_text_elements': total_text_elements,
            'total_word_count': total_word_count,
            'total_token_count': total_token_count,
            'avg_text_elements_per_page': total_text_elements / total_pages if total_pages > 0 else 0,
            'avg_word_count_per_page': total_word_count / total_pages if total_pages > 0 else 0,
            'avg_token_count_per_page': total_token_count / total_pages if total_pages > 0 else 0,
            'time_per_text_element': total_processing_time / total_text_elements if total_text_elements > 0 else 0,
            'time_per_word': total_processing_time / total_word_count if total_word_count > 0 else 0,
            'time_per_token': total_processing_time / total_token_count if total_token_count > 0 else 0,
            'avg_confidence': avg_confidence,
            'confidence_std': statistics.stdev(confidence_scores) if len(confidence_scores) > 1 else 0,
            'total_corrections': total_corrections,
            'total_akkadian_translations': total_akkadian,
            'avg_cpu_percent': avg_cpu_percent,
            'avg_memory_mb': avg_memory_mb,
            'confidence_scores': confidence_scores,
            # Smart LLM metrics
            'total_llm_lines_processed': total_llm_lines_processed,
            'total_llm_lines_changed': total_llm_lines_changed,
            'total_llm_lines_skipped': total_llm_lines_skipped,
            'total_llm_akkadian_lines': total_llm_akkadian_lines,
            'total_llm_low_conf_lines': total_llm_low_conf_lines,
            'llm_call_reduction_pct': (total_llm_lines_skipped / total_llm_lines_processed * 100) if total_llm_lines_processed > 0 else 0,
            'smart_llm_efficiency': (total_llm_lines_changed / total_llm_lines_processed * 100) if total_llm_lines_processed > 0 else 0,
            # Language Detection V3 metrics
            'language_detection_v3': language_detection_metrics
        }
    
    def compare_modes(self, mode_data: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Compare different evaluation modes."""
        comparison = {}
        
        for mode, data in mode_data.items():
            metrics = self.calculate_metrics(data)
            comparison[mode] = metrics
            self.analysis_results[mode] = metrics
        
        # Calculate relative improvements if we have multiple modes
        if len(comparison) >= 2:
            modes = list(comparison.keys())
            base_mode = modes[0]  # Use first mode as baseline
            
            for mode in modes[1:]:
                base_metrics = comparison[base_mode]
                mode_metrics = comparison[mode]
                
                improvements = {}
                for key in ['success_rate', 'avg_confidence', 'time_per_word', 'time_per_token']:
                    if key in base_metrics and key in mode_metrics:
                        base_val = base_metrics[key]
                        mode_val = mode_metrics[key]
                        if base_val > 0:
                            improvements[f'{key}_improvement'] = ((mode_val - base_val) / base_val) * 100
                        else:
                            improvements[f'{key}_improvement'] = 0
                
                comparison[f'{mode}_vs_{base_mode}'] = improvements
        
        return comparison
    
    def create_simple_charts(self, comparison_data: Dict[str, Any]) -> List[str]:
        """Create simple charts using matplotlib."""
        if not MATPLOTLIB_AVAILABLE:
            self.logger.warning("Matplotlib not available, skipping chart generation")
            return []
        
        charts_created = []
        modes = [k for k in comparison_data.keys() if not k.endswith('_vs_') and not k.endswith('_improvement')]
        
        if len(modes) < 2:
            self.logger.info("Need at least 2 modes for comparison charts")
            return []
        
        # Set up the plot style
        plt.style.use('default')
        
        # Create comparison chart
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('OCR Pipeline Mode Comparison', fontsize=14, fontweight='bold')
        
        # Success Rate
        success_rates = [comparison_data[mode].get('success_rate', 0) for mode in modes]
        axes[0, 0].bar(modes, success_rates, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(modes)])
        axes[0, 0].set_title('Success Rate by Mode')
        axes[0, 0].set_ylabel('Success Rate')
        axes[0, 0].set_ylim(0, 1)
        for i, v in enumerate(success_rates):
            axes[0, 0].text(i, v + 0.01, f'{v:.1%}', ha='center', va='bottom')
        
        # Average Confidence
        confidences = [comparison_data[mode].get('avg_confidence', 0) for mode in modes]
        axes[0, 1].bar(modes, confidences, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(modes)])
        axes[0, 1].set_title('Average Confidence by Mode')
        axes[0, 1].set_ylabel('Average Confidence')
        axes[0, 1].set_ylim(0, 1)
        for i, v in enumerate(confidences):
            axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # Time per Word (Cost of Compute)
        time_per_word = [comparison_data[mode].get('time_per_word', 0) for mode in modes]
        axes[1, 0].bar(modes, time_per_word, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(modes)])
        axes[1, 0].set_title('Time per Word (Cost of Compute)')
        axes[1, 0].set_ylabel('Time per Word (seconds)')
        for i, v in enumerate(time_per_word):
            axes[1, 0].text(i, v + max(time_per_word) * 0.01, f'{v:.4f}s', ha='center', va='bottom')
        
        # Time per Token
        time_per_token = [comparison_data[mode].get('time_per_token', 0) for mode in modes]
        axes[1, 1].bar(modes, time_per_token, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(modes)])
        axes[1, 1].set_title('Time per Token (Cost of Compute)')
        axes[1, 1].set_ylabel('Time per Token (seconds)')
        for i, v in enumerate(time_per_token):
            axes[1, 1].text(i, v + max(time_per_token) * 0.01, f'{v:.4f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        chart_path = self.output_dir / 'mode_comparison.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        charts_created.append(str(chart_path))
        
        return charts_created
    
    def generate_simple_report(self, comparison_data: Dict[str, Any]) -> str:
        """Generate a simple, readable analysis report."""
        report_lines = []
        report_lines.append("# OCR Pipeline Evaluation Analysis Report")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        modes = [k for k in comparison_data.keys() if not k.endswith('_vs_') and not k.endswith('_improvement')]
        
        # Executive Summary
        report_lines.append("## Executive Summary")
        report_lines.append("")
        
        for mode in modes:
            metrics = comparison_data[mode]
            report_lines.append(f"### {mode.upper()} Mode")
            report_lines.append(f"- **Documents Processed**: {metrics.get('total_documents', 0)}")
            report_lines.append(f"- **Total Pages**: {metrics.get('total_pages', 0)}")
            report_lines.append(f"- **Success Rate**: {metrics.get('success_rate', 0):.1%}")
            report_lines.append(f"- **Average Confidence**: {metrics.get('avg_confidence', 0):.3f}")
            report_lines.append(f"- **Processing Time**: {metrics.get('total_processing_time', 0):.1f}s")
            report_lines.append("")
        
        # Cost of Compute Analysis
        report_lines.append("## Cost of Compute Analysis")
        report_lines.append("")
        
        # Add clear explanation of resource usage
        report_lines.append("### Understanding Resource Usage")
        report_lines.append("")
        report_lines.append("**CPU Usage Explained:**")
        report_lines.append("- **100%** = One CPU core working at full capacity")
        report_lines.append("- **Higher percentages** = Multiple CPU cores working simultaneously")
        report_lines.append("- **Very high percentages** = All cores + hyperthreading + multiple processes")
        report_lines.append("- *Higher percentages mean more efficient use of your computer's processing power*")
        report_lines.append("")
        report_lines.append("**Memory Usage Explained:**")
        report_lines.append("- Shows how much RAM (memory) the OCR process uses")
        report_lines.append("- *Average* means typical usage throughout the entire run, not peak usage")
        report_lines.append("- Higher memory usage can indicate processing complex documents")
        report_lines.append("")
        report_lines.append("**Why This Matters:**")
        report_lines.append("- **High CPU usage** = Software is working efficiently")
        report_lines.append("- **Low CPU usage** = Software might be waiting or not optimized")
        report_lines.append("- **Memory usage** = Shows how much system resources are needed")
        report_lines.append("")
        
        for mode in modes:
            metrics = comparison_data[mode]
            report_lines.append(f"### {mode.upper()} Mode - Resource Usage")
            report_lines.append(f"- **Text Elements**: {metrics.get('total_text_elements', 0)} total, {metrics.get('avg_text_elements_per_page', 0):.1f} per page")
            report_lines.append(f"- **Word Count**: {metrics.get('total_word_count', 0)} total, {metrics.get('avg_word_count_per_page', 0):.1f} per page")
            report_lines.append(f"- **Token Count**: {metrics.get('total_token_count', 0)} total, {metrics.get('avg_token_count_per_page', 0):.1f} per page")
            report_lines.append(f"- **Time per Text Element**: {metrics.get('time_per_text_element', 0):.4f}s")
            report_lines.append(f"- **Time per Word**: {metrics.get('time_per_word', 0):.4f}s")
            report_lines.append(f"- **Time per Token**: {metrics.get('time_per_token', 0):.4f}s")
            # CPU Usage with clear explanation
            cpu_usage = metrics.get('avg_cpu_percent', 0)
            if cpu_usage > 100:
                # For multi-core systems: 100% = 1 core, 600% = 6 cores, 1000% = 6 cores + hyperthreading
                if cpu_usage <= 600:
                    cores_used = cpu_usage / 100
                    cpu_explanation = f" (using {cores_used:.1f} CPU cores simultaneously)"
                else:
                    cpu_explanation = f" (using all CPU cores + hyperthreading + subprocesses)"
            else:
                cpu_explanation = f" (using {cpu_usage:.1f}% of one CPU core)"
            report_lines.append(f"- **Average CPU Usage**: {cpu_usage:.1f}%{cpu_explanation}")
            report_lines.append(f"- **Average Memory Usage**: {metrics.get('avg_memory_mb', 0):.1f} MB")
            report_lines.append("")
            report_lines.append(f"*Note: 'Average' means the typical usage throughout the entire processing run, not peak usage*")
            report_lines.append("")
            
            # Cost of Compute Analysis
            time_per_word = metrics.get('time_per_word', 0)
            time_per_token = metrics.get('time_per_token', 0)
            total_words = metrics.get('total_word_count', 0)
            total_tokens = metrics.get('total_token_count', 0)
            total_time = metrics.get('total_processing_time', 0)
            
            report_lines.append(f"#### Cost Efficiency Analysis")
            # Use appropriate metric based on what's available
            if total_words > 0:
                # Modern languages: use word-based efficiency
                if time_per_word < 0.05:
                    efficiency_rating = "⚡ **Excellent** - Very fast processing"
                elif time_per_word < 0.1:
                    efficiency_rating = "✅ **Good** - Efficient processing"
                elif time_per_word < 0.2:
                    efficiency_rating = "⚠️ **Moderate** - Acceptable speed"
                else:
                    efficiency_rating = "❌ **Slow** - Needs optimization"
            else:
                # Akkadian/limited languages: use token-based efficiency
                if time_per_token < 0.1:
                    efficiency_rating = "⚡ **Excellent** - Very fast processing"
                elif time_per_token < 0.2:
                    efficiency_rating = "✅ **Good** - Efficient processing"
                elif time_per_token < 0.5:
                    efficiency_rating = "⚠️ **Moderate** - Acceptable speed"
                else:
                    efficiency_rating = "❌ **Slow** - Needs optimization"
            
            report_lines.append(f"- **Processing Speed**: {efficiency_rating}")
            # Calculate throughput based on what's available
            if total_words > 0:
                report_lines.append(f"- **Throughput**: {total_words/total_time:.1f} words/second, {total_tokens/total_time:.1f} tokens/second")
            else:
                report_lines.append(f"- **Throughput**: {total_tokens/total_time:.1f} tokens/second (text elements)")
            
            # Resource efficiency
            cpu_usage = metrics.get('avg_cpu_percent', 0)
            memory_usage = metrics.get('avg_memory_mb', 0)
            
            if cpu_usage > 500:
                cpu_efficiency = "🔥 **High CPU usage** - Consider optimization"
            elif cpu_usage > 200:
                cpu_efficiency = "⚠️ **Moderate CPU usage** - Monitor performance"
            else:
                cpu_efficiency = "✅ **Efficient CPU usage** - Good resource utilization"
            
            if memory_usage > 2000:
                memory_efficiency = "🧠 **High memory usage** - Consider memory optimization"
            elif memory_usage > 1000:
                memory_efficiency = "⚠️ **Moderate memory usage** - Monitor consumption"
            else:
                memory_efficiency = "✅ **Efficient memory usage** - Good memory management"
            
            report_lines.append(f"- **CPU Efficiency**: {cpu_efficiency}")
            report_lines.append(f"- **Memory Efficiency**: {memory_efficiency}")
            report_lines.append("")
        
        # Performance Comparison
        if len(modes) >= 2:
            report_lines.append("## Performance Comparison")
            report_lines.append("")
            
            # Find best performing mode for each metric
            best_success = max(modes, key=lambda m: comparison_data[m].get('success_rate', 0))
            best_confidence = max(modes, key=lambda m: comparison_data[m].get('avg_confidence', 0))
            best_speed = min(modes, key=lambda m: comparison_data[m].get('time_per_word', float('inf')))
            most_efficient = min(modes, key=lambda m: comparison_data[m].get('time_per_token', float('inf')))
            
            report_lines.append(f"- **Best Success Rate**: {best_success} ({comparison_data[best_success].get('success_rate', 0):.1%})")
            report_lines.append(f"- **Best Confidence**: {best_confidence} ({comparison_data[best_confidence].get('avg_confidence', 0):.3f})")
            report_lines.append(f"- **Fastest per Word**: {best_speed} ({comparison_data[best_speed].get('time_per_word', 0):.4f}s/word)")
            report_lines.append(f"- **Most Efficient per Token**: {most_efficient} ({comparison_data[most_efficient].get('time_per_token', 0):.4f}s/token)")
            report_lines.append("")
        
        # Recommendations
        report_lines.append("## Recommendations")
        report_lines.append("")
        
        if len(modes) >= 2:
            # Multi-mode comparison recommendations
            basic_mode = next((m for m in modes if 'basic' in m.lower()), modes[0])
            akkadian_mode = next((m for m in modes if 'akkadian' in m.lower()), None)
            
            if akkadian_mode and basic_mode:
                basic_metrics = comparison_data[basic_mode]
                akkadian_metrics = comparison_data[akkadian_mode]
                
                # Cost efficiency comparison
                basic_efficiency = basic_metrics.get('time_per_word', 0)
                akkadian_efficiency = akkadian_metrics.get('time_per_word', 0)
                
                if akkadian_efficiency < basic_efficiency:
                    report_lines.append(f"✅ **{akkadian_mode} mode is more cost-efficient** - {akkadian_efficiency:.4f}s/word vs {basic_efficiency:.4f}s/word")
                else:
                    report_lines.append(f"⚠️ **{basic_mode} mode is more cost-efficient** - {basic_efficiency:.4f}s/word vs {akkadian_efficiency:.4f}s/word")
                
                # Success rate comparison
                if akkadian_metrics.get('success_rate', 0) > basic_metrics.get('success_rate', 0):
                    report_lines.append(f"✅ **{akkadian_mode} mode has better success rate** - consider for critical documents")
                else:
                    report_lines.append(f"⚠️ **{basic_mode} mode has better success rate** - {akkadian_mode} mode may need optimization")
        else:
            # Single mode analysis recommendations
            mode_name = list(comparison_data.keys())[0]
            metrics = comparison_data[mode_name]
            
            # Performance analysis
            success_rate = metrics.get('success_rate', 0)
            if success_rate >= 0.95:
                report_lines.append(f"✅ **Excellent success rate** ({success_rate:.1%}) - {mode_name} mode is performing well")
            elif success_rate >= 0.90:
                report_lines.append(f"⚠️ **Good success rate** ({success_rate:.1%}) - {mode_name} mode is acceptable but could be improved")
            else:
                report_lines.append(f"❌ **Low success rate** ({success_rate:.1%}) - {mode_name} mode needs optimization")
            
            # Resource usage analysis
            cpu_usage = metrics.get('avg_cpu_percent', 0)
            if cpu_usage > 500:
                report_lines.append(f"🔥 **High CPU usage** ({cpu_usage:.1f}%) - Consider optimizing for better efficiency")
            elif cpu_usage > 200:
                report_lines.append(f"⚠️ **Moderate CPU usage** ({cpu_usage:.1f}%) - Monitor for performance impact")
            else:
                report_lines.append(f"✅ **Efficient CPU usage** ({cpu_usage:.1f}%) - Good resource utilization")
            
            # Memory usage analysis
            memory_usage = metrics.get('avg_memory_mb', 0)
            if memory_usage > 2000:
                report_lines.append(f"🧠 **High memory usage** ({memory_usage:.0f}MB) - Consider memory optimization")
            elif memory_usage > 1000:
                report_lines.append(f"⚠️ **Moderate memory usage** ({memory_usage:.0f}MB) - Monitor memory consumption")
            else:
                report_lines.append(f"✅ **Efficient memory usage** ({memory_usage:.0f}MB) - Good memory management")
            
            # Cost efficiency analysis
            time_per_word = metrics.get('time_per_word', 0)
            if time_per_word < 0.05:
                report_lines.append(f"⚡ **Excellent processing speed** ({time_per_word:.4f}s/word) - Very efficient")
            elif time_per_word < 0.1:
                report_lines.append(f"✅ **Good processing speed** ({time_per_word:.4f}s/word) - Efficient processing")
            else:
                report_lines.append(f"⚠️ **Slow processing speed** ({time_per_word:.4f}s/word) - Consider optimization")
        
        # Language Detection Metrics (V3)
        report_lines.append("")
        report_lines.append("## Language Detection Metrics")
        report_lines.append("")
        report_lines.append("### Understanding Language Detection")
        report_lines.append("")
        report_lines.append("**Advanced Detection:**")
        report_lines.append("- Uses statistical models (langid/langdetect) trained on large text corpora")
        report_lines.append("- Provides confidence scores and context-aware language identification")
        report_lines.append("- More accurate than character-based detection")
        report_lines.append("")
        report_lines.append("**Character-Based Detection:**")
        report_lines.append("- Fallback method using special characters (çğıöşü, äöüß, etc.)")
        report_lines.append("- Used when advanced detection fails or confidence is low")
        report_lines.append("- Less accurate but more reliable for edge cases")
        report_lines.append("")
        report_lines.append("**Fallback Detection:**")
        report_lines.append("- Defaults to English when no clear language is detected")
        report_lines.append("- Indicates potential language detection issues")
        report_lines.append("- High fallback rates may affect LLM correction quality")
        report_lines.append("")
        
        for mode in modes:
            metrics = comparison_data[mode]
            lang_metrics = metrics.get('language_detection_v3', {})
            
            if lang_metrics:
                report_lines.append(f"### {mode.upper()} Mode - Language Detection")
                report_lines.append(f"- **Pages Analyzed**: {lang_metrics.get('total_pages_analyzed', 0)}")
                report_lines.append(f"- **Advanced Detections**: {lang_metrics.get('advanced_detections', 0)} ({lang_metrics.get('advanced_success_rate', 0):.1%})")
                report_lines.append(f"- **Character-Based Detections**: {lang_metrics.get('character_based_detections', 0)} ({lang_metrics.get('character_based_success_rate', 0):.1%})")
                report_lines.append(f"- **Fallback Detections**: {lang_metrics.get('fallback_detections', 0)} ({lang_metrics.get('fallback_rate', 0):.1%})")
                report_lines.append(f"- **Average Fallback Percentage**: {lang_metrics.get('avg_fallback_percentage', 0):.1%}")
                report_lines.append(f"- **Threshold Exceeded Documents**: {lang_metrics.get('threshold_exceeded_documents', 0)}")
                
                # Language distribution
                lang_dist = lang_metrics.get('language_distribution', {})
                if lang_dist:
                    report_lines.append(f"- **Language Distribution**:")
                    for lang, count in sorted(lang_dist.items()):
                        percentage = (count / lang_metrics.get('total_pages_analyzed', 1)) * 100
                        report_lines.append(f"  - {lang.upper()}: {count} pages ({percentage:.1f}%)")
                
                # Analysis and recommendations
                fallback_rate = lang_metrics.get('fallback_rate', 0)
                if fallback_rate > 0.5:
                    report_lines.append(f"- **⚠️ High Fallback Rate**: {fallback_rate:.1%} of pages fell back to English")
                    report_lines.append(f"  - This may indicate language detection issues affecting LLM correction quality")
                    report_lines.append(f"  - Consider reviewing document types and language detection settings")
                elif fallback_rate > 0.2:
                    report_lines.append(f"- **⚠️ Moderate Fallback Rate**: {fallback_rate:.1%} of pages fell back to English")
                    report_lines.append(f"  - Monitor for potential language detection issues")
                else:
                    report_lines.append(f"- **✅ Good Language Detection**: {fallback_rate:.1%} fallback rate is acceptable")
                
                advanced_rate = lang_metrics.get('advanced_success_rate', 0)
                if advanced_rate > 0.7:
                    report_lines.append(f"- **✅ Excellent Advanced Detection**: {advanced_rate:.1%} success rate")
                elif advanced_rate > 0.5:
                    report_lines.append(f"- **⚠️ Moderate Advanced Detection**: {advanced_rate:.1%} success rate")
                else:
                    report_lines.append(f"- **❌ Low Advanced Detection**: {advanced_rate:.1%} success rate - consider language detection optimization")
                
                report_lines.append("")
        
        report_lines.append("")
        report_lines.append("## Notes")
        report_lines.append("- **Token Count**: Uses text_elements count (suitable for Akkadian/limited language detection)")
        report_lines.append("- **Word Count**: Only calculated for modern languages (English, German, French, Turkish, etc.)")
        report_lines.append("- **CPU Usage**: 100% = 1 CPU core. Higher % = multiple cores + hyperthreading + subprocesses")
        report_lines.append("- **Memory Usage**: Average RAM consumption during processing (not peak usage)")
        report_lines.append("- **Resource Monitoring**: Tracked throughout entire processing run, not just peak moments")
        report_lines.append("- **Cost of Compute**: Time per word/token shows processing efficiency")
        report_lines.append("- **Language Detection**: V3 mode provides enhanced detection with confidence scoring and fallback analysis")
        
        report_content = "\n".join(report_lines)
        
        # Save report
        report_path = self.output_dir / 'analysis_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        return str(report_path)
    
    def generate_json_summary(self, comparison_data: Dict[str, Any]) -> str:
        """Generate JSON summary for programmatic access."""
        summary = {
            'analysis_timestamp': datetime.now().isoformat(),
            'modes_analyzed': list(comparison_data.keys()),
            'comparison_data': comparison_data,
            'dependencies_available': {
                'matplotlib': MATPLOTLIB_AVAILABLE,
                'pandas': PANDAS_AVAILABLE
            }
        }
        
        json_path = self.output_dir / 'analysis_summary.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        return str(json_path)
    
    def run_analysis(self, eval_dirs: List[str]) -> Dict[str, str]:
        """Run complete analysis and return paths to generated files."""
        self.logger.info("Starting simplified analysis...")
        
        # Load data
        mode_data = self.load_evaluation_data(eval_dirs)
        if not mode_data:
            raise ValueError("No evaluation data found in specified directories")
        
        # Compare modes
        comparison_data = self.compare_modes(mode_data)
        
        # Generate charts
        charts = self.create_simple_charts(comparison_data)
        
        # Generate reports
        markdown_report = self.generate_simple_report(comparison_data)
        json_summary = self.generate_json_summary(comparison_data)
        
        results = {
            'markdown_report': markdown_report,
            'json_summary': json_summary,
            'charts': charts
        }
        
        self.logger.info(f"Analysis complete! Results saved to: {self.output_dir}")
        return results

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Simplified OCR Pipeline Evaluation Analysis")
    parser.add_argument("eval_dirs", nargs="+", help="Paths to evaluation output directories")
    parser.add_argument("-o", "--output", default="./data/analysis_output", 
                       help="Output directory for analysis results")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = SimpleAnalyzer(args.output)
    
    try:
        # Run analysis
        results = analyzer.run_analysis(args.eval_dirs)
        
        print("\n" + "="*60)
        print("📊 SIMPLIFIED ANALYSIS COMPLETE")
        print("="*60)
        print(f"📄 Report: {results['markdown_report']}")
        print(f"📊 JSON Summary: {results['json_summary']}")
        print(f"📈 Charts: {len(results['charts'])} generated")
        for chart in results['charts']:
            print(f"   - {chart}")
        print("="*60)
        
    except Exception as e:
        analyzer.logger.error(f"Analysis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

