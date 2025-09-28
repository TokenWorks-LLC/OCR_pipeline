#!/usr/bin/env python3
"""
Comprehensive Summary Analysis for OCR Pipeline Evaluation Results
Analyzes and compares evaluation outputs with visualizations and metrics
"""

import json
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import statistics
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class SummaryAnalyzer:
    """Analyzes OCR pipeline evaluation results and generates comprehensive reports."""
    
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
        self.comparison_data = {}
        
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
                    # Safe logging - encode problematic characters
                    safe_path = str(report_file).encode('ascii', 'replace').decode('ascii')
                    self.logger.info(f"Loaded: {safe_path}")
                except Exception as e:
                    self.logger.error(f"Failed to load {report_file}: {e}")
        
        return all_data
    
    def calculate_baseline_metrics(self, data: List[Dict]) -> Dict[str, Any]:
        """Calculate baseline metrics for OCR accuracy comparison."""
        if not data:
            return {}
        
        # Extract key metrics
        total_pages = sum(d.get('total_pages', 0) for d in data)
        successful_pages = sum(d.get('successful_pages', 0) for d in data)
        total_text_elements = sum(d.get('total_text_elements', 0) for d in data)
        total_processing_time = sum(d.get('total_processing_time', 0) for d in data)
        
        # Add new cost of compute metrics
        total_word_count = sum(d.get('total_word_count', 0) for d in data)
        total_token_count = sum(d.get('total_token_count', 0) for d in data)
        
        # Calculate confidence scores
        confidence_scores = []
        for doc in data:
            for page_stat in doc.get('page_statistics', []):
                if 'avg_confidence' in page_stat:
                    confidence_scores.append(page_stat['avg_confidence'])
        
        # Calculate corrections
        total_corrections = sum(d.get('total_corrections', 0) for d in data)
        
        # Calculate Akkadian translations
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
        
        # Calculate resource monitoring metrics
        resource_data = []
        for doc in data:
            for page_stat in doc.get('page_statistics', []):
                if 'resource_usage' in page_stat:
                    resource_data.append(page_stat['resource_usage'])
        
        avg_cpu_percent = 0.0
        avg_memory_mb = 0.0
        
        if resource_data:
            cpu_values = [r.get('cpu_percent', 0) for r in resource_data if r.get('available', False)]
            memory_values = [r.get('memory_mb', 0) for r in resource_data if r.get('available', False)]
            avg_cpu_percent = statistics.mean(cpu_values) if cpu_values else 0.0
            avg_memory_mb = statistics.mean(memory_values) if memory_values else 0.0
        
        return {
            'total_documents': len(data),
            'total_pages': total_pages,
            'successful_pages': successful_pages,
            'success_rate': successful_pages / total_pages if total_pages > 0 else 0,
            'total_text_elements': total_text_elements,
            'avg_text_elements_per_page': total_text_elements / total_pages if total_pages > 0 else 0,
            'total_processing_time': total_processing_time,
            'avg_processing_time_per_page': total_processing_time / total_pages if total_pages > 0 else 0,
            'avg_confidence': statistics.mean(confidence_scores) if confidence_scores else 0,
            'confidence_std': statistics.stdev(confidence_scores) if len(confidence_scores) > 1 else 0,
            'total_corrections': total_corrections,
            'total_akkadian_translations': total_akkadian,
            'confidence_scores': confidence_scores,
            # New cost of compute metrics
            'total_word_count': total_word_count,
            'total_token_count': total_token_count,
            'avg_word_count_per_page': total_word_count / total_pages if total_pages > 0 else 0,
            'avg_token_count_per_page': total_token_count / total_pages if total_pages > 0 else 0,
            'time_per_text_element': total_processing_time / total_text_elements if total_text_elements > 0 else 0,
            'time_per_word': total_processing_time / total_word_count if total_word_count > 0 else 0,
            'time_per_token': total_processing_time / total_token_count if total_token_count > 0 else 0,
            # Resource monitoring metrics
            'avg_cpu_percent': avg_cpu_percent,
            # Smart LLM metrics
            'total_llm_lines_processed': total_llm_lines_processed,
            'total_llm_lines_changed': total_llm_lines_changed,
            'total_llm_lines_skipped': total_llm_lines_skipped,
            'total_llm_akkadian_lines': total_llm_akkadian_lines,
            'total_llm_low_conf_lines': total_llm_low_conf_lines,
            'llm_call_reduction_pct': (total_llm_lines_skipped / total_llm_lines_processed * 100) if total_llm_lines_processed > 0 else 0,
            'smart_llm_efficiency': (total_llm_lines_changed / total_llm_lines_processed * 100) if total_llm_lines_processed > 0 else 0,
            'avg_memory_mb': avg_memory_mb,
            # Enhanced confidence metrics
            'confidence_analysis': self._analyze_confidence_metrics(data),
            'cost_benefit_analysis': self._calculate_cost_benefit_analysis(data, total_processing_time)
        }
    
    def compare_modes(self, mode_data: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Compare different evaluation modes."""
        comparison = {}
        
        for mode, data in mode_data.items():
            metrics = self.calculate_baseline_metrics(data)
            comparison[mode] = metrics
            
            # Store for later use
            self.analysis_results[mode] = metrics
        
        # Calculate relative improvements
        if len(comparison) >= 2:
            modes = list(comparison.keys())
            base_mode = modes[0]  # Use first mode as baseline
            
            for mode in modes[1:]:
                base_metrics = comparison[base_mode]
                mode_metrics = comparison[mode]
                
                improvements = {}
                for key in ['success_rate', 'avg_confidence', 'avg_text_elements_per_page']:
                    if key in base_metrics and key in mode_metrics:
                        base_val = base_metrics[key]
                        mode_val = mode_metrics[key]
                        if base_val > 0:
                            improvements[f'{key}_improvement'] = ((mode_val - base_val) / base_val) * 100
                        else:
                            improvements[f'{key}_improvement'] = 0
                
                comparison[f'{mode}_vs_{base_mode}'] = improvements
        
        return comparison
    
    def _analyze_confidence_metrics(self, data: List[Dict]) -> Dict[str, Any]:
        """Analyze confidence metrics with explanations."""
        if not data:
            return {}
        
        # Extract confidence data
        all_confidences = []
        word_level_confidences = []
        llm_corrected_words = []
        ocr_vs_llm_confidence = []
        
        for doc in data:
            # Page-level confidences
            for page_stat in doc.get('page_statistics', []):
                if 'avg_confidence' in page_stat:
                    all_confidences.append(page_stat['avg_confidence'])
            
            # Word-level confidences from confidence_metrics
            if 'confidence_metrics' in doc:
                conf_metrics = doc['confidence_metrics']
                word_level_confidences.extend(conf_metrics.get('word_level_confidences', []))
                llm_corrected_words.extend(conf_metrics.get('llm_corrected_words', []))
                ocr_vs_llm_confidence.extend(conf_metrics.get('ocr_vs_llm_confidence', []))
        
        # Calculate confidence statistics
        if all_confidences:
            avg_confidence = sum(all_confidences) / len(all_confidences)
            min_confidence = min(all_confidences)
            max_confidence = max(all_confidences)
            high_conf_pct = sum(1 for c in all_confidences if c >= 0.8) / len(all_confidences) * 100
            low_conf_pct = sum(1 for c in all_confidences if c < 0.5) / len(all_confidences) * 100
        else:
            avg_confidence = min_confidence = max_confidence = 0
            high_conf_pct = low_conf_pct = 0
        
        # Analyze LLM correction impact
        llm_impact = {}
        if ocr_vs_llm_confidence:
            ocr_confidences = [item.get('ocr_confidence', 0) for item in ocr_vs_llm_confidence]
            llm_confidences = [item.get('llm_confidence', 0) for item in ocr_vs_llm_confidence]
            
            if ocr_confidences and llm_confidences:
                avg_ocr_conf = sum(ocr_confidences) / len(ocr_confidences)
                avg_llm_conf = sum(llm_confidences) / len(llm_confidences)
                llm_improvement = ((avg_llm_conf - avg_ocr_conf) / avg_ocr_conf * 100) if avg_ocr_conf > 0 else 0
                
                llm_impact = {
                    'avg_ocr_confidence': avg_ocr_conf,
                    'avg_llm_confidence': avg_llm_conf,
                    'confidence_improvement_pct': llm_improvement,
                    'words_corrected': len(llm_corrected_words),
                    'correction_rate': len(llm_corrected_words) / len(word_level_confidences) * 100 if word_level_confidences else 0
                }
        
        return {
            'page_level_confidence': {
                'average': avg_confidence,
                'minimum': min_confidence,
                'maximum': max_confidence,
                'high_confidence_pages_pct': high_conf_pct,
                'low_confidence_pages_pct': low_conf_pct,
                'explanation': 'Page-level confidence represents the average OCR confidence for all text elements on each page. Higher values (>0.8) indicate more reliable text extraction.'
            },
            'word_level_confidence': {
                'total_words': len(word_level_confidences),
                'average': sum(word_level_confidences) / len(word_level_confidences) if word_level_confidences else 0,
                'distribution': self._categorize_confidence_distribution(word_level_confidences),
                'explanation': 'Word-level confidence shows individual text element reliability. This helps identify which specific words may need LLM correction.'
            },
            'llm_correction_impact': llm_impact,
            'confidence_interpretation': {
                'high_confidence': '>0.8 - Very reliable OCR results, minimal correction needed',
                'medium_confidence': '0.5-0.8 - Moderate reliability, may benefit from LLM correction',
                'low_confidence': '<0.5 - Poor OCR results, likely needs LLM correction or manual review'
            }
        }
    
    def _categorize_confidence_distribution(self, confidences: List[float]) -> Dict[str, int]:
        """Categorize confidence scores into high/medium/low."""
        if not confidences:
            return {'high': 0, 'medium': 0, 'low': 0}
        
        high = sum(1 for c in confidences if c >= 0.8)
        medium = sum(1 for c in confidences if 0.5 <= c < 0.8)
        low = sum(1 for c in confidences if c < 0.5)
        
        return {'high': high, 'medium': medium, 'low': low}
    
    def _calculate_cost_benefit_analysis(self, data: List[Dict], total_processing_time: float) -> Dict[str, Any]:
        """Calculate cost-benefit analysis for LLM usage."""
        if not data:
            return {}
        
        # Extract metrics
        total_pages = sum(d.get('total_pages', 0) for d in data)
        total_corrections = sum(d.get('total_corrections', 0) for d in data)
        total_word_count = sum(d.get('total_word_count', 0) for d in data)
        total_text_elements = sum(d.get('total_text_elements', 0) for d in data)
        
        # Calculate costs (processing time)
        base_processing_time = total_processing_time  # This includes LLM time
        estimated_ocr_only_time = total_processing_time * 0.7  # Estimate OCR-only time (30% faster)
        llm_overhead_time = total_processing_time - estimated_ocr_only_time
        
        # Calculate benefits (corrections made)
        correction_rate = (total_corrections / total_text_elements * 100) if total_text_elements > 0 else 0
        words_per_second = total_word_count / total_processing_time if total_processing_time > 0 else 0
        
        # Cost-benefit ratios
        time_per_correction = llm_overhead_time / total_corrections if total_corrections > 0 else 0
        corrections_per_second = total_corrections / total_processing_time if total_processing_time > 0 else 0
        
        # Efficiency metrics
        efficiency_score = (correction_rate / (llm_overhead_time / total_processing_time * 100)) if total_processing_time > 0 else 0
        
        return {
            'processing_costs': {
                'total_time_seconds': total_processing_time,
                'estimated_ocr_only_time': estimated_ocr_only_time,
                'llm_overhead_time': llm_overhead_time,
                'llm_overhead_percentage': (llm_overhead_time / total_processing_time * 100) if total_processing_time > 0 else 0,
                'explanation': 'Processing costs measure the time overhead of LLM correction compared to OCR-only processing.'
            },
            'correction_benefits': {
                'total_corrections': total_corrections,
                'correction_rate_percentage': correction_rate,
                'corrections_per_second': corrections_per_second,
                'time_per_correction_seconds': time_per_correction,
                'explanation': 'Correction benefits measure how many OCR errors the LLM successfully fixed.'
            },
            'cost_benefit_ratios': {
                'efficiency_score': efficiency_score,
                'time_overhead_vs_corrections': llm_overhead_time / total_corrections if total_corrections > 0 else 0,
                'corrections_per_overhead_second': total_corrections / llm_overhead_time if llm_overhead_time > 0 else 0,
                'explanation': 'Cost-benefit ratios help determine if LLM correction provides sufficient value for the processing time overhead.'
            },
            'recommendations': self._generate_llm_recommendations(efficiency_score, correction_rate, llm_overhead_time / total_processing_time if total_processing_time > 0 else 0)
        }
    
    def _generate_llm_recommendations(self, efficiency_score: float, correction_rate: float, overhead_pct: float) -> List[str]:
        """Generate recommendations based on cost-benefit analysis."""
        recommendations = []
        
        if efficiency_score > 2.0:
            recommendations.append("✅ EXCELLENT: LLM correction provides high value - continue using")
        elif efficiency_score > 1.0:
            recommendations.append("✅ GOOD: LLM correction provides positive value - recommended")
        elif efficiency_score > 0.5:
            recommendations.append("⚠️ MODERATE: LLM correction provides some value - consider optimization")
        else:
            recommendations.append("❌ POOR: LLM correction overhead exceeds benefits - consider disabling")
        
        if correction_rate < 5:
            recommendations.append("💡 Consider confidence-based filtering to only correct low-confidence words")
        
        if overhead_pct > 50:
            recommendations.append("⏱️ High processing overhead - consider using lighter LLM model")
        
        if correction_rate > 20:
            recommendations.append("🎯 High correction rate - LLM is finding many errors, very valuable")
        
        return recommendations
    
    def create_visualizations(self, comparison_data: Dict[str, Any]) -> List[str]:
        """Create visualization charts and save them."""
        charts_created = []
        
        # Set style
        try:
            plt.style.use('seaborn-v0_8')
        except OSError:
            # Fallback to default seaborn style
            plt.style.use('seaborn')
        sns.set_palette("husl")
        
        # Define modes at the beginning
        modes = [k for k in comparison_data.keys() if not k.endswith('_vs_') and not k.endswith('_improvement')]
        
        # 1. Success Rate Comparison
        if len(comparison_data) >= 2:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('OCR Pipeline Mode Comparison', fontsize=16, fontweight='bold')
            
            # Success Rate
            success_rates = [comparison_data[mode].get('success_rate', 0) for mode in modes]
            
            axes[0, 0].bar(modes, success_rates, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            axes[0, 0].set_title('Success Rate by Mode')
            axes[0, 0].set_ylabel('Success Rate')
            axes[0, 0].set_ylim(0, 1)
            for i, v in enumerate(success_rates):
                axes[0, 0].text(i, v + 0.01, f'{v:.1%}', ha='center', va='bottom')
            
            # Average Confidence
            confidences = [comparison_data[mode].get('avg_confidence', 0) for mode in modes]
            axes[0, 1].bar(modes, confidences, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            axes[0, 1].set_title('Average Confidence by Mode')
            axes[0, 1].set_ylabel('Average Confidence')
            axes[0, 1].set_ylim(0, 1)
            for i, v in enumerate(confidences):
                axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
            
            # Processing Time per Page
            times = [comparison_data[mode].get('avg_processing_time_per_page', 0) for mode in modes]
            axes[1, 0].bar(modes, times, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            axes[1, 0].set_title('Average Processing Time per Page')
            axes[1, 0].set_ylabel('Time (seconds)')
            for i, v in enumerate(times):
                axes[1, 0].text(i, v + max(times) * 0.01, f'{v:.1f}s', ha='center', va='bottom')
            
            # Text Elements per Page
            elements = [comparison_data[mode].get('avg_text_elements_per_page', 0) for mode in modes]
            axes[1, 1].bar(modes, elements, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            axes[1, 1].set_title('Average Text Elements per Page')
            axes[1, 1].set_ylabel('Text Elements')
            for i, v in enumerate(elements):
                axes[1, 1].text(i, v + max(elements) * 0.01, f'{v:.0f}', ha='center', va='bottom')
            
            plt.tight_layout()
            chart_path = self.output_dir / 'mode_comparison.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            charts_created.append(str(chart_path))
        
        # 2. Confidence Distribution
        if any('confidence_scores' in comparison_data[mode] for mode in modes):
            plt.figure(figsize=(12, 8))
            for mode in modes:
                scores = comparison_data[mode].get('confidence_scores', [])
                if scores:
                    plt.hist(scores, alpha=0.7, label=f'{mode} (n={len(scores)})', bins=20)
            
            plt.xlabel('Confidence Score')
            plt.ylabel('Frequency')
            plt.title('OCR Confidence Score Distribution by Mode')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            chart_path = self.output_dir / 'confidence_distribution.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            charts_created.append(str(chart_path))
        
        # 3. Performance Metrics Heatmap
        if len(modes) >= 2:
            metrics_df = pd.DataFrame({
                mode: {
                    'Success Rate': comparison_data[mode].get('success_rate', 0),
                    'Avg Confidence': comparison_data[mode].get('avg_confidence', 0),
                    'Text Elements/Page': comparison_data[mode].get('avg_text_elements_per_page', 0),
                    'Time/Page (s)': comparison_data[mode].get('avg_processing_time_per_page', 0)
                } for mode in modes
            }).T
            
            plt.figure(figsize=(10, 6))
            sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='RdYlBu_r', center=0.5)
            plt.title('Performance Metrics Heatmap')
            plt.tight_layout()
            
            chart_path = self.output_dir / 'metrics_heatmap.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            charts_created.append(str(chart_path))
        
        return charts_created
    
    def generate_detailed_report(self, comparison_data: Dict[str, Any]) -> str:
        """Generate a detailed text report."""
        report_lines = []
        report_lines.append("# OCR Pipeline Evaluation Analysis Report")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # Summary statistics
        modes = [k for k in comparison_data.keys() if not k.endswith('_vs_') and not k.endswith('_improvement')]
        
        report_lines.append("## Executive Summary")
        report_lines.append("")
        
        for mode in modes:
            metrics = comparison_data[mode]
            report_lines.append(f"### {mode.upper()} Mode")
            report_lines.append(f"- **Documents Processed**: {metrics.get('total_documents', 0)}")
            report_lines.append(f"- **Total Pages**: {metrics.get('total_pages', 0)}")
            report_lines.append(f"- **Success Rate**: {metrics.get('success_rate', 0):.1%}")
            report_lines.append(f"- **Average Confidence**: {metrics.get('avg_confidence', 0):.3f}")
            report_lines.append(f"- **Text Elements per Page**: {metrics.get('avg_text_elements_per_page', 0):.1f}")
            report_lines.append(f"- **Processing Time per Page**: {metrics.get('avg_processing_time_per_page', 0):.1f}s")
            report_lines.append(f"- **Total Corrections**: {metrics.get('total_corrections', 0)}")
            report_lines.append(f"- **Akkadian Translations**: {metrics.get('total_akkadian_translations', 0)}")
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
        
        # Performance comparison
        if len(modes) >= 2:
            report_lines.append("## Performance Comparison")
            report_lines.append("")
            
            # Find best performing mode for each metric
            best_success = max(modes, key=lambda m: comparison_data[m].get('success_rate', 0))
            best_confidence = max(modes, key=lambda m: comparison_data[m].get('avg_confidence', 0))
            best_speed = min(modes, key=lambda m: comparison_data[m].get('avg_processing_time_per_page', float('inf')))
            
            report_lines.append(f"- **Best Success Rate**: {best_success} ({comparison_data[best_success].get('success_rate', 0):.1%})")
            report_lines.append(f"- **Best Confidence**: {best_confidence} ({comparison_data[best_confidence].get('avg_confidence', 0):.3f})")
            report_lines.append(f"- **Fastest Processing**: {best_speed} ({comparison_data[best_speed].get('avg_processing_time_per_page', 0):.1f}s/page)")
            report_lines.append("")
        
        # Recommendations
        report_lines.append("## Recommendations")
        report_lines.append("")
        
        recommendations = self._generate_recommendations(comparison_data)
        if recommendations:
            for rec in recommendations:
                report_lines.append(rec)
        else:
            report_lines.append("No specific recommendations available for this analysis.")
        
        report_lines.append("")
        report_lines.append("## Baseline OCR Accuracy Assessment")
        report_lines.append("")
        report_lines.append("To establish OCR accuracy baselines, consider:")
        report_lines.append("1. **Ground Truth Comparison**: Compare OCR results against manually verified text")
        report_lines.append("2. **Character-Level Accuracy**: Measure character-by-character accuracy")
        report_lines.append("3. **Word-Level Accuracy**: Measure word recognition accuracy")
        report_lines.append("4. **Language-Specific Metrics**: Different languages may have different accuracy patterns")
        report_lines.append("5. **Document Type Analysis**: Academic papers vs. general documents may have different baselines")
        report_lines.append("")
        report_lines.append("### Suggested Baseline Implementation:")
        report_lines.append("```python")
        report_lines.append("# Add to your evaluation pipeline:")
        report_lines.append("def calculate_ocr_accuracy(ocr_text, ground_truth):")
        report_lines.append("    # Character-level accuracy")
        report_lines.append("    char_accuracy = calculate_character_accuracy(ocr_text, ground_truth)")
        report_lines.append("    # Word-level accuracy")
        report_lines.append("    word_accuracy = calculate_word_accuracy(ocr_text, ground_truth)")
        report_lines.append("    return {'char_accuracy': char_accuracy, 'word_accuracy': word_accuracy}")
        report_lines.append("```")
        
        # Enhanced Notes section
        report_lines.append("")
        report_lines.append("## Notes")
        report_lines.append("- **Token Count**: Uses text_elements count (suitable for Akkadian/limited language detection)")
        report_lines.append("- **Word Count**: Only calculated for modern languages (English, German, French, Turkish, etc.)")
        report_lines.append("- **CPU Usage**: 100% = 1 CPU core. Higher % = multiple cores + hyperthreading + subprocesses")
        report_lines.append("- **Memory Usage**: Average RAM consumption during processing (not peak usage)")
        report_lines.append("- **Resource Monitoring**: Tracked throughout entire processing run, not just peak moments")
        report_lines.append("- **Cost of Compute**: Time per word/token shows processing efficiency")
        
        report_content = "\n".join(report_lines)
        
        # Save report
        report_path = self.output_dir / 'detailed_analysis_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        return str(report_path)
    
    def generate_json_summary(self, comparison_data: Dict[str, Any]) -> str:
        """Generate JSON summary for programmatic access."""
        summary = {
            'analysis_timestamp': datetime.now().isoformat(),
            'modes_analyzed': list(comparison_data.keys()),
            'comparison_data': comparison_data,
            'recommendations': self._generate_recommendations(comparison_data)
        }
        
        json_path = self.output_dir / 'analysis_summary.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        return str(json_path)
    
    def _generate_recommendations(self, comparison_data: Dict[str, Any]) -> List[str]:
        """Generate automated recommendations based on analysis."""
        recommendations = []
        modes = [k for k in comparison_data.keys() if not k.endswith('_vs_') and not k.endswith('_improvement')]
        
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
                    recommendations.append(f"✅ **{akkadian_mode} mode is more cost-efficient** - {akkadian_efficiency:.4f}s/word vs {basic_efficiency:.4f}s/word")
                else:
                    recommendations.append(f"⚠️ **{basic_mode} mode is more cost-efficient** - {basic_efficiency:.4f}s/word vs {akkadian_efficiency:.4f}s/word")
                
                # Success rate comparison
                if akkadian_metrics.get('success_rate', 0) > basic_metrics.get('success_rate', 0):
                    recommendations.append(f"✅ **{akkadian_mode} mode has better success rate** - consider for critical documents")
                else:
                    recommendations.append(f"⚠️ **{basic_mode} mode has better success rate** - {akkadian_mode} mode may need optimization")
        else:
            # Single mode analysis recommendations
            mode_name = list(comparison_data.keys())[0]
            metrics = comparison_data[mode_name]
            
            # Performance analysis
            success_rate = metrics.get('success_rate', 0)
            if success_rate >= 0.95:
                recommendations.append(f"✅ **Excellent success rate** ({success_rate:.1%}) - {mode_name} mode is performing well")
            elif success_rate >= 0.90:
                recommendations.append(f"⚠️ **Good success rate** ({success_rate:.1%}) - {mode_name} mode is acceptable but could be improved")
            else:
                recommendations.append(f"❌ **Low success rate** ({success_rate:.1%}) - {mode_name} mode needs optimization")
            
            # Resource usage analysis
            cpu_usage = metrics.get('avg_cpu_percent', 0)
            if cpu_usage > 500:
                recommendations.append(f"🔥 **High CPU usage** ({cpu_usage:.1f}%) - Consider optimizing for better efficiency")
            elif cpu_usage > 200:
                recommendations.append(f"⚠️ **Moderate CPU usage** ({cpu_usage:.1f}%) - Monitor for performance impact")
            else:
                recommendations.append(f"✅ **Efficient CPU usage** ({cpu_usage:.1f}%) - Good resource utilization")
            
            # Memory usage analysis
            memory_usage = metrics.get('avg_memory_mb', 0)
            if memory_usage > 2000:
                recommendations.append(f"🧠 **High memory usage** ({memory_usage:.0f}MB) - Consider memory optimization")
            elif memory_usage > 1000:
                recommendations.append(f"⚠️ **Moderate memory usage** ({memory_usage:.0f}MB) - Monitor memory consumption")
            else:
                recommendations.append(f"✅ **Efficient memory usage** ({memory_usage:.0f}MB) - Good memory management")
            
            # Cost efficiency analysis
            time_per_word = metrics.get('time_per_word', 0)
            if time_per_word < 0.05:
                recommendations.append(f"⚡ **Excellent processing speed** ({time_per_word:.4f}s/word) - Very efficient")
            elif time_per_word < 0.1:
                recommendations.append(f"✅ **Good processing speed** ({time_per_word:.4f}s/word) - Efficient processing")
            else:
                recommendations.append(f"⚠️ **Slow processing speed** ({time_per_word:.4f}s/word) - Consider optimization")
        
        return recommendations
    
    def run_analysis(self, eval_dirs: List[str]) -> Dict[str, str]:
        """Run complete analysis and return paths to generated files."""
        self.logger.info("Starting comprehensive analysis...")
        
        # Load data
        mode_data = self.load_evaluation_data(eval_dirs)
        if not mode_data:
            raise ValueError("No evaluation data found in specified directories")
        
        # Compare modes
        comparison_data = self.compare_modes(mode_data)
        
        # Generate visualizations
        charts = self.create_visualizations(comparison_data)
        
        # Generate reports
        markdown_report = self.generate_detailed_report(comparison_data)
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
    parser = argparse.ArgumentParser(description="OCR Pipeline Evaluation Summary Analysis")
    parser.add_argument("eval_dirs", nargs="+", help="Paths to evaluation output directories")
    parser.add_argument("-o", "--output", default="./data/analysis_output", 
                       help="Output directory for analysis results")
    parser.add_argument("--compare", action="store_true", 
                       help="Compare all modes in the directories")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = SummaryAnalyzer(args.output)
    
    try:
        # Run analysis
        results = analyzer.run_analysis(args.eval_dirs)
        
        print("\n" + "="*60)
        print("📊 ANALYSIS COMPLETE")
        print("="*60)
        print(f"📄 Detailed Report: {results['markdown_report']}")
        print(f"📊 JSON Summary: {results['json_summary']}")
        print(f"📈 Charts Generated: {len(results['charts'])}")
        for chart in results['charts']:
            print(f"   - {chart}")
        print("="*60)
        
    except Exception as e:
        analyzer.logger.error(f"Analysis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

