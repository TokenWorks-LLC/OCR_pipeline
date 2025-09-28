#!/usr/bin/env python3
"""
Accuracy Measurement System - Before/After LLM Comparison
Measures OCR accuracy improvements with and without LLM correction using Gold Pages reference.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class AccuracyMetrics:
    """Accuracy metrics for before/after comparison."""
    character_accuracy: float
    word_accuracy: float
    line_accuracy: float
    overall_accuracy: float
    error_count: int
    total_elements: int
    confidence_score: float
    processing_time: float

@dataclass
class BeforeAfterComparison:
    """Comparison between before and after LLM correction."""
    before_metrics: AccuracyMetrics
    after_metrics: AccuracyMetrics
    improvement_character: float
    improvement_word: float
    improvement_line: float
    improvement_overall: float
    cost_benefit_ratio: float
    meets_success_threshold: bool

class AccuracyMeasurer:
    """Measures OCR accuracy with before/after LLM comparison."""
    
    def __init__(self, gold_pages_manager=None):
        """
        Initialize Accuracy Measurer.
        
        Args:
            gold_pages_manager: Gold Pages Manager instance for ground truth reference
        """
        self.gold_pages_manager = gold_pages_manager
        self.measurements: List[BeforeAfterComparison] = []
        
        # Success thresholds
        self.min_improvement_threshold = 0.05  # 5% minimum improvement
        self.max_processing_time_increase = 0.5  # 50% maximum time increase
        self.cost_per_accuracy_point = 0.10  # $0.10 per 1% improvement
    
    def measure_ocr_accuracy(self, ocr_text: str, ground_truth: str, 
                           processing_time: float = 0.0, confidence: float = 0.0) -> AccuracyMetrics:
        """
        Measure OCR accuracy against ground truth.
        
        Args:
            ocr_text: OCR extracted text
            ground_truth: Ground truth reference text
            processing_time: Processing time in seconds
            confidence: OCR confidence score
        
        Returns:
            AccuracyMetrics object
        """
        if not ocr_text or not ground_truth:
            return AccuracyMetrics(
                character_accuracy=0.0,
                word_accuracy=0.0,
                line_accuracy=0.0,
                overall_accuracy=0.0,
                error_count=0,
                total_elements=0,
                confidence_score=confidence,
                processing_time=processing_time
            )
        
        # Character-level accuracy
        char_accuracy = self._calculate_character_accuracy(ocr_text, ground_truth)
        
        # Word-level accuracy
        word_accuracy = self._calculate_word_accuracy(ocr_text, ground_truth)
        
        # Line-level accuracy (simplified)
        line_accuracy = self._calculate_line_accuracy(ocr_text, ground_truth)
        
        # Overall accuracy (weighted average)
        overall_accuracy = (char_accuracy * 0.4 + word_accuracy * 0.4 + line_accuracy * 0.2)
        
        # Error count
        error_count = self._count_errors(ocr_text, ground_truth)
        
        # Total elements
        total_elements = len(ground_truth.split())
        
        return AccuracyMetrics(
            character_accuracy=char_accuracy,
            word_accuracy=word_accuracy,
            line_accuracy=line_accuracy,
            overall_accuracy=overall_accuracy,
            error_count=error_count,
            total_elements=total_elements,
            confidence_score=confidence,
            processing_time=processing_time
        )
    
    def compare_before_after(self, before_metrics: AccuracyMetrics, 
                           after_metrics: AccuracyMetrics, 
                           llm_cost: float = 0.0) -> BeforeAfterComparison:
        """
        Compare before and after LLM correction metrics.
        
        Args:
            before_metrics: Accuracy metrics before LLM correction
            after_metrics: Accuracy metrics after LLM correction
            llm_cost: Cost of LLM processing
        
        Returns:
            BeforeAfterComparison object
        """
        # Calculate improvements
        char_improvement = after_metrics.character_accuracy - before_metrics.character_accuracy
        word_improvement = after_metrics.word_accuracy - before_metrics.word_accuracy
        line_improvement = after_metrics.line_accuracy - before_metrics.line_accuracy
        overall_improvement = after_metrics.overall_accuracy - before_metrics.overall_accuracy
        
        # Calculate cost-benefit ratio
        cost_benefit_ratio = 0.0
        if overall_improvement > 0 and llm_cost > 0:
            cost_benefit_ratio = llm_cost / (overall_improvement * 100)  # Cost per 1% improvement
        
        # Check if meets success threshold
        meets_threshold = (
            overall_improvement >= self.min_improvement_threshold and
            (after_metrics.processing_time - before_metrics.processing_time) / before_metrics.processing_time <= self.max_processing_time_increase
        )
        
        return BeforeAfterComparison(
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            improvement_character=char_improvement,
            improvement_word=word_improvement,
            improvement_line=line_improvement,
            improvement_overall=overall_improvement,
            cost_benefit_ratio=cost_benefit_ratio,
            meets_success_threshold=meets_threshold
        )
    
    def _calculate_character_accuracy(self, ocr_text: str, ground_truth: str) -> float:
        """Calculate character-level accuracy."""
        if not ocr_text or not ground_truth:
            return 0.0
        
        # Simple character comparison
        ocr_chars = list(ocr_text.lower().replace(' ', ''))
        gt_chars = list(ground_truth.lower().replace(' ', ''))
        
        if not gt_chars:
            return 1.0 if not ocr_chars else 0.0
        
        # Calculate matches
        matches = 0
        min_len = min(len(ocr_chars), len(gt_chars))
        
        for i in range(min_len):
            if ocr_chars[i] == gt_chars[i]:
                matches += 1
        
        return matches / len(gt_chars)
    
    def _calculate_word_accuracy(self, ocr_text: str, ground_truth: str) -> float:
        """Calculate word-level accuracy."""
        if not ocr_text or not ground_truth:
            return 0.0
        
        ocr_words = ocr_text.lower().split()
        gt_words = ground_truth.lower().split()
        
        if not gt_words:
            return 1.0 if not ocr_words else 0.0
        
        # Calculate matches
        matches = 0
        min_len = min(len(ocr_words), len(gt_words))
        
        for i in range(min_len):
            if ocr_words[i] == gt_words[i]:
                matches += 1
        
        return matches / len(gt_words)
    
    def _calculate_line_accuracy(self, ocr_text: str, ground_truth: str) -> float:
        """Calculate line-level accuracy (simplified)."""
        if not ocr_text or not ground_truth:
            return 0.0
        
        ocr_lines = ocr_text.split('\n')
        gt_lines = ground_truth.split('\n')
        
        if not gt_lines:
            return 1.0 if not ocr_lines else 0.0
        
        # Calculate matches
        matches = 0
        min_len = min(len(ocr_lines), len(gt_lines))
        
        for i in range(min_len):
            if ocr_lines[i].strip() == gt_lines[i].strip():
                matches += 1
        
        return matches / len(gt_lines)
    
    def _count_errors(self, ocr_text: str, ground_truth: str) -> int:
        """Count errors between OCR text and ground truth."""
        if not ocr_text or not ground_truth:
            return 0
        
        ocr_words = ocr_text.lower().split()
        gt_words = ground_truth.lower().split()
        
        errors = 0
        min_len = min(len(ocr_words), len(gt_words))
        
        for i in range(min_len):
            if ocr_words[i] != gt_words[i]:
                errors += 1
        
        # Add errors for length differences
        errors += abs(len(ocr_words) - len(gt_words))
        
        return errors
    
    def add_measurement(self, comparison: BeforeAfterComparison) -> None:
        """Add a measurement to the collection."""
        self.measurements.append(comparison)
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics for all measurements."""
        if not self.measurements:
            return {
                'total_measurements': 0,
                'average_improvement': 0.0,
                'success_rate': 0.0,
                'average_cost_benefit': 0.0,
                'meets_threshold_count': 0
            }
        
        total_measurements = len(self.measurements)
        successful_measurements = sum(1 for m in self.measurements if m.meets_success_threshold)
        
        avg_improvement = sum(m.improvement_overall for m in self.measurements) / total_measurements
        success_rate = successful_measurements / total_measurements
        
        # Calculate average cost-benefit ratio (excluding zero costs)
        cost_benefits = [m.cost_benefit_ratio for m in self.measurements if m.cost_benefit_ratio > 0]
        avg_cost_benefit = sum(cost_benefits) / len(cost_benefits) if cost_benefits else 0.0
        
        return {
            'total_measurements': total_measurements,
            'average_improvement': avg_improvement,
            'success_rate': success_rate,
            'average_cost_benefit': avg_cost_benefit,
            'meets_threshold_count': successful_measurements,
            'improvement_breakdown': {
                'character': sum(m.improvement_character for m in self.measurements) / total_measurements,
                'word': sum(m.improvement_word for m in self.measurements) / total_measurements,
                'line': sum(m.improvement_line for m in self.measurements) / total_measurements
            }
        }
    
    def export_measurements(self, output_file: str) -> None:
        """Export measurements to JSON file."""
        try:
            data = {
                'metadata': {
                    'export_date': datetime.now().isoformat(),
                    'total_measurements': len(self.measurements),
                    'success_threshold': self.min_improvement_threshold,
                    'max_time_increase': self.max_processing_time_increase
                },
                'measurements': []
            }
            
            for i, measurement in enumerate(self.measurements):
                data['measurements'].append({
                    'measurement_id': i,
                    'before': {
                        'character_accuracy': measurement.before_metrics.character_accuracy,
                        'word_accuracy': measurement.before_metrics.word_accuracy,
                        'line_accuracy': measurement.before_metrics.line_accuracy,
                        'overall_accuracy': measurement.before_metrics.overall_accuracy,
                        'error_count': measurement.before_metrics.error_count,
                        'processing_time': measurement.before_metrics.processing_time
                    },
                    'after': {
                        'character_accuracy': measurement.after_metrics.character_accuracy,
                        'word_accuracy': measurement.after_metrics.word_accuracy,
                        'line_accuracy': measurement.after_metrics.line_accuracy,
                        'overall_accuracy': measurement.after_metrics.overall_accuracy,
                        'error_count': measurement.after_metrics.error_count,
                        'processing_time': measurement.after_metrics.processing_time
                    },
                    'improvements': {
                        'character': measurement.improvement_character,
                        'word': measurement.improvement_word,
                        'line': measurement.improvement_line,
                        'overall': measurement.improvement_overall
                    },
                    'cost_benefit_ratio': measurement.cost_benefit_ratio,
                    'meets_success_threshold': measurement.meets_success_threshold
                })
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported {len(self.measurements)} measurements to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to export measurements: {e}")

# Convenience functions
def create_accuracy_measurer(gold_pages_manager=None) -> AccuracyMeasurer:
    """Create a new Accuracy Measurer instance."""
    return AccuracyMeasurer(gold_pages_manager)

def measure_ocr_accuracy(ocr_text: str, ground_truth: str, 
                        processing_time: float = 0.0, confidence: float = 0.0) -> AccuracyMetrics:
    """Quick function to measure OCR accuracy."""
    measurer = AccuracyMeasurer()
    return measurer.measure_ocr_accuracy(ocr_text, ground_truth, processing_time, confidence)
