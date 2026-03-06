#!/usr/bin/env python3
"""
Baseline OCR Accuracy Assessment
Provides functions to calculate and compare OCR accuracy against ground truth
"""

import re
import difflib
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
import logging

class BaselineAccuracyCalculator:
    """Calculates OCR accuracy metrics against ground truth data."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison by removing extra whitespace and standardizing."""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Convert to lowercase for case-insensitive comparison
        text = text.lower()
        
        # Remove common OCR artifacts
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        
        return text
    
    def calculate_character_accuracy(self, ocr_text: str, ground_truth: str) -> Dict[str, float]:
        """Calculate character-level accuracy metrics."""
        ocr_normalized = self.normalize_text(ocr_text)
        gt_normalized = self.normalize_text(ground_truth)
        
        if not gt_normalized:
            return {'accuracy': 0.0, 'error_rate': 1.0, 'characters_compared': 0}
        
        # Use difflib for character-level comparison
        matcher = difflib.SequenceMatcher(None, gt_normalized, ocr_normalized)
        
        # Calculate matches and mismatches
        matches = sum(block.size for block in matcher.get_matching_blocks())
        total_chars = len(gt_normalized)
        
        accuracy = matches / total_chars if total_chars > 0 else 0.0
        error_rate = 1.0 - accuracy
        
        return {
            'accuracy': accuracy,
            'error_rate': error_rate,
            'characters_compared': total_chars,
            'characters_matched': matches,
            'characters_missed': total_chars - matches
        }
    
    def calculate_word_accuracy(self, ocr_text: str, ground_truth: str) -> Dict[str, float]:
        """Calculate word-level accuracy metrics."""
        ocr_words = self.normalize_text(ocr_text).split()
        gt_words = self.normalize_text(ground_truth).split()
        
        if not gt_words:
            return {'accuracy': 0.0, 'error_rate': 1.0, 'words_compared': 0}
        
        # Use difflib for word-level comparison
        matcher = difflib.SequenceMatcher(None, gt_words, ocr_words)
        
        # Calculate matches and mismatches
        matches = sum(block.size for block in matcher.get_matching_blocks())
        total_words = len(gt_words)
        
        accuracy = matches / total_words if total_words > 0 else 0.0
        error_rate = 1.0 - accuracy
        
        return {
            'accuracy': accuracy,
            'error_rate': error_rate,
            'words_compared': total_words,
            'words_matched': matches,
            'words_missed': total_words - matches
        }
    
    def calculate_line_accuracy(self, ocr_text: str, ground_truth: str) -> Dict[str, float]:
        """Calculate line-level accuracy metrics."""
        ocr_lines = [line.strip() for line in ocr_text.split('\n') if line.strip()]
        gt_lines = [line.strip() for line in ground_truth.split('\n') if line.strip()]
        
        if not gt_lines:
            return {'accuracy': 0.0, 'error_rate': 1.0, 'lines_compared': 0}
        
        # Use difflib for line-level comparison
        matcher = difflib.SequenceMatcher(None, gt_lines, ocr_lines)
        
        # Calculate matches and mismatches
        matches = sum(block.size for block in matcher.get_matching_blocks())
        total_lines = len(gt_lines)
        
        accuracy = matches / total_lines if total_lines > 0 else 0.0
        error_rate = 1.0 - accuracy
        
        return {
            'accuracy': accuracy,
            'error_rate': error_rate,
            'lines_compared': total_lines,
            'lines_matched': matches,
            'lines_missed': total_lines - matches
        }
    
    def calculate_overall_accuracy(self, ocr_text: str, ground_truth: str) -> Dict[str, Any]:
        """Calculate comprehensive accuracy metrics."""
        char_metrics = self.calculate_character_accuracy(ocr_text, ground_truth)
        word_metrics = self.calculate_word_accuracy(ocr_text, ground_truth)
        line_metrics = self.calculate_line_accuracy(ocr_text, ground_truth)
        
        # Calculate overall score (weighted average)
        overall_score = (
            char_metrics['accuracy'] * 0.4 +  # Character accuracy is most important
            word_metrics['accuracy'] * 0.4 +  # Word accuracy is equally important
            line_metrics['accuracy'] * 0.2    # Line accuracy is less critical
        )
        
        return {
            'overall_score': overall_score,
            'character_accuracy': char_metrics,
            'word_accuracy': word_metrics,
            'line_accuracy': line_metrics,
            'text_length': {
                'ocr_chars': len(ocr_text),
                'gt_chars': len(ground_truth),
                'ocr_words': len(ocr_text.split()),
                'gt_words': len(ground_truth.split()),
                'ocr_lines': len([l for l in ocr_text.split('\n') if l.strip()]),
                'gt_lines': len([l for l in ground_truth.split('\n') if l.strip()])
            }
        }
    
    def compare_with_baseline(self, current_metrics: Dict[str, Any], 
                            baseline_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Compare current metrics against baseline metrics."""
        comparison = {}
        
        for metric_type in ['character_accuracy', 'word_accuracy', 'line_accuracy']:
            if metric_type in current_metrics and metric_type in baseline_metrics:
                current_acc = current_metrics[metric_type]['accuracy']
                baseline_acc = baseline_metrics[metric_type]['accuracy']
                
                improvement = current_acc - baseline_acc
                improvement_pct = (improvement / baseline_acc * 100) if baseline_acc > 0 else 0
                
                comparison[metric_type] = {
                    'current': current_acc,
                    'baseline': baseline_acc,
                    'improvement': improvement,
                    'improvement_percentage': improvement_pct,
                    'meets_baseline': current_acc >= baseline_acc
                }
        
        # Overall comparison
        current_overall = current_metrics.get('overall_score', 0)
        baseline_overall = baseline_metrics.get('overall_score', 0)
        
        comparison['overall'] = {
            'current': current_overall,
            'baseline': baseline_overall,
            'improvement': current_overall - baseline_overall,
            'improvement_percentage': ((current_overall - baseline_overall) / baseline_overall * 100) if baseline_overall > 0 else 0,
            'meets_baseline': current_overall >= baseline_overall
        }
        
        return comparison

class GroundTruthManager:
    """Manages ground truth data for baseline accuracy assessment."""
    
    def __init__(self, gt_dir: str = "./data/ground_truth"):
        self.gt_dir = Path(gt_dir)
        self.gt_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def save_ground_truth(self, document_id: str, page_num: int, 
                         ground_truth_text: str, metadata: Dict[str, Any] = None) -> str:
        """Save ground truth text for a specific document page."""
        gt_file = self.gt_dir / f"{document_id}_page_{page_num:03d}.txt"
        
        gt_data = {
            'document_id': document_id,
            'page_num': page_num,
            'ground_truth_text': ground_truth_text,
            'metadata': metadata or {},
            'created_at': str(Path().cwd())  # Simple timestamp placeholder
        }
        
        with open(gt_file, 'w', encoding='utf-8') as f:
            f.write(ground_truth_text)
        
        # Save metadata
        metadata_file = self.gt_dir / f"{document_id}_page_{page_num:03d}_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(gt_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved ground truth: {gt_file}")
        return str(gt_file)
    
    def load_ground_truth(self, document_id: str, page_num: int) -> Optional[str]:
        """Load ground truth text for a specific document page."""
        gt_file = self.gt_dir / f"{document_id}_page_{page_num:03d}.txt"
        
        if gt_file.exists():
            with open(gt_file, 'r', encoding='utf-8') as f:
                return f.read()
        
        return None
    
    def list_available_ground_truth(self) -> List[Dict[str, Any]]:
        """List all available ground truth data."""
        gt_files = list(self.gt_dir.glob("*_page_*.txt"))
        gt_data = []
        
        for gt_file in gt_files:
            # Extract document_id and page_num from filename
            match = re.match(r'(.+)_page_(\d+)\.txt', gt_file.name)
            if match:
                document_id, page_num = match.groups()
                gt_data.append({
                    'document_id': document_id,
                    'page_num': int(page_num),
                    'file_path': str(gt_file),
                    'file_size': gt_file.stat().st_size
                })
        
        return sorted(gt_data, key=lambda x: (x['document_id'], x['page_num']))

def create_baseline_from_evaluation(eval_data: List[Dict[str, Any]], 
                                  output_file: str = "./data/baseline_metrics.json") -> Dict[str, Any]:
    """Create baseline metrics from evaluation data."""
    calculator = BaselineAccuracyCalculator()
    
    # Calculate metrics for each document
    document_metrics = []
    overall_metrics = {
        'character_accuracy': [],
        'word_accuracy': [],
        'line_accuracy': [],
        'overall_scores': []
    }
    
    for doc_data in eval_data:
        # Extract text from page statistics
        doc_text = ""
        for page_stat in doc_data.get('page_statistics', []):
            # This would need to be adapted based on your actual data structure
            # For now, we'll use a placeholder
            doc_text += page_stat.get('extracted_text', '')
        
        # For demonstration, we'll create synthetic ground truth
        # In practice, you would load actual ground truth data
        ground_truth = doc_text  # This should be replaced with actual ground truth
        
        if doc_text and ground_truth:
            metrics = calculator.calculate_overall_accuracy(doc_text, ground_truth)
            document_metrics.append({
                'document_id': doc_data.get('pdf_path', 'unknown'),
                'metrics': metrics
            })
            
            # Collect for overall statistics
            overall_metrics['character_accuracy'].append(metrics['character_accuracy']['accuracy'])
            overall_metrics['word_accuracy'].append(metrics['word_accuracy']['accuracy'])
            overall_metrics['line_accuracy'].append(metrics['line_accuracy']['accuracy'])
            overall_metrics['overall_scores'].append(metrics['overall_score'])
    
    # Calculate baseline statistics
    baseline = {
        'created_at': str(Path().cwd()),  # Simple timestamp placeholder
        'total_documents': len(document_metrics),
        'baseline_metrics': {
            'character_accuracy': {
                'mean': sum(overall_metrics['character_accuracy']) / len(overall_metrics['character_accuracy']) if overall_metrics['character_accuracy'] else 0,
                'min': min(overall_metrics['character_accuracy']) if overall_metrics['character_accuracy'] else 0,
                'max': max(overall_metrics['character_accuracy']) if overall_metrics['character_accuracy'] else 0
            },
            'word_accuracy': {
                'mean': sum(overall_metrics['word_accuracy']) / len(overall_metrics['word_accuracy']) if overall_metrics['word_accuracy'] else 0,
                'min': min(overall_metrics['word_accuracy']) if overall_metrics['word_accuracy'] else 0,
                'max': max(overall_metrics['word_accuracy']) if overall_metrics['word_accuracy'] else 0
            },
            'line_accuracy': {
                'mean': sum(overall_metrics['line_accuracy']) / len(overall_metrics['line_accuracy']) if overall_metrics['line_accuracy'] else 0,
                'min': min(overall_metrics['line_accuracy']) if overall_metrics['line_accuracy'] else 0,
                'max': max(overall_metrics['line_accuracy']) if overall_metrics['line_accuracy'] else 0
            },
            'overall_score': {
                'mean': sum(overall_metrics['overall_scores']) / len(overall_metrics['overall_scores']) if overall_metrics['overall_scores'] else 0,
                'min': min(overall_metrics['overall_scores']) if overall_metrics['overall_scores'] else 0,
                'max': max(overall_metrics['overall_scores']) if overall_metrics['overall_scores'] else 0
            }
        },
        'document_metrics': document_metrics
    }
    
    # Save baseline
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(baseline, f, indent=2, ensure_ascii=False)
    
    return baseline

# Example usage and testing
if __name__ == "__main__":
    # Example usage
    calculator = BaselineAccuracyCalculator()
    
    # Example OCR text and ground truth
    ocr_text = "This is a test document with some OCR errors."
    ground_truth = "This is a test document with some OCR errors."
    
    # Calculate accuracy
    metrics = calculator.calculate_overall_accuracy(ocr_text, ground_truth)
    
    print("OCR Accuracy Metrics:")
    print(f"Overall Score: {metrics['overall_score']:.3f}")
    print(f"Character Accuracy: {metrics['character_accuracy']['accuracy']:.3f}")
    print(f"Word Accuracy: {metrics['word_accuracy']['accuracy']:.3f}")
    print(f"Line Accuracy: {metrics['line_accuracy']['accuracy']:.3f}")

