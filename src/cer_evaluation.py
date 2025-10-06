"""
Character Error Rate (CER) and performance evaluation for OCR pipeline.
Provides baseline measurements and language-specific accuracy metrics.
"""
import logging
import re
import time
import unicodedata
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import json
from collections import defaultdict
import statistics
import pandas as pd

logger = logging.getLogger(__name__)

def normalize_text(text: str) -> str:
    """Normalize text for CER calculation."""
    # Convert to lowercase
    text = text.lower()
    
    # Unicode normalization
    text = unicodedata.normalize('NFKD', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove punctuation for CER calculation (optional)
    # text = re.sub(r'[^\w\s]', '', text)
    
    return text

def calculate_cer(reference: str, hypothesis: str) -> float:
    """
    Calculate Character Error Rate (CER) between reference and hypothesis text.
    
    CER = (S + D + I) / N
    where S=substitutions, D=deletions, I=insertions, N=total characters in reference
    """
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    
    # Calculate edit distance (Levenshtein distance)
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    
    # Initialize base cases
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    
    # Fill the DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # deletion
                    dp[i][j-1],    # insertion
                    dp[i-1][j-1]   # substitution
                )
    
    edit_distance = dp[n][m]
    
    # CER calculation
    if n == 0:
        return 1.0 if m > 0 else 0.0
    
    return edit_distance / n

def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate (WER)."""
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()
    
    n, m = len(ref_words), len(hyp_words)
    
    if n == 0:
        return 1.0 if m > 0 else 0.0
    
    # Simple word-level edit distance
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[n][m] / n

def detect_language_from_text(text: str) -> str:
    """Simple language detection based on character patterns."""
    text = text.lower()
    
    # Turkish specific characters
    turkish_chars = set('çğıöşüâêîôû')
    if any(char in turkish_chars for char in text):
        return 'turkish'
    
    # German specific characters
    german_chars = set('äöüß')
    if any(char in german_chars for char in text):
        return 'german'
    
    # French specific characters
    french_chars = set('àâäéèêëïîôöùûüÿç')
    if any(char in french_chars for char in text):
        return 'french'
    
    # Italian specific characters
    italian_chars = set('àáèéìíîòóùú')
    if any(char in italian_chars for char in text):
        return 'italian'
    
    return 'english'

class CEREvaluator:
    """Character Error Rate evaluator for OCR pipeline."""
    
    def __init__(self):
        self.results: List[Dict[str, Any]] = []
        self.language_stats: Dict[str, List[float]] = defaultdict(list)
        
    def evaluate_page(self, reference: str, hypothesis: str, 
                     page_id: str, language: str = None) -> Dict[str, Any]:
        """Evaluate a single page."""
        if language is None:
            language = detect_language_from_text(reference)
        
        cer = calculate_cer(reference, hypothesis)
        wer = calculate_wer(reference, hypothesis)
        
        # Character-level statistics
        ref_chars = len(normalize_text(reference))
        hyp_chars = len(normalize_text(hypothesis))
        char_ratio = hyp_chars / max(ref_chars, 1)
        
        # Word-level statistics
        ref_words = len(normalize_text(reference).split())
        hyp_words = len(normalize_text(hypothesis).split())
        word_ratio = hyp_words / max(ref_words, 1)
        
        result = {
            'page_id': page_id,
            'language': language,
            'cer': cer,
            'wer': wer,
            'reference_chars': ref_chars,
            'hypothesis_chars': hyp_chars,
            'char_ratio': char_ratio,
            'reference_words': ref_words,
            'hypothesis_words': hyp_words,
            'word_ratio': word_ratio,
            'reference_text': reference,
            'hypothesis_text': hypothesis
        }
        
        self.results.append(result)
        self.language_stats[language].append(cer)
        
        return result
    
    def get_language_baselines(self) -> Dict[str, Dict[str, float]]:
        """Get baseline CER statistics by language."""
        baselines = {}
        
        for language, cer_values in self.language_stats.items():
            if cer_values:
                baselines[language] = {
                    'mean_cer': statistics.mean(cer_values),
                    'median_cer': statistics.median(cer_values),
                    'min_cer': min(cer_values),
                    'max_cer': max(cer_values),
                    'std_cer': statistics.stdev(cer_values) if len(cer_values) > 1 else 0.0,
                    'samples': len(cer_values)
                }
        
        return baselines
    
    def get_overall_baseline(self) -> Dict[str, float]:
        """Get overall baseline statistics."""
        if not self.results:
            return {}
        
        all_cer = [r['cer'] for r in self.results]
        all_wer = [r['wer'] for r in self.results]
        
        return {
            'mean_cer': statistics.mean(all_cer),
            'median_cer': statistics.median(all_cer),
            'mean_wer': statistics.mean(all_wer),
            'median_wer': statistics.median(all_wer),
            'total_pages': len(self.results),
            'languages': list(self.language_stats.keys())
        }
    
    def export_results(self, filepath: str) -> None:
        """Export evaluation results to JSON."""
        export_data = {
            'overall_baseline': self.get_overall_baseline(),
            'language_baselines': self.get_language_baselines(),
            'detailed_results': self.results,
            'evaluation_timestamp': time.time()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"CER evaluation results exported to {filepath}")
    
    def create_summary_report(self) -> str:
        """Create a human-readable summary report."""
        if not self.results:
            return "No evaluation results available."
        
        overall = self.get_overall_baseline()
        language_baselines = self.get_language_baselines()
        
        report = []
        report.append("=== OCR Pipeline CER Baseline Report ===\n")
        
        report.append(f"Total Pages Evaluated: {overall['total_pages']}")
        report.append(f"Overall Mean CER: {overall['mean_cer']:.3f}")
        report.append(f"Overall Mean WER: {overall['mean_wer']:.3f}")
        report.append(f"Languages: {', '.join(overall['languages'])}\n")
        
        report.append("=== Language-Specific Baselines ===")
        for language, stats in language_baselines.items():
            report.append(f"\n{language.title()}:")
            report.append(f"  Mean CER: {stats['mean_cer']:.3f}")
            report.append(f"  Median CER: {stats['median_cer']:.3f}")
            report.append(f"  Min CER: {stats['min_cer']:.3f}")
            report.append(f"  Max CER: {stats['max_cer']:.3f}")
            report.append(f"  Std Dev: {stats['std_cer']:.3f}")
            report.append(f"  Samples: {stats['samples']}")
        
        # Performance thresholds
        report.append("\n=== Performance Assessment ===")
        mean_cer = overall['mean_cer']
        if mean_cer < 0.1:
            assessment = "Excellent (CER < 10%)"
        elif mean_cer < 0.2:
            assessment = "Good (CER < 20%)"
        elif mean_cer < 0.3:
            assessment = "Fair (CER < 30%)"
        else:
            assessment = "Needs Improvement (CER ≥ 30%)"
        
        report.append(f"Overall Performance: {assessment}")
        
        return "\n".join(report)

def evaluate_gold_standard(gold_csv_path: str, ocr_results_dir: str, 
                          output_dir: str = None) -> CEREvaluator:
    """
    Evaluate OCR pipeline against gold standard data.
    
    Args:
        gold_csv_path: Path to gold standard CSV file
        ocr_results_dir: Directory containing OCR pipeline results
        output_dir: Directory to save evaluation results
    
    Returns:
        CEREvaluator with results
    """
    evaluator = CEREvaluator()
    
    # Load gold standard data
    try:
        gold_df = pd.read_csv(gold_csv_path)
        logger.info(f"Loaded {len(gold_df)} gold standard entries")
    except Exception as e:
        logger.error(f"Failed to load gold standard CSV: {e}")
        return evaluator
    
    ocr_results_path = Path(ocr_results_dir)
    
    # Process each gold standard entry
    for _, row in gold_df.iterrows():
        pdf_name = row.iloc[0]  # First column: PDF name
        page_num = row.iloc[1]  # Second column: page number
        gold_text = row.iloc[2]  # Third column: gold standard text
        
        # Find corresponding OCR result
        page_id = f"page_{page_num:03d}"
        
        # Look for OCR result file (could be in CSV format)
        ocr_file_pattern = f"*{pdf_name}*"
        ocr_files = list(ocr_results_path.glob(ocr_file_pattern))
        
        if not ocr_files:
            logger.warning(f"No OCR result found for {pdf_name} page {page_num}")
            continue
        
        # Read OCR result (assuming CSV format)
        try:
            ocr_df = pd.read_csv(ocr_files[0])
            
            # Find the specific page
            page_rows = ocr_df[ocr_df['page_id'] == page_id]
            
            if page_rows.empty:
                logger.warning(f"Page {page_id} not found in OCR results for {pdf_name}")
                continue
            
            # Combine all text elements for this page
            ocr_text = ' '.join(page_rows['text'].astype(str))
            
            # Evaluate this page
            result = evaluator.evaluate_page(
                reference=gold_text,
                hypothesis=ocr_text,
                page_id=f"{pdf_name}_{page_id}"
            )
            
            logger.info(f"Evaluated {pdf_name} {page_id}: CER={result['cer']:.3f}")
            
        except Exception as e:
            logger.error(f"Error processing OCR results for {pdf_name}: {e}")
            continue
    
    # Export results if output directory specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Export detailed results
        evaluator.export_results(str(output_path / "cer_evaluation.json"))
        
        # Save summary report
        summary = evaluator.create_summary_report()
        with open(output_path / "cer_baseline_report.txt", 'w', encoding='utf-8') as f:
            f.write(summary)
        
        logger.info(f"Evaluation results saved to {output_path}")
    
    return evaluator

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate OCR pipeline CER baseline')
    parser.add_argument('gold_csv', help='Path to gold standard CSV file')
    parser.add_argument('ocr_results', help='Directory containing OCR results')
    parser.add_argument('-o', '--output', help='Output directory for evaluation results')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run evaluation
    evaluator = evaluate_gold_standard(args.gold_csv, args.ocr_results, args.output)
    
    # Print summary
    print(evaluator.create_summary_report())