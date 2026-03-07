#!/usr/bin/env python3
"""
Confidence calibration tool for OCR engines.
Fits temperature scaling per engine×language using gold CSV data.
"""
import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
from scipy.optimize import minimize_scalar
from collections import defaultdict
import matplotlib.pyplot as plt

# Add src to path for imports
sys.path.append('src')
from recognition_router import RecognitionRouter, RecognitionResult
from lang_and_akkadian import analyze_text_line

logger = logging.getLogger(__name__)


class ConfidenceCalibrator:
    """Calibrates OCR engine confidences using temperature scaling."""
    
    def __init__(self, gold_csv_path: str, input_dir: str):
        self.gold_csv_path = gold_csv_path
        self.input_dir = input_dir
        self.calibration_data = {}
        
        # Load gold data
        self.gold_pages = self._load_gold_data()
        
    def _load_gold_data(self) -> List[Dict[str, str]]:
        """Load gold data from CSV."""
        gold_pages = []
        
        with open(self.gold_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if ('PDF LINK' in row and 'PAGE' in row and 'HANDTYPED' in row):
                    pdf_link = row['PDF LINK'].strip()
                    page_spec = row['PAGE'].strip()
                    handtyped = row['HANDTYPED'].strip()
                    
                    if handtyped and handtyped.lower() not in ['nan', 'null', '']:
                        gold_pages.append({
                            'pdf_filename': pdf_link,
                            'page_spec': page_spec,
                            'ground_truth': handtyped
                        })
        
        logger.info(f"Loaded {len(gold_pages)} gold pages for calibration")
        return gold_pages
    
    def collect_engine_outputs(self, engines: List[str], max_pages: int = 50) -> Dict[str, List[Dict]]:
        """
        Collect OCR outputs from different engines on gold data.
        
        Args:
            engines: List of engine names to test
            max_pages: Maximum pages to process for calibration
            
        Returns:
            Dict mapping engine×language to list of {confidence, accuracy} records
        """
        from tools.run_baseline_eval import run_ocr_on_page, parse_page_spec, calculate_cer
        
        engine_data = defaultdict(list)
        processed_pages = 0
        
        for page_data in self.gold_pages:
            if processed_pages >= max_pages:
                break
                
            pdf_name = page_data['pdf_filename']
            page_spec = page_data['page_spec']
            ground_truth = page_data['ground_truth']
            
            # Parse page specification
            page_numbers = parse_page_spec(page_spec)
            if not page_numbers:
                continue
            
            pdf_path = f"{self.input_dir}/{pdf_name}"
            if not Path(pdf_path).exists():
                continue
            
            for page_num in page_numbers:
                if processed_pages >= max_pages:
                    break
                
                logger.info(f"Processing {pdf_name} page {page_num} for calibration")
                
                # Detect language from ground truth
                lang_analysis = analyze_text_line(ground_truth)
                language = lang_analysis['language']
                
                # Test each engine
                for engine in engines:
                    try:
                        # Run OCR
                        hypothesis = run_ocr_on_page(pdf_path, page_num, 'fast', engine)
                        
                        if not hypothesis:
                            continue
                        
                        # Calculate accuracy (1 - CER)
                        cer = calculate_cer(ground_truth, hypothesis)
                        accuracy = 1.0 - cer
                        
                        # Get confidence (this is simplified - in practice we'd need
                        # to modify the engines to return confidence scores)
                        confidence = self._estimate_confidence(hypothesis, engine)
                        
                        # Store data point
                        key = f"{engine}_{language}"
                        engine_data[key].append({
                            'confidence': confidence,
                            'accuracy': accuracy,
                            'cer': cer,
                            'text_length': len(hypothesis),
                            'ground_truth_length': len(ground_truth)
                        })
                        
                    except Exception as e:
                        logger.warning(f"Engine {engine} failed on {pdf_name} page {page_num}: {e}")
                        continue
                
                processed_pages += 1
        
        logger.info(f"Collected calibration data for {len(engine_data)} engine×language combinations")
        return dict(engine_data)
    
    def _estimate_confidence(self, text: str, engine: str) -> float:
        """
        Estimate confidence for engines that don't provide it.
        This is a placeholder - in practice, engines should be modified
        to return actual confidence scores.
        """
        if not text:
            return 0.0
        
        # Simple heuristics based on text characteristics
        alpha_ratio = sum(1 for c in text if c.isalpha()) / len(text)
        digit_ratio = sum(1 for c in text if c.isdigit()) / len(text)
        space_ratio = sum(1 for c in text if c.isspace()) / len(text)
        
        # Good text has high alpha ratio, reasonable spaces
        base_conf = alpha_ratio * 0.8 + (0.1 <= space_ratio <= 0.25) * 0.2
        
        # Engine-specific adjustments
        engine_factors = {
            'paddle': 1.0,
            'doctr': 1.1,  # Tends to be more confident
            'easyocr': 0.9,
            'tesseract': 0.8,
            'trocr': 0.7
        }
        
        factor = engine_factors.get(engine, 1.0)
        confidence = min(1.0, base_conf * factor)
        
        return confidence
    
    def fit_temperature_scaling(self, engine_data: Dict[str, List[Dict]]) -> Dict[str, Dict]:
        """
        Fit temperature scaling parameters for each engine×language.
        
        Args:
            engine_data: Collected confidence/accuracy data
            
        Returns:
            Dict mapping engine×language to calibration parameters
        """
        calibration_results = {}
        
        for key, data_points in engine_data.items():
            if len(data_points) < 5:  # Need minimum data
                logger.warning(f"Insufficient data for {key}: {len(data_points)} points")
                continue
            
            # Extract confidences and binary accuracy labels
            confidences = np.array([d['confidence'] for d in data_points])
            # Use accuracy > 0.8 as "correct" threshold
            binary_correct = np.array([d['accuracy'] > 0.8 for d in data_points])
            
            if len(np.unique(binary_correct)) < 2:  # Need both correct and incorrect
                logger.warning(f"No variation in accuracy for {key}")
                continue
            
            # Fit temperature parameter
            best_temp = self._find_optimal_temperature(confidences, binary_correct)
            
            # Calculate calibration metrics
            original_ece = self._expected_calibration_error(confidences, binary_correct)
            calibrated_confs = confidences ** (1.0 / best_temp)
            calibrated_ece = self._expected_calibration_error(calibrated_confs, binary_correct)
            
            calibration_results[key] = {
                'temperature': best_temp,
                'original_ece': original_ece,
                'calibrated_ece': calibrated_ece,
                'improvement': original_ece - calibrated_ece,
                'num_samples': len(data_points),
                'mean_confidence': float(np.mean(confidences)),
                'mean_accuracy': float(np.mean([d['accuracy'] for d in data_points]))
            }
            
            logger.info(f"{key}: T={best_temp:.3f}, ECE {original_ece:.3f}→{calibrated_ece:.3f} (Δ={calibration_results[key]['improvement']:.3f})")
        
        return calibration_results
    
    def _find_optimal_temperature(self, confidences: np.ndarray, 
                                 binary_correct: np.ndarray) -> float:
        """Find optimal temperature parameter using ECE minimization."""
        def objective(temperature):
            if temperature <= 0.01:  # Avoid division by zero
                return 1.0
            
            calibrated = confidences ** (1.0 / temperature)
            return self._expected_calibration_error(calibrated, binary_correct)
        
        # Search in reasonable range
        result = minimize_scalar(objective, bounds=(0.1, 5.0), method='bounded')
        return float(result.x)
    
    def _expected_calibration_error(self, confidences: np.ndarray, 
                                   binary_correct: np.ndarray, num_bins: int = 10) -> float:
        """Calculate Expected Calibration Error (ECE)."""
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        total_samples = len(confidences)
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = binary_correct[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                
                ece += abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def save_calibration(self, calibration_results: Dict[str, Dict], 
                        output_path: str = "data/.cache/calibration.json"):
        """Save calibration parameters to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Extract just the temperature parameters for runtime use
        runtime_calibration = {}
        for key, results in calibration_results.items():
            runtime_calibration[key] = {
                'temperature': results['temperature']
            }
        
        with open(output_path, 'w') as f:
            json.dump(runtime_calibration, f, indent=2)
        
        logger.info(f"Saved calibration parameters to {output_path}")
        
        # Also save detailed results
        detailed_path = output_path.parent / "calibration_detailed.json"
        with open(detailed_path, 'w') as f:
            json.dump(calibration_results, f, indent=2)
        
        logger.info(f"Saved detailed calibration results to {detailed_path}")
    
    def plot_calibration_curves(self, engine_data: Dict[str, List[Dict]], 
                               calibration_results: Dict[str, Dict],
                               output_dir: str = "reports/calibration"):
        """Generate calibration curve plots."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for key, data_points in engine_data.items():
            if key not in calibration_results:
                continue
            
            confidences = np.array([d['confidence'] for d in data_points])
            binary_correct = np.array([d['accuracy'] > 0.8 for d in data_points])
            temperature = calibration_results[key]['temperature']
            
            # Create reliability diagram
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Before calibration
            self._plot_reliability_diagram(ax1, confidences, binary_correct, 
                                         f"{key} - Original")
            
            # After calibration
            calibrated_confs = confidences ** (1.0 / temperature)
            self._plot_reliability_diagram(ax2, calibrated_confs, binary_correct, 
                                         f"{key} - Calibrated (T={temperature:.2f})")
            
            plt.tight_layout()
            plot_path = output_dir / f"calibration_{key}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved calibration plot: {plot_path}")
    
    def _plot_reliability_diagram(self, ax, confidences, binary_correct, title):
        """Plot reliability diagram."""
        num_bins = 10
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
        
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for i in range(num_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            
            if in_bin.sum() > 0:
                bin_accuracy = binary_correct[in_bin].mean()
                bin_confidence = confidences[in_bin].mean()
                bin_count = in_bin.sum()
            else:
                bin_accuracy = 0
                bin_confidence = bin_centers[i]
                bin_count = 0
            
            bin_accuracies.append(bin_accuracy)
            bin_confidences.append(bin_confidence)
            bin_counts.append(bin_count)
        
        # Plot bars
        ax.bar(bin_centers, bin_accuracies, width=0.08, alpha=0.7, 
               label='Accuracy', edgecolor='black')
        
        # Plot confidence line
        ax.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
        
        # Plot average confidence in each bin
        ax.scatter(bin_confidences, bin_accuracies, s=100, alpha=0.8, 
                  c='red', label='Bin Average')
        
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)


def main():
    parser = argparse.ArgumentParser(description="Calibrate OCR engine confidences")
    parser.add_argument('--gold-csv', required=True, help="Path to gold CSV file")
    parser.add_argument('--input-dir', default='data/input_pdfs', help="Input PDFs directory")
    parser.add_argument('--engines', nargs='+', default=['paddle', 'doctr', 'easyocr'], 
                       help="Engines to calibrate")
    parser.add_argument('--max-pages', type=int, default=20, 
                       help="Maximum pages for calibration")
    parser.add_argument('--output', default='data/.cache/calibration.json',
                       help="Output calibration file")
    parser.add_argument('--plot', action='store_true', 
                       help="Generate calibration plots")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        # Create calibrator
        calibrator = ConfidenceCalibrator(args.gold_csv, args.input_dir)
        
        # Collect engine outputs
        logger.info(f"Collecting outputs from engines: {args.engines}")
        engine_data = calibrator.collect_engine_outputs(args.engines, args.max_pages)
        
        if not engine_data:
            logger.error("No calibration data collected")
            return 1
        
        # Fit calibration parameters
        logger.info("Fitting temperature scaling parameters")
        calibration_results = calibrator.fit_temperature_scaling(engine_data)
        
        if not calibration_results:
            logger.error("No calibration parameters fitted")
            return 1
        
        # Save results
        calibrator.save_calibration(calibration_results, args.output)
        
        # Generate plots if requested
        if args.plot:
            logger.info("Generating calibration plots")
            calibrator.plot_calibration_curves(engine_data, calibration_results)
        
        # Print summary
        print("\nCalibration Summary:")
        print("=" * 50)
        for key, results in calibration_results.items():
            print(f"{key}:")
            print(f"  Temperature: {results['temperature']:.3f}")
            print(f"  ECE improvement: {results['improvement']:.3f}")
            print(f"  Samples: {results['num_samples']}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Calibration failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())