#!/usr/bin/env python3
"""
Evaluate translation pairing quality.
Computes Precision, Recall, F1 from ground truth labels vs predicted pairs.
"""

import sys
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class PairingLabel:
    """Ground truth pairing label."""
    pdf_id: str
    page: int
    akk_block_id: str
    expected_trans_id: str


@dataclass
class PairingMetrics:
    """Pairing evaluation metrics."""
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    total_predictions: int = 0
    total_labels: int = 0


def load_labels(labels_path: Path) -> List[PairingLabel]:
    """
    Load ground truth pairing labels from JSON.
    
    Expected format:
    {
        "pairings": [
            {
                "pdf_id": "doc1",
                "page": 1,
                "akk_block_id": "block_1",
                "trans_block_id": "block_2"
            },
            ...
        ]
    }
    """
    logger.info(f"Loading labels from: {labels_path}")
    
    with open(labels_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    labels = []
    for entry in data.get('pairings', []):
        label = PairingLabel(
            pdf_id=entry['pdf_id'],
            page=entry['page'],
            akk_block_id=entry['akk_block_id'],
            expected_trans_id=entry['trans_block_id']
        )
        labels.append(label)
    
    logger.info(f"Loaded {len(labels)} ground truth pairings")
    return labels


def load_predictions(csv_paths: List[Path]) -> Dict[Tuple[str, int, str], str]:
    """
    Load predicted pairings from translations.csv files.
    
    Returns:
        Dict mapping (pdf_id, page, akk_block_id) -> trans_block_id
    """
    predictions = {}
    
    for csv_path in csv_paths:
        if not csv_path.exists():
            logger.warning(f"Predictions file not found: {csv_path}")
            continue
        
        logger.info(f"Loading predictions from: {csv_path}")
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pdf_id = row['pdf_id']
                page = int(row['page'])
                akk_block_id = row['akk_block_id']
                trans_block_id = row['trans_block_id']
                
                key = (pdf_id, page, akk_block_id)
                predictions[key] = trans_block_id
    
    logger.info(f"Loaded {len(predictions)} predicted pairings")
    return predictions


def compute_metrics(
    labels: List[PairingLabel],
    predictions: Dict[Tuple[str, int, str], str]
) -> PairingMetrics:
    """
    Compute Precision, Recall, F1.
    
    TP: Predicted pair matches ground truth
    FP: Predicted pair doesn't match ground truth or no ground truth exists
    FN: Ground truth pair missing from predictions
    """
    metrics = PairingMetrics()
    
    # Convert labels to dict for easier lookup
    label_dict = {}
    for label in labels:
        key = (label.pdf_id, label.page, label.akk_block_id)
        label_dict[key] = label.expected_trans_id
    
    # Check predictions
    for key, predicted_trans_id in predictions.items():
        if key in label_dict:
            expected_trans_id = label_dict[key]
            if predicted_trans_id == expected_trans_id:
                metrics.true_positives += 1
            else:
                metrics.false_positives += 1
        else:
            # Prediction without ground truth label
            metrics.false_positives += 1
    
    # Check for missing predictions
    for key, expected_trans_id in label_dict.items():
        if key not in predictions:
            metrics.false_negatives += 1
    
    # Calculate metrics
    metrics.total_predictions = len(predictions)
    metrics.total_labels = len(labels)
    
    if metrics.true_positives + metrics.false_positives > 0:
        metrics.precision = metrics.true_positives / (metrics.true_positives + metrics.false_positives)
    
    if metrics.true_positives + metrics.false_negatives > 0:
        metrics.recall = metrics.true_positives / (metrics.true_positives + metrics.false_negatives)
    
    if metrics.precision + metrics.recall > 0:
        metrics.f1 = 2 * (metrics.precision * metrics.recall) / (metrics.precision + metrics.recall)
    
    return metrics


def find_translation_csvs(output_root: Path) -> List[Path]:
    """Find all translations.csv files in output directory."""
    csv_files = list(output_root.rglob("translations.csv"))
    logger.info(f"Found {len(csv_files)} translations.csv files in {output_root}")
    return csv_files


def print_metrics_report(metrics: PairingMetrics, output_path: Path = None):
    """Print metrics report."""
    print()
    print("=" * 60)
    print("PAIRING EVALUATION METRICS")
    print("=" * 60)
    print()
    print(f"Ground Truth Labels:    {metrics.total_labels}")
    print(f"Predicted Pairings:     {metrics.total_predictions}")
    print()
    print(f"True Positives:         {metrics.true_positives}")
    print(f"False Positives:        {metrics.false_positives}")
    print(f"False Negatives:        {metrics.false_negatives}")
    print()
    print(f"Precision:              {metrics.precision:.3f}")
    print(f"Recall:                 {metrics.recall:.3f}")
    print(f"F1 Score:               {metrics.f1:.3f}")
    print()
    print("=" * 60)
    
    # Acceptance check
    if metrics.f1 >= 0.80:
        print("✅ ACCEPTANCE: F1 ≥ 0.80 PASSED")
    else:
        print(f"❌ ACCEPTANCE: F1 < 0.80 FAILED (got {metrics.f1:.3f})")
    
    print("=" * 60)
    
    # Save metrics
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        metrics_dict = {
            'total_labels': metrics.total_labels,
            'total_predictions': metrics.total_predictions,
            'true_positives': metrics.true_positives,
            'false_positives': metrics.false_positives,
            'false_negatives': metrics.false_negatives,
            'precision': metrics.precision,
            'recall': metrics.recall,
            'f1': metrics.f1,
            'acceptance_passed': metrics.f1 >= 0.80
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_dict, f, indent=2)
        
        logger.info(f"Metrics saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate pairing quality')
    parser.add_argument('--labels', required=True,
                       help='Path to pairing_labels.json with ground truth')
    parser.add_argument('--predictions', nargs='+',
                       help='Paths to translations.csv files (optional if using --output-root)')
    parser.add_argument('--output-root',
                       help='Root directory to scan for translations.csv files')
    parser.add_argument('--output', default='pairing_eval.json',
                       help='Output metrics file (default: pairing_eval.json)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PAIRING EVALUATION")
    print("=" * 60)
    
    # Load labels
    labels_path = Path(args.labels)
    if not labels_path.exists():
        logger.error(f"Labels file not found: {labels_path}")
        sys.exit(1)
    
    labels = load_labels(labels_path)
    
    if not labels:
        logger.error("No labels loaded!")
        sys.exit(1)
    
    # Find prediction files
    csv_paths = []
    
    if args.predictions:
        csv_paths = [Path(p) for p in args.predictions]
    elif args.output_root:
        output_root = Path(args.output_root)
        csv_paths = find_translation_csvs(output_root)
    else:
        logger.error("Must specify either --predictions or --output-root")
        sys.exit(1)
    
    if not csv_paths:
        logger.error("No prediction files found!")
        sys.exit(1)
    
    # Load predictions
    predictions = load_predictions(csv_paths)
    
    if not predictions:
        logger.error("No predictions loaded!")
        sys.exit(1)
    
    # Compute metrics
    metrics = compute_metrics(labels, predictions)
    
    # Print report
    output_path = Path(args.output)
    print_metrics_report(metrics, output_path)
    
    # Return exit code based on acceptance
    return 0 if metrics.f1 >= 0.80 else 1


if __name__ == "__main__":
    sys.exit(main())
