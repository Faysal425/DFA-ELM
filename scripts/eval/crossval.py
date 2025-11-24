#!/usr/bin/env python3
"""
Perform cross-validation evaluation using fixed subject-independent folds.

This script runs 5-fold CV evaluation on trained models using pre-defined
subject splits to ensure no data leakage.

Usage:
    python crossval.py --model_dir ../weights --results_dir ../results --dataset MAT
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

from dfaelm.train import evaluate_model
from dfaelm.data import load_fold_splits


def load_fold_results(results_dir, dataset, model_type):
    """
    Load evaluation results for all folds

    Args:
        results_dir: Directory containing fold results
        dataset: Dataset name
        model_type: 'reconstruction' or 'classification'

    Returns:
        fold_results: Dictionary of fold results
    """
    results_path = Path(results_dir) / dataset.upper()
    fold_results = {}

    if model_type == 'classification':
        results_path = results_path / 'clf_reconstructed'

    # Find all fold result files
    result_files = list(results_path.glob("metrics_fold*.json"))

    for result_file in result_files:
        fold_idx = int(result_file.stem.split('fold')[-1])
        with open(result_file, 'r') as f:
            fold_results[fold_idx] = json.load(f)

    return fold_results


def calculate_cv_statistics(fold_results):
    """
    Calculate cross-validation statistics

    Args:
        fold_results: Dictionary of fold results

    Returns:
        cv_stats: Dictionary with mean, std, ci for each metric
    """
    if not fold_results:
        return {}

    # Extract all metrics
    all_metrics = set()
    for fold_data in fold_results.values():
        all_metrics.update(fold_data.keys())

    cv_stats = {}

    for metric in all_metrics:
        values = []
        for fold_data in fold_results.values():
            if metric in fold_data:
                values.append(fold_data[metric])

        if values:
            values = np.array(values)
            mean_val = np.mean(values)
            std_val = np.std(values)
            ci_95 = 1.96 * std_val / np.sqrt(len(values))

            cv_stats[metric] = {
                'mean': float(mean_val),
                'std': float(std_val),
                'ci_95': float(ci_95),
                'values': values.tolist()
            }

    return cv_stats


def save_cv_summary(cv_stats, output_path):
    """
    Save cross-validation summary to JSON

    Args:
        cv_stats: CV statistics dictionary
        output_path: Output file path
    """
    with open(output_path, 'w') as f:
        json.dump(cv_stats, f, indent=2)


def print_cv_summary(cv_stats, dataset, model_type):
    """
    Print formatted CV summary

    Args:
        cv_stats: CV statistics dictionary
        dataset: Dataset name
        model_type: Model type
    """
    print(f"\n{'='*60}")
    print(f"Cross-Validation Results - {dataset.upper()} {model_type.title()}")
    print(f"{'='*60}")

    for metric, stats in cv_stats.items():
        mean_val = stats['mean']
        std_val = stats['std']
        ci_val = stats['ci_95']

        if metric in ['accuracy', 'precision', 'recall', 'f1_score', 'specificity', 'sensitivity']:
            # Percentage metrics
            print(".1f")
        elif metric in ['mcc', 'auc', 'pr_auc']:
            # Score metrics
            print(".3f")
        elif metric in ['brier_score']:
            # Brier score
            print(".4f")
        else:
            # Other metrics
            print(".4f")

    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Cross-validation evaluation")
    parser.add_argument('--results_dir', type=str, default='../results',
                       help='Directory containing fold results')
    parser.add_argument('--output_dir', type=str, default='../results',
                       help='Output directory for CV summary')
    parser.add_argument('--dataset', type=str, default='MAT',
                       help='Dataset name (MAT or SELF_COLLECTED)')
    parser.add_argument('--model_type', type=str, default='classification',
                       choices=['reconstruction', 'classification'],
                       help='Model type to evaluate')

    args = parser.parse_args()

    print(f"Performing cross-validation analysis for {args.dataset} {args.model_type}...")

    # Load fold results
    fold_results = load_fold_results(args.results_dir, args.dataset, args.model_type)

    if not fold_results:
        print(f"No fold results found in {args.results_dir}")
        return

    print(f"Loaded results for {len(fold_results)} folds")

    # Calculate CV statistics
    cv_stats = calculate_cv_statistics(fold_results)

    # Print summary
    print_cv_summary(cv_stats, args.dataset, args.model_type)

    # Save CV summary
    output_dir = Path(args.output_dir) / args.dataset.upper()
    if args.model_type == 'classification':
        output_dir = output_dir / 'clf_reconstructed'

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_file = output_dir / 'summary_cv.json'

    save_cv_summary(cv_stats, summary_file)
    print(f"CV summary saved to {summary_file}")


if __name__ == '__main__':
    main()
