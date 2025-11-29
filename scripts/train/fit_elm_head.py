#!/usr/bin/env python3
"""
Fit ELM classifier head analytically using ridge regression.

This script loads pre-trained CNN features and computes optimal ELM weights
using closed-form solution: (H^T H + αI)^(-1) H^T T

Usage:
    python fit_elm_head.py --features_dir ../data/features --output_dir ../weights/classification
"""

import os
import argparse
import numpy as np
import torch
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

from dfaelm.models import ELMClassifier


def load_features_and_labels(features_dir, fold_idx=None):
    """
    Load features and labels for training

    Args:
        features_dir: Directory containing feature files
        fold_idx: Specific fold to load (None for all)

    Returns:
        features: Concatenated feature matrix
        labels: Concatenated labels
    """
    features_list = []
    labels_list = []

    features_path = Path(features_dir)

    if fold_idx is not None:
        # Load specific fold
        fold_files = list(features_path.glob(f"*fold{fold_idx}_features.npy"))
        label_files = list(features_path.glob(f"*fold{fold_idx}_labels.npy"))
    else:
        # Load all folds
        fold_files = list(features_path.glob("*_features.npy"))
        label_files = list(features_path.glob("*_labels.npy"))

    for feat_file in fold_files:
        # Find corresponding label file
        label_file = feat_file.parent / feat_file.name.replace('features', 'labels')

        if label_file.exists():
            features = np.load(feat_file)
            labels = np.load(label_file)

            features_list.append(features)
            labels_list.append(labels)

    if features_list:
        all_features = np.concatenate(features_list, axis=0)
        all_labels = np.concatenate(labels_list, axis=0)
        return all_features, all_labels
    else:
        return None, None


def fit_elm_classifier(features, labels, alpha=0.01, random_state=42):
    """
    Fit ELM classifier using analytical solution

    Args:
        features: Feature matrix [n_samples, n_features]
        labels: Target labels [n_samples]
        alpha: Ridge regularization parameter
        random_state: Random seed

    Returns:
        elm_classifier: Fitted ELMClassifier instance
    """
    # Initialize ELM classifier
    n_features = features.shape[1]
    n_classes = len(np.unique(labels))

    elm = ELMClassifier(
        input_size=n_features,
        hidden_size=[512, 1024, 256],  # Standard ELM hidden size
        output_size=n_classes,
        alpha=alpha,
        random_state=random_state
    )

    # Fit the classifier
    elm.fit(features, labels)

    return elm


def evaluate_elm_classifier(elm_classifier, features, labels):
    """
    Evaluate fitted ELM classifier

    Args:
        elm_classifier: Fitted ELMClassifier
        features: Test features
        labels: Test labels

    Returns:
        metrics: Dictionary of evaluation metrics
    """
    # Get predictions
    predictions = elm_classifier.predict(features)

    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)

    # Classification report
    report = classification_report(labels, predictions, output_dict=True)

    metrics = {
        'accuracy': accuracy,
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1_score': report['weighted avg']['f1-score']
    }

    return metrics, predictions


def save_elm_weights(elm_classifier, output_path):
    """
    Save ELM weights to file

    Args:
        elm_classifier: Fitted ELMClassifier
        output_path: Path to save weights
    """
    weights = {
        'input_weights': elm_classifier.input_weights,
        'bias': elm_classifier.bias,
        'output_weights': elm_classifier.output_weights,
        'alpha': elm_classifier.alpha,
        'input_size': elm_classifier.input_size,
        'hidden_size': elm_classifier.hidden_size,
        'output_size': elm_classifier.output_size
    }

    torch.save(weights, output_path)
    print(f"ELM weights saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Fit ELM classifier head")
    parser.add_argument('--features_dir', type=str, required=True,
                       help='Directory containing extracted features')
    parser.add_argument('--output_dir', type=str, default='../weights/classification',
                       help='Output directory for ELM weights')
    parser.add_argument('--alpha', type=float, default=0.01,
                       help='Ridge regularization parameter')
    parser.add_argument('--fold', type=int, default=None,
                       help='Specific fold to train on (None for all)')
    parser.add_argument('--dataset', type=str, default='MAT',
                       help='Dataset name (MAT or SELF_COLLECTED)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Fitting ELM classifier for {args.dataset} dataset...")

    # Load features and labels
    features, labels = load_features_and_labels(args.features_dir, args.fold)

    if features is None or labels is None:
        print("No feature files found!")
        return

    print(f"Loaded {features.shape[0]} samples with {features.shape[1]} features")
    print(f"Class distribution: {np.bincount(labels)}")

    # Fit ELM classifier
    print("Fitting ELM classifier...")
    elm_classifier = fit_elm_classifier(
        features, labels,
        alpha=args.alpha,
        random_state=args.seed
    )

    # Evaluate on training data
    print("Evaluating on training data...")
    train_metrics, train_predictions = evaluate_elm_classifier(
        elm_classifier, features, labels
    )

    print("Training metrics:")
    for metric, value in train_metrics.items():
        print(".4f")

    # Save ELM weights
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.fold is not None:
        weight_file = output_dir / f"{args.dataset.lower()}_elm_fold{args.fold}.pth"
    else:
        weight_file = output_dir / f"{args.dataset.lower()}_elm_full.pth"

    save_elm_weights(elm_classifier, weight_file)

    # Save training metrics
    metrics_file = weight_file.with_suffix('.json')
    import json
    with open(metrics_file, 'w') as f:
        json.dump(train_metrics, f, indent=2)

    print(f"Training completed. Weights saved to {weight_file}")


if __name__ == '__main__':
    main()
