#!/usr/bin/env python3
"""
Build dataset index for training with no data leakage.

This script creates train/val/test splits ensuring no subject overlap
between splits and maintains class balance.

Usage:
    python build_dataset_index.py --data_dir ../data/features --output_dir ../splits
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')


def load_subject_data(features_dir, subject_id, dataset_name):
    """
    Load all data for a subject

    Args:
        features_dir: Directory containing features
        subject_id: Subject identifier
        dataset_name: 'MAT' or 'SELF_COLLECTED'

    Returns:
        subject_data: Dictionary with subject information
    """
    subject_dir = Path(features_dir) / dataset_name.lower() / f"subject_{subject_id}"

    if not subject_dir.exists():
        return None

    subject_data = {
        'subject_id': subject_id,
        'dataset': dataset_name,
        'conditions': {},
        'total_windows': 0
    }

    # Load each condition
    for condition_file in subject_dir.glob("*_features.npy"):
        condition = condition_file.stem.replace('_features', '')

        # Load features and metadata
        features = np.load(condition_file)
        meta_file = condition_file.parent / f"{condition}_features_meta.json"

        if meta_file.exists():
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}

        subject_data['conditions'][condition] = {
            'n_windows': features.shape[0],
            'features_path': str(condition_file.relative_to(features_dir)),
            'metadata': metadata
        }

        subject_data['total_windows'] += features.shape[0]

    return subject_data if subject_data['conditions'] else None


def collect_all_subjects(features_dir):
    """
    Collect data from all subjects

    Args:
        features_dir: Directory containing features

    Returns:
        all_subjects: List of subject data dictionaries
    """
    all_subjects = []

    for dataset_name in ['MAT', 'SELF_COLLECTED']:
        dataset_dir = Path(features_dir) / dataset_name.lower()

        if not dataset_dir.exists():
            continue

        # Find all subject directories
        subject_dirs = list(dataset_dir.glob("subject_*"))

        for subject_dir in subject_dirs:
            subject_id = subject_dir.name.replace('subject_', '')

            subject_data = load_subject_data(features_dir, subject_id, dataset_name)
            if subject_data:
                all_subjects.append(subject_data)

    return all_subjects


def create_cross_validation_splits(all_subjects, n_splits=5, random_state=42):
    """
    Create cross-validation splits ensuring no subject overlap

    Args:
        all_subjects: List of subject data
        n_splits: Number of CV folds
        random_state: Random state for reproducibility

    Returns:
        cv_splits: Dictionary with fold information
    """
    # Prepare data for stratification
    subject_ids = [s['subject_id'] for s in all_subjects]
    datasets = [s['dataset'] for s in all_subjects]

    # Create stratified splits based on dataset
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    cv_splits = {}

    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(subject_ids, datasets)):
        # Split train_val into train and validation (80/20)
        train_val_subjects = [all_subjects[i] for i in train_val_idx]
        test_subjects = [all_subjects[i] for i in test_idx]

        # Further split train_val
        n_train_val = len(train_val_subjects)
        n_train = int(0.8 * n_train_val)

        # Shuffle train_val subjects
        np.random.seed(random_state + fold_idx)
        train_val_indices = np.random.permutation(n_train_val)

        train_indices = train_val_indices[:n_train]
        val_indices = train_val_indices[n_train:]

        train_subjects = [train_val_subjects[i] for i in train_indices]
        val_subjects = [train_val_subjects[i] for i in val_indices]

        cv_splits[f'fold_{fold_idx}'] = {
            'train_subjects': [s['subject_id'] for s in train_subjects],
            'val_subjects': [s['subject_id'] for s in val_subjects],
            'test_subjects': [s['subject_id'] for s in test_subjects],
            'train_datasets': list(set([s['dataset'] for s in train_subjects])),
            'val_datasets': list(set([s['dataset'] for s in val_subjects])),
            'test_datasets': list(set([s['dataset'] for s in test_subjects]))
        }

    return cv_splits


def create_sample_index(subject_data, split_name, fold_name):
    """
    Create index of all samples for a split

    Args:
        subject_data: Subject data dictionary
        split_name: 'train', 'val', or 'test'
        fold_name: Fold identifier

    Returns:
        samples: List of sample dictionaries
    """
    samples = []

    for condition, condition_data in subject_data['conditions'].items():
        features_path = condition_data['features_path']

        # Create sample entries for each window
        for window_idx in range(condition_data['n_windows']):
            sample = {
                'subject_id': subject_data['subject_id'],
                'dataset': subject_data['dataset'],
                'condition': condition,
                'window_idx': window_idx,
                'features_path': features_path,
                'split': split_name,
                'fold': fold_name,
                'label': 1 if condition.lower() == 'workload' else 0
            }
            samples.append(sample)

    return samples


def build_dataset_index(features_dir, output_dir, n_splits=5, random_state=42):
    """
    Build complete dataset index

    Args:
        features_dir: Directory containing features
        output_dir: Output directory for splits
        n_splits: Number of CV folds
        random_state: Random state
    """
    # Collect all subject data
    print("Collecting subject data...")
    all_subjects = collect_all_subjects(features_dir)

    if not all_subjects:
        print("No subject data found")
        return

    print(f"Found {len(all_subjects)} subjects")

    # Create CV splits
    print("Creating cross-validation splits...")
    cv_splits = create_cross_validation_splits(all_subjects, n_splits, random_state)

    # Create subject lookup
    subject_lookup = {s['subject_id']: s for s in all_subjects}

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save subject IDs for each dataset
    for dataset_name in ['MAT', 'SELF_COLLECTED']:
        dataset_subjects = [s['subject_id'] for s in all_subjects if s['dataset'] == dataset_name]

        subject_ids_file = output_path / dataset_name.lower() / 'subject_ids.json'
        subject_ids_file.parent.mkdir(parents=True, exist_ok=True)

        with open(subject_ids_file, 'w') as f:
            json.dump(dataset_subjects, f, indent=2)

    # Create fold directories and indices
    for fold_name, fold_info in cv_splits.items():
        print(f"Processing {fold_name}...")

        fold_dir = output_path / 'MAT' / 'kfold_5'  # Using MAT structure for now
        fold_dir.mkdir(parents=True, exist_ok=True)

        fold_samples = {'train': [], 'val': [], 'test': []}

        # Collect samples for each split
        for split_name in ['train', 'val', 'test']:
            subject_ids = fold_info[f'{split_name}_subjects']

            for subject_id in subject_ids:
                if subject_id in subject_lookup:
                    subject_data = subject_lookup[subject_id]
                    samples = create_sample_index(subject_data, split_name, fold_name)
                    fold_samples[split_name].extend(samples)

        # Save fold index
        fold_file = fold_dir / f'{fold_name}.json'
        with open(fold_file, 'w') as f:
            json.dump(fold_samples, f, indent=2)

    # Save global seeds
    seeds = {
        'global_seed': random_state,
        'cv_seed': random_state,
        'split_seed': random_state + 1,
        'noise_seed': random_state + 2
    }

    seeds_file = output_path.parent / 'seeds' / 'global_seeds.yaml'
    seeds_file.parent.mkdir(parents=True, exist_ok=True)

    import yaml
    with open(seeds_file, 'w') as f:
        yaml.dump(seeds, f, default_flow_style=False)

    # Create summary
    summary = {
        'total_subjects': len(all_subjects),
        'datasets': {},
        'cv_folds': n_splits,
        'splits': cv_splits
    }

    # Dataset summaries
    for dataset_name in ['MAT', 'SELF_COLLECTED']:
        dataset_subjects = [s for s in all_subjects if s['dataset'] == dataset_name]
        summary['datasets'][dataset_name] = {
            'n_subjects': len(dataset_subjects),
            'total_windows': sum([s['total_windows'] for s in dataset_subjects]),
            'conditions': {}
        }

        # Condition breakdown
        for subject in dataset_subjects:
            for condition, cond_data in subject['conditions'].items():
                if condition not in summary['datasets'][dataset_name]['conditions']:
                    summary['datasets'][dataset_name]['conditions'][condition] = 0
                summary['datasets'][dataset_name]['conditions'][condition] += cond_data['n_windows']

    # Save summary
    summary_file = output_path / 'dataset_index_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Dataset index created with {n_splits} folds")
    print(f"Summary saved to {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Build dataset index for training")
    parser.add_argument('--data_dir', type=str, default='../data/features',
                       help='Input directory with extracted features')
    parser.add_argument('--output_dir', type=str, default='../splits',
                       help='Output directory for dataset splits')
    parser.add_argument('--n_splits', type=int, default=5,
                       help='Number of cross-validation folds')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for reproducibility')

    args = parser.parse_args()

    build_dataset_index(args.data_dir, args.output_dir, args.n_splits, args.random_state)


if __name__ == '__main__':
    main()
