#!/usr/bin/env python3
"""
Generate nested cross-validation splits for the MAT dataset.

This script creates a nested CV structure with 5 outer folds and 4 inner folds
per outer fold, ensuring reproducible splits using fixed seeds.

Usage:
    python nested_crossval.py
"""

import os
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold

def generate_nested_cv_splits(dataset='MAT', seed=2001):
    """
    Generate nested cross-validation splits.

    Args:
        dataset: Dataset name (default: 'MAT')
        seed: Random seed for reproducibility
    """
    # Load subject IDs
    subject_ids_path = Path(__file__).parent.parent.parent / 'splits' / dataset / 'subject_ids.json'
    with open(subject_ids_path, 'r') as f:
        subjects = json.load(f)

    # Create nested_cv directory
    nested_cv_dir = Path(__file__).parent.parent.parent / 'splits' / dataset / 'nested_cv'
    nested_cv_dir.mkdir(parents=True, exist_ok=True)

    # Set seed
    np.random.seed(seed)

    # Outer 5-fold CV
    kf_outer = KFold(n_splits=5, shuffle=True, random_state=seed)

    for outer_idx, (train_idx, test_idx) in enumerate(kf_outer.split(subjects)):
        outer_fold_dir = nested_cv_dir / f'outer_fold{outer_idx}'
        outer_fold_dir.mkdir(exist_ok=True)

        outer_train_subjects = [subjects[i] for i in train_idx]
        outer_test_subjects = [subjects[i] for i in test_idx]

        # Save outer train and test subjects
        with open(outer_fold_dir / 'train_subjects.json', 'w') as f:
            json.dump(outer_train_subjects, f, indent=2)

        with open(outer_fold_dir / 'test_subjects.json', 'w') as f:
            json.dump(outer_test_subjects, f, indent=2)

        # Inner 4-fold CV on outer train subjects
        kf_inner = KFold(n_splits=4, shuffle=True, random_state=seed + outer_idx + 1)

        for inner_idx, (inner_train_idx, inner_val_idx) in enumerate(kf_inner.split(outer_train_subjects)):
            inner_train = [outer_train_subjects[i] for i in inner_train_idx]
            inner_val = [outer_train_subjects[i] for i in inner_val_idx]

            inner_fold_data = {
                'train_subjects': inner_train,
                'val_subjects': inner_val
            }

            with open(outer_fold_dir / f'inner_fold{inner_idx}.json', 'w') as f:
                json.dump(inner_fold_data, f, indent=2)

    print(f"Nested CV splits generated in {nested_cv_dir}")

if __name__ == '__main__':
    generate_nested_cv_splits()
