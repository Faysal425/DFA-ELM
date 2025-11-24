# Nested Cross-Validation Structure

This directory contains the nested cross-validation splits for the MAT dataset.

## Structure

- `outer_fold0/` to `outer_fold4/`: 5 outer folds for performance estimation
  - `train_subjects.json`: Subjects used for training (80% of total)
  - `test_subjects.json`: Subjects held out for testing (20% of total)
  - `inner_fold0.json` to `inner_fold3.json`: 4 inner folds for hyperparameter tuning on the training subjects

## Usage

The outer loop is used for unbiased performance estimation. For each outer fold:
1. Use the inner folds to tune hyperparameters and select the best model
2. Apply the best model to the outer test set to get performance estimates

This ensures that model selection is performed independently of the final performance evaluation.
