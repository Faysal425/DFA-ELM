# Evaluation Protocol

## Nested Cross-Validation Protocol

### Outer Loop
- Purpose: Performance estimation
- Each fold: 20% of subjects held out for testing

### Inner Loop
- Purpose: Hyperparameter tuning
- Used for model selection
- Best model applied to outer test set

### Implementation
See: `scripts/eval/nested_crossval.py`
Outer folds: `splits/MAT/nested_cv/outer_fold*/`
Inner folds: `splits/MAT/nested_cv/outer_fold*/inner_fold*/`
