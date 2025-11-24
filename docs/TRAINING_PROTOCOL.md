# DFA-ELM Training Protocol

## Stage 1: Reconstruction Model Training

- **Subjects Used:** Training set only (28 subjects from splits/MAT/holdout_split/train_subjects.json)
- **Data:** Clean EEG signals (no noise)
- **Purpose:** Learn clean signal representations
- **Validation:** 3 subjects (val_subjects.json)
- **Test:** NOT USED in this phase
- **Checkpoint:** weights/reconstruction/mat_s9_clean.pth

- **Subjects Used:** SAME 28 training subjects
- **Data:** Clean signals + synthetic noise
- **Purpose:** Learn to reconstruct from noisy to clean
- **Validation:** SAME 3 subjects (with noise)
- **Test:** NOT USED in this phase
- **Important:** Test subjects (5) completely excluded to prevent memorization
- **Checkpoint:** weights/reconstruction/mat_s9_noisy.pth

## Stage 2: Classification Model Training

- **Training:** 28 subjects (reconstructed signals)
- **Validation:** 3 subjects (reconstructed signals)
- **Test:** 5 subjects (reconstructed for first time using pretrained model)
- **No fine-tuning:** Test reconstruction uses frozen model