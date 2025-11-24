# Data Directory

This directory should contain the EEG datasets used for training and evaluation.

## MAT Dataset
- **Source**: Public dataset (cite appropriately)
- **Subjects**: 36 subjects × 23 channels × 500 Hz
- **Conditions**: Rest vs. Workload
- **Files**: Place `.edf` or `.EDF` files in `data/MAT/`
- **Download**: [Link to dataset source]

## Self-Collected Dataset
- **Subjects**: 40 subjects × 16 channels × 200 Hz
- **Conditions**: Rest vs. Workload (realistic scenarios)
- **Files**: Place `.edf` or `.EDF` files in `data/SELF_COLLECTED/`
- **Ethics**: Approved by institutional review board

## Preprocessing Output
After running preprocessing scripts, this directory will contain:
- `*.npy` files for NumPy arrays
- `*.json` files for metadata
- Processed windows and features

## Usage
1. Download datasets manually
2. Place files in appropriate subdirectories
3. Run preprocessing pipeline:
   ```bash
   python scripts/preprocess/edf_to_numpy.py
   python scripts/preprocess/filter_and_window.py
   python scripts/preprocess/feature_extract.py
   python scripts/preprocess/build_dataset_index.py
