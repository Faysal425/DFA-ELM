# DFA-ELM: Denoising Feature-Aware Extreme Learning Machine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.4](https://img.shields.io/badge/PyTorch-2.4-ee4c2c.svg)](https://pytorch.org/)

**Official Implementation** of the paper:  
*"An Adaptive Denoising-Driven EEG Signal Fusion Framework for Cognitive Load Monitoring in Real-World Human-Machine Environments"*

> 📄 **Paper:** [IEEE Transactions on Neural Networks and Learning Systems](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=5962385)  [Under Review]

> 📊 **Self Collected Dataset:** [https://doi.org/10.6084/m9.figshare.29528219.v3](https://doi.org/10.6084/m9.figshare.29528219.v3)  
> 🎯 **Code:** [https://github.com/Faysal425/Denoising_EEG_Signal/](https://github.com/Faysal425/Denoising_EEG_Signal/)

---

## 🎯 Overview

DFA-ELM is a **dual-stage neural framework** designed for robust EEG-based cognitive workload monitoring in real-world environments. The system addresses the critical challenge of physiological artifacts (EMG, EOG, ECG, respiratory, power-line interference) that degrade EEG signal quality in practical settings.

### Key Features

✅ **Two-Stage Architecture**
- **Stage 1:** Signal reconstruction via attention-enhanced autoencoder
- **Stage 2:** Cognitive workload classification via MR-ELM

✅ **Robust Attention Mechanisms**
- Channel Attention (CA) for spatial feature refinement
- Temporal Attention (TA) for sequential patterns
- Multi-Head Self-Attention (MHSA) for global dependencies
- Multi-Scale Feature Fusion (MSFF) for hierarchical integration
---

## 📁 Repository Structure

```
DFA-ELM/
├── 📂 configs/                         # Configuration files
│   ├── reconstruction_config.yaml      # Stage 1 hyperparameters
│   ├── classification_config.yaml      # Stage 2 hyperparameters
│   └── evaluation_config.yaml          # Evaluation settings
│
├── 📂 data/                            # Dataset directory
│   ├── MAT/                            # MAT dataset (download separately)
│   └── SELF_COLLECTED/                 # Self-collected dataset
│
├── 📂 docs/                            # Documentation
│   ├── TRAINING_PROTOCOL.md            # 3-phase training guide
│   ├── EVALUATION_PROTOCOL.md          # Standardized evaluation
│   └── FEATURE_EXTRACTION.md           # 680 features documentation
│
├── 📂 scripts/                         # Executable scripts
│   ├── preprocess/                     # Data preprocessing
│   ├── noise/                          # Noise injection
│   ├── train/                          # Training scripts
│   ├── eval/                           # Evaluation scripts
│   └── utils/                          # Helper utilities
│
├── 📂 src/dfaelm/                      # Core source code
│   ├── models/                         # Neural network architectures
│   ├── data/                           # Data loaders
│   └── train/                          # Training utilities
│
├── 📂 splits/                          # Pre-generated data splits
│   ├── MAT/
│   │   ├── holdout_split/              # 80/10/10 split (28/3/5 subjects)
│   │   ├── kfold_5/                    # 5-fold cross-validation
│   │   └── nested_cv/                  # Nested CV structure
│   └── SELF/
│       ├── holdout_split/
│       └── kfold_5/
│
├── 📂 seeds/                           # Reproducibility seeds
│   ├── global_seeds.yaml               # Master RNG control
│   ├── noise_injection_seeds.json      # Noise determinism
│   └── split_seeds.json                # Data split seeds
│
├── 📂 weights/                         # Pre-trained models
│   ├── reconstruction/
│   └── classification/
│
└── README.md                           # This file
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.8 or higher
- CUDA 11.3+ (for GPU training)
- 16GB RAM (minimum), 32GB recommended

### Option 1: Using pip (Recommended)

```bash
# Clone the repository
git clone https://github.com/Faysal425/Denoising_EEG_Signal.git
cd DFA-ELM

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .
```

### Option 2: Using Conda

```bash
# Clone the repository
git clone https://github.com/Faysal425/Denoising_EEG_Signal.git
cd DFA-ELM

# Create conda environment
conda env create -f environment.yml
conda activate dfaelm

# Install the package
pip install -e .
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch: 2.4
CUDA Available: True
```

---

## 📊 Dataset Preparation

### 1. Download Datasets

**MAT Dataset:**
```bash
# Download the MAT Dataset from PhysioNet
wget -r -N -c -np https://physionet.org/files/eegmat/1.0.0/
```

**Self-Collected Dataset:**
```bash
# Download from Figshare
wget https://doi.org/10.6084/m9.figshare.29528219.v3 -O self_collected.zip
```

### 2. Preprocess Raw EEG Data

```bash
# Step 1: Convert EDF to NumPy (MAT dataset)
python scripts/preprocess/edf_to_numpy.py \
    --input_dir data/MAT/raw/ \
    --output_dir data/MAT/processed/ \
    --dataset MAT

# Step 2: Apply filtering and windowing (128 Hz resampling, 5-second windows)
python scripts/preprocess/filter_and_window.py \
    --input_dir data/MAT/processed/ \
    --output_dir data/MAT/filtered/ \
    --config configs/reconstruction_config.yaml \
    --dataset MAT

# Step 3: Extract 680 features for classification
python scripts/preprocess/feature_extract.py \
    --input_dir data/MAT/filtered/ \
    --output_dir data/MAT/features/ \
    --config configs/classification_config.yaml

# Step 4: Build dataset index (prevents data leakage)
python scripts/preprocess/build_dataset_index.py \
    --data_dir data/MAT/features/ \
    --splits_dir splits/MAT/ \
    --output_file data/MAT/dataset_index.json
```
### 3. Synthetic Noise

```bash
# Generate synthetic noise for all artifact types
python scripts/noise/inject_noise.py \
    --input_dir data/MAT/filtered/ \
    --output_dir data/MAT/noisy/ \
    --noise_config scripts/noise/noise_profiles.yaml \
    --noise_types all \
    --seed_file seeds/noise_injection_seeds.json

# Validate synthetic noise 
python scripts/noise/validate_synthetic_noise.py \
    --synthetic_dir data/MAT/noisy/ \
    --real_dir data/SELF_COLLECTED/raw/ \
    --output_dir results/noise_validation/
```

---

## 📦 Pre-trained Model Weights

### **MAT Dataset**
| Model | Download Link |
|-------|---------------|
| Reconstruction Model | [Google Drive](https://drive.google.com/file/d/1WV58LaWgiK9SDESm4XzEb8bb19mAcqyg/view?usp=sharing) |
| Classification Model — Phase 1 | [Google Drive](https://drive.google.com/file/d/1YJPN9X8cVjNHgmmogh__dR1yvcIIJJUA/view?usp=sharing) |
| Classification Model — Phase 2 | [Google Drive](https://drive.google.com/file/d/17pAmiXiyvFjuKyi436wZEO8LsMfC5Xsx/view?usp=sharing) |

---

### **SELF Dataset**
| Model | Download Link |
|-------|---------------|
| Reconstruction Model | [Google Drive](https://drive.google.com/file/d/YOUR_RECONSTRUCTION_SELF_ID) |
| Classification Model — Phase 1 | [Google Drive](https://drive.google.com/file/d/YOUR_CLASSIFICATION_SELF_PHASE1_ID) |
| Classification Model — Phase 2 | [Google Drive](https://drive.google.com/file/d/YOUR_CLASSIFICATION_SELF_PHASE2_ID) |


### Model Architecture Details

| Component | Parameters | FLOPs | Description |
|-----------|-----------|-------|-------------|
| **Reconstruction Model** | 2.137M | 9.5M | 3-level encoder-decoder with CA, TA, MHSA, MSFF |
| **Classification Model** | 0.828M | 2.39M | 2 encoders + MSFF + MR-ELM |
| **Total** | 2.965M | 11.89M | End-to-end pipeline |

---

##  Training

### Stage 1: Reconstruction Model (Denoising Autoencoder)

#### Phase 1a: Train on Clean Signals

```bash
# Train autoencoder on clean EEG (28 training subjects)
python scripts/train/train_reconstruction.py \
    --config configs/reconstruction_config.yaml \
    --data_dir data/MAT/filtered/ \
    --split_file splits/MAT/holdout_split/train_subjects.json \
    --phase clean \
    --output_dir checkpoints/reconstruction/phase1a/ \
    --epochs 100 \
    --batch_size 64 \
    --device cuda

# Monitor training with TensorBoard
tensorboard --logdir logs/reconstruction/phase1a/
```

#### Phase 1b: Retrain on Noisy Signals

```bash
# Retrain on noisy signals (test excluded)
python scripts/train/train_reconstruction.py \
    --config configs/reconstruction_config.yaml \
    --data_dir data/MAT/noisy/ \
    --clean_dir data/MAT/filtered/ \
    --split_file splits/MAT/holdout_split/train_subjects.json \
    --phase noisy \
    --pretrained checkpoints/reconstruction/phase1a/best_model.pth \
    --output_dir checkpoints/reconstruction/phase1b/ \
    --epochs 50 \
    --batch_size 64 \
    --device cuda
```

**Automated Reconstruction Training (Both Phases):**
```bash
bash scripts/run_reconstruction_training.sh \
    --dataset MAT \
    --config configs/reconstruction_config.yaml \
    --device cuda
```

---

### Stage 2: Classification Model

#### Reconstruct Signals for All Subjects

```bash
# Apply trained reconstruction model to all subjects (including test)
python scripts/inference/inference.py \
    --checkpoint checkpoints/reconstruction/phase1b/best_model.pth \
    --input_dir data/MAT/noisy/ \
    --output_dir data/MAT/reconstructed/ \
    --split_file splits/MAT/holdout_split/test_subjects.json \
    --batch_size 64 \
    --device cuda
```

#### Train Classification Model

```bash
# Train DFA-ELM classifier on reconstructed signals
python scripts/train/train_classification.py \
    --config configs/classification_config.yaml \
    --data_dir data/MAT/reconstructed/ \
    --features_dir data/MAT/features/ \
    --split_file splits/MAT/holdout_split/train_subjects.json \
    --output_dir checkpoints/classification/ \
    --epochs 50 \
    --batch_size 64 \
    --device cuda

# Fit ELM head (analytical solution)
python scripts/train/fit_elm_head.py \
    --checkpoint checkpoints/classification/best_encoder.pth \
    --data_dir data/MAT/reconstructed/ \
    --split_file splits/MAT/holdout_split/train_subjects.json \
    --output checkpoints/classification/final_model.pth \
    --alpha 0.001
```

**Automated Classification Training:**
```bash
bash scripts/run_classification_training.sh \
    --dataset MAT \
    --config configs/classification_config.yaml \
    --device cuda
```

---

## 📈 Evaluation

```bash
# Evaluate on test subjects (never seen during training/reconstruction)
python scripts/evaluate/evaluate_model.py \
    --checkpoint checkpoints/classification/final_model.pth \
    --data_dir data/MAT/reconstructed/ \
    --split_file splits/MAT/holdout_split/test_subjects.json \
    --output_dir results/MAT/holdout_test/ \
    --phase 1
```
```bash
python scripts/eval/crossval.py \
    --config configs/classification_config.yaml \
    --dataset MAT \
    --splits_dir splits/MAT/kfold_5/ \
    --data_dir data/MAT/reconstructed/ \
    --output_dir results/MAT/cv_5fold/ \
    --phase 1 \
    --device cuda
```

```bash
# Run nested cross-validation (outer: performance, inner: hyperparameter tuning)
python scripts/eval/nested_crossval.py \
    --config configs/classification_config.yaml \
    --dataset MAT \
    --splits_dir splits/MAT/nested_cv/ \
    --data_dir data/MAT/reconstructed/ \
    --output_dir results/MAT/nested_cv/ \
    --device cuda
```



## 💻 Hardware Requirements
### Recommended (Training)
- **CPU:** Intel Core i7-1165G7 (4 cores / 8 threads @ 2.8 GHz)
- **GPU:** Intel Iris Xe Graphics (128 MB VRAM) 
- **RAM:** 16 GB
- **Storage:** 477 GB SSD
- **OS:** Windows 10 Pro (64-bit)
- **Framework:** PyTorch 2.4
- **GPU:** NVIDIA RTX 3080 or higher (8–10+ GB VRAM)
- **CUDA:** CUDA 11.3 / 11.8 (PyTorch-compatible)
- **Drivers:** Latest NVIDIA GPU drivers for Windows or Linux
---




## 📖 Citation

If you use this code or dataset in your research, please cite (TBA):

```bibtex
@article{dfaelm2024,
  title={An Adaptive Denoising-Driven EEG Signal Fusion Framework for Cognitive Load Monitoring in Real-World Human-Machine Environments},
  author={[Author Names]},
  journal={[Journal Name]},
  year={2024},
  doi={[DOI]},
  url={}
}
```
---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---


## 🔗 Related Resources

- **Paper:** [Under Review]
- **Supplementary Materials:** [Link to supplementary PDF]

---

**Last Updated:** 2024-11-18  
