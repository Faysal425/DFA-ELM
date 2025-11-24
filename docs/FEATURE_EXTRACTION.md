# 680 Features for Classification (Manuscript Section III.L)

## Overview

Each 5-second EEG segment is analyzed to compute 680 features, including time-domain, frequency-domain, nonlinear, and wavelet-based descriptors. Features are extracted from 20 EEG channels and undergo feature selection to reduce dimensionality while preserving discriminative power.

## Time-Domain Features (120 features × 20 channels = 2400, reduced via selection)

### Basic Statistical Features (8 × 20 = 160 features)
- Mean
- Variance
- Standard Deviation
- Minimum
- Maximum
- Peak-to-peak amplitude
- Skewness
- Kurtosis

### Hjorth Parameters (3 × 20 = 60 features)
- Activity (variance of signal)
- Mobility (square root of variance of first derivative divided by variance of signal)
- Complexity (mobility of first derivative divided by mobility of signal)

### Signal Characteristics (4 × 20 = 80 features)
- Zero-crossings
- Shannon entropy
- Line length (sum of absolute differences)
- RMS (Root Mean Square)

## Frequency-Domain Features (12 types × 5 bands × 20 channels = 1200, reduced)

### Band Powers (5 × 20 = 100 features)
- Delta (1-4 Hz): Power
- Theta (4-8 Hz): Power
- Alpha (8-13 Hz): Power
- Beta (13-30 Hz): Power
- Gamma (30-45 Hz): Power

### Band Ratios (4 × 20 = 80 features)
- Alpha/Theta ratio
- Beta/Alpha ratio
- Activation index: (Alpha + Beta) / (Delta + Theta)
- Cognitive load index: Theta / (Alpha + Beta)

### Spectral Descriptors (3 × 20 = 60 features)
- Spectral centroid
- Spectral roll-off (85% cumulative power)
- Peak frequency

## Nonlinear Features (5 × 20 = 100 features)

### Entropy Measures (2 × 20 = 40 features)
- Permutation entropy (order 3, delay 1)
- Sample entropy (m=2, r=0.2)

### Higher-Order Statistics (3 × 20 = 60 features)
- Second-order zero crossings
- Third-order zero crossings
- Fourth-order zero crossings

## Wavelet Features (db4 basis, 4 levels)

### Per-Level Features (3 × 4 levels × 20 = 240 features)
- Wavelet energy
- Standard deviation
- Shannon entropy

### Cross-Channel Features (6 coherence values × 20 channel pairs = variable)
- Wavelet coherence between channel pairs

## Inter-Channel Features

### Phase Synchrony (Hilbert transform)
- Phase locking value (PLV)
- Phase lag index (PLI)

### Connectivity Measures
- Coherence between channels
- Cross-correlation

## Feature Selection

After extraction of ~2400+ features, dimensionality reduction is performed to select the most discriminative 680 features for classification. Selection criteria include:

- Mutual information with target labels
- Feature stability across folds
- Redundancy elimination
- Computational efficiency

## Implementation

See `scripts/preprocess/feature_extract.py` for the complete implementation.

## References

Manuscript Section III.L - Preprocessing of Reconstructed EEG
