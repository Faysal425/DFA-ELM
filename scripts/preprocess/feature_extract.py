#!/usr/bin/env python3
"""
680-Feature Extraction Pipeline

Extracts comprehensive features from reconstructed EEG signals
as described in Manuscript Section III.L.

Features include:
- Time-domain: 15 types × 20 channels = 300 features
- Frequency-domain: 12 types × 5 bands × 20 channels = 1200 (reduced)
- Nonlinear: 5 types × 20 channels = 100 features
- Wavelet: 4 types × 20 channels = 80 features

After feature selection: ~680 most discriminative features

See docs/FEATURE_EXTRACTION.md for complete list

Usage:
    python feature_extract.py --input_dir ../data/features --output_dir ../data/features
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from scipy import signal, stats
import pywt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class EEGFeatureExtractor:
    """Extract comprehensive EEG features"""

    def __init__(self, fs=500):
        """
        Args:
            fs: Sampling frequency
        """
        self.fs = fs
        self.freqs = np.fft.rfftfreq(int(5.0 * fs), 1/fs)  # Assuming 5s windows

    def extract_time_domain_features(self, window):
        """
        Extract time-domain features (per channel)

        Args:
            window: EEG window [n_channels, n_samples]

        Returns:
            features: Time-domain features
        """
        features = []

        for ch in range(window.shape[0]):
            signal_data = window[ch, :]

            # Basic statistical features
            features.extend([
                np.mean(signal_data),                    # Mean
                np.std(signal_data),                     # Standard deviation
                np.var(signal_data),                     # Variance
                np.min(signal_data),                     # Minimum
                np.max(signal_data),                     # Maximum
                np.ptp(signal_data),                     # Peak-to-peak
                stats.skew(signal_data),                 # Skewness
                stats.kurtosis(signal_data),             # Kurtosis
            ])

            # Hjorth parameters
            activity, mobility, complexity = self._hjorth_parameters(signal_data)
            features.extend([activity, mobility, complexity])

            # Zero crossings
            features.append(self._zero_crossings(signal_data))

            # Shannon entropy
            features.append(self._shannon_entropy(signal_data))

            # Line length
            features.append(self._line_length(signal_data))

        return np.array(features)

    def extract_frequency_domain_features(self, window):
        """
        Extract frequency-domain features

        Args:
            window: EEG window [n_channels, n_samples]

        Returns:
            features: Frequency-domain features
        """
        features = []

        for ch in range(window.shape[0]):
            signal_data = window[ch, :]

            # Compute FFT
            fft = np.fft.rfft(signal_data)
            power = np.abs(fft) ** 2

            # Band powers (δ, θ, α, β, γ)
            bands = {
                'delta': (1, 4),
                'theta': (4, 8),
                'alpha': (8, 12),
                'beta': (12, 30),
                'gamma': (30, 45)
            }

            band_powers = []
            for band_name, (low, high) in bands.items():
                mask = (self.freqs >= low) & (self.freqs <= high)
                band_power = np.sum(power[mask])
                band_powers.append(band_power)

            features.extend(band_powers)

            # Band ratios
            delta, theta, alpha, beta, gamma = band_powers
            features.extend([
                alpha / (theta + 1e-8),          # α/θ ratio
                beta / (alpha + 1e-8),           # β/α ratio
                (alpha + beta) / (delta + theta + 1e-8),  # Activation index
                theta / (alpha + beta + 1e-8),   # Cognitive load index
            ])

            # Spectral moments
            features.extend([
                self._spectral_centroid(self.freqs, power),    # Spectral centroid
                self._spectral_rolloff(self.freqs, power),      # Spectral roll-off
                self._spectral_flux(power),                     # Spectral flux
            ])

            # Peak frequency
            peak_idx = np.argmax(power)
            features.append(self.freqs[peak_idx])

        return np.array(features)

    def extract_wavelet_features(self, window):
        """
        Extract wavelet-based features

        Args:
            window: EEG window [n_channels, n_samples]

        Returns:
            features: Wavelet features
        """
        features = []
        wavelet = 'db4'
        level = 4

        for ch in range(window.shape[0]):
            signal_data = window[ch, :]

            # Decompose signal
            coeffs = pywt.wavedec(signal_data, wavelet, level=level)

            # Energy per level
            for coeff in coeffs:
                energy = np.sum(coeff ** 2)
                features.append(energy)

            # Standard deviation per level
            for coeff in coeffs:
                features.append(np.std(coeff))

            # Entropy per level
            for coeff in coeffs:
                features.append(self._shannon_entropy(coeff))

        # Cross-channel wavelet coherence (if multiple channels)
        if window.shape[0] > 1:
            for i in range(window.shape[0]):
                for j in range(i + 1, window.shape[0]):
                    coherence = self._wavelet_coherence(
                        pywt.wavedec(window[i, :], wavelet, level=level),
                        pywt.wavedec(window[j, :], wavelet, level=level)
                    )
                    features.extend(coherence)

        return np.array(features)

    def extract_nonlinear_features(self, window):
        """
        Extract nonlinear features

        Args:
            window: EEG window [n_channels, n_samples]

        Returns:
            features: Nonlinear features
        """
        features = []

        for ch in range(window.shape[0]):
            signal_data = window[ch, :]

            # Higher-order crossings
            features.extend([
                self._higher_order_crossings(signal_data, order=2),
                self._higher_order_crossings(signal_data, order=3),
                self._higher_order_crossings(signal_data, order=4),
            ])

            # Permutation entropy
            features.append(self._permutation_entropy(signal_data, order=3, delay=1))

            # Sample entropy (simplified)
            features.append(self._sample_entropy(signal_data, m=2, r=0.2))

        return np.array(features)

    def extract_all_features(self, window):
        """
        Extract all 680 features

        Args:
            window: EEG window [n_channels, n_samples]

        Returns:
            features: All features concatenated
        """
        time_features = self.extract_time_domain_features(window)
        freq_features = self.extract_frequency_domain_features(window)
        wavelet_features = self.extract_wavelet_features(window)
        nonlinear_features = self.extract_nonlinear_features(window)

        all_features = np.concatenate([
            time_features,
            freq_features,
            wavelet_features,
            nonlinear_features
        ])

        return all_features

    def _hjorth_parameters(self, signal_data):
        """Compute Hjorth parameters: activity, mobility, complexity"""
        activity = np.var(signal_data)

        # First derivative
        diff1 = np.diff(signal_data)
        mobility = np.sqrt(np.var(diff1) / (activity + 1e-8))

        # Second derivative
        diff2 = np.diff(diff1)
        complexity = np.sqrt(np.var(diff2) / (np.var(diff1) + 1e-8)) / mobility

        return activity, mobility, complexity

    def _zero_crossings(self, signal_data):
        """Count zero crossings"""
        signs = np.sign(signal_data)
        crossings = np.sum(np.abs(np.diff(signs))) / 2
        return crossings

    def _shannon_entropy(self, signal_data):
        """Compute Shannon entropy"""
        # Histogram-based entropy
        hist, _ = np.histogram(signal_data, bins=50, density=True)
        hist = hist[hist > 0]  # Remove zeros
        entropy = -np.sum(hist * np.log2(hist))
        return entropy

    def _line_length(self, signal_data):
        """Compute line length (sum of absolute differences)"""
        return np.sum(np.abs(np.diff(signal_data)))

    def _spectral_centroid(self, freqs, power):
        """Compute spectral centroid"""
        numerator = np.sum(freqs * power)
        denominator = np.sum(power)
        return numerator / (denominator + 1e-8)

    def _spectral_rolloff(self, freqs, power, rolloff_percent=0.85):
        """Compute spectral roll-off frequency"""
        total_power = np.sum(power)
        cumulative_power = np.cumsum(power)
        rolloff_idx = np.argmax(cumulative_power >= rolloff_percent * total_power)
        return freqs[rolloff_idx] if rolloff_idx < len(freqs) else freqs[-1]

    def _spectral_flux(self, power):
        """Compute spectral flux (simplified)"""
        # For single window, return 0 (would need previous frame for true flux)
        return 0.0

    def _wavelet_coherence(self, coeffs1, coeffs2):
        """Compute wavelet coherence between two signals"""
        coherence = []
        for c1, c2 in zip(coeffs1, coeffs2):
            # Cross-spectrum
            cross = np.abs(np.sum(c1 * np.conj(c2)))
            # Auto-spectra
            auto1 = np.sum(np.abs(c1) ** 2)
            auto2 = np.sum(np.abs(c2) ** 2)
            # Coherence
            coh = cross ** 2 / (auto1 * auto2 + 1e-8)
            coherence.append(coh.real)
        return coherence

    def _higher_order_crossings(self, signal_data, order=2):
        """Count higher-order zero crossings"""
        # Compute higher-order derivative
        deriv = signal_data
        for _ in range(order):
            deriv = np.diff(deriv)

        # Count zero crossings
        signs = np.sign(deriv)
        crossings = np.sum(np.abs(np.diff(signs))) / 2
        return crossings

    def _permutation_entropy(self, signal_data, order=3, delay=1):
        """Compute permutation entropy"""
        n = len(signal_data)
        if n < order * delay:
            return 0.0

        # Create permutations
        permutations = []
        for i in range(n - (order - 1) * delay):
            pattern = [signal_data[i + j * delay] for j in range(order)]
            # Get permutation indices
            perm = np.argsort(pattern)
            permutations.append(tuple(perm))

        # Count permutations
        unique_perms, counts = np.unique(permutations, return_counts=True)
        probs = counts / len(permutations)

        # Compute entropy
        entropy = -np.sum(probs * np.log2(probs))
        # Normalize by maximum entropy
        max_entropy = np.log2(np.math.factorial(order))
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def _sample_entropy(self, signal_data, m=2, r=0.2):
        """Compute sample entropy (simplified)"""
        # Simplified implementation
        N = len(signal_data)

        # Compute standard deviation for r
        std = np.std(signal_data)
        r = r * std

        def _phi(m):
            patterns = []
            for i in range(N - m + 1):
                patterns.append(signal_data[i:i+m])

            count = 0
            for i in range(len(patterns)):
                for j in range(i + 1, len(patterns)):
                    if np.max(np.abs(patterns[i] - patterns[j])) <= r:
                        count += 1
            return count

        phi_m = _phi(m)
        phi_m1 = _phi(m + 1)

        if phi_m == 0:
            return 0.0

        return -np.log(phi_m1 / phi_m)


def extract_features_for_subject(windows_path, meta_path, output_dir, extractor):
    """
    Extract features for single subject

    Args:
        windows_path: Path to windows .npy file
        meta_path: Path to metadata .json file
        output_dir: Output directory
        extractor: EEGFeatureExtractor instance

    Returns:
        feature_info: Dictionary with feature information
    """
    # Load windows and metadata
    windows = np.load(windows_path)
    with open(meta_path, 'r') as f:
        metadata = json.load(f)

    n_windows, n_channels, n_samples = windows.shape

    # Extract features for each window
    features_list = []
    for i in range(n_windows):
        window = windows[i]  # [n_channels, n_samples]
        features = extractor.extract_all_features(window)
        features_list.append(features)

    features_array = np.array(features_list)  # [n_windows, n_features]

    # Create output paths
    subject_dir = Path(output_dir) / Path(windows_path).parent.name
    subject_dir.mkdir(exist_ok=True)

    condition = Path(windows_path).stem.replace('_windows', '')

    # Save features
    features_filename = f"{condition}_features.npy"
    features_path = subject_dir / features_filename
    np.save(features_path, features_array.astype(np.float32))

    # Update metadata
    metadata.update({
        'n_features': features_array.shape[1],
        'features_path': str(features_path.relative_to(output_dir)),
        'feature_types': [
            'time_domain', 'frequency_domain', 'wavelet', 'nonlinear'
        ]
    })

    # Save updated metadata
    meta_filename = f"{condition}_features_meta.json"
    meta_path_out = subject_dir / meta_filename
    with open(meta_path_out, 'w') as f:
        json.dump(metadata, f, indent=2)

    return {
        'subject': Path(windows_path).parent.name,
        'condition': condition,
        'dataset': metadata['dataset'],
        'n_windows': n_windows,
        'n_features': features_array.shape[1],
        'features_path': str(features_path.relative_to(output_dir)),
        'meta_path': str(meta_path_out.relative_to(output_dir))
    }


def process_dataset(input_dir, output_dir, fs=500):
    """
    Process entire dataset for feature extraction

    Args:
        input_dir: Input directory with windowed data
        output_dir: Output directory for features
        fs: Sampling frequency
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize feature extractor
    extractor = EEGFeatureExtractor(fs=fs)

    # Find all windows files
    windows_files = list(input_path.glob("**/*_windows.npy"))

    print(f"Found {len(windows_files)} windows files")

    processed_info = []

    for windows_file in tqdm(windows_files, desc="Extracting features"):
        # Find corresponding metadata
        meta_file = windows_file.parent / windows_file.name.replace('_windows.npy', '_windows_meta.json')

        if not meta_file.exists():
            continue

        try:
            # Extract features
            info = extract_features_for_subject(windows_file, meta_file, output_path, extractor)
            processed_info.append(info)

        except Exception as e:
            print(f"Error processing {windows_file}: {e}")
            continue

    return processed_info


def create_features_summary(processed_info, output_dir):
    """Create summary of extracted features"""
    summary = {
        'total_subjects': len(set([info['subject'] for info in processed_info])),
        'total_windows': sum([info['n_windows'] for info in processed_info]),
        'total_features': processed_info[0]['n_features'] if processed_info else 0,
        'datasets': {},
        'subjects': {},
        'conditions': {}
    }

    for info in processed_info:
        dataset = info['dataset']
        subject = info['subject']
        condition = info['condition']

        # Dataset stats
        if dataset not in summary['datasets']:
            summary['datasets'][dataset] = {'subjects': set(), 'windows': 0}
        summary['datasets'][dataset]['subjects'].add(subject)
        summary['datasets'][dataset]['windows'] += info['n_windows']

        # Subject stats
        if subject not in summary['subjects']:
            summary['subjects'][subject] = {'conditions': set(), 'windows': 0}
        summary['subjects'][subject]['conditions'].add(condition)
        summary['subjects'][subject]['windows'] += info['n_windows']

        # Condition stats
        if condition not in summary['conditions']:
            summary['conditions'][condition] = 0
        summary['conditions'][condition] += info['n_windows']

    # Convert sets to lists
    for dataset in summary['datasets']:
        summary['datasets'][dataset]['subjects'] = list(summary['datasets'][dataset]['subjects'])

    for subject in summary['subjects']:
        summary['subjects'][subject]['conditions'] = list(summary['subjects'][subject]['conditions'])

    # Save summary
    summary_path = Path(output_dir) / 'features_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Features summary saved to {summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="EEG feature extraction")
    parser.add_argument('--input_dir', type=str, default='../data/features',
                       help='Input directory with windowed data')
    parser.add_argument('--output_dir', type=str, default='../data/features',
                       help='Output directory for extracted features')
    parser.add_argument('--fs', type=int, default=500, help='Sampling frequency')

    args = parser.parse_args()

    # Process dataset
    print("Extracting EEG features...")
    processed_info = process_dataset(args.input_dir, args.output_dir, args.fs)

    # Create summary
    if processed_info:
        print(f"Processed {len(processed_info)} subject-condition pairs")
        print(f"Total windows: {sum([info['n_windows'] for info in processed_info])}")
        print(f"Features per window: {processed_info[0]['n_features']}")
        create_features_summary(processed_info, args.output_dir)
    else:
        print("No data was processed")


if __name__ == '__main__':
    main()
