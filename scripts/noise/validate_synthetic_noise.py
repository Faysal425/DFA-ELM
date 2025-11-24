#!/usr/bin/env python3
"""
Validate synthetic noise against real physiological artifacts.

This script compares synthetic noise parameters with real EEG artifacts
by analyzing power spectral density (PSD) and amplitude distributions.

Usage:
    python validate_synthetic_noise.py [--artifact_type ARTIFACT_TYPE]
"""

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path
import argparse

def load_noise_profiles():
    """Load noise profiles from YAML file."""
    profile_path = Path(__file__).parent / 'noise_profiles.yaml'
    with open(profile_path, 'r') as f:
        profiles = yaml.safe_load(f)
    return profiles

def generate_synthetic_noise(artifact_type, duration=10.0, fs=250.0):
    """
    Generate synthetic noise for given artifact type.

    Args:
        artifact_type: Type of artifact ('EOG', 'EMG', etc.)
        duration: Duration in seconds
        fs: Sampling frequency

    Returns:
        numpy array: Synthetic noise signal
    """
    profiles = load_noise_profiles()
    profile = profiles.get(artifact_type.upper())

    if not profile:
        raise ValueError(f"Unknown artifact type: {artifact_type}")

    np.random.seed(profile['seed'])
    n_samples = int(duration * fs)
    t = np.linspace(0, duration, n_samples)

    if artifact_type.upper() == 'EOG':
        # Gaussian blinks
        blink_rate = np.random.uniform(*profile['frequency_range'])
        n_blinks = int(duration * blink_rate)
        blink_times = np.sort(np.random.uniform(0, duration, n_blinks))

        noise = np.zeros(n_samples)
        for blink_time in blink_times:
            start_idx = int((blink_time - profile['duration']/2) * fs)
            end_idx = int((blink_time + profile['duration']/2) * fs)
            if start_idx >= 0 and end_idx < n_samples:
                gaussian = np.exp(-0.5 * ((t[start_idx:end_idx] - blink_time) / (profile['duration']/4))**2)
                noise[start_idx:end_idx] += profile['amplitude_std'] * gaussian

    elif artifact_type.upper() == 'EMG':
        # High-frequency bursts
        noise = np.zeros(n_samples)
        burst_prob = profile['burst_probability']
        burst_duration_range = profile['burst_duration']

        for i in range(n_samples):
            if np.random.random() < burst_prob:
                burst_duration = np.random.uniform(*burst_duration_range)
                burst_samples = int(burst_duration * fs)
                if i + burst_samples < n_samples:
                    freq = np.random.uniform(*profile['frequency_range'])
                    burst_signal = profile['amplitude_std'] * np.sin(2 * np.pi * freq * t[i:i+burst_samples])
                    noise[i:i+burst_samples] += burst_signal

    elif artifact_type.upper() == 'ECG':
        # R-peak waveform
        heart_rate = profile['heart_rate']
        period = 60.0 / heart_rate  # seconds per beat
        n_beats = int(duration / period) + 1

        noise = np.zeros(n_samples)
        for beat in range(n_beats):
            beat_time = beat * period
            if beat_time > duration:
                break

            # Simple QRS complex approximation
            qrs_start = beat_time
            qrs_end = beat_time + profile['qrs_duration']
            qrs_indices = (t >= qrs_start) & (t <= qrs_end)
            if np.any(qrs_indices):
                qrs_signal = profile['amplitude_std'] * np.sin(np.pi * (t[qrs_indices] - qrs_start) / profile['qrs_duration'])
                noise[qrs_indices] += qrs_signal

    elif artifact_type.upper() == 'RESPIRATORY':
        # Sinusoidal wave
        freq = np.random.uniform(*profile['frequency_range'])
        noise = profile['amplitude_std'] * np.sin(2 * np.pi * freq * t)

    elif artifact_type.upper() == 'POWERLINE':
        # 50 Hz sinusoid with harmonics
        noise = np.zeros(n_samples)
        for harmonic in profile['harmonics']:
            noise += profile['amplitude_std'] / len(profile['harmonics']) * np.sin(2 * np.pi * harmonic * t)

    else:
        raise ValueError(f"Synthetic generation not implemented for {artifact_type}")

    return noise

def load_real_noise_data(artifact_type, data_path=None):
    """
    Load real noise data for comparison.

    Args:
        artifact_type: Type of artifact
        data_path: Path to real data file (if None, generate placeholder)

    Returns:
        numpy array: Real noise signal
    """
    if data_path and os.path.exists(data_path):
        # Load real data (assuming .npy format)
        return np.load(data_path)
    else:
        # Generate placeholder real data with similar characteristics
        print(f"Warning: No real data provided for {artifact_type}, using synthetic placeholder")
        return generate_synthetic_noise(artifact_type) * 0.8  # Slightly different amplitude

def compute_psd(signal_data, fs=250.0):
    """Compute power spectral density."""
    freqs, psd = signal.welch(signal_data, fs=fs, nperseg=1024)
    return freqs, psd

def compute_amplitude_distribution(signal_data, bins=50):
    """Compute amplitude distribution."""
    hist, bin_edges = np.histogram(signal_data, bins=bins, density=True)
    return bin_edges[:-1], hist

def validate_artifact(artifact_type, real_data_path=None, plot=True):
    """
    Validate synthetic noise against real data.

    Args:
        artifact_type: Type of artifact to validate
        real_data_path: Path to real data file
        plot: Whether to generate plots
    """
    print(f"Validating {artifact_type} artifact...")

    # Generate synthetic noise
    synthetic_noise = generate_synthetic_noise(artifact_type)

    # Load real noise
    real_noise = load_real_noise_data(artifact_type, real_data_path)

    # Ensure same length
    min_len = min(len(synthetic_noise), len(real_noise))
    synthetic_noise = synthetic_noise[:min_len]
    real_noise = real_noise[:min_len]

    # Compute PSD
    fs = 250.0
    synth_freqs, synth_psd = compute_psd(synthetic_noise, fs)
    real_freqs, real_psd = compute_psd(real_noise, fs)

    # Compute amplitude distributions
    synth_bins, synth_hist = compute_amplitude_distribution(synthetic_noise)
    real_bins, real_hist = compute_amplitude_distribution(real_noise)

    # Compute metrics
    psd_correlation = np.corrcoef(synth_psd, real_psd)[0, 1]
    amplitude_kl_div = np.sum(synth_hist * np.log((synth_hist + 1e-10) / (real_hist + 1e-10)))

    print(".4f")
    print(".4f")

    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Time series
        axes[0, 0].plot(synthetic_noise[:1000], label='Synthetic')
        axes[0, 0].plot(real_noise[:1000], label='Real', alpha=0.7)
        axes[0, 0].set_title(f'{artifact_type} - Time Series (first 1000 samples)')
        axes[0, 0].legend()

        # PSD
        axes[0, 1].semilogy(synth_freqs, synth_psd, label='Synthetic')
        axes[0, 1].semilogy(real_freqs, real_psd, label='Real', alpha=0.7)
        axes[0, 1].set_title(f'{artifact_type} - Power Spectral Density')
        axes[0, 1].set_xlabel('Frequency (Hz)')
        axes[0, 1].set_ylabel('Power')
        axes[0, 1].legend()

        # Amplitude distribution
        axes[1, 0].plot(synth_bins, synth_hist, label='Synthetic')
        axes[1, 0].plot(real_bins, real_hist, label='Real', alpha=0.7)
        axes[1, 0].set_title(f'{artifact_type} - Amplitude Distribution')
        axes[1, 0].set_xlabel('Amplitude')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].legend()

        # Metrics
        axes[1, 1].text(0.1, 0.8, f'PSD Correlation: {psd_correlation:.4f}', fontsize=12)
        axes[1, 1].text(0.1, 0.6, f'KL Divergence: {amplitude_kl_div:.4f}', fontsize=12)
        axes[1, 1].set_title('Validation Metrics')
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(f'validation_{artifact_type.lower()}.png', dpi=300, bbox_inches='tight')
        plt.show()

    return {
        'psd_correlation': psd_correlation,
        'amplitude_kl_divergence': amplitude_kl_div
    }

def main():
    parser = argparse.ArgumentParser(description='Validate synthetic noise against real artifacts')
    parser.add_argument('--artifact_type', type=str, default='EOG',
                        choices=['EOG', 'EMG', 'ECG', 'RESPIRATORY', 'POWERLINE'],
                        help='Artifact type to validate')
    parser.add_argument('--real_data_path', type=str, default=None,
                        help='Path to real noise data file (.npy format)')
    parser.add_argument('--no_plot', action='store_true',
                        help='Disable plotting')

    args = parser.parse_args()

    results = validate_artifact(args.artifact_type, args.real_data_path, not args.no_plot)

    print(f"\nValidation complete for {args.artifact_type}")
    print(f"Results: {results}")

if __name__ == '__main__':
    main()
