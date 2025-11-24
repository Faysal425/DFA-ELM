#!/usr/bin/env python3
"""
Inject deterministic noise artifacts into clean EEG signals.

This script adds controlled amounts of EOG, EMG, ECG, respiratory,
and powerline artifacts to create realistic noisy EEG data for training.

Usage:
    python inject_noise.py --input_dir ../data/processed --output_dir ../data/processed
"""

import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))


class EEGNoiseInjector:
    """Inject controlled noise artifacts into EEG signals"""

    def __init__(self, noise_profiles_path, fs=500):
        """
        Args:
            noise_profiles_path: Path to YAML file with noise profiles
            fs: Sampling frequency
        """
        self.fs = fs
        self.load_noise_profiles(noise_profiles_path)

        # Initialize random generators with fixed seeds for reproducibility
        self.noise_seeds = {
            'eog': 1001,
            'emg': 1002,
            'ecg': 1003,
            'resp': 1004,
            'pl': 1005
        }

    def load_noise_profiles(self, profiles_path):
        """Load noise profile configurations"""
        with open(profiles_path, 'r') as f:
            self.profiles = yaml.safe_load(f)

    def generate_eog_noise(self, duration_samples, n_channels, seed=None):
        """Generate EOG (eye movement) artifacts"""
        if seed:
            np.random.seed(seed)

        profile = self.profiles['eog']
        amplitude = profile['amplitude_range']

        # EOG typically affects frontal channels
        eog_channels = [0, 1, 2]  # Fp1, Fp2, F3 (assuming standard 10-20)

        # Generate slow eye movements (0.1-0.5 Hz)
        t = np.arange(duration_samples) / self.fs
        freq = np.random.uniform(0.1, 0.5)
        phase = np.random.uniform(0, 2*np.pi)

        eog_signal = np.random.uniform(amplitude[0], amplitude[1]) * \
                    np.sin(2 * np.pi * freq * t + phase)

        # Add blinks (sharp transients)
        blink_times = np.random.choice(duration_samples,
                                     size=int(duration_samples * profile['blink_rate']),
                                     replace=False)

        for blink_time in blink_times:
            start = max(0, blink_time - int(0.1 * self.fs))
            end = min(duration_samples, blink_time + int(0.1 * self.fs))
            blink_amplitude = np.random.uniform(amplitude[0]*2, amplitude[1]*3)
            eog_signal[start:end] += blink_amplitude * \
                                   np.exp(-((np.arange(end-start) - 0.05*self.fs)/(0.02*self.fs))**2)

        # Create noise matrix
        noise = np.zeros((n_channels, duration_samples))
        for ch in eog_channels:
            if ch < n_channels:
                noise[ch, :] = eog_signal

        return noise

    def generate_emg_noise(self, duration_samples, n_channels, seed=None):
        """Generate EMG (muscle activity) artifacts"""
        if seed:
            np.random.seed(seed)

        profile = self.profiles['emg']
        amplitude = profile['amplitude_range']

        # EMG affects temporal and frontal channels
        emg_channels = [7, 8, 9, 10]  # T7, T8, F7, F8 (assuming standard 10-20)

        # Generate high-frequency noise (20-100 Hz)
        noise = np.zeros((n_channels, duration_samples))

        for ch in emg_channels:
            if ch < n_channels:
                # Band-limited white noise
                nyquist = self.fs / 2
                low = 20 / nyquist
                high = 100 / nyquist
                b, a = signal.butter(4, [low, high], btype='band')

                white_noise = np.random.randn(duration_samples)
                filtered_noise = signal.filtfilt(b, a, white_noise)
                noise[ch, :] = np.random.uniform(amplitude[0], amplitude[1]) * filtered_noise

        return noise

    def generate_ecg_noise(self, duration_samples, n_channels, seed=None):
        """Generate ECG artifacts"""
        if seed:
            np.random.seed(seed)

        profile = self.profiles['ecg']
        amplitude = profile['amplitude_range']

        # ECG affects multiple channels, strongest in chest leads
        # For scalp EEG, it's weaker and affects various channels
        affected_channels = list(range(min(8, n_channels)))  # First 8 channels

        # Generate ECG-like waveform (approximate)
        t = np.arange(duration_samples) / self.fs
        heart_rate = np.random.uniform(60, 100)  # BPM
        freq = heart_rate / 60

        # Simple ECG approximation
        ecg_signal = np.zeros(duration_samples)
        for i in range(int(duration_samples * freq / self.fs)):
            peak_time = int(i / freq)
            if peak_time < duration_samples:
                # QRS complex approximation
                start = max(0, peak_time - int(0.05 * self.fs))
                end = min(duration_samples, peak_time + int(0.1 * self.fs))
                if start < end:
                    qrs = np.exp(-((np.arange(end-start) - 0.02*self.fs)/(0.01*self.fs))**2)
                    ecg_signal[start:end] += qrs

        noise = np.zeros((n_channels, duration_samples))
        for ch in affected_channels:
            amp = np.random.uniform(amplitude[0], amplitude[1])
            noise[ch, :] = amp * ecg_signal

        return noise

    def generate_respiratory_noise(self, duration_samples, n_channels, seed=None):
        """Generate respiratory artifacts"""
        if seed:
            np.random.seed(seed)

        profile = self.profiles['respiratory']
        amplitude = profile['amplitude_range']

        # Respiratory artifacts affect multiple channels
        affected_channels = list(range(n_channels))

        # Generate slow respiratory rhythm (0.2-0.5 Hz)
        t = np.arange(duration_samples) / self.fs
        freq = np.random.uniform(0.2, 0.5)
        phase = np.random.uniform(0, 2*np.pi)

        resp_signal = np.sin(2 * np.pi * freq * t + phase)

        noise = np.zeros((n_channels, duration_samples))
        for ch in affected_channels:
            amp = np.random.uniform(amplitude[0], amplitude[1])
            noise[ch, :] = amp * resp_signal

        return noise

    def generate_powerline_noise(self, duration_samples, n_channels, seed=None):
        """Generate powerline (50/60 Hz) interference"""
        if seed:
            np.random.seed(seed)

        profile = self.profiles['powerline']
        amplitude = profile['amplitude_range']

        # Powerline affects all channels
        t = np.arange(duration_samples) / self.fs

        # 50 Hz fundamental + harmonics
        noise = np.zeros((n_channels, duration_samples))

        for harmonic in range(1, 4):  # Fundamental + 2 harmonics
            freq = 50 * harmonic
            phase = np.random.uniform(0, 2*np.pi)

            for ch in range(n_channels):
                amp = np.random.uniform(amplitude[0], amplitude[1]) / harmonic
                noise[ch, :] += amp * np.sin(2 * np.pi * freq * t + phase)

        return noise

    def inject_noise(self, clean_signal, snr_db=None, noise_types=None):
        """
        Inject noise into clean EEG signal

        Args:
            clean_signal: Clean EEG [n_channels, n_samples]
            snr_db: Target SNR in dB (if None, use profile defaults)
            noise_types: List of noise types to inject

        Returns:
            noisy_signal: Signal with added noise
        """
        if noise_types is None:
            noise_types = ['eog', 'emg', 'ecg', 'respiratory', 'powerline']

        n_channels, n_samples = clean_signal.shape
        noisy_signal = clean_signal.copy()

        total_noise = np.zeros_like(clean_signal)

        for noise_type in noise_types:
            if noise_type == 'eog':
                noise = self.generate_eog_noise(n_samples, n_channels, self.noise_seeds['eog'])
            elif noise_type == 'emg':
                noise = self.generate_emg_noise(n_samples, n_channels, self.noise_seeds['emg'])
            elif noise_type == 'ecg':
                noise = self.generate_ecg_noise(n_samples, n_channels, self.noise_seeds['ecg'])
            elif noise_type == 'respiratory':
                noise = self.generate_respiratory_noise(n_samples, n_channels, self.noise_seeds['resp'])
            elif noise_type == 'powerline':
                noise = self.generate_powerline_noise(n_samples, n_channels, self.noise_seeds['pl'])
            else:
                continue

            total_noise += noise

        # Scale noise to achieve target SNR if specified
        if snr_db is not None:
            # Calculate current SNR
            signal_power = np.mean(clean_signal ** 2)
            noise_power = np.mean(total_noise ** 2)

            if noise_power > 0:
                current_snr = 10 * np.log10(signal_power / noise_power)
                target_snr_linear = 10 ** (snr_db / 10)
                current_snr_linear = 10 ** (current_snr / 10)

                # Scale noise
                scale_factor = np.sqrt(signal_power / (noise_power * target_snr_linear))
                total_noise *= scale_factor

        noisy_signal += total_noise
        return noisy_signal


def process_subject_data(input_path, output_path, injector, snr_db=None):
    """
    Process single subject's data

    Args:
        input_path: Path to clean data
        output_path: Path to save noisy data
        injector: EEGNoiseInjector instance
        snr_db: Target SNR
    """
    # Load clean data
    clean_data = np.load(input_path)

    # Inject noise
    noisy_data = injector.inject_noise(clean_data, snr_db)

    # Save noisy data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, noisy_data.astype(np.float32))

    return noisy_data.shape


def main():
    parser = argparse.ArgumentParser(description="Inject noise into clean EEG data")
    parser.add_argument('--input_dir', type=str, default='../data/processed',
                       help='Input directory with clean data')
    parser.add_argument('--output_dir', type=str, default='../data/processed',
                       help='Output directory for noisy data')
    parser.add_argument('--noise_profiles', type=str, default='../scripts/noise/noise_profiles.yaml',
                       help='Path to noise profiles YAML')
    parser.add_argument('--snr_db', type=float, default=None,
                       help='Target SNR in dB (optional)')
    parser.add_argument('--datasets', nargs='+', default=['MAT', 'SELF_COLLECTED'],
                       help='Datasets to process')

    args = parser.parse_args()

    # Initialize noise injector
    injector = EEGNoiseInjector(args.noise_profiles)

    # Process each dataset
    for dataset in args.datasets:
        print(f"Processing {dataset} dataset...")

        input_dataset_dir = Path(args.input_dir) / dataset.lower()
        output_dataset_dir = Path(args.output_dir) / dataset.lower()

        if not input_dataset_dir.exists():
            print(f"Input directory {input_dataset_dir} does not exist")
            continue

        # Find all clean data files
        clean_files = list(input_dataset_dir.glob("**/*_data.npy"))

        for clean_file in clean_files:
            # Create corresponding noisy file path
            relative_path = clean_file.relative_to(input_dataset_dir)
            noisy_file = output_dataset_dir / relative_path

            try:
                shape = process_subject_data(clean_file, noisy_file, injector, args.snr_db)
                print(f"Processed {clean_file.name}: {shape}")
            except Exception as e:
                print(f"Error processing {clean_file}: {e}")


if __name__ == '__main__':
    main()
