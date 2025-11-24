#!/usr/bin/env python3
"""
Preprocessing Pipeline (Manuscript Section III.K)

1. Resample: 500Hz/200Hz → 128 Hz
2. Notch filter: 50 Hz (powerline)
3. Bandpass: 1-30 Hz
4. Z-score normalization
5. Windowing: 5 seconds (640 samples)

Preprocess EEG data: filtering, windowing, and feature extraction.

This script applies filtering (notch + bandpass), creates overlapping windows,
and prepares data for training.

Usage:
    python filter_and_window.py --input_dir ../data/processed --output_dir ../data/features
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from scipy import signal
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class EEGPreprocessor:
    """EEG preprocessing pipeline"""

    def __init__(self, fs=500, window_size=5.0, overlap=0.5):
        """
        Args:
            fs: Sampling frequency
            window_size: Window size in seconds
            overlap: Overlap fraction (0-1)
        """
        self.fs = fs
        self.window_size = window_size
        self.overlap = overlap
        self.window_samples = int(window_size * fs)
        self.step = int(self.window_samples * (1 - overlap))

        # Design filters
        self._design_filters()

    def _design_filters(self):
        """Design notch and bandpass filters"""
        # Notch filter for powerline noise (50 Hz)
        nyquist = self.fs / 2
        notch_freq = 50.0 / nyquist
        self.notch_b, self.notch_a = signal.iirnotch(notch_freq, Q=30)

        # Bandpass filter for EEG (1-30 Hz)
        low = 1.0 / nyquist
        high = 30.0 / nyquist
        self.bp_b, self.bp_a = signal.butter(4, [low, high], btype='band')

    def apply_filters(self, data, filter_type='raw'):
        """
        Apply appropriate filtering

        Args:
            data: EEG data [n_channels, n_samples]
            filter_type: 'raw' or 'reconstructed'
        """
        if filter_type == 'raw':
            # Raw EEG: notch + bandpass
            filtered = signal.filtfilt(self.notch_b, self.notch_a, data, axis=1)
            filtered = signal.filtfilt(self.bp_b, self.bp_a, filtered, axis=1)
        else:
            # Reconstructed EEG: only bandpass (3-40 Hz)
            nyquist = self.fs / 2
            low = 3.0 / nyquist
            high = 40.0 / nyquist
            rec_b, rec_a = signal.butter(1, [low, high], btype='band')
            filtered = signal.filtfilt(rec_b, rec_a, data, axis=1)

        return filtered

    def apply_average_reference(self, data):
        """Apply average reference montage"""
        avg_ref = np.mean(data, axis=0, keepdims=True)
        return data - avg_ref

    def z_score_normalize(self, data):
        """Z-score normalization per channel"""
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)
        return (data - mean) / (std + 1e-8)

    def create_windows(self, data):
        """
        Create overlapping windows from continuous signal

        Args:
            data: EEG data [n_channels, n_samples]

        Returns:
            windows: [n_windows, n_channels, window_samples]
        """
        n_channels, n_samples = data.shape
        windows = []

        for start in range(0, n_samples - self.window_samples + 1, self.step):
            end = start + self.window_samples
            window = data[:, start:end]
            windows.append(window)

        if windows:
            return np.stack(windows, axis=0)
        else:
            return np.empty((0, n_channels, self.window_samples))

    def process_subject_data(self, data_path, metadata_path, filter_type='raw'):
        """
        Process single subject's data

        Args:
            data_path: Path to .npy file
            metadata_path: Path to metadata .json file
            filter_type: 'raw' or 'reconstructed'

        Returns:
            processed_data, metadata
        """
        # Load data and metadata
        data = np.load(data_path)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Apply preprocessing
        filtered = self.apply_filters(data, filter_type)
        avg_ref = self.apply_average_reference(filtered)
        normalized = self.z_score_normalize(avg_ref)
        windows = self.create_windows(normalized)

        # Update metadata
        metadata.update({
            'filter_type': filter_type,
            'window_size_seconds': self.window_size,
            'window_overlap': self.overlap,
            'window_samples': self.window_samples,
            'step_samples': self.step,
            'n_windows': windows.shape[0] if windows.size > 0 else 0,
            'processed_shape': list(windows.shape) if windows.size > 0 else []
        })

        return windows, metadata


def process_dataset(input_dir, output_dir, filter_type='raw', fs=500, window_size=5.0, overlap=0.5):
    """
    Process entire dataset

    Args:
        input_dir: Input directory with processed data
        output_dir: Output directory for features
        filter_type: 'raw' or 'reconstructed'
        fs: Sampling frequency
        window_size: Window size in seconds
        overlap: Overlap fraction
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir) / f"{filter_type}_features"
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize preprocessor
    preprocessor = EEGPreprocessor(fs=fs, window_size=window_size, overlap=overlap)

    # Find all subject directories
    subject_dirs = []
    for dataset_dir in ['mat', 'self_collected']:
        dataset_path = input_path / dataset_dir
        if dataset_path.exists():
            subject_dirs.extend(list(dataset_path.glob("subject_*")))

    print(f"Found {len(subject_dirs)} subject directories")

    processed_info = []

    for subject_dir in tqdm(subject_dirs, desc=f"Processing {filter_type} data"):
        # Find data and metadata files
        data_files = list(subject_dir.glob("*_data.npy"))
        meta_files = list(subject_dir.glob("*_metadata.json"))

        if not data_files or not meta_files:
            continue

        # Process each condition
        for data_file in data_files:
            condition = data_file.stem.replace('_data', '')

            # Find corresponding metadata
            meta_file = subject_dir / f"{condition}_metadata.json"
            if not meta_file.exists():
                continue

            try:
                # Process data
                windows, metadata = preprocessor.process_subject_data(
                    data_file, meta_file, filter_type
                )

                if windows.size == 0:
                    print(f"No windows created for {subject_dir.name} {condition}")
                    continue

                # Create output paths
                subject_output_dir = output_path / subject_dir.name
                subject_output_dir.mkdir(exist_ok=True)

                # Save windows
                windows_filename = f"{condition}_windows.npy"
                windows_path = subject_output_dir / windows_filename
                np.save(windows_path, windows.astype(np.float32))

                # Save updated metadata
                meta_filename = f"{condition}_windows_meta.json"
                meta_path = subject_output_dir / meta_filename

                with open(meta_path, 'w') as f:
                    json.dump(metadata, f, indent=2)

                # Store info for summary
                processed_info.append({
                    'subject': subject_dir.name,
                    'condition': condition,
                    'dataset': metadata['dataset'],
                    'windows_path': str(windows_path.relative_to(output_path)),
                    'meta_path': str(meta_path.relative_to(output_path)),
                    'n_windows': windows.shape[0],
                    'metadata': metadata
                })

            except Exception as e:
                print(f"Error processing {subject_dir.name} {condition}: {e}")
                continue

    return processed_info


def create_windows_summary(processed_info, output_dir, filter_type):
    """Create summary of windowed data"""
    summary = {
        'filter_type': filter_type,
        'total_subjects': len(set([info['subject'] for info in processed_info])),
        'total_windows': sum([info['n_windows'] for info in processed_info]),
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
    summary_path = Path(output_dir) / f"{filter_type}_windows_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Windows summary saved to {summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="EEG filtering and windowing")
    parser.add_argument('--input_dir', type=str, default='../data/processed',
                       help='Input directory with processed EDF data')
    parser.add_argument('--output_dir', type=str, default='../data/features',
                       help='Output directory for windowed features')
    parser.add_argument('--filter_types', nargs='+', default=['raw', 'reconstructed'],
                       help='Filter types to process')
    parser.add_argument('--fs', type=int, default=500, help='Sampling frequency')
    parser.add_argument('--window_size', type=float, default=5.0, help='Window size in seconds')
    parser.add_argument('--overlap', type=float, default=0.5, help='Overlap fraction (0-1)')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_processed = []

    # Process each filter type
    for filter_type in args.filter_types:
        print(f"\nProcessing {filter_type} data...")
        processed = process_dataset(
            args.input_dir, args.output_dir, filter_type,
            args.fs, args.window_size, args.overlap
        )
        all_processed.extend(processed)

        # Create summary for this filter type
        if processed:
            create_windows_summary(processed, args.output_dir, filter_type)

    # Overall summary
    if all_processed:
        print(f"\nTotal processed: {len(all_processed)} subject-condition pairs")
        print(f"Total windows: {sum([info['n_windows'] for info in all_processed])}")
    else:
        print("No data was processed")


if __name__ == '__main__':
    main()
