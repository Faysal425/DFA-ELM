#!/usr/bin/env python3
"""
Convert raw EEG recordings (.edf) to NumPy format with metadata.

This script processes EDF files from MAT and self-collected datasets,
converts them to NumPy arrays, and saves metadata for each subject.

Usage:
    python edf_to_numpy.py --data_dir ../data --output_dir ../data/processed
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import mne
from tqdm import tqdm


def load_edf_file(filepath):
    """
    Load EDF file using MNE

    Args:
        filepath: Path to EDF file

    Returns:
        Raw MNE object
    """
    try:
        raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
        return raw
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def process_subject_edf(edf_path, output_dir, dataset_name, subject_id):
    """
    Process single EDF file and save as NumPy

    Args:
        edf_path: Path to EDF file
        output_dir: Output directory
        dataset_name: 'MAT' or 'SELF_COLLECTED'
        subject_id: Subject identifier
    """
    # Load EDF file
    raw = load_edf_file(edf_path)
    if raw is None:
        return None

    # Get data and info
    data = raw.get_data()  # Shape: [n_channels, n_samples]
    sfreq = raw.info['sfreq']
    ch_names = raw.ch_names
    n_samples = data.shape[1]

    # Create metadata
    metadata = {
        'dataset': dataset_name,
        'subject_id': subject_id,
        'sampling_rate': sfreq,
        'n_channels': len(ch_names),
        'n_samples': n_samples,
        'channel_names': ch_names,
        'duration_seconds': n_samples / sfreq,
        'file_path': str(edf_path)
    }

    # Determine condition from filename
    filename = edf_path.stem.lower()
    if 'rest' in filename or 'baseline' in filename:
        condition = 'rest'
    elif 'workload' in filename or 'task' in filename:
        condition = 'workload'
    else:
        condition = 'unknown'

    metadata['condition'] = condition

    # Create output paths
    subject_dir = output_dir / dataset_name.lower() / f"subject_{subject_id}"
    subject_dir.mkdir(parents=True, exist_ok=True)

    # Save data
    data_filename = f"{condition}_data.npy"
    data_path = subject_dir / data_filename
    np.save(data_path, data.astype(np.float32))

    # Save metadata
    meta_filename = f"{condition}_metadata.json"
    meta_path = subject_dir / meta_filename

    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return {
        'data_path': str(data_path.relative_to(output_dir)),
        'metadata_path': str(meta_path.relative_to(output_dir)),
        'metadata': metadata
    }


def process_dataset(data_dir, output_dir, dataset_name):
    """
    Process all EDF files in a dataset

    Args:
        data_dir: Input data directory
        output_dir: Output directory
        dataset_name: 'MAT' or 'SELF_COLLECTED'
    """
    data_path = Path(data_dir) / dataset_name
    output_path = Path(output_dir) / dataset_name.lower()

    if not data_path.exists():
        print(f"Data directory {data_path} does not exist")
        return []

    # Find all EDF files
    edf_files = list(data_path.glob("**/*.edf")) + list(data_path.glob("**/*.EDF"))

    if not edf_files:
        print(f"No EDF files found in {data_path}")
        return []

    print(f"Found {len(edf_files)} EDF files in {dataset_name}")

    processed_files = []

    for edf_file in tqdm(edf_files, desc=f"Processing {dataset_name}"):
        # Extract subject ID from filename
        filename = edf_file.stem

        # Try different patterns for subject ID extraction
        subject_id = None
        if dataset_name == 'MAT':
            # MAT dataset: assume format like "S01_rest.edf" or "subject01_workload.edf"
            if 'S' in filename and filename.split('S')[1][:2].isdigit():
                subject_id = filename.split('S')[1][:2]
            elif 'subject' in filename.lower():
                parts = filename.lower().split('subject')[1].split('_')[0]
                subject_id = parts.zfill(2)
        else:  # SELF_COLLECTED
            # Self-collected: assume format like "P01_rest.edf" or "participant01_workload.edf"
            if 'P' in filename and filename.split('P')[1][:2].isdigit():
                subject_id = filename.split('P')[1][:2]
            elif 'participant' in filename.lower():
                parts = filename.lower().split('participant')[1].split('_')[0]
                subject_id = parts.zfill(2)

        if subject_id is None:
            print(f"Could not extract subject ID from {filename}, skipping")
            continue

        # Process file
        result = process_subject_edf(edf_file, output_path, dataset_name, subject_id)
        if result:
            processed_files.append(result)

    return processed_files


def create_dataset_summary(processed_files, output_dir):
    """
    Create summary of processed dataset

    Args:
        processed_files: List of processed file info
        output_dir: Output directory
    """
    summary = {
        'total_files': len(processed_files),
        'datasets': {},
        'subjects': {},
        'conditions': {}
    }

    for file_info in processed_files:
        meta = file_info['metadata']

        dataset = meta['dataset']
        subject = meta['subject_id']
        condition = meta['condition']

        # Dataset stats
        if dataset not in summary['datasets']:
            summary['datasets'][dataset] = {'files': 0, 'subjects': set(), 'conditions': set()}
        summary['datasets'][dataset]['files'] += 1
        summary['datasets'][dataset]['subjects'].add(subject)
        summary['datasets'][dataset]['conditions'].add(condition)

        # Subject stats
        if subject not in summary['subjects']:
            summary['subjects'][subject] = {'files': 0, 'datasets': set(), 'conditions': set()}
        summary['subjects'][subject]['files'] += 1
        summary['subjects'][subject]['datasets'].add(dataset)
        summary['subjects'][subject]['conditions'].add(condition)

        # Condition stats
        if condition not in summary['conditions']:
            summary['conditions'][condition] = 0
        summary['conditions'][condition] += 1

    # Convert sets to lists for JSON serialization
    for dataset in summary['datasets']:
        summary['datasets'][dataset]['subjects'] = list(summary['datasets'][dataset]['subjects'])
        summary['datasets'][dataset]['conditions'] = list(summary['datasets'][dataset]['conditions'])

    for subject in summary['subjects']:
        summary['subjects'][subject]['datasets'] = list(summary['subjects'][subject]['datasets'])
        summary['subjects'][subject]['conditions'] = list(summary['subjects'][subject]['conditions'])

    # Save summary
    summary_path = Path(output_dir) / 'dataset_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Dataset summary saved to {summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Convert EDF files to NumPy format")
    parser.add_argument('--data_dir', type=str, default='../data',
                       help='Input data directory containing MAT and SELF_COLLECTED folders')
    parser.add_argument('--output_dir', type=str, default='../data/processed',
                       help='Output directory for processed data')
    parser.add_argument('--datasets', nargs='+', default=['MAT', 'SELF_COLLECTED'],
                       help='Datasets to process')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_processed = []

    # Process each dataset
    for dataset in args.datasets:
        print(f"\nProcessing {dataset} dataset...")
        processed = process_dataset(args.data_dir, args.output_dir, dataset)
        all_processed.extend(processed)

    # Create summary
    if all_processed:
        print(f"\nProcessed {len(all_processed)} files total")
        create_dataset_summary(all_processed, args.output_dir)
    else:
        print("No files were processed")


if __name__ == '__main__':
    main()
