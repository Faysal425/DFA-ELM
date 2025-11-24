#!/usr/bin/env python3
"""
Global reproducibility seeding utilities.

This module provides functions to set all random number generators
to ensure deterministic behavior across runs.
"""

import os
import random
import numpy as np
import torch
import yaml
from pathlib import Path


def seed_everything(seed: int = 1337, deterministic: bool = True):
    """
    Set all random number generators for reproducibility

    Args:
        seed: Random seed value
        deterministic: Whether to enable deterministic behavior in PyTorch
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # PyTorch deterministic behavior
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    print(f"Random seed set to {seed}")


def load_global_seeds(config_path: str = '../seeds/global_seeds.yaml'):
    """
    Load global seed configuration from YAML file

    Args:
        config_path: Path to seed configuration file

    Returns:
        seeds: Dictionary of seed values
    """
    config_file = Path(config_path)

    if config_file.exists():
        with open(config_file, 'r') as f:
            seeds = yaml.safe_load(f)
    else:
        # Default seeds
        seeds = {
            'python': 1337,
            'numpy': 1337,
            'torch': 1337,
            'torch_deterministic': True,
            'cudnn_benchmark': False
        }

    return seeds


def apply_global_seeds(config_path: str = '../seeds/global_seeds.yaml'):
    """
    Apply global seed configuration

    Args:
        config_path: Path to seed configuration file
    """
    seeds = load_global_seeds(config_path)

    seed_everything(
        seed=seeds.get('python', 1337),
        deterministic=seeds.get('torch_deterministic', True)
    )

    # Additional NumPy seed if different
    if 'numpy' in seeds:
        np.random.seed(seeds['numpy'])

    # Additional PyTorch seed if different
    if 'torch' in seeds:
        torch.manual_seed(seeds['torch'])
        torch.cuda.manual_seed(seeds['torch'])
        torch.cuda.manual_seed_all(seeds['torch'])


def get_noise_seed(artifact_type: str, config_path: str = '../seeds/noise_injection_seeds.json'):
    """
    Get fixed seed for specific noise artifact type

    Args:
        artifact_type: Type of artifact ('eog', 'emg', 'ecg', 'resp', 'pl')
        config_path: Path to noise seed configuration

    Returns:
        seed: Fixed seed for the artifact type
    """
    config_file = Path(config_path)

    if config_file.exists():
        import json
        with open(config_file, 'r') as f:
            noise_seeds = json.load(f)
        return noise_seeds.get(artifact_type, 1337)
    else:
        # Default seeds
        default_seeds = {
            'eog': 5001,
            'emg': 5002,
            'ecg': 5003,
            'resp': 5004,
            'pl': 5050
        }
        return default_seeds.get(artifact_type, 1337)


def get_split_seed(dataset: str, config_path: str = '../seeds/split_seeds.json'):
    """
    Get fixed seed for dataset splitting

    Args:
        dataset: Dataset name ('MAT' or 'SELF_COLLECTED')
        config_path: Path to split seed configuration

    Returns:
        seed: Fixed seed for the dataset
    """
    config_file = Path(config_path)

    if config_file.exists():
        import json
        with open(config_file, 'r') as f:
            split_seeds = json.load(f)
        return split_seeds.get(dataset.lower(), 1337)
    else:
        # Default seeds
        default_seeds = {
            'mat': 2001,
            'self_collected': 2002
        }
        return default_seeds.get(dataset.lower(), 1337)


if __name__ == '__main__':
    # Test seeding
    print("Testing global seeding...")
    apply_global_seeds()

    # Test reproducibility
    a = np.random.randn(5)
    b = torch.randn(5)

    print("NumPy random array:", a)
    print("PyTorch random tensor:", b)

    # Reset and check if same
    apply_global_seeds()
    a2 = np.random.randn(5)
    b2 = torch.randn(5)

    print("Reproducibility check:")
    print("NumPy arrays match:", np.allclose(a, a2))
    print("PyTorch tensors match:", torch.allclose(b, b2))
