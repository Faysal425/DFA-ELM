#!/usr/bin/env python3
"""
Safe file I/O utilities for model checkpoints, metrics, and reproducibility logs.

This module provides functions for saving and loading model checkpoints,
JSON metrics, and other data with proper error handling and validation.
"""

import os
import json
import pickle
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Union
import warnings
warnings.filterwarnings('ignore')


def save_checkpoint(model: torch.nn.Module,
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   loss: float,
                   filepath: Union[str, Path],
                   additional_data: Optional[Dict[str, Any]] = None) -> bool:
    """
    Save model checkpoint with metadata

    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        filepath: Save path
        additional_data: Additional data to save

    Returns:
        success: Whether save was successful
    """
    try:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'timestamp': str(filepath.stat().st_mtime) if filepath.exists() else None
        }

        if additional_data:
            checkpoint.update(additional_data)

        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
        return True

    except Exception as e:
        print(f"Error saving checkpoint: {e}")
        return False


def load_checkpoint(filepath: Union[str, Path],
                   model: torch.nn.Module,
                   optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, Any]:
    """
    Load model checkpoint

    Args:
        filepath: Checkpoint path
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)

    Returns:
        checkpoint: Loaded checkpoint data
    """
    try:
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")

        checkpoint = torch.load(filepath, map_location='cpu')

        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"Checkpoint loaded from {filepath}")
        return checkpoint

    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return {}


def save_metrics(metrics: Dict[str, Any], filepath: Union[str, Path]) -> bool:
    """
    Save metrics dictionary to JSON file

    Args:
        metrics: Metrics dictionary
        filepath: Save path

    Returns:
        success: Whether save was successful
    """
    try:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy types to Python types
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                                np.int16, np.int32, np.int64, np.uint8,
                                np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.bool_)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj

        serializable_metrics = convert_to_serializable(metrics)

        with open(filepath, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)

        print(f"Metrics saved to {filepath}")
        return True

    except Exception as e:
        print(f"Error saving metrics: {e}")
        return False


def load_metrics(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load metrics from JSON file

    Args:
        filepath: Metrics file path

    Returns:
        metrics: Loaded metrics dictionary
    """
    try:
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Metrics file not found: {filepath}")

        with open(filepath, 'r') as f:
            metrics = json.load(f)

        print(f"Metrics loaded from {filepath}")
        return metrics

    except Exception as e:
        print(f"Error loading metrics: {e}")
        return {}


def save_training_history(history: list, filepath: Union[str, Path]) -> bool:
    """
    Save training history to JSON file

    Args:
        history: List of epoch dictionaries
        filepath: Save path

    Returns:
        success: Whether save was successful
    """
    return save_metrics({'history': history}, filepath)


def load_training_history(filepath: Union[str, Path]) -> list:
    """
    Load training history from JSON file

    Args:
        filepath: History file path

    Returns:
        history: Training history list
    """
    metrics = load_metrics(filepath)
    return metrics.get('history', [])


def save_numpy_array(array: np.ndarray, filepath: Union[str, Path]) -> bool:
    """
    Save NumPy array with error handling

    Args:
        array: NumPy array to save
        filepath: Save path

    Returns:
        success: Whether save was successful
    """
    try:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        np.save(filepath, array)
        print(f"Array saved to {filepath}")
        return True

    except Exception as e:
        print(f"Error saving array: {e}")
        return False


def load_numpy_array(filepath: Union[str, Path]) -> Optional[np.ndarray]:
    """
    Load NumPy array with error handling

    Args:
        filepath: Array file path

    Returns:
        array: Loaded NumPy array or None if failed
    """
    try:
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Array file not found: {filepath}")

        array = np.load(filepath)
        print(f"Array loaded from {filepath}")
        return array

    except Exception as e:
        print(f"Error loading array: {e}")
        return None


def save_pickle(data: Any, filepath: Union[str, Path]) -> bool:
    """
    Save data using pickle with error handling

    Args:
        data: Data to pickle
        filepath: Save path

    Returns:
        success: Whether save was successful
    """
    try:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        print(f"Data pickled to {filepath}")
        return True

    except Exception as e:
        print(f"Error pickling data: {e}")
        return False


def load_pickle(filepath: Union[str, Path]) -> Any:
    """
    Load data from pickle file

    Args:
        filepath: Pickle file path

    Returns:
        data: Unpickled data or None if failed
    """
    try:
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Pickle file not found: {filepath}")

        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        print(f"Data unpickled from {filepath}")
        return data

    except Exception as e:
        print(f"Error unpickling data: {e}")
        return None


def get_file_checksum(filepath: Union[str, Path]) -> str:
    """
    Calculate SHA256 checksum of file

    Args:
        filepath: File path

    Returns:
        checksum: SHA256 checksum string
    """
    try:
        import hashlib

        filepath = Path(filepath)
        if not filepath.exists():
            return ""

        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)

        return sha256.hexdigest()

    except Exception as e:
        print(f"Error calculating checksum: {e}")
        return ""


def validate_file_integrity(filepath: Union[str, Path], expected_checksum: str) -> bool:
    """
    Validate file integrity using checksum

    Args:
        filepath: File path
        expected_checksum: Expected SHA256 checksum

    Returns:
        valid: Whether file integrity is valid
    """
    actual_checksum = get_file_checksum(filepath)
    return actual_checksum == expected_checksum


def create_reproducibility_log(config: Dict[str, Any],
                              git_commit: Optional[str] = None,
                              filepath: Union[str, Path] = 'reproducibility_log.json') -> bool:
    """
    Create reproducibility log with all relevant information

    Args:
        config: Configuration dictionary
        git_commit: Git commit hash
        filepath: Log file path

    Returns:
        success: Whether log creation was successful
    """
    try:
        import platform
        import torch
        import numpy as np

        log_data = {
            'timestamp': str(Path(filepath).stat().st_mtime) if Path(filepath).exists() else None,
            'platform': {
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor()
            },
            'python_version': platform.python_version(),
            'libraries': {
                'torch': torch.__version__,
                'numpy': np.__version__,
                'cuda_available': torch.cuda.is_available(),
                'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
                'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None
            },
            'git_commit': git_commit,
            'config': config
        }

        return save_metrics(log_data, filepath)

    except Exception as e:
        print(f"Error creating reproducibility log: {e}")
        return False


if __name__ == '__main__':
    # Test I/O functions
    import numpy as np

    # Test metrics
    test_metrics = {
        'accuracy': 0.95,
        'f1_score': 0.94,
        'loss': 0.123,
        'array': np.array([1, 2, 3])
    }

    save_metrics(test_metrics, 'test_metrics.json')
    loaded_metrics = load_metrics('test_metrics.json')
    print("Metrics I/O test:", loaded_metrics.keys())

    # Test array
    test_array = np.random.randn(10, 20)
    save_numpy_array(test_array, 'test_array.npy')
    loaded_array = load_numpy_array('test_array.npy')
    print("Array I/O test:", loaded_array.shape if loaded_array is not None else "Failed")

    # Clean up
    import os
    for f in ['test_metrics.json', 'test_array.npy']:
        if os.path.exists(f):
            os.remove(f)
