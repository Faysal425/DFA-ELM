"""
Dataset classes for EEG data loading
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import json
import os


class EEGDataset(Dataset):
    """
    Base EEG Dataset class

    Args:
        data_path (str): Path to data directory
        split (str): 'train', 'val', or 'test'
        transform (callable, optional): Optional transform to be applied
    """
    def __init__(self, data_path, split='train', transform=None):
        self.data_path = data_path
        self.split = split
        self.transform = transform

        # Load dataset index
        self.index_path = os.path.join(data_path, 'dataset_index.json')
        with open(self.index_path, 'r') as f:
            self.dataset_index = json.load(f)

        # Filter by split
        self.samples = [s for s in self.dataset_index if s['split'] == split]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load data
        data_path = os.path.join(self.data_path, sample['data_path'])
        data = np.load(data_path)

        # Load labels if available
        if 'label_path' in sample:
            label_path = os.path.join(self.data_path, sample['label_path'])
            labels = np.load(label_path)
        else:
            labels = sample.get('label', 0)

        # Convert to tensors
        data = torch.from_numpy(data).float()
        labels = torch.tensor(labels).long()

        # Apply transform if provided
        if self.transform:
            data = self.transform(data)

        return data, labels


class EEGReconstructionDataset(Dataset):
    """
    Dataset for EEG reconstruction task (paired clean and noisy data)

    Args:
        clean_data_path (str): Path to clean EEG data
        noisy_data_path (str): Path to noisy EEG data
        transform (callable, optional): Optional transform
    """
    def __init__(self, clean_data_path, noisy_data_path, transform=None):
        self.clean_data_path = clean_data_path
        self.noisy_data_path = noisy_data_path
        self.transform = transform

        # Load data lists
        self.clean_files = self._get_files(clean_data_path)
        self.noisy_files = self._get_files(noisy_data_path)

        assert len(self.clean_files) == len(self.noisy_files), "Clean and noisy data must have same number of files"

    def _get_files(self, path):
        """Get list of .npy files in directory"""
        files = []
        for file in os.listdir(path):
            if file.endswith('.npy'):
                files.append(os.path.join(path, file))
        return sorted(files)

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        # Load clean and noisy data
        clean = np.load(self.clean_files[idx])
        noisy = np.load(self.noisy_files[idx])

        # Convert to tensors
        clean = torch.from_numpy(clean).float()
        noisy = torch.from_numpy(noisy).float()

        # Apply transform if provided
        if self.transform:
            clean = self.transform(clean)
            noisy = self.transform(noisy)

        return noisy, clean  # Input: noisy, Target: clean


class EEGClassificationDataset(Dataset):
    """
    Dataset for EEG classification task

    Args:
        data_path (str): Path to feature data
        labels_path (str): Path to labels
        transform (callable, optional): Optional transform
    """
    def __init__(self, data_path, labels_path, transform=None):
        self.data_path = data_path
        self.labels_path = labels_path
        self.transform = transform

        # Load data
        self.data = np.load(data_path)
        self.labels = np.load(labels_path)

        assert len(self.data) == len(self.labels), "Data and labels must have same length"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]

        # Convert to tensors
        data = torch.from_numpy(data).float()
        label = torch.tensor(label).long()

        # Apply transform if provided
        if self.transform:
            data = self.transform(data)

        return data, label


class MATDataset(EEGDataset):
    """
    Dataset class for MAT EEG dataset

    Args:
        data_path (str): Path to MAT data
        split (str): 'train', 'val', or 'test'
        transform (callable, optional): Optional transform
    """
    def __init__(self, data_path, split='train', transform=None):
        super().__init__(data_path, split, transform)

        # MAT specific preprocessing
        self.sampling_rate = 500  # Hz
        self.channels = 23
        self.subjects = list(range(1, 37))  # Subjects 1-36


class SelfCollectedDataset(EEGDataset):
    """
    Dataset class for self-collected EEG dataset

    Args:
        data_path (str): Path to self-collected data
        split (str): 'train', 'val', or 'test'
        transform (callable, optional): Optional transform
    """
    def __init__(self, data_path, split='train', transform=None):
        super().__init__(data_path, split, transform)

        # Self-collected specific preprocessing
        self.sampling_rate = 200  # Hz
        self.channels = 16
        self.subjects = list(range(1, 41))  # Subjects 1-40


def create_data_loaders(dataset, batch_size=32, shuffle=True, num_workers=4):
    """
    Create PyTorch DataLoaders

    Args:
        dataset: Dataset instance
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle
        num_workers (int): Number of workers

    Returns:
        DataLoader: PyTorch DataLoader
    """
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

    return loader
