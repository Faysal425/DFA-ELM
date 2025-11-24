"""
DFA-ELM: Dual-Stage EEG Framework for Cognitive Workload Assessment

This package provides implementations of:
- EEG signal reconstruction with attention mechanisms
- Cognitive workload classification using DFA-ELM architecture
"""

from .models import reconstruction, classification, blocks
from .data import datasets, transforms
from .train import trainer_utils

__all__ = [
    "reconstruction",
    "classification",
    "blocks",
    "datasets",
    "transforms",
    "trainer_utils",
    "plots",
    "topomap"
]
