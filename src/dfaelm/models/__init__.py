"""
Models package for DFA-ELM framework
"""

from .blocks import (
    ChannelAttention,
    TemporalAttention,
    MultiHeadSelfAttention,
    MultiscaleAttentionFusion
)
from .reconstruction import (
    EEGReconstructionModel,
    EEGReconstructionLoss,
    create_eeg_reconstruction_model
)
from .classification import (
    ELMClassifier,
    DFAELM_C,
    create_dfaelm_classifier
)

__all__ = [
    "ChannelAttention",
    "TemporalAttention",
    "MultiHeadSelfAttention",
    "MultiscaleAttentionFusion",
    "EEGReconstructionModel",
    "EEGReconstructionLoss",
    "create_eeg_reconstruction_model",
    "ELMClassifier",
    "DFAELM_C",
    "create_dfaelm_classifier"
]
