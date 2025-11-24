#!/usr/bin/env python3
"""
Evaluation metrics for EEG classification and reconstruction tasks.

This module provides comprehensive metrics for assessing model performance
including accuracy, F1-score, MCC, AUC, PR-AUC, sensitivity, specificity,
and Brier score.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    matthews_corrcoef, brier_score_loss
)
from scipy import stats


def calculate_classification_metrics(y_true, y_pred, y_prob=None):
    """
    Calculate comprehensive classification metrics

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities (for AUC metrics)

    Returns:
        metrics: Dictionary of all metrics
    """
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Confusion matrix for additional metrics
    cm = confusion_matrix(y_true, y_pred)

    # Matthews Correlation Coefficient
    mcc = matthews_corrcoef(y_true, y_pred)

    # Sensitivity and Specificity (for binary classification)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        # For multiclass, use macro averages
        sensitivity = recall_score(y_true, y_pred, average='macro', zero_division=0)
        specificity = precision_score(y_true, y_pred, average='macro', zero_division=0)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'mcc': mcc,
        'sensitivity': sensitivity,
        'specificity': specificity
    }

    # AUC metrics (if probabilities provided)
    if y_prob is not None:
        try:
            # For binary classification
            if len(np.unique(y_true)) == 2:
                auc = roc_auc_score(y_true, y_prob[:, 1])
                pr_auc = average_precision_score(y_true, y_prob[:, 1])
                brier = brier_score_loss(y_true, y_prob[:, 1])
            else:
                # For multiclass, use macro average
                auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
                pr_auc = average_precision_score(y_true, y_prob, average='macro')
                # Brier score for multiclass (simplified)
                brier = np.mean([brier_score_loss((y_true == i).astype(int),
                                                y_prob[:, i]) for i in np.unique(y_true)])

            metrics.update({
                'auc': auc,
                'pr_auc': pr_auc,
                'brier_score': brier
            })
        except Exception as e:
            print(f"Warning: Could not calculate AUC metrics: {e}")

    return metrics


def calculate_reconstruction_metrics(clean_signal, reconstructed_signal, fs=500):
    """
    Calculate reconstruction quality metrics

    Args:
        clean_signal: Original clean signal [n_channels, n_samples]
        reconstructed_signal: Reconstructed signal [n_channels, n_samples]
        fs: Sampling frequency

    Returns:
        metrics: Dictionary of reconstruction metrics
    """
    # Ensure same shape
    if clean_signal.shape != reconstructed_signal.shape:
        min_samples = min(clean_signal.shape[1], reconstructed_signal.shape[1])
        clean_signal = clean_signal[:, :min_samples]
        reconstructed_signal = reconstructed_signal[:, :min_samples]

    # MSE and MAE
    mse = np.mean((clean_signal - reconstructed_signal) ** 2)
    mae = np.mean(np.abs(clean_signal - reconstructed_signal))

    # SNR
    signal_power = np.mean(clean_signal ** 2)
    noise_power = np.mean((clean_signal - reconstructed_signal) ** 2)
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')

    # Correlation coefficient
    correlations = []
    for ch in range(clean_signal.shape[0]):
        corr = np.corrcoef(clean_signal[ch, :], reconstructed_signal[ch, :])[0, 1]
        if not np.isnan(corr):
            correlations.append(corr)

    mean_correlation = np.mean(correlations) if correlations else 0

    # Frequency domain metrics
    freq_metrics = calculate_frequency_domain_metrics(clean_signal, reconstructed_signal, fs)

    metrics = {
        'mse': mse,
        'mae': mae,
        'snr_db': snr,
        'correlation': mean_correlation,
        **freq_metrics
    }

    return metrics


def calculate_frequency_domain_metrics(clean_signal, reconstructed_signal, fs=500):
    """
    Calculate frequency domain reconstruction metrics

    Args:
        clean_signal: Clean signal
        reconstructed_signal: Reconstructed signal
        fs: Sampling frequency

    Returns:
        metrics: Frequency domain metrics
    """
    # Compute FFT
    clean_fft = np.fft.rfft(clean_signal, axis=1)
    recon_fft = np.fft.rfft(reconstructed_signal, axis=1)

    # Frequency bins
    freqs = np.fft.rfftfreq(clean_signal.shape[1], 1/fs)

    # Magnitude spectra
    clean_mag = np.abs(clean_fft)
    recon_mag = np.abs(recon_fft)

    # Frequency bands
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 12),
        'beta': (12, 30),
        'gamma': (30, 45)
    }

    band_similarities = {}
    for band_name, (low_freq, high_freq) in bands.items():
        band_mask = (freqs >= low_freq) & (freqs <= high_freq)

        if np.any(band_mask):
            # Cosine similarity in frequency domain
            clean_band = clean_mag[:, band_mask]
            recon_band = recon_mag[:, band_mask]

            # Average across channels
            clean_avg = np.mean(clean_band, axis=0)
            recon_avg = np.mean(recon_band, axis=0)

            # Cosine similarity
            similarity = np.dot(clean_avg, recon_avg) / (
                np.linalg.norm(clean_avg) * np.linalg.norm(recon_avg)
            ) if np.linalg.norm(clean_avg) > 0 and np.linalg.norm(recon_avg) > 0 else 0

            band_similarities[f'{band_name}_similarity'] = similarity
        else:
            band_similarities[f'{band_name}_similarity'] = 0

    # Overall frequency domain MSE
    freq_mse = np.mean(np.abs(clean_fft - recon_fft) ** 2)

    metrics = {
        'frequency_mse': freq_mse,
        **band_similarities
    }

    return metrics


def calculate_topographic_metrics(clean_signal, reconstructed_signal, channel_positions=None):
    """
    Calculate topographic similarity metrics

    Args:
        clean_signal: Clean signal
        reconstructed_signal: Reconstructed signal
        channel_positions: Channel positions for topographic analysis

    Returns:
        metrics: Topographic metrics
    """
    # For now, return channel-wise correlations
    correlations = []
    for ch in range(min(clean_signal.shape[0], reconstructed_signal.shape[0])):
        corr = np.corrcoef(clean_signal[ch, :], reconstructed_signal[ch, :])[0, 1]
        if not np.isnan(corr):
            correlations.append(corr)

    metrics = {
        'topographic_correlation_mean': np.mean(correlations) if correlations else 0,
        'topographic_correlation_std': np.std(correlations) if correlations else 0,
        'channel_correlations': correlations
    }

    return metrics


def bootstrap_confidence_intervals(y_true, y_pred, n_bootstraps=1000, alpha=0.95):
    """
    Calculate bootstrap confidence intervals for metrics

    Args:
        y_true: True labels
        y_pred: Predicted labels
        n_bootstraps: Number of bootstrap samples
        alpha: Confidence level

    Returns:
        ci: Dictionary of confidence intervals
    """
    metrics_boot = []

    n_samples = len(y_true)
    for _ in range(n_bootstraps):
        # Bootstrap sample
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]

        # Calculate metrics
        metrics = calculate_classification_metrics(y_true_boot, y_pred_boot)
        metrics_boot.append(metrics)

    # Calculate confidence intervals
    ci = {}
    for metric_name in metrics_boot[0].keys():
        values = [m[metric_name] for m in metrics_boot]
        lower = np.percentile(values, (1 - alpha) / 2 * 100)
        upper = np.percentile(values, (1 + alpha) / 2 * 100)
        ci[metric_name] = {'lower': lower, 'upper': upper}

    return ci


def print_metrics_summary(metrics, title="Metrics Summary"):
    """
    Print formatted metrics summary

    Args:
        metrics: Dictionary of metrics
    """
    print(f"\n{title}")
    print("=" * len(title))

    for key, value in metrics.items():
        if isinstance(value, float):
            if key in ['accuracy', 'precision', 'recall', 'f1_score', 'sensitivity', 'specificity', 'auc', 'pr_auc']:
                print(".4f")
            elif key in ['mcc']:
                print(".4f")
            elif key in ['brier_score']:
                print(".4f")
            elif key in ['snr_db']:
                print(".2f")
            elif 'similarity' in key:
                print(".4f")
            else:
                print(".6f")
        else:
            print(f"{key}: {value}")


if __name__ == '__main__':
    # Test metrics
    np.random.seed(42)

    # Classification test
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)
    y_prob = np.random.rand(100, 2)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)

    clf_metrics = calculate_classification_metrics(y_true, y_pred, y_prob)
    print_metrics_summary(clf_metrics, "Classification Metrics Test")

    # Reconstruction test
    clean = np.random.randn(23, 2500)
    noisy = clean + 0.1 * np.random.randn(23, 2500)

    recon_metrics = calculate_reconstruction_metrics(clean, noisy)
    print_metrics_summary(recon_metrics, "Reconstruction Metrics Test")
