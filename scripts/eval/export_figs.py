#!/usr/bin/env python3
"""
Export visualization figures for analysis and publications.

This script generates:
- Power Spectral Density (PSD) similarity plots
- Topographic correlation maps
- t-SNE embeddings for noisy vs reconstructed features
- Training history plots

Usage:
    python export_figs.py --results_dir ../results --output_dir ../figures
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import signal
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

from dfaelm.viz import create_psd_comparison, create_topomap, plot_tsne_embeddings


def load_reconstruction_results(results_dir, dataset):
    """
    Load reconstruction model results

    Args:
        results_dir: Directory containing results
        dataset: Dataset name

    Returns:
        results: Dictionary of results
    """
    results_path = Path(results_dir) / dataset.upper() / 'recon_performance.json'

    if results_path.exists():
        import json
        with open(results_path, 'r') as f:
            return json.load(f)
    else:
        print(f"Reconstruction results not found: {results_path}")
        return {}


def load_classification_results(results_dir, dataset):
    """
    Load classification results for all folds

    Args:
        results_dir: Directory containing results
        dataset: Dataset name

    Returns:
        results: Dictionary of fold results
    """
    results_path = Path(results_dir) / dataset.upper() / 'clf_reconstructed'
    fold_results = {}

    if results_path.exists():
        import json
        for fold_file in results_path.glob('metrics_fold*.json'):
            fold_idx = int(fold_file.stem.split('fold')[-1])
            with open(fold_file, 'r') as f:
                fold_results[fold_idx] = json.load(f)

    return fold_results


def plot_psd_similarity(results, output_path):
    """
    Plot PSD similarity between clean and reconstructed signals

    Args:
        results: Reconstruction results dictionary
        output_path: Output figure path
    """
    if 'psd_similarity' not in results:
        print("PSD similarity data not found")
        return

    psd_data = results['psd_similarity']

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot PSD similarity across frequency bands
    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    similarities = [psd_data.get(band, 0) for band in bands]

    bars = ax.bar(bands, similarities, color='skyblue', alpha=0.7)
    ax.set_ylabel('PSD Similarity')
    ax.set_title('Power Spectral Density Similarity: Clean vs Reconstructed')
    ax.set_ylim(0, 1)

    # Add value labels on bars
    for bar, sim in zip(bars, similarities):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                '.3f', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"PSD similarity plot saved to {output_path}")


def plot_topographic_correlation(results, output_path):
    """
    Plot topographic correlation map

    Args:
        results: Reconstruction results dictionary
        output_path: Output figure path
    """
    if 'topographic_correlation' not in results:
        print("Topographic correlation data not found")
        return

    corr_data = results['topographic_correlation']

    # Create topographic map
    fig = create_topomap(corr_data, title='Topographic Correlation: Clean vs Reconstructed')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Topographic correlation map saved to {output_path}")


def plot_tsne_features(results_dir, dataset, output_path):
    """
    Plot t-SNE embeddings of features

    Args:
        results_dir: Directory containing results
        dataset: Dataset name
        output_path: Output figure path
    """
    # This would require loading actual feature data
    # For now, create a placeholder
    print("t-SNE plotting requires feature data loading - placeholder implementation")

    # Placeholder data
    np.random.seed(42)
    clean_features = np.random.randn(100, 50)
    noisy_features = np.random.randn(100, 50)
    reconstructed_features = np.random.randn(100, 50)

    fig = plot_tsne_embeddings(
        [clean_features, noisy_features, reconstructed_features],
        ['Clean', 'Noisy', 'Reconstructed'],
        title='t-SNE: Feature Space Comparison'
    )
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"t-SNE plot saved to {output_path}")


def plot_training_history(log_file, output_path):
    """
    Plot training history from log file

    Args:
        log_file: Path to training log file
        output_path: Output figure path
    """
    if not log_file.exists():
        print(f"Training log not found: {log_file}")
        return

    # Load training history (assuming JSON format)
    import json
    with open(log_file, 'r') as f:
        history = json.load(f)

    if not history:
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

    epochs = list(range(1, len(history) + 1))

    # Loss
    if 'train_loss' in history[0]:
        train_losses = [h.get('train_loss', 0) for h in history]
        val_losses = [h.get('val_loss', 0) for h in history]
        ax1.plot(epochs, train_losses, 'b-', label='Train', linewidth=2)
        ax1.plot(epochs, val_losses, 'r-', label='Validation', linewidth=2)
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # Accuracy
    if 'train_acc' in history[0]:
        train_accs = [h.get('train_acc', 0) for h in history]
        val_accs = [h.get('val_acc', 0) for h in history]
        ax2.plot(epochs, train_accs, 'b-', label='Train', linewidth=2)
        ax2.plot(epochs, val_accs, 'r-', label='Validation', linewidth=2)
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # SNR
    if 'train_snr' in history[0]:
        train_snrs = [h.get('train_snr', 0) for h in history]
        val_snrs = [h.get('val_snr', 0) for h in history]
        ax3.plot(epochs, train_snrs, 'g-', label='Train', linewidth=2)
        ax3.plot(epochs, val_snrs, 'm-', label='Validation', linewidth=2)
        ax3.set_title('Signal-to-Noise Ratio')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('SNR (dB)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # Learning rate
    if 'lr' in history[0]:
        lrs = [h.get('lr', 0) for h in history]
        ax4.plot(epochs, lrs, 'k-', linewidth=2)
        ax4.set_title('Learning Rate Schedule')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Training history plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export visualization figures")
    parser.add_argument('--results_dir', type=str, default='../results',
                       help='Directory containing results')
    parser.add_argument('--output_dir', type=str, default='../figures',
                       help='Output directory for figures')
    parser.add_argument('--dataset', type=str, default='MAT',
                       help='Dataset name (MAT or SELF_COLLECTED)')
    parser.add_argument('--log_file', type=str, default=None,
                       help='Path to training log file for history plot')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting figures for {args.dataset} dataset...")

    # Load results
    recon_results = load_reconstruction_results(args.results_dir, args.dataset)
    clf_results = load_classification_results(args.results_dir, args.dataset)

    # Generate figures
    if recon_results:
        # PSD similarity plot
        psd_path = output_dir / f'{args.dataset.lower()}_psd_similarity.png'
        plot_psd_similarity(recon_results, psd_path)

        # Topographic correlation
        topo_path = output_dir / f'{args.dataset.lower()}_topographic_correlation.png'
        plot_topographic_correlation(recon_results, topo_path)

    # t-SNE features
    tsne_path = output_dir / f'{args.dataset.lower()}_tsne_features.png'
    plot_tsne_features(args.results_dir, args.dataset, tsne_path)

    # Training history
    if args.log_file:
        history_path = output_dir / f'{args.dataset.lower()}_training_history.png'
        plot_training_history(Path(args.log_file), history_path)

    print(f"All figures exported to {output_dir}")


if __name__ == '__main__':
    main()
