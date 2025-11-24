#!/usr/bin/env python3
"""
Train DFA-ELM classification model for cognitive workload assessment.

This script trains the complete DFA-ELM pipeline: CNN feature extraction,
attention mechanisms, and ELM classification.

Usage:
    python train_classification.py --config configs/classification_config.yaml
"""

import os
import yaml
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

from dfaelm.models import create_dfaelm_classifier
from dfaelm.data import EEGClassificationDataset, create_data_loaders
from dfaelm.train import (
    create_optimizer, create_scheduler, train_epoch, validate_epoch,
    train_model, ModelCheckpoint, TrainingLogger, compute_metrics
)


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_data_loaders(config, fold_idx=None):
    """Setup data loaders for training"""
    data_config = config['data']

    # Load dataset index
    with open(data_config['dataset_index'], 'r') as f:
        dataset_index = json.load(f)

    if fold_idx is not None:
        # Use specific fold
        fold_key = f'fold_{fold_idx}'
        if fold_key not in dataset_index:
            raise ValueError(f"Fold {fold_idx} not found in dataset index")

        fold_data = dataset_index[fold_key]
    else:
        # Use all data (for final training)
        fold_data = dataset_index

    # Create datasets
    train_dataset = EEGClassificationDataset(
        data_path=data_config['features_path'],
        labels_path=None,  # Labels will be extracted from index
        transform=None
    )

    val_dataset = EEGClassificationDataset(
        data_path=data_config['features_path'],
        labels_path=None,
        transform=None
    )

    # Filter samples based on fold
    train_samples = [s for s in fold_data['train'] if s['split'] == 'train']
    val_samples = [s for s in fold_data['val'] if s['split'] == 'val']

    # Create data loaders
    train_loader = create_data_loaders(
        train_dataset,
        batch_size=data_config['batch_size'],
        shuffle=True,
        num_workers=data_config['num_workers']
    )

    val_loader = create_data_loaders(
        val_dataset,
        batch_size=data_config['batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers']
    )

    return train_loader, val_loader, fold_data


def train_classification_model(config, fold_idx=None):
    """Train the classification model"""
    # Set random seeds
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    model = create_dfaelm_classifier(
        input_size=config['model']['input_size'],
        num_classes=config['model']['num_classes'],
        num_heads=config['model']['num_heads'],
        elm_hidden_size=config['model']['elm_hidden_size']
    )

    model = model.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Setup data loaders
    train_loader, val_loader, fold_data = setup_data_loaders(config, fold_idx)

    # Create optimizer and scheduler
    optimizer = create_optimizer(
        model,
        optimizer_name=config['training']['optimizer'],
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )

    scheduler = create_scheduler(
        optimizer,
        scheduler_name=config['training']['scheduler'],
        **config['training']['scheduler_params']
    )

    # Setup training utilities
    checkpoint_dir = Path(config['training']['checkpoint_dir'])
    if fold_idx is not None:
        checkpoint_dir = checkpoint_dir / f'fold_{fold_idx}'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = ModelCheckpoint(
        save_path=str(checkpoint_dir),
        monitor='val_f1',
        mode='max',
        save_best_only=True
    )

    logger = TrainingLogger(
        log_path=str(checkpoint_dir / 'training_logs.json')
    )

    # Training loop
    best_f1 = 0.0
    patience_counter = 0

    print("Starting training...")

    for epoch in range(config['training']['num_epochs']):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scheduler)
        logger.log(epoch + 1, {'train_loss': train_loss, 'train_acc': train_acc})

        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        # Compute additional metrics
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_preds.append(outputs)
                val_targets.append(targets)

        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)

        val_metrics = compute_metrics(val_preds, val_targets, config['model']['num_classes'])

        # Log metrics
        epoch_metrics = {
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_f1': val_metrics['f1'],
            'val_auc': val_metrics.get('auc', 0)
        }
        logger.log(epoch + 1, epoch_metrics)

        # Update scheduler
        if config['training']['scheduler'] == 'reduce_on_plateau':
            scheduler.step(val_loss)
        elif config['training']['scheduler'] != 'one_cycle':
            scheduler.step()

        # Logging
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{config['training']['num_epochs']} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_metrics['f1']:.4f}, "
              f"LR: {current_lr:.6f}")

        # Checkpoint
        logs = {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'lr': current_lr,
            'epoch': epoch + 1,
            **epoch_metrics
        }
        checkpoint(model, epoch, logs)

        # Early stopping based on F1 score
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config['training']['patience']:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print("Training completed!")

    # Fit ELM classifier on best model
    print("Fitting ELM classifier...")
    best_model_path = checkpoint_dir / 'best_model.pth'
    if best_model_path.exists():
        checkpoint_data = torch.load(best_model_path)
        model.load_state_dict(checkpoint_data['model_state_dict'])

    # Fit ELM on training data
    model.fit_elm(train_loader, device)

    # Save final model (with fitted ELM)
    final_model_path = checkpoint_dir / 'final_model_with_elm.pth'
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'best_f1': best_f1,
        'elm_fitted': True
    }, final_model_path)

    print(f"Final model saved to {final_model_path}")

    return best_f1


def cross_validate(config):
    """Perform cross-validation training"""
    n_folds = config['training']['n_folds']
    fold_results = []

    print(f"Starting {n_folds}-fold cross-validation...")

    for fold_idx in range(n_folds):
        print(f"\nTraining fold {fold_idx + 1}/{n_folds}")
        best_f1 = train_classification_model(config, fold_idx=fold_idx)
        fold_results.append(best_f1)

    # Compute CV results
    cv_mean = np.mean(fold_results)
    cv_std = np.std(fold_results)

    print(f"\nCross-validation results:")
    print(f"Mean F1: {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"Fold results: {fold_results}")

    # Save CV results
    cv_results = {
        'n_folds': n_folds,
        'fold_results': fold_results,
        'mean_f1': cv_mean,
        'std_f1': cv_std
    }

    cv_results_path = Path(config['training']['checkpoint_dir']) / 'cv_results.json'
    with open(cv_results_path, 'w') as f:
        json.dump(cv_results, f, indent=2)

    print(f"CV results saved to {cv_results_path}")


def main():
    parser = argparse.ArgumentParser(description="Train DFA-ELM classification model")
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration YAML file')
    parser.add_argument('--cross_validate', action='store_true',
                       help='Perform cross-validation instead of single training')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    if args.cross_validate:
        cross_validate(config)
    else:
        train_classification_model(config)


if __name__ == '__main__':
    main()
