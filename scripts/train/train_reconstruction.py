#!/usr/bin/env python3
"""
Train EEG reconstruction model with attention mechanisms.

This script trains the DFA-ELM reconstruction model to denoise EEG signals
and reconstruct missing channels.

Usage:
    python train_reconstruction.py --config configs/reconstruction_config.yaml
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

from dfaelm.models import create_eeg_reconstruction_model
from dfaelm.data import EEGReconstructionDataset, create_data_loaders
from dfaelm.train import (
    create_optimizer, create_scheduler, train_epoch, validate_epoch,
    train_model, ModelCheckpoint, TrainingLogger
)


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_data_loaders(config):
    """Setup data loaders for training"""
    data_config = config['data']

    # Create datasets
    train_dataset = EEGReconstructionDataset(
        clean_data_path=data_config['clean_data_path'],
        noisy_data_path=data_config['noisy_data_path']
    )

    # For reconstruction, we don't have separate val/test sets
    # Use portion of training data for validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config['seed'])
    )

    # Create data loaders
    train_loader = create_data_loaders(
        train_subset,
        batch_size=data_config['batch_size'],
        shuffle=True,
        num_workers=data_config['num_workers']
    )

    val_loader = create_data_loaders(
        val_subset,
        batch_size=data_config['batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers']
    )

    return train_loader, val_loader


def train_reconstruction_model(config):
    """Train the reconstruction model"""
    # Set random seeds
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model and loss function
    model, criterion = create_eeg_reconstruction_model(
        input_channels=config['model']['input_channels'],
        input_length=config['model']['input_length'],
        base_channels=config['model']['base_channels'],
        latent_dim=config['model']['latent_dim'],
        num_heads=config['model']['num_heads']
    )

    model = model.to(device)
    criterion = criterion.to(device)

    # Setup data loaders
    train_loader, val_loader = setup_data_loaders(config)

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
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = ModelCheckpoint(
        save_path=str(checkpoint_dir),
        monitor='val_loss',
        mode='min',
        save_best_only=True
    )

    logger = TrainingLogger(
        log_path=str(checkpoint_dir / 'training_logs.json')
    )

    # Training loop
    best_loss = float('inf')
    patience_counter = 0

    print("Starting training...")

    for epoch in range(config['training']['num_epochs']):
        # Train
        train_loss, _ = train_epoch(model, train_loader, criterion, optimizer, device, scheduler)
        logger.log(epoch + 1, {'train_loss': train_loss})

        # Validate
        val_loss, _ = validate_epoch(model, val_loader, criterion, device)
        logger.log(epoch + 1, {'val_loss': val_loss})

        # Update scheduler
        if config['training']['scheduler'] == 'reduce_on_plateau':
            scheduler.step(val_loss)
        elif config['training']['scheduler'] != 'one_cycle':
            scheduler.step()

        # Logging
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{config['training']['num_epochs']} - "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")

        # Checkpoint
        logs = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': current_lr,
            'epoch': epoch + 1
        }
        checkpoint(model, epoch, logs)

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config['training']['patience']:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print("Training completed!")

    # Save final model
    final_model_path = checkpoint_dir / 'final_model.pth'
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'best_loss': best_loss
    }, final_model_path)

    print(f"Final model saved to {final_model_path}")


def main():
    parser = argparse.ArgumentParser(description="Train EEG reconstruction model")
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration YAML file')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Train model
    train_reconstruction_model(config)


if __name__ == '__main__':
    main()
