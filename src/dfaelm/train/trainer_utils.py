"""
Training utilities for DFA-ELM framework
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
import os
import json
from datetime import datetime


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, loss, model):
        if self.best_loss is None:
            self.best_loss = loss
            self.save_checkpoint(model)
        elif loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights:
                self.restore_checkpoint(model)
            return True
        return False

    def save_checkpoint(self, model):
        """Save model weights"""
        self.best_weights = model.state_dict().copy()

    def restore_checkpoint(self, model):
        """Restore best model weights"""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


class ModelCheckpoint:
    """Save model checkpoints during training"""
    def __init__(self, save_path, monitor='val_loss', mode='min', save_best_only=True):
        self.save_path = save_path
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_value = float('inf') if mode == 'min' else float('-inf')

        os.makedirs(save_path, exist_ok=True)

    def __call__(self, model, epoch, logs):
        current_value = logs.get(self.monitor)

        if current_value is None:
            return

        is_better = (current_value < self.best_value) if self.mode == 'min' else (current_value > self.best_value)

        if is_better:
            self.best_value = current_value
            checkpoint_path = os.path.join(self.save_path, f'best_model_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': logs.get('optimizer_state_dict'),
                'logs': logs
            }, checkpoint_path)
            print(f"Saved best model at epoch {epoch+1} with {self.monitor}: {current_value:.4f}")

        if not self.save_best_only:
            checkpoint_path = os.path.join(self.save_path, f'model_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'logs': logs
            }, checkpoint_path)


class TrainingLogger:
    """Logger for training metrics"""
    def __init__(self, log_path):
        self.log_path = log_path
        self.logs = []
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def log(self, epoch, metrics):
        """Log metrics for an epoch"""
        log_entry = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        self.logs.append(log_entry)

        # Save to file
        with open(self.log_path, 'w') as f:
            json.dump(self.logs, f, indent=2)

    def get_logs(self):
        """Get all logged metrics"""
        return self.logs


def create_optimizer(model, optimizer_name='adam', lr=1e-3, weight_decay=1e-4):
    """Create optimizer"""
    if optimizer_name.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    return optimizer


def create_scheduler(optimizer, scheduler_name='reduce_on_plateau', **kwargs):
    """Create learning rate scheduler"""
    if scheduler_name.lower() == 'reduce_on_plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', **kwargs)
    elif scheduler_name.lower() == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, **kwargs)
    elif scheduler_name.lower() == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, **kwargs)
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    return scheduler


def train_epoch(model, dataloader, criterion, optimizer, device, scheduler=None):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Calculate accuracy for classification
        if outputs.shape[-1] > 1:  # Multi-class
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total if total > 0 else 0

    if scheduler and isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
        scheduler.step()

    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()

            # Calculate accuracy for classification
            if outputs.shape[-1] > 1:  # Multi-class
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total if total > 0 else 0

    return epoch_loss, epoch_acc


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                device, num_epochs=100, patience=10, save_path=None):
    """
    Complete training loop with early stopping and checkpointing

    Args:
        model: PyTorch model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        num_epochs: Maximum number of epochs
        patience: Early stopping patience
        save_path: Path to save checkpoints

    Returns:
        dict: Training history
    """
    # Initialize training utilities
    early_stopping = EarlyStopping(patience=patience)
    checkpoint = ModelCheckpoint(save_path, monitor='val_loss', mode='min') if save_path else None
    logger = TrainingLogger(os.path.join(save_path, 'training_logs.json')) if save_path else None

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scheduler)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Update scheduler
        if scheduler and not isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Log metrics
        logs = {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': optimizer.param_groups[0]['lr']
        }

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, LR: {logs['lr']:.6f}")

        # Checkpoint
        if checkpoint:
            checkpoint(model, epoch, logs)

        # Log to file
        if logger:
            logger.log(epoch + 1, logs)

        # Early stopping
        if early_stopping(val_loss, model):
            print(f"Early stopping at epoch {epoch+1}")
            break

    return history


def compute_metrics(outputs, targets, num_classes=2):
    """Compute comprehensive classification metrics"""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, matthews_corrcoef,
        brier_score_loss, confusion_matrix
    )

    # Convert to numpy
    preds = torch.softmax(outputs, dim=1).cpu().numpy()
    pred_labels = np.argmax(preds, axis=1)
    true_labels = targets.cpu().numpy()

    # Binary classification metrics
    if num_classes == 2:
        metrics = {
            'accuracy': accuracy_score(true_labels, pred_labels),
            'precision': precision_score(true_labels, pred_labels, average='binary'),
            'recall': recall_score(true_labels, pred_labels, average='binary'),
            'f1': f1_score(true_labels, pred_labels, average='binary'),
            'auc': roc_auc_score(true_labels, preds[:, 1]),
            'pr_auc': average_precision_score(true_labels, preds[:, 1]),
            'mcc': matthews_corrcoef(true_labels, pred_labels),
            'brier': brier_score_loss(true_labels, preds[:, 1]),
            'sensitivity': recall_score(true_labels, pred_labels, pos_label=1),
            'specificity': recall_score(true_labels, pred_labels, pos_label=0)
        }
    else:
        # Multi-class metrics
        metrics = {
            'accuracy': accuracy_score(true_labels, pred_labels),
            'precision': precision_score(true_labels, pred_labels, average='macro'),
            'recall': recall_score(true_labels, pred_labels, average='macro'),
            'f1': f1_score(true_labels, pred_labels, average='macro'),
            'auc': roc_auc_score(true_labels, preds, multi_class='ovr'),
            'mcc': matthews_corrcoef(true_labels, pred_labels)
        }

    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    metrics['confusion_matrix'] = cm.tolist()

    return metrics
