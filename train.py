"""
Main training script for LULC Classification.

What: Entry point for model training. Orchestrates data loading, model initialization,
       training loop, and result visualization.
Why: Separates training logic from notebooks, makes experiments reproducible and scriptable.

Usage:
    python train.py [--modality RGB|MS] [--epochs 50]
"""

import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser

import config
from data_loader import get_data_loaders
from model import ResNet50Classifier
from engine import Trainer


def set_seed(seed=42):
    """Set seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot_training_curves(history, modality, output_dir):
    """Plot and save training/validation curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy curves
    axes[0].plot(history['train_acc'], label='Train Accuracy', marker='o', markersize=3)
    axes[0].plot(history['val_acc'], label='Validation Accuracy', marker='s', markersize=3)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title(f'{modality} Model — Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss curves
    axes[1].plot(history['train_loss'], label='Train Loss', marker='o', markersize=3)
    axes[1].plot(history['val_loss'], label='Validation Loss', marker='s', markersize=3)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title(f'{modality} Model — Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / f'{modality.lower()}_training_curves.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f" Plot saved: {plot_path}")
    plt.close()


def plot_confusion_matrix(cm, class_names, modality, output_dir):
    """Plot and save confusion matrix."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names,
                yticklabels=class_names, cbar_kws={'label': 'Count'}, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'{modality} Model — Confusion Matrix on Test Set')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45)
    plt.tight_layout()

    cm_path = output_dir / f'{modality.lower()}_confusion_matrix.png'
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    print(f" Confusion matrix saved: {cm_path}")
    plt.close()


def train_model(modality='RGB', num_epochs=config.NUM_EPOCHS):
    """
    Complete training pipeline for a single modality.

    Args:
        modality: 'RGB' or 'MS'
        num_epochs: Maximum number of training epochs
    """
    print(f"\n{'='*70}")
    print(f"TRAINING {modality} MODEL")
    print(f"{'='*70}\n")

    # Set seed for reproducibility
    set_seed(config.SEED)

    # Load data
    print(f" Loading {modality} data...")
    train_loader, val_loader, test_loader = get_data_loaders(modality, config.BATCH_SIZE)

    # Initialize model
    print(f" Initializing {modality} ResNet50 model...")
    input_channels = 3 if modality == 'RGB' else 13
    model = ResNet50Classifier(
        input_channels=input_channels,
        num_classes=config.NUM_CLASSES,
        freeze_backbone=config.BACKBONE_FREEZE
    )
    print(f"   Total parameters: {model.get_total_params():,}")
    print(f"   Trainable parameters: {model.get_trainable_params():,}")

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    # Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        checkpoint_dir=config.MODELS_DIR,
        device=config.DEVICE,
        use_amp=True  # Enable Automatic Mixed Precision
    )

    # Training
    print(f"\n Starting training...")
    history = trainer.fit(num_epochs)

    # Evaluation on test set
    print(f"\n Evaluating on test set...")
    test_results = trainer.test()
    test_accuracy = test_results['accuracy']
    cm = test_results['confusion_matrix']

    print(f"\n{'='*70}")
    print(f"{modality} TEST RESULTS")
    print(f"{'='*70}")
    print(f" Test Accuracy: {test_accuracy*100:.2f}%\n")

    # Save plots
    plot_training_curves(history, modality, config.OUTPUTS_PLOTS)
    plot_confusion_matrix(cm, config.CLASS_NAMES, modality, config.OUTPUTS_PLOTS)

    # Save metrics to JSON
    metrics = {
        'modality': modality,
        'test_accuracy': float(test_accuracy),
        'best_val_loss': float(trainer.best_val_loss),
        'num_epochs_trained': len(history['train_loss']),
        'input_channels': input_channels,
        'num_parameters': model.get_total_params(),
        'trainable_parameters': model.get_trainable_params(),
    }

    metrics_path = config.OUTPUTS_DIR / f'{modality.lower()}_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f" Metrics saved: {metrics_path}")

    # Save final model
    model_path = config.MODELS_DIR / f'{modality.lower()}_best.pth'
    torch.save(model.state_dict(), model_path)
    print(f" Final model saved: {model_path}")

    return test_accuracy


def main():
    """Main entry point."""
    parser = ArgumentParser(description='Train LULC classification models')
    parser.add_argument('--modality', type=str, choices=['RGB', 'MS', 'both'],
                       default='both', help='Which model to train')
    parser.add_argument('--epochs', type=int, default=config.NUM_EPOCHS,
                       help='Maximum number of training epochs')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu', 'auto'],
                       default='auto', help='Device to use for training')
    args = parser.parse_args()

    # Override device if specified
    if args.device != 'auto':
        config.DEVICE = torch.device(args.device)

    print(f" Device: {config.DEVICE}")
    print(f" Seed: {config.SEED}")
    print(f" Data: RGB={config.DATA_RGB}, MS={config.DATA_MS}")
    print(f" Outputs: {config.OUTPUTS_DIR}")

    # Train models
    results = {}
    if args.modality.upper() in ['RGB', 'BOTH']:
        results['RGB'] = train_model('RGB', args.epochs)
    if args.modality.upper() in ['MS', 'BOTH']:
        results['MS'] = train_model('MS', args.epochs)

    # Summary
    print(f"\n{'='*70}")
    print(f" TRAINING COMPLETE")
    print(f"{'='*70}")
    for modality, accuracy in results.items():
        print(f"   {modality:4s} Test Accuracy: {accuracy*100:6.2f}%")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
