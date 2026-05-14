"""
Training engine for LULC Classification.

What: Implements training loop with AMP, learning rate scheduling, early stopping, and checkpointing.
Why: Encapsulates training logic cleanly, ensures reproducible training with best practices,
     enables fast local training with mixed precision acceleration.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json
from pathlib import Path
from tqdm import tqdm

from config import (
    DEVICE, SCHEDULER_PATIENCE, SCHEDULER_FACTOR,
    EARLY_STOPPING_PATIENCE, EARLY_STOPPING_MIN_DELTA
)


class Trainer:
    """
    Training engine with Automatic Mixed Precision, learning rate scheduling, and early stopping.

    What: Manages the full training lifecycle (train/val/test epochs), saves best checkpoints,
          tracks metrics history.
    Why: DRY principle — centralizes training logic, prevents code duplication across different
         runs, ensures consistent hyperparameter application.
    """

    def __init__(self, model, optimizer, criterion, train_loader, val_loader, test_loader,
                 checkpoint_dir, device=DEVICE, use_amp=True):
        """
        Args:
            model: PyTorch model instance
            optimizer: torch.optim optimizer
            criterion: Loss function (typically CrossEntropyLoss)
            train_loader: DataLoader for training split
            val_loader: DataLoader for validation split
            test_loader: DataLoader for test split
            checkpoint_dir: Directory to save model checkpoints
            device: torch.device (cuda or cpu)
            use_amp: Whether to use Automatic Mixed Precision
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device
        self.use_amp = use_amp

        # AMP components
        if use_amp:
            self.scaler = GradScaler()
        else:
            self.scaler = None

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=SCHEDULER_FACTOR,
            patience=SCHEDULER_PATIENCE
        )

        # Early stopping tracking
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        self.early_stop = False

        # Metrics history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }

    def _process_batch(self, batch, training=True):
        """
        Process a single batch and return loss + predictions.

        What: Unified batch processing (forward pass, loss compute, backward if training).
        Why: DRY principle — avoids duplicating forward/backward logic in train/val/test.
        """
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)

        if training:
            self.optimizer.zero_grad()

        # Forward pass with AMP if enabled
        if self.use_amp and training:
            with autocast():
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            if training:
                loss.backward()
                self.optimizer.step()

        _, preds = torch.max(outputs, 1)
        return loss.item(), preds, labels

    def train_epoch(self):
        """
        Train for one epoch.

        Returns:
            avg_loss: Average training loss
            avg_acc: Average training accuracy
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(self.train_loader, desc='Training', leave=False)
        for batch in loop:
            loss, preds, labels = self._process_batch(batch, training=True)
            total_loss += loss
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            loop.set_postfix({'loss': loss, 'acc': correct / total})

        avg_loss = total_loss / len(self.train_loader)
        avg_acc = correct / total

        return avg_loss, avg_acc

    def validate(self):
        """
        Validate on validation set.

        Returns:
            avg_loss: Average validation loss
            avg_acc: Average validation accuracy
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            loop = tqdm(self.val_loader, desc='Validating', leave=False)
            for batch in loop:
                loss, preds, labels = self._process_batch(batch, training=False)
                total_loss += loss
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                loop.set_postfix({'loss': loss, 'acc': correct / total})

        avg_loss = total_loss / len(self.val_loader)
        avg_acc = correct / total

        return avg_loss, avg_acc

    def test(self):
        """
        Evaluate on test set and return detailed metrics.

        Returns:
            dict: Contains accuracy, per-class metrics, confusion matrix, predictions
        """
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc='Testing', leave=False):
                _, preds, labels = self._process_batch(batch, training=False)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        cm = confusion_matrix(all_labels, all_preds)

        return {
            'accuracy': accuracy,
            'predictions': all_preds,
            'labels': all_labels,
            'confusion_matrix': cm
        }

    def save_checkpoint(self, model_name='model.pth'):
        """
        Save model checkpoint.

        What: Persists model state to disk for later inference or resumption.
        Why: Enables experiment reproducibility, allows loading best weights post-training.
        """
        checkpoint_path = self.checkpoint_dir / model_name
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f" Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, model_name='model.pth'):
        """Load model checkpoint from disk."""
        checkpoint_path = self.checkpoint_dir / model_name
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        print(f" Checkpoint loaded: {checkpoint_path}")

    def fit(self, num_epochs):
        """
        Full training loop with learning rate scheduling and early stopping.

        Args:
            num_epochs: Maximum number of epochs to train

        Returns:
            dict: Training history (losses, accuracies per epoch)
        """
        print(f"\n{'='*60}")
        print(f"Training on {self.device} | AMP: {self.use_amp}")
        print(f"Epochs: {num_epochs} | LR Scheduler Patience: {SCHEDULER_PATIENCE}")
        print(f"Early Stopping Patience: {EARLY_STOPPING_PATIENCE}")
        print(f"{'='*60}\n")

        for epoch in range(num_epochs):
            if self.early_stop:
                print(f"\n Early stopping triggered at epoch {epoch}")
                break

            # Training phase
            train_loss, train_acc = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)

            # Validation phase
            val_loss, val_acc = self.validate()
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            # Learning rate from optimizer
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rates'].append(current_lr)

            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                  f"LR: {current_lr:.2e}")

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            # Checkpointing: save if validation loss improved
            if val_loss < self.best_val_loss - EARLY_STOPPING_MIN_DELTA:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                self.save_checkpoint('best_model.pth')
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                    print(f"\n  No improvement for {EARLY_STOPPING_PATIENCE} epochs. Stopping.")
                    self.early_stop = True

        print(f"\n Training complete. Best validation loss: {self.best_val_loss:.4f}")

        # Load best checkpoint before returning
        self.load_checkpoint('best_model.pth')

        return self.history
