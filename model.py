"""
Model architecture for LULC Classification.

What: Implements ResNet50 classifier with dynamic input layer for RGB (3) or MS (13) channels.
Why: Enables code reuse — single model class handles both modalities by adapting the first
     convolutional layer, avoiding redundant model definitions.
"""

import torch
import torch.nn as nn
from torchvision import models

from config import NUM_CLASSES, FC_HIDDEN_DIMS, FC_DROPOUT


class ResNet50Classifier(nn.Module):
    """
    ResNet50 backbone with custom first layer and classification head.

    What: Adapts ResNet50 to accept 3 or 13 input channels (RGB or MS),
          replaces final FC layer with custom classifier.
    Why: Transfer learning with input flexibility — maintains pretrained weights where applicable,
         enables learning on both RGB and multispectral data.
    """

    def __init__(self, input_channels=3, num_classes=NUM_CLASSES, freeze_backbone=True):
        """
        Args:
            input_channels: 3 for RGB, 13 for Multispectral
            num_classes: Number of classification targets (10 for EuroSAT)
            freeze_backbone: If True, freeze all layers except conv1 and fc layers
        """
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes

        # Load pretrained ResNet50
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # ─────────────────────────────────────────────────────────────────────
        # Adapt first convolutional layer if input_channels != 3
        # ─────────────────────────────────────────────────────────────────────
        if input_channels != 3:
            # Save original conv1 weight (3, 64, 7, 7)
            original_conv1 = self.backbone.conv1
            original_weight = original_conv1.weight.data  # (64, 3, 7, 7)

            # Create new conv1 with input_channels
            self.backbone.conv1 = nn.Conv2d(
                input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

            # Initialize new conv1 weights:
            # For MS: replicate RGB weights across extra channels, then average
            with torch.no_grad():
                if input_channels > 3:
                    # Replicate the 3 RGB channels and pad with small random values
                    new_weight = self.backbone.conv1.weight.data
                    new_weight[:, :3, :, :] = original_weight
                    # Initialize remaining channels with small random values
                    new_weight[:, 3:, :, :] = torch.randn_like(new_weight[:, 3:, :, :]) * 0.01
                else:
                    # If fewer than 3 channels, take the average
                    self.backbone.conv1.weight.data = original_weight.mean(dim=1, keepdim=True)

        # ─────────────────────────────────────────────────────────────────────
        # Freeze backbone if requested (keep conv1 trainable)
        # ─────────────────────────────────────────────────────────────────────
        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if 'conv1' not in name:  # Keep conv1 trainable for adaptation
                    param.requires_grad = False

        # ─────────────────────────────────────────────────────────────────────
        # Replace final classification head
        # ─────────────────────────────────────────────────────────────────────
        num_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, FC_HIDDEN_DIMS),
            nn.ReLU(inplace=True),
            nn.Dropout(p=FC_DROPOUT),
            nn.Linear(FC_HIDDEN_DIMS, num_classes)
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_channels, 64, 64)

        Returns:
            logits: Tensor of shape (batch_size, num_classes) — raw predictions
        """
        return self.backbone(x)

    def get_trainable_params(self):
        """Return count of trainable parameters (for logging)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self):
        """Return total parameter count."""
        return sum(p.numel() for p in self.parameters())
