"""
Data loading module for LULC Classification.

What: Implements unified EuroSATDataset class for both RGB and Multispectral data,
       with error handling and efficient batch loading.
Why: Consolidates data pipeline logic, supports both modalities seamlessly, prevents
     data leakage through split-specific transforms.
"""

import torch
import numpy as np
from pathlib import Path
from PIL import Image
import rasterio
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

from config import (
    DATA_RGB, DATA_MS, IMAGE_SIZE, NUM_WORKERS, PIN_MEMORY,
    BATCH_SIZE, RGB_MEAN, RGB_STD, MS_MEAN, MS_STD, CLASS_NAMES
)


class EuroSATDataset(Dataset):
    """
    Unified PyTorch Dataset for RGB or Multispectral (TIFF) images.

    What: Loads satellite images from disk, handles both JPG (RGB) and TIFF (13-band MS),
          applies per-split transforms.
    Why: DRY principle — single dataset class avoids duplication of RGB/MS loaders.
    """

    def __init__(self, root_dir, modality='RGB', transform=None):
        """
        Args:
            root_dir: Path to dataset root (e.g., EuroSAT_RGB or EuroSAT_MS)
            modality: 'RGB' or 'MS' (determines file extension and loader)
            transform: torchvision.transforms.Compose or None
        """
        self.root_dir = Path(root_dir)
        self.modality = modality.upper()
        self.transform = transform
        self.samples = []
        self.targets = []
        self.class_to_idx = {cls: idx for idx, cls in enumerate(CLASS_NAMES)}

        # Build file list from directory structure
        for class_name in CLASS_NAMES:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                print(f"[WARN] Warning: class directory not found: {class_dir}")
                continue

            # File extension based on modality
            pattern = '*.jpg' if self.modality == 'RGB' else '*.tif'
            files = sorted(class_dir.glob(pattern))

            for file_path in files:
                self.samples.append((str(file_path), self.class_to_idx[class_name]))
                self.targets.append(self.class_to_idx[class_name])

        print(f"[OK] Loaded {len(self.samples)} {self.modality} samples from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Load and return a single sample (image, label)."""
        path, label = self.samples[idx]

        try:
            if self.modality == 'RGB':
                img = self._load_rgb(path)
            else:  # MS
                img = self._load_ms(path)

            if self.transform:
                img = self.transform(img)

            return img, label

        except Exception as e:
            print(f"[ERROR] Error loading {path}: {e}")
            # Return a dummy tensor to avoid breaking the loader
            channels = 3 if self.modality == 'RGB' else 13
            return torch.zeros(channels, IMAGE_SIZE, IMAGE_SIZE), label

    @staticmethod
    def _load_rgb(path):
        """Load RGB JPG image as PIL Image."""
        try:
            img = Image.open(path).convert('RGB')
            return img
        except Exception as e:
            raise IOError(f"Failed to load RGB image {path}: {e}")

    @staticmethod
    def _load_ms(path):
        """Load Multispectral TIFF as float32 tensor (C, H, W)."""
        try:
            with rasterio.open(path) as src:
                # Read all 13 bands as (13, H, W) float32
                arr = src.read().astype(np.float32)
            return torch.from_numpy(arr)
        except Exception as e:
            raise IOError(f"Failed to load MS image {path}: {e}")


class TransformSubset(Dataset):
    """
    Wrapper around Subset to apply per-split transforms.

    What: Enables train/val/test-specific augmentation without modifying the main Dataset.
    Why: Prevents transform cross-contamination (e.g., val should not use aggressive augmentation).
    """

    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        # Get raw sample from underlying dataset
        sample_idx = self.subset.indices[idx]
        path, label = self.subset.dataset.samples[sample_idx]
        img = self.subset.dataset[sample_idx][0]  # Get the image part only

        # Apply split-specific transform
        if self.transform:
            img = self.transform(img)

        return img, label


def get_transforms(modality='RGB', split='train'):
    """
    Create and return transforms for a given modality and split.

    What: Factory function for augmentation/normalization pipelines.
    Why: Centralizes transform logic, makes it easy to adjust augmentation strategy.
    """
    if modality.upper() == 'RGB':
        mean, std = RGB_MEAN, RGB_STD
        if split == 'train':
            transform = transforms.Compose([
                transforms.Resize(72),
                transforms.RandomCrop(IMAGE_SIZE),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=20),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        else:  # val or test
            transform = transforms.Compose([
                transforms.Resize(IMAGE_SIZE),
                transforms.CenterCrop(IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
    else:  # MS
        mean, std = MS_MEAN, MS_STD
        if split == 'train':
            transform = transforms.Compose([
                transforms.Resize((72, 72), antialias=True),
                transforms.RandomCrop(IMAGE_SIZE),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=20),
                transforms.Normalize(mean=mean, std=std),
            ])
        else:  # val or test
            transform = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
                transforms.CenterCrop(IMAGE_SIZE),
                transforms.Normalize(mean=mean, std=std),
            ])

    return transform


def get_data_loaders(modality='RGB', batch_size=BATCH_SIZE):
    """
    Create and return train/val/test DataLoaders.

    What: Factory function for reproducible data loading.
    Why: Encapsulates split logic, ensures consistent batch sizes and worker config.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    from sklearn.model_selection import train_test_split

    # Load full dataset
    data_dir = DATA_RGB if modality.upper() == 'RGB' else DATA_MS
    dataset = EuroSATDataset(data_dir, modality=modality, transform=None)

    # Create indices for train/val/test split (80/10/10)
    all_indices = list(range(len(dataset)))
    all_labels = dataset.targets

    train_idx, temp_idx, _, temp_labels = train_test_split(
        all_indices, all_labels,
        test_size=0.20, stratify=all_labels, random_state=42
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, stratify=temp_labels, random_state=42
    )

    # Create subsets
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    test_subset = Subset(dataset, test_idx)

    # Wrap with split-specific transforms
    train_data = TransformSubset(train_subset, get_transforms(modality, 'train'))
    val_data = TransformSubset(val_subset, get_transforms(modality, 'val'))
    test_data = TransformSubset(test_subset, get_transforms(modality, 'test'))

    # Create DataLoaders
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    print(f"[OK] {modality} DataLoaders created:")
    print(f"   Train: {len(train_data)} samples | Val: {len(val_data)} samples | Test: {len(test_data)} samples")

    return train_loader, val_loader, test_loader
