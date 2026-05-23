"""
Configuration module for LULC Classification System.

What: Centralizes all hyperparameters, paths, and normalization statistics.
Why: Eliminates hardcoding, makes experiments reproducible, enables easy config changes without modifying code.
"""

from pathlib import Path
import torch

# ─────────────────────────────────────────────────────────────────────────────
# PATHS — Cross-platform using pathlib
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_RGB = PROJECT_ROOT / "EuroSAT" / "EuroSAT_RGB"
DATA_MS = PROJECT_ROOT / "EuroSAT_MS"
MODELS_DIR = PROJECT_ROOT
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_PLOTS = OUTPUTS_DIR / "plots"
OUTPUTS_METRICS = OUTPUTS_DIR / "metrics.json"

# Create output directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)
OUTPUTS_PLOTS.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATASET & CLASSES
# ─────────────────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
    'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'
]
NUM_CLASSES = len(CLASS_NAMES)
IMAGE_SIZE = 64

# ─────────────────────────────────────────────────────────────────────────────
# DATA SPLIT RATIOS
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
SEED = 42

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
BATCH_SIZE = 32
NUM_EPOCHS = 50  # Early stopping will likely end before this
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
PIN_MEMORY = True

# ─────────────────────────────────────────────────────────────────────────────
# SCHEDULER & EARLY STOPPING
# ─────────────────────────────────────────────────────────────────────────────
SCHEDULER_PATIENCE = 3  # ReduceLROnPlateau: lower LR after 3 epochs no improvement
SCHEDULER_FACTOR = 0.5  # Multiply LR by this factor when patience exceeded
EARLY_STOPPING_PATIENCE = 7  # Stop if no improvement for 7 epochs
EARLY_STOPPING_MIN_DELTA = 1e-4  # Minimum change to qualify as an improvement

# ─────────────────────────────────────────────────────────────────────────────
# RGB NORMALIZATION STATISTICS (computed from training split)
# ─────────────────────────────────────────────────────────────────────────────
RGB_MEAN = [0.3441, 0.3801, 0.4076]
RGB_STD = [0.2021, 0.1365, 0.1152]
RGB_CHANNELS = 3

# ─────────────────────────────────────────────────────────────────────────────
# MULTISPECTRAL (MS) NORMALIZATION STATISTICS (13 Sentinel-2 bands)
# Computed from training split — per-band statistics
# ─────────────────────────────────────────────────────────────────────────────
MS_MEAN = [
    1353.48, 1116.89, 1041.34, 945.85, 1198.16, 2001.29, 2372.07,
    2299.16, 730.52, 12.10, 1820.51, 1117.43, 2597.99
]
MS_STD = [
    241.68, 329.60, 391.10, 588.77, 561.34, 852.64, 1076.47,
    1108.48, 398.09, 4.52, 994.75, 754.05, 1219.94
]
MS_CHANNELS = 13

# Band name mapping (for visualization)
MS_BAND_NAMES = [
    'B01 Coastal', 'B02 Blue', 'B03 Green', 'B04 Red',
    'B05 RE-1', 'B06 RE-2', 'B07 RE-3', 'B08 NIR',
    'B8A NIR-n', 'B09 WV', 'B10 Cirrus', 'B11 SWIR1', 'B12 SWIR2'
]

# Band indices for False Color Composite and spectral indices
BAND_INDICES = {
    'BLUE': 1,      # B02
    'GREEN': 2,     # B03
    'RED': 3,       # B04
    'NIR': 7,       # B08
    'SWIR1': 11     # B11
}

# ─────────────────────────────────────────────────────────────────────────────
# DEVICE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─────────────────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
BACKBONE_FREEZE = True  # Freeze backbone layers except first conv
FC_HIDDEN_DIMS = 512
FC_DROPOUT = 0.5

# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION SETTINGS (Streamlit App)
# ─────────────────────────────────────────────────────────────────────────────
STREAMLIT_MAX_UPLOAD_MB = 5
STREAMLIT_THEME = "light"
