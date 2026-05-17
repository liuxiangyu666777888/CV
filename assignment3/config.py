"""Shared configuration for Assignment 3: U-Net Semantic Segmentation."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Dataset paths
IMAGES_DIR = Path("d:/CV/data/images/images")
ANNOTATIONS_DIR = Path("d:/CV/data/annotations/annotations")
TRIMAPS_DIR = ANNOTATIONS_DIR / "trimaps"
TRAINVAL_LIST = ANNOTATIONS_DIR / "trainval.txt"
TEST_LIST = ANNOTATIONS_DIR / "test.txt"

# Output
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Training hyperparameters
IMAGE_SIZE = 256
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 50
VAL_SPLIT = 0.2  # 80/20 train/val split from trainval.txt
NUM_WORKERS = 0

# 3-class segmentation: 0=pet, 1=background, 2=boundary
NUM_CLASSES = 3
TRIMAP_MAPPING = {1: 0, 2: 1, 3: 2}  # original → 0-indexed
CLASS_NAMES = ["pet", "background", "boundary"]

# Random seed
SEED = 42
