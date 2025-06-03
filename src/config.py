import os

# Image parameters
IMG_HEIGHT = 256
IMG_WIDTH = 256

# At the end of config.py

# Aliases for compatibility
IMAGE_HEIGHT = IMG_HEIGHT
IMAGE_WIDTH = IMG_WIDTH

IMG_CHANNELS = 3

# Training parameters
BATCH_SIZE = 3
EPOCHS = 50
BUFFER_SIZE = 1000
LEARNING_RATE = 2e-4

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_HAZY_DIR = os.path.join(DATA_DIR, "train", "hazy")
TRAIN_CLEAN_DIR = os.path.join(DATA_DIR, "train", "clean")
VAL_HAZY_DIR = os.path.join(DATA_DIR, "val", "hazy")
VAL_CLEAN_DIR = os.path.join(DATA_DIR, "val", "clean")

# Output paths
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
