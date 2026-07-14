import os
import random
import numpy as np
import torch

# ==========================================================
# PROJECT INFORMATION
# ==========================================================

PROJECT_NAME = "Women Safety AI"
VERSION = "1.0.0"

# ==========================================================
# MODEL CONFIGURATION
# ==========================================================

MODEL_NAME = "xlm-roberta-base"

# NOTE: dataset/raw/Women_Safety_Final_v3_50k.csv contains only these 5
# categories - "Cyber Bullying" and "Stalking" have zero labeled examples
# anywhere in the source data. NUM_CLASSES/CLASS_NAMES must match what the
# data actually supports; add those 2 classes back only after real labeled
# examples for them exist in the raw dataset.
NUM_CLASSES = 5

CLASS_NAMES = [
    "Safe",
    "Offensive",
    "Sexual Harassment",
    "Threat",
    "Hate Speech"
]

MAX_LENGTH = 128

DROPOUT = 0.30

LABEL_SMOOTHING = 0.10

# ==========================================================
# TRAINING CONFIGURATION
# ==========================================================

BATCH_SIZE = 16

EPOCHS = 8

# Backbone (pretrained encoder) learning rate - kept low so the
# pretrained multilingual representations are fine-tuned gently.
LEARNING_RATE = 2e-5

# Classifier head is randomly initialized, so it is trained with a
# higher learning rate than the backbone (discriminative fine-tuning).
HEAD_LEARNING_RATE = 1e-4

WARMUP_RATIO = 0.10

WEIGHT_DECAY = 0.01

# ==========================================================
# DATA SPLIT
# ==========================================================

TRAIN_SPLIT = 0.70

VALID_SPLIT = 0.15

TEST_SPLIT = 0.15

# ==========================================================
# DATASET CONFIGURATION
# ==========================================================

TEXT_COLUMN = "comment"

LABEL_COLUMN = "category"

RANDOM_SEED = 42

# ==========================================================
# DEVICE
# ==========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

NUM_WORKERS = 2

PIN_MEMORY = torch.cuda.is_available()

# ==========================================================
# RANDOM SEED
# ==========================================================

SEED = RANDOM_SEED

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "processed"
)

TRAIN_FILE = os.path.join(
    DATASET_DIR,
    "train.csv"
)

VALID_FILE = os.path.join(
    DATASET_DIR,
    "validation.csv"
)

TEST_FILE = os.path.join(
    DATASET_DIR,
    "test.csv"
)

# ==========================================================
# MODEL PATHS
# ==========================================================

MODEL_DIR = os.path.join(
    BASE_DIR,
    "saved_models"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_model.pt"
)

LAST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "last_model.pt"
)

TOKENIZER_DIR = os.path.join(
    MODEL_DIR,
    "tokenizer"
)

os.makedirs(
    TOKENIZER_DIR,
    exist_ok=True
)

# ==========================================================
# LABEL MAP
# ==========================================================

LABEL_MAP = {
    0: "Safe",
    1: "Offensive",
    2: "Sexual Harassment",
    3: "Threat",
    4: "Hate Speech"
}

ID_MAP = {
    value: key
    for key, value in LABEL_MAP.items()
}

# ==========================================================
# TRAINING OPTIONS
# ==========================================================

USE_MIXED_PRECISION = torch.cuda.is_available()

EARLY_STOPPING = True

PATIENCE = 2

SAVE_EVERY_EPOCH = True

PRINT_EVERY = 50