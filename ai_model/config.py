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

# Raised from 0.30: the previous run overfit hard after epoch 2 (train F1
# 0.81->0.87 while valid F1 fell 0.81->0.79) - the 278M-param backbone was
# nearly fully trainable against only ~32K train rows. Stronger dropout
# slows memorization of the training set.
DROPOUT = 0.40

LABEL_SMOOTHING = 0.10

# ==========================================================
# TRAINING CONFIGURATION
# ==========================================================

# Raised from 8: smaller batches gave noisier gradient estimates, which
# compounds overfitting on top of the encoder having more capacity than
# the ~32K-row train set needs. Larger batches also roughly halve the
# number of optimizer steps per epoch.
BATCH_SIZE = 16

# Lowered from 8: early stopping (PATIENCE=2) has triggered at epoch 4 in
# practice, so the cosine LR schedule below - sized off EPOCHS - was
# planned for a horizon training never reached, leaving LR still near its
# peak when it stopped. Matching EPOCHS to the realistic run length lets
# the schedule actually anneal LR down through the epochs where
# overfitting starts.
EPOCHS = 5

# Backbone (pretrained encoder) learning rate - kept low so the
# pretrained multilingual representations are fine-tuned gently.
# Lowered from 2e-5: at that rate the (mostly-unfrozen) backbone
# fit the ~32K-row train set faster than it generalized.
LEARNING_RATE = 1e-5

# Classifier head is randomly initialized, so it is trained with a
# higher learning rate than the backbone (discriminative fine-tuning).
HEAD_LEARNING_RATE = 1e-4

WARMUP_RATIO = 0.10

# Raised from 0.01 alongside DROPOUT to fight the same overfitting pattern.
WEIGHT_DECAY = 0.02

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

NUM_WORKERS = 0

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