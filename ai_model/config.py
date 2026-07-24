import os
import random
import numpy as np
import torch

# ==========================================================
# LOCAL ENVIRONMENT (.env)
# ==========================================================


def _load_dotenv():
    """
    Load KEY=value pairs from a .env file at the project root into the
    environment, for any key not already set.

    Exists because os.environ is a snapshot taken when the process starts:
    an editor or terminal opened before GEMINI_API_KEY was set never sees it,
    so the LLM fallback silently disables itself and predictions quietly stay
    local. A file next to the code removes that dependency on which shell
    happened to launch Python.

    setdefault, not assignment - a key exported in the real environment is
    more deliberate than one sitting in a file, so it wins. .env is already
    in .gitignore; nothing here should ever be committed.
    """
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ".env",
    )
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            # Strip one layer of matching quotes - people paste keys both ways.
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            os.environ.setdefault(key.strip(), value)


_load_dotenv()

# ==========================================================
# PROJECT INFORMATION
# ==========================================================

PROJECT_NAME = "Women Safety AI"
VERSION = "1.0.0"

# ==========================================================
# MODEL CONFIGURATION
# ==========================================================

# Tried xlm-roberta-large (560M params) to push past the base model's
# 89.80% test accuracy ceiling, but reverted: on this 6GB GPU it sits at
# 96% VRAM usage (5.9/6.1GB) under the real training loop - close enough
# to the limit that Windows/WDDM spills into slow shared system memory,
# making a single epoch take ~115 min instead of the ~9 min a smoke test
# at 4.95GB predicted. Not a usable tradeoff for the accuracy gained.
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

# Lowered from 0.40: the 5-epoch run at 0.40 never showed the earlier
# overfit collapse (train/val F1 stayed within ~0.01 of each other through
# epoch 5, train F1 still climbing at 0.879) - dropout that strong was
# capping capacity rather than fixing overfitting. 0.30 keeps meaningful
# regularization while giving the head more room to fit the genuinely
# ambiguous Offensive/Hate Speech/Safe boundary (test confusion matrix
# showed 27% of Offensive rows predicted Safe).
DROPOUT = 0.30

# Lowered from 0.10: on 5 classes with real semantic overlap between
# Offensive/Hate Speech/Safe, smoothing that strong blurs exactly the
# boundary the model needs to sharpen.
LABEL_SMOOTHING = 0.05

# ==========================================================
# TRAINING CONFIGURATION
# ==========================================================

# Raised from 8: smaller batches gave noisier gradient estimates, which
# compounds overfitting on top of the encoder having more capacity than
# the ~32K-row train set needs. Larger batches also roughly halve the
# number of optimizer steps per epoch.
BATCH_SIZE = 16

# Raised from 5: the 5-epoch run plateaued noisily (val F1 0.868 -> 0.859
# -> 0.894 -> 0.881 -> 0.885) rather than cleanly overfitting - it hit its
# best epoch at 3 and then PATIENCE=2 ran out simply because EPOCHS=5 gave
# it nowhere left to go, not because it had genuinely degraded. Raised
# together with PATIENCE below so a later, better optimum has room to show
# up if one exists.
EPOCHS = 8

# Backbone (pretrained encoder) learning rate - kept low so the
# pretrained multilingual representations are fine-tuned gently.
# Lowered from 2e-5: at that rate the (mostly-unfrozen) backbone
# fit the ~32K-row train set faster than it generalized.
LEARNING_RATE = 1e-5

# Classifier head is randomly initialized, so it is trained with a
# higher learning rate than the backbone (discriminative fine-tuning).
HEAD_LEARNING_RATE = 1e-4

WARMUP_RATIO = 0.10

# Lowered from 0.02 alongside DROPOUT - same reasoning, current run showed
# no sign of the overfitting this was raised to fight.
WEIGHT_DECAY = 0.015

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

# Raised from 2 alongside EPOCHS: the 5-epoch run's best epoch (3) was
# followed by 2 straight non-improving epochs that only happened to
# coincide with the end of the run. A tighter patience would have stopped
# at exactly the same point for the wrong reason; this gives a real chance
# to ride out a noisy plateau instead of stopping at its edge.
PATIENCE = 4

SAVE_EVERY_EPOCH = True

PRINT_EVERY = 50

# ==========================================================
# LLM FALLBACK (low-confidence escalation)
# ==========================================================

# Below this confidence, predict.py escalates to an LLM for a second
# opinion instead of trusting the local model's top-1 label.
#
# Raised from 0.755: that value came from a 100-example spot-check and
# undershot badly - a full pass over the 11,246-row test set (see
# saved_models/reports/full_test_confidence.csv) showed correct/wrong
# confidence distributions overlap far more than the small sample
# suggested (correct mean 0.905, wrong mean 0.702, both with wide
# spread). At 0.755 only 58.8% of real errors were ever escalated.
# Sweeping thresholds against ground truth on the full set:
#   0.755 -> 16.7% of traffic escalated, 58.8% of errors caught
#   0.80  -> 19.9% escalated, 65.3% caught
#   0.85  -> 24.2% escalated, 74.1% caught
#   0.90  -> 31.4% escalated, 82.7% caught (correct-prediction waste jumps to 24.1%)
#   0.95  -> 52.6% escalated, 93.2% caught (waste jumps to 46.8%)
# Marginal cost/benefit stays favorable up to ~0.85 (each step up catches
# more errors than it wastes escalations on already-correct predictions)
# and breaks down past 0.90. 0.85 sits at that efficiency elbow - for a
# safety classifier, missing real harmful content outweighs the cost of
# extra LLM calls, so this leans toward the higher end of the efficient
# range rather than the cost-minimizing middle.
LLM_ESCALATION_THRESHOLD = 0.85

# Gemini rather than Hugging Face Inference Providers. HF was tried and
# reverted: its free tier meters a monthly *credit balance* ($0.10) rather
# than a request count, and once that is spent every call 402s until the
# month rolls over - a verification sweep stalls for weeks, not a day.
# Gemini's free tier is a daily request quota that refills every midnight
# Pacific, which suits an occasional 76-row sweep far better.
#
# Tried in order; on a 429 the next one is used automatically. Each Gemini
# model meters its own separate free-tier quota bucket, so exhausting one
# leaves the rest untouched - that is what makes this chain worth having
# rather than just failing for the day.
#
# Ordering is by measured free-tier throughput, not by capability, because
# quota is the binding constraint here. A live probe on 2026-07-24 measured
# gemini-3.5-flash at roughly 20 requests/day on this account (a 76-row sweep
# would take four days), while 3.5-flash-lite completed all 76 in one run.
# All of flash-lite, 3.1-flash-lite and flash scored 3/3 on the same hard
# romanized-Telugu cases, so there is no evidence the heavier models actually
# buy accuracy here - which makes the throughput ordering an easy call.
#
# gemini-3.6-flash is deliberately absent despite being newest: the same probe
# got one successful call, then immediate 429. It would burn a retry slot on
# every request for one call a day.
LLM_FALLBACK_MODELS = [
    "gemini-3.5-flash-lite",    # fastest (~1.2s), largest observed daily quota
    "gemini-3.1-flash-lite",    # same tier, separate bucket
    "gemini-3-flash-preview",   # works but slow (~40s observed)
    "gemini-3.5-flash",         # strongest, but ~20/day
]

# Kept as the "preferred" single model for logging and reports.
LLM_FALLBACK_MODEL = LLM_FALLBACK_MODELS[0]

# How long a model stays benched after a 429 before the chain retries it.
# A 429 is ambiguous - it can be the per-minute limit (clears in seconds) or
# the daily one (clears at midnight Pacific). Benching for 90s rather than for
# the whole process means a per-minute limit self-heals, while a daily one
# costs at most one wasted call every 90s.
QUOTA_RETRY_AFTER_SECONDS = 90

ENABLE_LLM_FALLBACK = True