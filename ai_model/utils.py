"""
====================================================
Women Safety AI Project
File : utils.py
====================================================
Not imported anywhere in the pipeline (train.py / predict.py /
evaluate.py / dataset.py / model.py all read settings directly from
config.py). This module used to hold its own hardcoded copy of the
config values, which drifted out of sync (it still said 7 classes
after config.py was updated to 5). Re-exporting from config.py instead
of duplicating literals means it can never drift again.
"""

from ai_model.config import (  # noqa: F401
    MODEL_NAME,
    NUM_CLASSES,
    TRAIN_FILE,
    VALID_FILE as VALIDATION_FILE,
    TEST_FILE,
    TEXT_COLUMN,
    LABEL_COLUMN,
    MAX_LENGTH,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    DEVICE,
    MODEL_DIR,
    BEST_MODEL_PATH as MODEL_PATH,
    LABEL_MAP as LABELS,
    SEED,
)
