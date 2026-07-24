import logging
import re
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from ai_model.config import (
    TEXT_COLUMN,
    LABEL_COLUMN,
    MAX_LENGTH,
    BATCH_SIZE,
    SEED,
    TRAIN_FILE,
    VALID_FILE,
    TEST_FILE,
    ID_MAP,
    CLASS_NAMES,
    NUM_WORKERS,
    PIN_MEMORY
    )

from ai_model.tokenizer import (
    get_tokenizer,
    tokenize_text
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# Matches letter-by-letter spaced-out words used as a filter-evasion
# tactic in some toxic-comment sources (e.g. "e a s t a s i a n s"),
# which XLM-R's subword tokenizer otherwise shreds into a run of
# meaningless single-character tokens.
_SPACED_LETTERS_RE = re.compile(r"\b(?:[A-Za-z][ ]){2,}[A-Za-z]\b")


def normalize_spaced_letters(text):
    return _SPACED_LETTERS_RE.sub(lambda m: m.group(0).replace(" ", ""), text)


class WomenSafetyDataset(Dataset):

    def __init__(
        self,
        texts,
        labels,
        tokenizer,
        max_length=MAX_LENGTH
    ):

        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):

        return len(self.texts)

    def __getitem__(self, index):

        text = str(self.texts[index])

        label = int(self.labels[index])

        encoding = tokenize_text(
            tokenizer=self.tokenizer,
            text=text,
            max_length=self.max_length
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(
                label,
                dtype=torch.long
            )
        }
def load_dataset(csv_path, keep_language=False):
    """
    Load a CSV split and encode its label column using the fixed
    ID_MAP defined in config.py (rather than a per-file sklearn
    LabelEncoder). This guarantees class id 0 always means "Safe",
    id 1 always means "Offensive", etc. across train/validation/test
    and at inference time - a per-split LabelEncoder would silently
    assign different ids on each split (alphabetical order) and break
    that guarantee.

    keep_language=True also keeps the raw "language" column (needed by
    compute_sample_weights() to build language-aware sampling weights;
    ignored by everything else, which only needs text+label).
    """
    logger.info(f"Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"{TEXT_COLUMN} column not found.")
    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"{LABEL_COLUMN} column not found.")

    columns = [TEXT_COLUMN, LABEL_COLUMN] + (["language"] if keep_language else [])
    df = df[columns].dropna()
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str).apply(normalize_spaced_letters)

    unknown_labels = set(df[LABEL_COLUMN].unique()) - set(ID_MAP.keys())
    if unknown_labels:
        raise ValueError(
            f"{csv_path} contains unknown label(s) {sorted(unknown_labels)}. "
            f"Expected one of {CLASS_NAMES}."
        )

    df[LABEL_COLUMN] = df[LABEL_COLUMN].map(ID_MAP).astype(int)

    logger.info(f"Total Samples : {len(df)}")
    logger.info(
        f"Class distribution : {df[LABEL_COLUMN].value_counts().sort_index().to_dict()}"
    )
    return df


def create_dataloader(csv_path, shuffle=False, sampler=None):
    tokenizer = get_tokenizer()
    df = load_dataset(csv_path)

    dataset = WomenSafetyDataset(
        texts=df[TEXT_COLUMN].tolist(),
        labels=df[LABEL_COLUMN].tolist(),
        tokenizer=tokenizer,
        max_length=MAX_LENGTH
        )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        # shuffle and sampler are mutually exclusive in DataLoader - a
        # sampler already defines the draw order/weighting.
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        )
    return loader


def compute_sample_weights():
    """
    Per-row inverse-frequency weight over the joint (language, category)
    distribution in train.csv, for a WeightedRandomSampler.

    compute_class_weights() below only reweights the loss by category,
    globally across all languages - it can't see that "Safe" makes up
    84% of Tanglish rows and 78% of Tamil rows in train.csv vs. 6.5% of
    English rows (dataset/README audit). The model picks up on that as a
    language -> Safe shortcut: on a held-out multilingual eval it scored
    100% on English but as low as 22% on Tanglish, with 56 of 62 misses
    predicting Safe.

    Uses sqrt (temperature=2) smoothing rather than full 1/count
    equalization - the (language, category) cells are wildly uneven in
    size (Kanglish Sexual Harassment has exactly 1 row; Tanglish Safe has
    5,611), so full equalization would resample that single Kanglish row
    thousands of times more often per epoch than an average row, teaching
    the model to memorize one exact sentence instead of the category.
    1/sqrt(count) still strongly upweights rare combinations (Tamil Hate
    Speech is drawn far more often than the raw data would give it) but
    tempers the most extreme cells enough to avoid that memorization
    failure mode. Standard technique for multilingual corpus balancing
    (e.g. how XLM-R itself samples its pretraining languages).
    """
    df = load_dataset(TRAIN_FILE, keep_language=True)
    group_sizes = df.groupby(["language", LABEL_COLUMN])[TEXT_COLUMN].transform("count")
    weights = 1.0 / group_sizes.pow(0.5)
    return torch.tensor(weights.values, dtype=torch.double)


def compute_class_weights():
    """
    Inverse-frequency class weights from the actual train.csv label
    distribution, for use in a weighted CrossEntropyLoss. Needed
    because balance_dataset.py now keeps majority classes (Safe,
    Hate Speech) at their natural size instead of undersampling
    them down to the smallest class - see dataset/balance_dataset.py
    for why.
    """
    df = load_dataset(TRAIN_FILE)
    counts = df[LABEL_COLUMN].value_counts().sort_index()
    total = counts.sum()
    num_classes = len(CLASS_NAMES)
    weights = [total / (num_classes * counts.get(i, 1)) for i in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float)


def create_train_loader():
    # WeightedRandomSampler (not shuffle=True) so every (language, category)
    # combination is drawn about equally often instead of natural frequency -
    # see compute_sample_weights() docstring. replacement=True lets rare
    # combinations (e.g. Kannada Threat, 44 real rows) be resampled enough
    # to reach parity with common ones in an epoch.
    weights = compute_sample_weights()
    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(weights),
        replacement=True,
    )
    return create_dataloader(TRAIN_FILE, sampler=sampler)


def create_validation_loader():
    return create_dataloader(VALID_FILE, shuffle=False)


def create_test_loader():
    return create_dataloader(TEST_FILE, shuffle=False)