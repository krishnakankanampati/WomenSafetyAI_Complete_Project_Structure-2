import logging
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader

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
def load_dataset(csv_path):
    """
    Load a CSV split and encode its label column using the fixed
    ID_MAP defined in config.py (rather than a per-file sklearn
    LabelEncoder). This guarantees class id 0 always means "Safe",
    id 1 always means "Offensive", etc. across train/validation/test
    and at inference time - a per-split LabelEncoder would silently
    assign different ids on each split (alphabetical order) and break
    that guarantee.
    """
    logger.info(f"Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"{TEXT_COLUMN} column not found.")
    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"{LABEL_COLUMN} column not found.")

    df = df[[TEXT_COLUMN, LABEL_COLUMN]].dropna()
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str)

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


def create_dataloader(csv_path, shuffle=False):
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
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        )
    return loader


def create_train_loader():
    return create_dataloader(TRAIN_FILE, shuffle=True)


def create_validation_loader():
    return create_dataloader(VALID_FILE, shuffle=False)


def create_test_loader():
    return create_dataloader(TEST_FILE, shuffle=False)