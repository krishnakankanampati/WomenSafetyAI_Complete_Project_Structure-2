import os
import logging

from transformers import AutoTokenizer

from ai_model.config import (
    MODEL_NAME,
    TOKENIZER_DIR,
)

# ==========================================================
# LOGGING
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ==========================================================
# TOKENIZER MANAGER
# ==========================================================

class TokenizerManager:

    def __init__(self):

        self.model_name = MODEL_NAME
        self.save_path = TOKENIZER_DIR
        self.tokenizer = None

    def load(self):

        if (
            os.path.exists(self.save_path)
            and len(os.listdir(self.save_path)) > 0
        ):

            logger.info("Loading tokenizer from local directory...")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.save_path
            )

        else:

            logger.info("Downloading tokenizer...")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name
            )

            os.makedirs(
                self.save_path,
                exist_ok=True
            )

            self.tokenizer.save_pretrained(
                self.save_path
            )

            logger.info("Tokenizer saved successfully.")

        return self.tokenizer


# ==========================================================
# SINGLETON TOKENIZER
# ==========================================================

tokenizer_manager = TokenizerManager()


def get_tokenizer():

    return tokenizer_manager.load()


# ==========================================================
# TOKENIZE SINGLE TEXT
# ==========================================================

def tokenize_text(
    tokenizer,
    text,
    max_length
):

    return tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt"
    )


# ==========================================================
# TOKENIZE MULTIPLE TEXTS
# ==========================================================

def tokenize_batch(
    tokenizer,
    texts,
    max_length
):

    return tokenizer(
        texts,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt"
    )


# ==========================================================
# DECODE TOKENS
# ==========================================================

def decode_text(
    tokenizer,
    input_ids
):

    return tokenizer.decode(
        input_ids,
        skip_special_tokens=True
    )


# ==========================================================
# GET VOCAB SIZE
# ==========================================================

def get_vocab_size():

    tokenizer = get_tokenizer()

    return tokenizer.vocab_size