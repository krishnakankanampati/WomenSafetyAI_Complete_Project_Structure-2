# cleaning module
"""
====================================================
Women Safety AI Project
Module 1 : Advanced Text Cleaning
File      : clean_dataset.py
====================================================
"""

import re
import unicodedata
import pandas as pd


INPUT_FILE = "dataset/processed/clean_dataset.csv"
OUTPUT_FILE = "dataset/processed/final_clean_dataset.csv"


def normalize_unicode(text):
    if pd.isna(text):
        return ""
    return unicodedata.normalize("NFKC", str(text))


def remove_extra_spaces(text):
    return re.sub(r"\s+", " ", text).strip()


def remove_extra_punctuation(text):
    text = re.sub(r"[!]{2,}", "!", text)
    text = re.sub(r"[?]{2,}", "?", text)
    text = re.sub(r"[.]{2,}", ".", text)
    return text


def normalize_repeated_characters(text):
    # loooove -> loove
    return re.sub(r"(.)\1{2,}", r"\1\1", text)


def clean_comment(text):

    text = normalize_unicode(text)
    text = remove_extra_spaces(text)
    text = remove_extra_punctuation(text)
    text = normalize_repeated_characters(text)

    return text


def main():

    print("=" * 60)
    print("Loading Clean Dataset...")
    print("=" * 60)

    df = pd.read_csv(INPUT_FILE)

    df["comment"] = df["comment"].apply(clean_comment)

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nAdvanced Cleaning Completed Successfully.")
    print("Saved :", OUTPUT_FILE)

    print("\nSample Comments")
    print(df["comment"].head())


if __name__ == "__main__":
    main()