# train/validation/test split
"""
====================================================
Women Safety AI Project
Module 1 : Train / Validation / Test Split
File      : split_dataset.py
====================================================
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

# -------------------------------------------------
# Input / Output
# -------------------------------------------------

INPUT_FILE = "dataset/processed/balanced_dataset.csv"
OUTPUT_FOLDER = "dataset/processed"

TRAIN_FILE = os.path.join(OUTPUT_FOLDER, "train.csv")
VALIDATION_FILE = os.path.join(OUTPUT_FOLDER, "validation.csv")
TEST_FILE = os.path.join(OUTPUT_FOLDER, "test.csv")


def main():

    print("=" * 60)
    print("Loading Balanced Dataset...")
    print("=" * 60)

    if not os.path.exists(INPUT_FILE):
        print("ERROR : balanced_dataset.csv not found")
        return

    df = pd.read_csv(INPUT_FILE)

    print(f"Total Records : {len(df)}")

    print("\nSplitting Dataset...")

    # 70% Train
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        stratify=df["label"]
    )

    # Remaining 30% -> 15% Validation + 15% Test
    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["label"]
    )

    train_df.to_csv(TRAIN_FILE, index=False, encoding="utf-8-sig")
    validation_df.to_csv(VALIDATION_FILE, index=False, encoding="utf-8-sig")
    test_df.to_csv(TEST_FILE, index=False, encoding="utf-8-sig")

    print("\nDataset Split Completed Successfully\n")

    print(f"Train Records      : {len(train_df)}")
    print(f"Validation Records : {len(validation_df)}")
    print(f"Test Records       : {len(test_df)}")

    print("\nFiles Created")
    print("-------------------------")
    print(TRAIN_FILE)
    print(VALIDATION_FILE)
    print(TEST_FILE)

    print("=" * 60)


if __name__ == "__main__":
    main()