# balancing module
"""
====================================================
Women Safety AI Project
Module 1 : Dataset Balancing
File      : balance_dataset.py
====================================================
"""

import os
import pandas as pd

INPUT_FILE = "dataset/processed/encoded_dataset.csv"
OUTPUT_FILE = "dataset/processed/balanced_dataset.csv"

# Target number of samples per class
TARGET_PER_CLASS = 5000


def balance_dataset(df):

    balanced_data = []

    print("\nOriginal Dataset Distribution\n")
    print(df["label"].value_counts().sort_index())

    labels = sorted(df["label"].dropna().unique())

    print("\nBalancing Dataset...\n")

    for label in labels:

        class_df = df[df["label"] == label]

        print(f"Processing Label {label}")

        # If enough records → Random Sampling
        if len(class_df) >= TARGET_PER_CLASS:

            sampled = class_df.sample(
                n=TARGET_PER_CLASS,
                random_state=42
            )

        # If fewer records → Oversampling
        else:

            sampled = class_df.sample(
                n=TARGET_PER_CLASS,
                replace=True,
                random_state=42
            )

        balanced_data.append(sampled)

    balanced_df = pd.concat(
        balanced_data,
        ignore_index=True
    )

    # Shuffle dataset
    balanced_df = balanced_df.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    return balanced_df


def main():

    print("=" * 60)
    print("Loading Encoded Dataset")
    print("=" * 60)

    if not os.path.exists(INPUT_FILE):

        print("ERROR : encoded_dataset.csv not found")
        return

    df = pd.read_csv(INPUT_FILE)

    print(f"\nTotal Rows : {len(df)}")

    balanced_df = balance_dataset(df)

    balanced_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nBalanced Dataset Saved Successfully")
    print(OUTPUT_FILE)

    print("\nBalanced Dataset Distribution\n")
    print(balanced_df["label"].value_counts().sort_index())

    print("\nTotal Rows :", len(balanced_df))

    print("=" * 60)


if __name__ == "__main__":
    main()