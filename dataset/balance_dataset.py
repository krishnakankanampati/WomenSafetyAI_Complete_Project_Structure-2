# balancing module
"""
====================================================
Women Safety AI Project
Module 1 : Dataset Balancing
File      : balance_dataset.py

Balances ONLY the train split (train_raw.csv, produced by
split_dataset.py). Validation/test are left at their natural,
already-split, duplicate-free distribution so evaluation reflects
real generalization instead of memorized oversampled duplicates.

Forcing every class down to one flat target (the original approach)
discarded most of the real data for large classes - e.g. Safe had
20,418 real rows and was cut to 3,500, throwing away 83% of real,
organic "Safe" diversity. Diagnostics on the trained model showed
this mattered: ~31% of real Safe test comments were misclassified
as Offensive/Hate Speech, and 95.7% of those misclassifications had
no vulgar/profane word in them at all - the model simply hadn't seen
enough real Safe examples to generalize, not a labeling problem.

So classes with enough real data now keep up to MAJORITY_CAP rows
(instead of being cut to the same small floor as the smallest
class), and only classes below MINORITY_FLOOR get oversampled with
augmentation. The residual imbalance is handled by class-weighted
loss (see ai_model/model.py) instead of by discarding data.
====================================================
"""

import os
import pandas as pd

from text_augment import augment_text

INPUT_FILE = "dataset/processed/train_raw.csv"
OUTPUT_FILE = "dataset/processed/train.csv"

# Classes with more real rows than this are subsampled down to it
# (still far more than the old flat 3500 target for big classes).
MAJORITY_CAP = 8000

# Classes with fewer real rows than this are oversampled (with
# text augmentation, not exact duplication) up to this floor.
MINORITY_FLOOR = 2500


def balance_dataset(df):

    balanced_data = []

    print("\nOriginal Dataset Distribution\n")
    print(df["label"].value_counts().sort_index())

    labels = sorted(df["label"].dropna().unique())

    print("\nBalancing Dataset...\n")

    for label in labels:

        class_df = df[df["label"] == label]

        print(f"Processing Label {label} ({len(class_df)} real rows)")

        if len(class_df) > MAJORITY_CAP:

            sampled = class_df.sample(
                n=MAJORITY_CAP,
                random_state=42
            )

        elif len(class_df) < MINORITY_FLOOR:

            # Keep every real row once, then fill the remainder with
            # augmented (not exact-duplicate) variants of randomly
            # resampled rows, so the model sees lexical variety
            # instead of memorizing repeated identical text.
            deficit = MINORITY_FLOOR - len(class_df)

            extra = class_df.sample(
                n=deficit,
                replace=True,
                random_state=42
            ).copy()

            extra["comment"] = [
                augment_text(text, seed=42 + i)
                for i, text in enumerate(extra["comment"])
            ]

            sampled = pd.concat([class_df, extra], ignore_index=True)

        else:

            # Between the floor and the cap - keep every real row,
            # no subsampling or oversampling needed.
            sampled = class_df

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

        print("ERROR : train_raw.csv not found. Run split_dataset.py first.")
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