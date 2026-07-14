# label encoding module
"""
====================================================
Women Safety AI Project
Module 1 : Label Encoder
File      : label_encoder.py
====================================================
"""

import pandas as pd

# -------------------------------------------------
# Input / Output
# -------------------------------------------------

INPUT_FILE = "dataset/processed/final_clean_dataset.csv"
OUTPUT_FILE = "dataset/processed/encoded_dataset.csv"

# -------------------------------------------------
# Label Mapping
# -------------------------------------------------

# Must match ai_model/config.py LABEL_MAP. Only 5 classes are mapped here
# because dataset/raw/Women_Safety_Final_v3_50k.csv has zero labeled
# examples for "Cyber Bullying" or "Stalking" - add them back to both
# files once real labeled data for them exists in the raw dataset.
LABEL_MAPPING = {
    "Safe": 0,
    "Offensive": 1,
    "Sexual Harassment": 2,
    "Threat": 3,
    "Hate Speech": 4
}


def main():

    print("=" * 60)
    print("Loading Dataset...")
    print("=" * 60)

    df = pd.read_csv(INPUT_FILE)

    print("Dataset Loaded Successfully")
    print("Rows :", len(df))

    print("\nEncoding Labels...")

    # Check unknown categories
    unknown = set(df["category"]) - set(LABEL_MAPPING.keys())

    if len(unknown) > 0:
        print("\nUnknown Categories Found:")
        print(unknown)

    # Create numeric label column
    df["label"] = df["category"].map(LABEL_MAPPING)

    # Check missing labels
    missing = df["label"].isna().sum()

    print("Missing Labels :", missing)

    # Convert to integer
    df["label"] = df["label"].astype("Int64")

    # Save
    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nEncoded Dataset Saved Successfully")
    print(OUTPUT_FILE)

    print("\nLabel Distribution")

    print(df["label"].value_counts().sort_index())

    print("\nLabel Mapping")

    for key, value in LABEL_MAPPING.items():
        print(f"{value} -> {key}")

    print("=" * 60)


if __name__ == "__main__":
    main()