"""
====================================================
Women Safety AI Project
Module : Merge newly-supplied raw data into the main raw dataset
File   : merge_new_data.py

The user supplied a new raw file (dataset/raw/Women_Safety_SocialMedia_India_25k.csv,
25,000 rows) that turned out to be 90%+ exact-duplicate text
(only 2,388 unique comments). Of those, all 2,388 are genuinely
new versus the existing Women_Safety_Final_v3_50k.csv, with real
(not synthetic) Telugu / Roman Telugu Threat and Sexual Harassment
examples the old data was missing entirely.

This appends only the deduplicated, genuinely-new rows onto the
original 50k raw file (re-IDed to avoid comment_id collisions,
since both files independently used C000001.. numbering) and
overwrites dataset/raw/Women_Safety_Final_v3_50k.csv in place, so
downstream scripts (preprocessing.py etc.) don't need path changes.
====================================================
"""

import pandas as pd

OLD_FILE = "dataset/raw/Women_Safety_Final_v3_50k.csv"
NEW_FILE = "dataset/raw/Women_Safety_SocialMedia_India_25k.csv"


def main():
    old = pd.read_csv(OLD_FILE)
    new = pd.read_csv(NEW_FILE)

    new_unique = new.drop_duplicates(subset=["comment"])
    old_texts = set(old["comment"])
    genuinely_new = new_unique[~new_unique["comment"].isin(old_texts)].copy()

    start_id = len(old) + 1
    genuinely_new["comment_id"] = [
        f"C{start_id + i:06d}" for i in range(len(genuinely_new))
    ]

    print(f"Old raw rows          : {len(old)}")
    print(f"New file rows         : {len(new)}")
    print(f"New file unique rows  : {len(new_unique)}")
    print(f"Genuinely new rows    : {len(genuinely_new)}")
    print()
    print("Genuinely-new rows by category:")
    print(genuinely_new["category"].value_counts())

    combined = pd.concat([old, genuinely_new], ignore_index=True)

    combined.to_csv(OLD_FILE, index=False, encoding="utf-8-sig")

    print()
    print(f"Combined rows saved -> {OLD_FILE}: {len(combined)} total")
    print()
    print("Combined category distribution:")
    print(combined["category"].value_counts())


if __name__ == "__main__":
    main()
