"""
====================================================
Women Safety AI Project
Module : Add regional-language data to Threat / Sexual
         Harassment classes (train split only)
File   : add_regional_data.py

Appends the Telugu / Tamil / Kannada seed sentences from
regional_seed_data.py to dataset/processed/train_raw.csv (each
English source line contributes 4 rows: english + 3 regional
translations). Run this BEFORE balance_dataset.py so the added
rows get mixed into the oversampling pool for Threat and Sexual
Harassment instead of relying only on English text-augmentation.

Validation/test are intentionally left untouched - they must stay
organic (real raw data only) so evaluation reflects true
generalization, not synthetic content.
====================================================
"""

import pandas as pd

from regional_seed_data import THREAT_SAMPLES, SEXUAL_HARASSMENT_SAMPLES

TRAIN_RAW_FILE = "dataset/processed/train_raw.csv"

LABEL_MAP = {
    "Threat": 3,
    "Sexual Harassment": 2,
}

LANG_SCRIPT = {
    "English": "Latin",
    "Telugu": "Telugu",
    "Tamil": "Tamil",
    "Kannada": "Kannada",
}


def build_rows(samples, category, start_id):
    rows = []
    cid = start_id
    for english, telugu, tamil, kannada in samples:
        for lang, text in (
            ("English", english),
            ("Telugu", telugu),
            ("Tamil", tamil),
            ("Kannada", kannada),
        ):
            rows.append({
                "comment_id": f"SYN{cid:06d}",
                "platform": "Synthetic",
                "platform_original": "Synthetic",
                "comment": text,
                "language": lang,
                "script": LANG_SCRIPT[lang],
                "english_translation": english,
                "translation_status": "synthetic_seed",
                "category": category,
                "category_original": category,
                "label": LABEL_MAP[category],
                "severity": "High" if category == "Threat" else "Medium",
                "toxicity_score": 0.9 if category == "Threat" else 0.7,
                "sentiment": "Negative",
            })
            cid += 1
    return rows, cid


def main():
    df = pd.read_csv(TRAIN_RAW_FILE)

    next_id = 900000
    threat_rows, next_id = build_rows(THREAT_SAMPLES, "Threat", next_id)
    sh_rows, next_id = build_rows(SEXUAL_HARASSMENT_SAMPLES, "Sexual Harassment", next_id)

    new_df = pd.DataFrame(threat_rows + sh_rows)

    print(f"Existing train_raw rows : {len(df)}")
    print(f"New synthetic rows      : {len(new_df)}")
    print(new_df.groupby(["category", "language"]).size())

    combined = pd.concat([df, new_df], ignore_index=True)
    combined.to_csv(TRAIN_RAW_FILE, index=False, encoding="utf-8-sig")

    print(f"\nUpdated train_raw.csv saved -> {TRAIN_RAW_FILE}")
    print(f"Total rows now : {len(combined)}")
    print(combined["category"].value_counts())


if __name__ == "__main__":
    main()
