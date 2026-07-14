# preprocessing module
"""
====================================================
Women Safety AI Project
Module 1 : Dataset Preprocessing
Author : Krishna
====================================================
"""

import os
import re
import pandas as pd


class DatasetPreprocessor:

    def __init__(self, input_file, output_folder):

        self.input_file = input_file
        self.output_folder = output_folder

        os.makedirs(self.output_folder, exist_ok=True)

    # -------------------------------
    # Load Dataset
    # -------------------------------
    def load_dataset(self):

        print("=" * 60)
        print("Loading Dataset...")
        print("=" * 60)

        self.df = pd.read_csv(self.input_file)

        print("Dataset Loaded Successfully")
        print(f"Rows    : {len(self.df)}")
        print(f"Columns : {len(self.df.columns)}")

    # -------------------------------
    # Remove Duplicate Rows
    # -------------------------------
    def remove_duplicates(self):

        before = len(self.df)

        self.df.drop_duplicates(inplace=True)

        after = len(self.df)

        print(f"Duplicates Removed : {before-after}")

    # -------------------------------
    # Remove Missing Comments
    # -------------------------------
    def remove_missing_comments(self):

        before = len(self.df)

        self.df.dropna(subset=["comment"], inplace=True)

        after = len(self.df)

        print(f"Missing Comments Removed : {before-after}")

    # -------------------------------
    # Clean Comment
    # -------------------------------
    def clean_comment(self, text):

        if pd.isna(text):
            return ""

        text = str(text)

        # Remove URLs
        text = re.sub(r"http\\S+", "", text)

        # Remove HTML
        text = re.sub(r"<.*?>", "", text)

        # Remove @username
        text = re.sub(r"@\\w+", "", text)

        # Remove #
        text = text.replace("#", "")

        # Remove extra spaces
        text = re.sub(r"\\s+", " ", text)

        return text.strip()

    # -------------------------------
    # Apply Cleaning
    # -------------------------------
    def preprocess_comments(self):

        print("Cleaning Comments...")

        self.df["comment"] = self.df["comment"].apply(
            self.clean_comment
        )

    # -------------------------------
    # Fill Translation Column
    # -------------------------------
    def fill_translation(self):

        if "english_translation" in self.df.columns:

            self.df["english_translation"] = self.df[
                "english_translation"
            ].fillna("")

    # -------------------------------
    # Save Dataset
    # -------------------------------
    def save_dataset(self):

        output_file = os.path.join(

            self.output_folder,

            "clean_dataset.csv"

        )

        self.df.to_csv(

            output_file,

            index=False,

            encoding="utf-8-sig"

        )

        print("=" * 60)
        print("Dataset Saved Successfully")
        print(output_file)
        print("=" * 60)

    # -------------------------------
    # Statistics
    # -------------------------------
    def show_statistics(self):

        print("\n")

        print("=" * 60)

        print("DATASET STATISTICS")

        print("=" * 60)

        print("Total Rows :", len(self.df))

        print("\nLanguages")

        print(self.df["language"].value_counts())

        print("\nCategories")

        print(self.df["category"].value_counts())

        print("\nPlatforms")

        print(self.df["platform"].value_counts())

        print("=" * 60)

    # -------------------------------
    # Run Complete Pipeline
    # -------------------------------
    def run(self):

        self.load_dataset()

        self.remove_duplicates()

        self.remove_missing_comments()

        self.preprocess_comments()

        self.fill_translation()

        self.show_statistics()

        self.save_dataset()


# ===========================================
# MAIN
# ===========================================

if __name__ == "__main__":

    INPUT_FILE = "dataset/raw/Women_Safety_Final_v3_50k.csv"

    OUTPUT_FOLDER = "dataset/processed"

    processor = DatasetPreprocessor(

        INPUT_FILE,

        OUTPUT_FOLDER

    )

    processor.run()