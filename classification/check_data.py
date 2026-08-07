"""
ID-Aaker Data Quality Check
=============================
Quick integrity check: column names, row count, class distribution,
missing values, language codes.

Usage: python check_data.py
"""

import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

# TODO: Update to the path of Aaker_Brand_Personality.csv
CSV_PATH = "data/Aaker_Brand_Personality.csv"

# ============================================================
df = pd.read_csv(CSV_PATH)

print("=== File Info ===")
print(f"Rows: {len(df):,}")
print(f"Columns ({len(df.columns)}): {list(df.columns)}")

print("\n=== Missing Values ===")
print(df.isnull().sum())

print("\n=== Class Distribution ===")
print(df["personality"].value_counts())

print("\n=== Language Distribution ===")
print(df["lang"].value_counts().head(10))

print("\n=== Word Count Summary ===")
print(df["word_count"].describe())

print("\n=== Engagement Summary ===")
for col in ["favorite_count", "retweet_count", "reply_count", "quote_count"]:
    print(f"  {col}: mean={df[col].mean():.1f}, max={df[col].max():,}")

print("\n✅ Check complete.")
