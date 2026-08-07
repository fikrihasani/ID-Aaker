"""
ID-Aaker Text Cleaning & Anonymization
=======================================
Reads the raw internal CSV (with PII) and produces the public anonymized CSV.

Input:  RAW_DATA.csv (internal, with PII columns)
Output: Aaker_Brand_Personality.csv (public, 8 columns, fully anonymized)

The script:
  1. Drops PII columns (user IDs, usernames, tweet URLs, image URLs, etc.)
  2. Replaces @username mentions with random pseudonyms (e.g., @user_4921)
  3. Replaces all URLs with a uniform dummy link (https://dummy-link.com/ref)
  4. Computes word_count from the cleaned text
  5. Retains engagement metrics and language code from the raw data

Usage:
    1. Update INPUT_PATH below to point to the internal raw CSV
    2. Update OUTPUT_PATH below for the output location
    3. python clean_text.py
"""

import pandas as pd
import re
import random

# ============================================================
# CONFIGURATION
# ============================================================

# TODO: Update to the path of your internal raw CSV (pre-anonymization)
INPUT_PATH = "data/RAW_DATA.csv"

# TODO: Update to desired output path
OUTPUT_PATH = "data/Aaker_Brand_Personality.csv"

# ============================================================
# 1. LOAD & DROP PII COLUMNS
# ============================================================
df = pd.read_csv(INPUT_PATH)

columns_to_drop = [
    "conversation_id_str",
    "id_str",
    "image_url",
    "in_reply_to_screen_name",
    "tweet_url",
    "user_id_str",
    "username",
]
df = df.drop(columns=columns_to_drop, errors="ignore")

# ============================================================
# 2. ANONYMIZE
# ============================================================
def clean_text_generative(text):
    """Replace @mentions with random pseudonyms and URLs with dummy links."""
    if not isinstance(text, str):
        return ""

    # Replace @username with @user_NNNN (consistent per unique username)
    mentions = set(re.findall(r"@\w+", text))

    def _replacer(match):
        username = match.group(0)
        # Deterministic pseudonym based on username hash
        seed = hash(username) % 10000
        return f"@user_{abs(seed):04d}"

    # Use a deterministic mapping for reproducibility
    mention_map = {}
    for m in mentions:
        seed = abs(hash(m)) % 10000
        mention_map[m] = f"@user_{seed}"

    for original, pseudo in mention_map.items():
        text = text.replace(original, pseudo)

    # Replace URLs
    text = re.sub(r"https?://\S+", "https://dummy-link.com/ref", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


df["cleaned_text"] = df["full_text"].apply(clean_text_generative)

# ============================================================
# 3. COMPUTE WORD COUNT
# ============================================================
df["word_count"] = df["cleaned_text"].apply(lambda x: len(x.split()))

# ============================================================
# 4. SELECT FINAL COLUMNS & EXPORT
# ============================================================
output_columns = [
    "favorite_count",
    "lang",
    "quote_count",
    "reply_count",
    "retweet_count",
    "personality",
    "cleaned_text",
    "word_count",
]

df[output_columns].to_csv(OUTPUT_PATH, index=False)
print(f"Anonymized CSV saved to: {OUTPUT_PATH}")
print(f"Rows: {len(df)}, Columns: {len(output_columns)}")
