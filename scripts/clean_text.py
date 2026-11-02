"""
ID-Aaker Text Cleaning & Anonymization
=======================================
Reads the raw internal CSV (with PII) and produces the public anonymized CSV.

Input:  RAW_DATA.csv (internal, with PII columns)
Output: Aaker_Brand_Personality.csv (public, 8 columns, fully anonymized)

Input: id_aaker_raw.csv as written by collect_id_aaker.py (14 columns, text in `text`),
       or a legacy raw file that carries the post text in `full_text`.

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

import hashlib

import pandas as pd
import re

# ============================================================
# CONFIGURATION
# ============================================================

# TODO: Update to the path of your internal raw CSV (pre-anonymization)
INPUT_PATH = "data/id_aaker_raw.csv"

# TODO: Update to desired output path
OUTPUT_PATH = "data/Aaker_Brand_Personality.csv"

# Fixed salt for the pseudonym mapping. Changing it changes every pseudonym,
# so keep it as is to reproduce the released file.
PSEUDONYM_SALT = "id-aaker-v1"

# ============================================================
# 1. LOAD & DROP PII COLUMNS
# ============================================================
df = pd.read_csv(INPUT_PATH)

columns_to_drop = [
    # names written by collect_id_aaker.py
    "tweet_id",
    "author_username",
    "author_id",
    "created_at",
    "conversation_id",
    "in_reply_to_user_id",
    "is_reply",
    # names used by the earlier raw export
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

    # Replace @username with @user_NNNN. The mapping is a salted SHA-256 of the
    # username, so the same mention always receives the same pseudonym and two
    # runs of this script produce identical files. The released file was built
    # from a different draw, so a re-run gives equivalent pseudonyms in the same
    # @user_NNNN form, not the identical strings. Python's built-in hash() is
    # randomised per process and must not be used here.
    mentions = set(re.findall(r"@\w+", text))

    mention_map = {}
    for m in mentions:
        digest = hashlib.sha256((PSEUDONYM_SALT + m).encode("utf-8")).hexdigest()
        mention_map[m] = f"@user_{int(digest[:8], 16) % 10000:04d}"

    for original, pseudo in mention_map.items():
        text = text.replace(original, pseudo)

    # Replace URLs
    text = re.sub(r"https?://\S+", "https://dummy-link.com/ref", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# The collector writes the post text to `text`; the earlier raw export used
# `full_text`. Accept either so the two scripts chain without editing.
text_column = "full_text" if "full_text" in df.columns else "text"
if text_column not in df.columns:
    raise SystemExit(
        "No text column found in the input file: expected 'text' (collect_id_aaker.py) "
        "or 'full_text' (legacy raw export)."
    )

df["cleaned_text"] = df[text_column].apply(clean_text_generative)

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
