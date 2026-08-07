"""
ID-Aaker Top Terms Visualization (Figure 2)
=============================================
Extracts top-10 TF-IDF weighted unigrams per brand personality class
and produces the bar chart grid.

Output: top_terms_per_personality.png

Usage:
    1. Update CSV_PATH below to your local Aaker_Brand_Personality.csv
    2. python visualization_term.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

# ============================================================
# CONFIGURATION
# ============================================================

# TODO: Update to the path of Aaker_Brand_Personality.csv
CSV_PATH = "data/Aaker_Brand_Personality.csv"
N_TOP_TERMS = 10

# ============================================================
# 1. LOAD DATA
# ============================================================
df = pd.read_csv(CSV_PATH)
df["cleaned_text"] = df["cleaned_text"].fillna("")

# ============================================================
# 2. TF-IDF PER CLASS
# ============================================================
personalities = sorted(df["personality"].unique())

fig, axes = plt.subplots(1, len(personalities), figsize=(18, 5))
fig.suptitle("Top 10 TF-IDF Terms per Brand Personality Class", fontsize=14)

for i, personality in enumerate(personalities):
    subset = df[df["personality"] == personality]["cleaned_text"]

    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    tfidf_matrix = tfidf.fit_transform(subset)
    feature_names = np.array(tfidf.get_feature_names_out())

    # Mean TF-IDF score per term across all documents in this class
    mean_scores = np.array(tfidf_matrix.mean(axis=0)).flatten()
    top_indices = mean_scores.argsort()[-N_TOP_TERMS:][::-1]

    top_terms = feature_names[top_indices]
    top_scores = mean_scores[top_indices]

    ax = axes[i]
    sns.barplot(x=top_scores, y=top_terms, ax=ax, palette="viridis",
                hue=top_terms, legend=False)
    ax.set_title(personality.capitalize(), fontsize=12)
    ax.set_xlabel("Mean TF-IDF")

plt.tight_layout()
plt.savefig("top_terms_per_personality.png", dpi=150, bbox_inches="tight")
plt.close()
print("Chart saved: top_terms_per_personality.png")
