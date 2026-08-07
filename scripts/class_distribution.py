"""
ID-Aaker Class Distribution & Descriptive Statistics
=====================================================
Generates the class distribution bar chart and descriptive
statistics from the public ID-Aaker CSV.

Output:
  - Bar chart: class_distribution.png
  - Console: descriptive statistics for word count and engagement metrics

Usage:
    1. Update CSV_PATH below to point to your local Aaker_Brand_Personality.csv
    2. python class_distribution.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# CONFIGURATION
# ============================================================

# TODO: Update to the path of Aaker_Brand_Personality.csv
CSV_PATH = "data/Aaker_Brand_Personality.csv"

# ============================================================
# 1. LOAD DATA
# ============================================================
df = pd.read_csv(CSV_PATH)

# ============================================================
# 2. CLASS DISTRIBUTION (Table 2)
# ============================================================
class_counts = df["personality"].value_counts()
class_counts.index = class_counts.index.str.capitalize()

print("=== Class Distribution ===")
print(class_counts)
print()

# ============================================================
# 3. WORD COUNT STATISTICS (Table 2)
# ============================================================
print("=== Word Count Statistics ===")
for p in ["competence", "sincerity", "sophistication", "excitement", "ruggedness"]:
    subset = df[df["personality"] == p]["word_count"]
    print(
        f"  {p.capitalize():<16} count={len(subset):>6,}  "
        f"mean={subset.mean():.2f}  min={subset.min()}  "
        f"max={subset.max()}  std={subset.std():.2f}"
    )

# ============================================================
# 4. ENGAGEMENT METRICS (Table 2b)
# ============================================================
metrics = ["favorite_count", "retweet_count", "reply_count", "quote_count"]
print("\n=== Engagement Metrics ===")
for p in ["competence", "sincerity", "sophistication", "excitement", "ruggedness"]:
    subset = df[df["personality"] == p]
    print(f"\n  {p.capitalize()}:")
    for m in metrics:
        vals = subset[m]
        print(f"    {m:<20} mean={vals.mean():>8.1f}  max={vals.max():>8,}")

# ============================================================
# 5. BAR CHART (Figure 1)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(
    x=class_counts.index,
    y=class_counts.values,
    ax=ax,
    palette="viridis",
    hue=class_counts.index,
    legend=False,
)
for p in ax.patches:
    ax.annotate(
        format(int(p.get_height()), ","),
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="center",
        xytext=(0, 5),
        textcoords="offset points",
    )
ax.set_title("Distribution of Aaker Brand Personality Classes", fontsize=14)
ax.set_xlabel("Personality Class")
ax.set_ylabel("Number of Instances")
plt.tight_layout()
plt.savefig("class_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nChart saved: class_distribution.png")
