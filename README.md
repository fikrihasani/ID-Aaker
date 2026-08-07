# ID-Aaker — Reproducibility Repository

Companion code for the Data in Brief article:

> **ID-Aaker: An Indonesian Social Media Dataset Annotated with Aaker's Five Brand Personality Dimensions**

Mendeley Data: [10.17632/rvfcystzn6.1](https://data.mendeley.com/datasets/rvfcystzn6/1)

---

## Folder Structure

```
ID-Aaker/
│
├── classification/                # Baseline classifier (Table 3)
│   ├── main.py                    #   Training + evaluation (70/15/15 split)
│   ├── finalmodel.py              #   Optional: train on 100% data (TST evaluator)
│   └── check_data.py              #   Quick data quality check
│
├── scripts/                       # Data pipeline & visualizations
│   ├── collect_id_aaker.py        #   X API v2 data collection
│   ├── clean_text.py              #   Anonymization & preprocessing
│   ├── class_distribution.py      #   Figure 1 + Table 2 & 2b
│   └── visualization_term.py      #   Figure 2 (TF-IDF top terms)
│
├── data/                          # Place CSV files here (not tracked)
│   └── (Aaker_Brand_Personality.csv)
│
└── README.md
```

---

## Quick Start

### 1. Download the Dataset

Download `Aaker_Brand_Personality.csv` from [Mendeley Data](https://data.mendeley.com/datasets/rvfcystzn6/1) and place it in the `data/` folder:

```
data/
└── Aaker_Brand_Personality.csv
```

### 2. Install Dependencies

```bash
pip install pandas matplotlib seaborn scikit-learn
```

For the classifier, also install:

```bash
pip install torch transformers datasets
```

For data collection, also install:

```bash
pip install tweepy python-dotenv
```

### 3. Update CSV Paths

Each script has a `# TODO` marker near the top. Update the path to point to your local copy of the CSV.

### 4. Run

```bash
# Data quality check
python classification/check_data.py

# Figure 1 + Table 2 & 2b
python scripts/class_distribution.py

# Figure 2
python scripts/visualization_term.py

# Baseline classifier (requires GPU recommended)
python classification/main.py
```

---

## Scripts — Manuscript Correspondence

| Script | Produces | Manuscript Reference |
|--------|----------|---------------------|
| `scripts/collect_id_aaker.py` | Raw CSV (14 columns) | Data Acquisition section |
| `scripts/clean_text.py` | `Aaker_Brand_Personality.csv` (8 columns) | Data Preprocessing section |
| `scripts/class_distribution.py` | Figure 1, Table 2, Table 2b | Data Description section |
| `scripts/visualization_term.py` | Figure 2 | Data Description section |
| `classification/main.py` | Table 3 (baseline benchmark) | Data Description section |
| `classification/finalmodel.py` | Trained model (optional) | For TST downstream use |
| `classification/check_data.py` | Console report | Quick verification |

---

## Script Details

### `scripts/collect_id_aaker.py`

**Purpose:** Collect tweets from 24 Indonesian brand accounts using X API v2.

**Requires:** X API Pro-tier Bearer Token (pay-per-use, ~$0.005/post read).

**Endpoint:** `GET /2/tweets/search/all`  
**Query:** `from:{username} -is:retweet`  
**Output:** `id_aaker_raw.csv` (14 columns: tweet metadata + personality label)

**Usage:**
```bash
export X_BEARER_TOKEN="your-token-here"
python scripts/collect_id_aaker.py
```

### `scripts/clean_text.py`

**Purpose:** Anonymize raw CSV → public CSV.

**Input:** Internal raw CSV (with PII columns: `username`, `user_id_str`, etc.)  
**Output:** `Aaker_Brand_Personality.csv` (8 columns, no PII)

**Steps:**
1. Drop PII columns (`user_id_str`, `username`, `tweet_url`, `image_url`, etc.)
2. Replace `@username` mentions → `@user_NNNN` (deterministic pseudonyms)
3. Replace all URLs → `https://dummy-link.com/ref`
4. Compute `word_count`

### `scripts/class_distribution.py`

**Purpose:** Descriptive statistics for the dataset.

**Output:**
- Console: class counts, word count stats, engagement metrics
- `class_distribution.png`: Bar chart of class distribution (Figure 1)

### `scripts/visualization_term.py`

**Purpose:** TF-IDF top terms per personality class.

**Output:** `top_terms_per_personality.png` (Figure 2)

### `classification/main.py`

**Purpose:** Reproduce the baseline classification benchmark.

**Configuration (matches manuscript):**

| Parameter | Value |
|-----------|-------|
| Model | `flax-community/indonesian-roberta-base` |
| Split | 70% train / 15% val / 15% test |
| Stratification | Label-stratified (`stratify=df['label']`) |
| Learning rate | 2e-5 |
| Batch size | 32 |
| Epochs | 10 |
| Early stopping | patience = 2 |
| Loss | Class-weighted CrossEntropyLoss |
| Seed | 42 |

**Output:** `classification_results/Indonesian-RoBERTa-Base_predictions.txt`

### `classification/finalmodel.py` (optional)

**Purpose:** Train a classifier on 100% of the data for use as an evaluator in Text Style Transfer experiments. No validation/test split — trains on the full dataset.

### `classification/check_data.py`

**Purpose:** Quick integrity check — row count, columns, missing values, class distribution, language codes, engagement summary.

---

## Dataset Columns

| Column | Type | Description |
|--------|------|-------------|
| `favorite_count` | int | Number of likes |
| `lang` | string | X API language code (mostly `in`) |
| `quote_count` | int | Number of quote tweets |
| `reply_count` | int | Number of replies |
| `retweet_count` | int | Number of retweets |
| `personality` | string | Aaker dimension (competence/sincerity/sophistication/excitement/ruggedness) |
| `cleaned_text` | string | Pre-processed, anonymized text |
| `word_count` | int | Number of words in `cleaned_text` |

---

## Citation

If you use the ID-Aaker dataset or this code, please cite:

```bibtex
@article{hasani2026idaaker,
  title   = {ID-Aaker: An Indonesian Social Media Dataset Annotated with
             Aaker's Five Brand Personality Dimensions},
  author  = {Hasani, Muhammad Fikri and Isa, Sani Muhammad and
             Arifin, Yulyani and Budiharto, Widodo},
  journal = {Data in Brief},
  year    = {2026},
  doi     = {10.17632/rvfcystzn6.1}
}
```

---

## License

Dataset: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)  
Code: MIT
