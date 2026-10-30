# ID-Aaker — Reproducibility Repository

Companion code for the Data in Brief article:

> **ID-Aaker: An Indonesian Social Media Dataset Annotated with Aaker's Five Brand Personality Dimensions**

Mendeley Data: [10.17632/rvfcystzn6](https://data.mendeley.com/datasets/rvfcystzn6) — the DOI resolves to the latest version of the dataset (`Aaker_Brand_Personality.csv`, UTF-8, 8 columns).


---

## Folder Structure

```
ID-Aaker/
│
├── classification/                # Baseline classifier (Table 5)
│   ├── main.py                    #   Training + evaluation (70/15/15 split)
│   ├── finalmodel.py              #   Optional: train on 100% data (TST evaluator)
│   └── check_data.py              #   Quick data quality check
│
├── scripts/                       # Data pipeline & visualizations
│   ├── collect_id_aaker.py        #   X API v2 data collection
│   ├── clean_text.py              #   Anonymization & preprocessing
│   ├── class_distribution.py      #   Figure 1 + Tables 3 and 4
│   └── visualization_term.py      #   Figure 2 (TF-IDF top terms)
│
├── data/                          # Place CSV files here (not tracked)
│   └── (Aaker_Brand_Personality.csv)
│
├── requirements.txt
├── LICENSE                        # CC BY 4.0
└── README.md
```

---

## Quick Start

### 1. Download the Dataset

Download `Aaker_Brand_Personality.csv` from [Mendeley Data](https://data.mendeley.com/datasets/rvfcystzn6) and place it in the `data/` folder:

```
data/
└── Aaker_Brand_Personality.csv
```

### 2. Install Dependencies

All three groups are listed in `requirements.txt`. Install the group you need:

```bash
# statistics & visualizations
pip install pandas==3.0.3 numpy==2.5.1 matplotlib==3.11.0 seaborn==0.13.2 scikit-learn

# classifier
pip install torch transformers datasets accelerate

# data collection (requires X API pay-per-use account)
pip install tweepy python-dotenv
```

### 3. Update CSV Paths

Each script has a `# TODO` marker near the top. Update the path to point to your local copy of the CSV.

### 4. Run

```bash
# Data quality check
python classification/check_data.py

# Figure 1 + Tables 3 and 4
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
| `scripts/class_distribution.py` | Figure 1, Tables 3 and 4 | Data Description section |
| `scripts/visualization_term.py` | Figure 2 | Data Description section |
| `classification/main.py` | Table 5 (baseline benchmark) | Data Description section |
| `classification/finalmodel.py` | Trained model (optional) | For TST downstream use |
| `classification/check_data.py` | Console report | Quick verification |

---

## Script Details

### `scripts/collect_id_aaker.py`

**Purpose:** Collect tweets from 24 Indonesian brand accounts using X API v2.

**Endpoint:** `GET /2/tweets/search/all` (full-archive search)  
**Query:** `from:{username} -is:retweet`  
**Output:** `id_aaker_raw.csv` (14 columns: tweet metadata + personality label)

> **Access & cost:** The full-archive search endpoint is available to all X API pay-per-use developers. At ~$0.005 per post read, reproducing the full 75,756-instance dataset costs approximately $379. The pay-per-use monthly cap is 2 million reads. Register at [developer.x.com](https://developer.x.com) to obtain a Bearer Token.

**Usage:**
```bash
export X_BEARER_TOKEN="your-bearer-token"
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
| Model | `indobenchmark/indobert-large-p1` |
| Split | 70% train / 15% val / 15% test |
| Stratification | Label-stratified (`stratify=df['label']`) |
| Learning rate | 2e-5 |
| Batch size | 4 per device + gradient_accumulation_steps=4 (effective 16) |
| Epochs | 10 |
| Early stopping | patience = 2 |
| Gradient checkpointing | Enabled (for 12 GB VRAM) |
| Loss | Class-weighted CrossEntropyLoss |
| Seed | 42 |

**Output:** `classification_results/IndoBERT-Large_predictions.txt`

### `classification/check_data.py`

**Purpose:** Quick integrity check — row count, columns, missing values, class distribution, language codes, engagement summary.

---

## Dataset Columns

The file is UTF-8 encoded, without a byte-order mark, and holds 75,756 rows.

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

---

## License

The dataset (`Aaker_Brand_Personality.csv`) and the code in this repository are
released under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/)
(CC BY 4.0). The full legal text is in [`LICENSE`](LICENSE).

The post texts were collected through the official X API and anonymized before
release; redistribution of the raw API payload is not permitted under the X
Developer Agreement and Policy, so the repository ships the processed corpus
together with the collection script rather than the raw responses.
