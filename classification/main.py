"""
ID-Aaker Brand Personality Classification -- Baseline Benchmark
===============================================================
Reproducible single-model baseline using IndoBERT-Large.
Matches the configuration reported in the Data in Brief manuscript:
  - Model:  "indobenchmark/indobert-large-p1"
  - Split:  70% train, 15% validation, 15% test (label-stratified)
  - LR:     2e-5
  - Batch:  4 per device + gradient_accumulation_steps=4 (effective 16)
  - Epochs: 10 (early stopping patience = 2)
  - Loss:   Class-weighted CrossEntropyLoss
  - Seed:   42
  - GPU:    ~12 GB VRAM required (uses gradient checkpointing)

Requirements:
    pip install torch transformers datasets scikit-learn pandas

Usage:
    1. Update CSV_PATH below to point to your local Aaker_Brand_Personality.csv
    2. python main.py

Output:
    classification_results/IndoBERT-Large_predictions.txt
"""

import os
import gc
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    set_seed,
    EvalPrediction,
)

# ============================================================
# CONFIGURATION
# ============================================================

# TODO: Update this path to your local copy of Aaker_Brand_Personality.csv
# The CSV can be downloaded from Mendeley Data: https://data.mendeley.com/datasets/rvfcystzn6/1
CSV_PATH = "data/Aaker_Brand_Personality.csv"

MODEL_ID = "indobenchmark/indobert-large-p1"
OUTPUT_DIR = "./classification_results"
SEED = 42

# ============================================================
# SETUP
# ============================================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
set_seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. LOAD & PREPROCESS
# ============================================================
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"CSV file not found at '{CSV_PATH}'.\n"
        "Download from https://data.mendeley.com/datasets/rvfcystzn6/1 "
        "and update CSV_PATH at the top of this script."
    )

df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["cleaned_text", "personality"])
df["cleaned_text"] = df["cleaned_text"].astype(str)

label2id = {label: i for i, label in enumerate(sorted(df["personality"].unique()))}
id2label = {i: label for label, i in label2id.items()}
df["label"] = df["personality"].map(label2id)
num_classes = len(label2id)

classes = np.unique(df["label"])
class_weights = torch.tensor(
    compute_class_weight("balanced", classes=classes, y=df["label"].values),
    dtype=torch.float32,
)

print(f"Classes: {num_classes}")
print(f"Distribution:\n{df['personality'].value_counts()}")
print(f"Class weights: {dict(zip(label2id.keys(), class_weights.tolist()))}")

# ============================================================
# 2. TRAIN / VAL / TEST SPLIT (70 / 15 / 15, label-stratified)
# ============================================================
train_df, temp_df = train_test_split(
    df, test_size=0.3, stratify=df["label"], random_state=SEED
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df["label"], random_state=SEED
)

train_dataset = Dataset.from_pandas(train_df[["cleaned_text", "label"]])
val_dataset = Dataset.from_pandas(val_df[["cleaned_text", "label"]])
test_dataset = Dataset.from_pandas(test_df[["cleaned_text", "label"]])

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# ============================================================
# 3. CUSTOM TRAINER WITH CLASS WEIGHTS
# ============================================================
class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(outputs.logits.device)
        )
        loss = loss_fct(
            outputs.logits.view(-1, self.model.config.num_labels), labels.view(-1)
        )
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred: EvalPrediction):
    predictions = eval_pred.predictions
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    preds = np.argmax(predictions, axis=-1)
    return {"macro_f1": f1_score(eval_pred.label_ids, preds, average="macro")}


# ============================================================
# 4. TOKENIZE & TRAIN
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def tokenize(examples):
    return tokenizer(examples["cleaned_text"], truncation=True, max_length=128)


train_tok = train_dataset.map(tokenize, batched=True, remove_columns=["cleaned_text"])
val_tok = val_dataset.map(tokenize, batched=True, remove_columns=["cleaned_text"])
test_tok = test_dataset.map(tokenize, batched=True, remove_columns=["cleaned_text"])

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    num_labels=num_classes,
    id2label=id2label,
    label2id=label2id,
)

# Enable gradient checkpointing for large model memory efficiency
if hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable()

# IndoBERT-Large: per_device_batch_size=4 + gradient_accumulation_steps=4
# gives an effective batch size of 16 (matches manuscript)
training_args = TrainingArguments(
    output_dir="./checkpoints",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=64,
    gradient_accumulation_steps=4,
    num_train_epochs=10,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    logging_dir="./logs",
    logging_steps=50,
    seed=SEED,
    fp16=torch.cuda.is_available(),
    gradient_checkpointing=True,
    report_to="none",
    remove_unused_columns=False,
    dataloader_num_workers=0,
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=val_tok,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    class_weights=class_weights,
)

# ============================================================
# 5. TRAIN & EVALUATE
# ============================================================
print("\nStarting training ...")
trainer.train()

print("\nEvaluating on test set ...")
pred_output = trainer.predict(test_tok)
macro_f1 = pred_output.metrics.get("test_macro_f1", 0)

logits = pred_output.predictions
if isinstance(logits, tuple):
    logits = logits[0]

true_labels = pred_output.label_ids
preds = np.argmax(logits, axis=-1)

report = classification_report(
    true_labels, preds, target_names=[id2label[i] for i in range(num_classes)]
)

print(f"\nTest Macro-F1: {macro_f1:.4f}")
print(report)

# ============================================================
# 6. EXPORT
# ============================================================
probs = torch.softmax(torch.tensor(logits).cpu(), dim=-1).numpy()
confidences = np.max(probs, axis=-1)

export_df = pd.DataFrame(
    {
        "True_Label": [id2label[t] for t in true_labels],
        "Predicted_Label": [id2label[p] for p in preds],
        "Confidence": np.round(confidences, 4),
        "Is_Correct": [
            id2label[p] == id2label[t] for p, t in zip(preds, true_labels)
        ],
        "Cleaned_Text": test_df["cleaned_text"].values,
    }
)

out_path = os.path.join(OUTPUT_DIR, "IndoBERT-Large_predictions.txt")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(f"Model: IndoBERT-Large\n")
    f.write(f"Model ID: {MODEL_ID}\n")
    f.write(f"Test Macro-F1: {macro_f1:.4f}\n")
    f.write("=" * 70 + "\n")
    f.write("CLASSIFICATION REPORT:\n")
    f.write(report + "\n")
    f.write("=" * 70 + "\n")
    f.write("DETAILED PREDICTIONS (Tab-Separated):\n")
    export_df.to_csv(f, sep="\t", index=False)

print(f"\nResults saved to: {out_path}")
