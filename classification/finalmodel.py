"""
ID-Aaker Final Evaluator Model (Optional)
===========================================
Trains IndoBERT-Large on 100% of the data for downstream use
(e.g., as an evaluator in Text Style Transfer experiments).
This is NOT the baseline benchmark -- use main.py for that.

Configuration matches main.py where applicable.
Note: No validation split (trains on full dataset).

Usage:
    1. Update CSV_PATH below
    2. python finalmodel.py
"""

import os
import gc
import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

# TODO: Update to the path of Aaker_Brand_Personality.csv
CSV_PATH = "data/Aaker_Brand_Personality.csv"

MODEL_ID = "indobenchmark/indobert-large-p1"
FINAL_MODEL_DIR = "./final_evaluator_indobert_large"
SEED = 42

# ============================================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
set_seed(SEED)
os.makedirs(FINAL_MODEL_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["cleaned_text", "personality"])
df["cleaned_text"] = df["cleaned_text"].astype(str)

label2id = {label: i for i, label in enumerate(sorted(df["personality"].unique()))}
id2label = {i: label for label, i in label2id.items()}
df["label"] = df["personality"].map(label2id)

classes = np.unique(df["label"])
class_weights = torch.tensor(
    compute_class_weight("balanced", classes=classes, y=df["label"].values),
    dtype=torch.float32,
)

full_dataset = Dataset.from_pandas(df[["cleaned_text", "label"]])

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def tokenize(examples):
    return tokenizer(examples["cleaned_text"], truncation=True, max_length=128)


full_tok = full_dataset.map(tokenize, batched=True, remove_columns=["cleaned_text"])

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID, num_labels=len(label2id), id2label=id2label, label2id=label2id
)

if hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable()


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


training_args = TrainingArguments(
    output_dir=os.path.join(FINAL_MODEL_DIR, "checkpoints"),
    save_strategy="epoch",
    save_total_limit=2,
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir=os.path.join(FINAL_MODEL_DIR, "logs"),
    logging_steps=100,
    seed=SEED,
    fp16=torch.cuda.is_available(),
    gradient_checkpointing=True,
    report_to="none",
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=full_tok,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    class_weights=class_weights,
)

print(f"Training final evaluator on {len(df):,} instances ...")
trainer.train()
trainer.save_model(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)
print(f"Model saved to: {FINAL_MODEL_DIR}")
