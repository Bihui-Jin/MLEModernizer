# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Predict the score of student essays.

## Metric
Quadratic weighted kappa.

## Submission Format
For each `essay_id` in the test set, you must predict the corresponding `score` (between 1-6, see [rubric](https://storage.googleapis.com/kaggle-forum-message-attachments/2733927/20538/Rubric_%20Holistic%20Essay%20Scoring.pdf) for more details). The file should contain a header and have the following format:

```
essay_id,score
000d118,3
000fe60,3
001ab80,4
...
```

## Dataset
- **train.csv** - Essays and scores to be used as training data.
    - `essay_id` - The unique ID of the essay
    - `full_text` - The full essay response
    - `score` - Holistic score of the essay on a 1-6 scale
- **test.csv** - The essays to be used as test data. Contains the same fields as `train.csv`, aside from exclusion of `score`.
- **sample_submission.csv** - A submission file in the correct format.
    - `essay_id` - The unique ID of the essay
    - `score` - The predicted holistic score of the essay on a 1-6 scale

# 2. Python version

3.12

# 3. Installed packages

datasets==4.4.1
geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
tensorflow-datasets==4.9.9
transformers==4.53.3
vega-datasets==0.9.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (153 lines)
            sample_submission.csv (1732 lines)
            sample_submission.csv.zip (9.4 kB)
            test.csv (15336 lines)
            test.csv.zip (1.2 MB)
            train.csv (139231 lines)
            train.csv.zip (11.0 MB)
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
        input/
            description.md (153 lines)
            sample_submission.csv (1732 lines)
            sample_submission.csv.zip (9.4 kB)
            test.csv (15336 lines)
            test.csv.zip (1.2 MB)
            train.csv (139231 lines)
            train.csv.zip (11.0 MB)
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
        working/
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
```

-> data/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv has 1731 rows and 2 columns.
The columns are: essay_id, score

-> data/learning-agency-lab-automated-essay-scoring-2/test.csv has 15335 rows and 2 columns.
The columns are: essay_id, full_text

-> data/learning-agency-lab-automated-essay-scoring-2/train.csv has 139230 rows and 3 columns.
The columns are: essay_id, full_text, score

-> data/sample_submission.csv has 1731 rows and 2 columns.
The columns are: essay_id, score

-> data/test.csv has 15335 rows and 2 columns.
The columns are: essay_id, full_text

-> data/train.csv has 139230 rows and 3 columns.
The columns are: essay_id, full_text, score

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import re
import random
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

SEED = 42
set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




## === cell 1
train_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
test_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

MAX_LEN = 512
MODEL_ID = "bert-base-uncased"

print("train_df:", train_df.shape, "test_df:", test_df.shape)
print("train columns:", train_df.columns.tolist())
print("test columns:", test_df.columns.tolist())




## === cell 2
def clean_text_series(s: pd.Series) -> pd.Series:
    s = s.astype(str)
    s = s.str.replace(r"\s+", " ", regex=True)
    s = s.str.replace(r"[^a-zA-Z0-9]", " ", regex=True)
    s = s.str.strip()
    return s


train_df["full_text"] = clean_text_series(train_df["full_text"])
test_df["full_text"] = clean_text_series(test_df["full_text"])

train_df["label"] = (train_df["score"].astype(int) - 1).clip(0, 5)

print(train_df[["essay_id", "score", "label"]].head())




## === cell 3
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=6)

model.config.num_labels = 6

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)




## === cell 4
class TokenizedEssayDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = (
            None if labels is None else torch.as_tensor(labels, dtype=torch.long)
        )

    def __len__(self):
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


val_frac = 0.05
val_size = max(1, int(len(train_df) * val_frac))
train_split = train_df.iloc[:-val_size].reset_index(drop=True)
val_split = train_df.iloc[-val_size:].reset_index(drop=True)

train_enc = tokenizer(
    train_split["full_text"].tolist(),
    padding="max_length",
    truncation=True,
    max_length=MAX_LEN,
    return_tensors="pt",
)
val_enc = tokenizer(
    val_split["full_text"].tolist(),
    padding="max_length",
    truncation=True,
    max_length=MAX_LEN,
    return_tensors="pt",
)
test_enc = tokenizer(
    test_df["full_text"].tolist(),
    padding="max_length",
    truncation=True,
    max_length=MAX_LEN,
    return_tensors="pt",
)

train_dataset = TokenizedEssayDataset(train_enc, train_split["label"].to_numpy())
val_dataset = TokenizedEssayDataset(val_enc, val_split["label"].to_numpy())
test_dataset = TokenizedEssayDataset(test_enc, None)

print("train/val/test sizes:", len(train_dataset), len(val_dataset), len(test_dataset))




## === cell 5
num_workers = min(4, os.cpu_count() or 1)

args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    report_to="none",
    seed=SEED,
    dataloader_drop_last=False,
    do_train=True,
    do_eval=True,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="no",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.06,
    logging_steps=50,
    fp16=False,  # keep deterministic and compatible
    dataloader_num_workers=num_workers,
    dataloader_pin_memory=True,
)

trainer = Trainer(
    model=model,
    args=args,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

train_result = trainer.train()
eval_result = trainer.evaluate()
print("train_result:", {k: float(v) for k, v in train_result.metrics.items()})
print("eval_result:", {k: float(v) for k, v in eval_result.items()})




## === cell 6
pred_out = trainer.predict(test_dataset)
preds = pred_out.predictions

if isinstance(preds, (tuple, list)):
    preds = preds[0]

pred_labels = np.argmax(preds, axis=1).astype(int)
scores = (pred_labels + 1).clip(1, 6).astype(int)

sub = test_df[["essay_id"]].copy()
sub["score"] = scores
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
assert list(sub.columns) == ["essay_id", "score"]
assert sub.shape[0] == test_df.shape[0]
assert os.path.exists("submission.csv")
