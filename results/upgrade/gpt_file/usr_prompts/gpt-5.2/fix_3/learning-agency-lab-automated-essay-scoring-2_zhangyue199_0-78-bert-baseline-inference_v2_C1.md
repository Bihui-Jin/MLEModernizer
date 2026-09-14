# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


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

# 5. Target score

0.7788391867553548

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
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



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text))
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    return text.strip()


train_df["full_text"] = train_df["full_text"].astype(str).apply(clean_text)
test_df["full_text"] = test_df["full_text"].astype(str).apply(clean_text)

train_df["label"] = (train_df["score"].astype(int) - 1).clip(0, 5)

print(train_df[["essay_id", "score", "label"]].head())



## === cell 3
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=6)

model.config.num_labels = 6

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)




## === cell 4
class EssayDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = list(texts)
        self.labels = None if labels is None else list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item


val_frac = 0.05
val_size = max(1, int(len(train_df) * val_frac))
train_split = train_df.iloc[:-val_size].reset_index(drop=True)
val_split = train_df.iloc[-val_size:].reset_index(drop=True)

train_dataset = EssayDataset(
    train_split["full_text"].values, train_split["label"].values, tokenizer, MAX_LEN
)
val_dataset = EssayDataset(
    val_split["full_text"].values, val_split["label"].values, tokenizer, MAX_LEN
)
test_dataset = EssayDataset(test_df["full_text"].values, None, tokenizer, MAX_LEN)

print("train/val/test sizes:", len(train_dataset), len(val_dataset), len(test_dataset))



## === cell 5
args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    report_to="none",
    seed=SEED,
    dataloader_drop_last=False,
    do_train=True,
    do_eval=True,
    evaluation_strategy="steps",
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



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3597435635.py in <cell line: 0>()
      2 # Minimal fine-tuning so predictions are meaningful (base model would otherwise be near-random).
      3 # Kept conservative to fit 600s; no approximations or early stopping added.
----> 4 args = TrainingArguments(
      5     output_dir="./output",
      6     overwrite_output_dir=True,

TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'evaluation_strategy'

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

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3683462711.py in <cell line: 0>()
      1 # Predict on test
----> 2 pred_out = trainer.predict(test_dataset)
      3 preds = pred_out.predictions
      4 
      5 # Handle possible tuple output

NameError: name 'trainer' is not defined
