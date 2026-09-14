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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import random
import contextlib
from datasets import Dataset
import pandas as pd
import numpy as np
import re
from transformers import (
    AutoTokenizer,  # use fast tokenizer
    BertForSequenceClassification,
    AutoConfig,
)
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
else:
    torch.set_num_threads(os.cpu_count() or 1)



## === cell 1
MODEL_NAME = "bert-base-uncased"
MAX_LEN = 512
BATCH_SIZE = 128  # larger batch reduces optimizer steps
EPOCHS = 1  # single epoch as before
LR = 2e-5

train_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
test_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)


def clean_text(series):
    return (
        series.str.replace(r"\s+", " ", regex=True)  # collapse whitespace
        .str.replace(r"[^a-zA-Z0-9]", " ", regex=True)  # keep alphanumerics
        .str.strip()
    )


train_df["full_text"] = clean_text(train_df["full_text"])
test_df["full_text"] = clean_text(test_df["full_text"])



## === cell 2
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=6)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, config=config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

if device.type == "cuda":
    model.half()  # optional fp16 for inference later (kept from original)



## === cell 3
train_encodings = tokenizer(
    train_df["full_text"].tolist(),
    padding=True,
    truncation=True,
    max_length=MAX_LEN,
    return_token_type_ids=False,
    return_tensors="pt",
)

train_input_ids = train_encodings["input_ids"]
train_attention_mask = train_encodings["attention_mask"]
train_labels = torch.tensor(train_df["score"].values - 1, dtype=torch.long)  # 0‑5

train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    num_workers=os.cpu_count() // 2 or 1,  # parallel data loading
    persistent_workers=True,
)

test_encodings = tokenizer(
    test_df["full_text"].tolist(),
    padding=True,
    truncation=True,
    max_length=MAX_LEN,
    return_token_type_ids=False,
    return_tensors="pt",
)

test_input_ids = test_encodings["input_ids"]
test_attention_mask = test_encodings["attention_mask"]



## === cell 4
model.train()
optimizer = AdamW(model.parameters(), lr=LR)
criterion = torch.nn.CrossEntropyLoss()

scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

for epoch in range(EPOCHS):
    epoch_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Training epoch {epoch+1}"):
        ids_batch, mask_batch, label_batch = batch
        ids_batch = ids_batch.to(device, non_blocking=True)
        mask_batch = mask_batch.to(device, non_blocking=True)
        label_batch = label_batch.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            outputs = model(input_ids=ids_batch, attention_mask=mask_batch)
            logits = outputs.logits
            loss = criterion(logits, label_batch)

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()
    print(f"Epoch {epoch+1} average loss: {epoch_loss/len(train_loader):.4f}")



## === cell 5
model.eval()
all_logits = []
batch_size_test = 128
test_dataset = TensorDataset(test_input_ids, test_attention_mask)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size_test,
    pin_memory=True,
    num_workers=os.cpu_count() // 2 or 1,
    persistent_workers=True,
)

if device.type == "cuda":
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
else:
    autocast_ctx = contextlib.nullcontext()

with torch.inference_mode():
    with autocast_ctx:
        for ids_batch, mask_batch in test_loader:
            ids_batch = ids_batch.to(device, non_blocking=True)
            mask_batch = mask_batch.to(device, non_blocking=True)
            outputs = model(input_ids=ids_batch, attention_mask=mask_batch)
            logits = outputs.logits
            all_logits.append(logits.detach().cpu().float())

logits_tensor = torch.cat(all_logits, dim=0).numpy()
scores = np.argmax(logits_tensor, axis=1) + 1  # convert 0‑5 → 1‑6



## === cell 6
sub = test_df[["essay_id"]].copy()
sub["score"] = scores
sub.to_csv("submission.csv", index=False)
display(sub.head())
