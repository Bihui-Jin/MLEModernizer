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
Given pairs of phrases (an `anchor` and a `target` phrase), build a model to rate how similar they are.  

## Metric
Pearson correlation coefficient.

## Submission Format
For each `id` (representing a pair of phrases) in the test set, you must predict the similarity `score`. The file should contain a header and have the following format:

```
id,score
4112d61851461f60,0
09e418c93a776564,0.25
36baf228038e314b,1
etc.

```

## Dataset
The scores are in the 0-1 range with increments of 0.25 with the following meanings:

- **1.0** - Very close match. This is typically an exact match except possibly for differences in conjugation, quantity (e.g. singular vs. plural), and addition or removal of stopwords (e.g. "the", "and", "or").
- **0.75** - Close synonym, e.g. "mobile phone" vs. "cellphone". This also includes abbreviations, e.g. "TCP" -> "transmission control protocol".
- **0.5** - Synonyms which don't have the same meaning (same function, same properties). This includes broad-narrow (hyponym) and narrow-broad (hypernym) matches.
- **0.25** - Somewhat related, e.g. the two phrases are in the same high level domain but are not synonyms. This also includes antonyms.
- **0.0** - Unrelated.

Files
-----

- **train.csv** - the training set, containing phrases, contexts, and their similarity scores
- **test.csv** - the test set set, identical in structure to the training set but without the score
- **sample_submission.csv** - a sample submission file in the correct format

Columns
-------

- `id` - a unique identifier for a pair of phrases
- `anchor` - the first phrase
- `target` - the second phrase
- `context` - the [CPC classification (version 2021.05)](https://en.wikipedia.org/wiki/Cooperative_Patent_Classification), which indicates the subject within which the similarity is to be scored
- `score` - the similarity. This is sourced from a combination of one or more manual expert ratings.

# 2. Python version

3.13

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (118 lines)
            sample_submission.csv (3649 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (3649 lines)
            test.csv.zip (86.4 kB)
            train.csv (32826 lines)
            train.csv.zip (790.5 kB)
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
        input/
            description.md (118 lines)
            sample_submission.csv (3649 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (3649 lines)
            test.csv.zip (86.4 kB)
            train.csv (32826 lines)
            train.csv.zip (790.5 kB)
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
        working/
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
```

-> data/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> data/test.csv has 3648 rows and 4 columns.
The columns are: id, anchor, target, context

-> data/train.csv has 32825 rows and 5 columns.
The columns are: id, anchor, target, context, score

-> data/us-patent-phrase-to-phrase-matching/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> data/us-patent-phrase-to-phrase-matching/test.csv has 3648 rows and 4 columns.
The columns are: id, anchor, target, context

-> data/us-patent-phrase-to-phrase-matching/train.csv has 32825 rows and 5 columns.
The columns are: id, anchor, target, context, score

-> input/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> (stopped after 10 files for performance)

# 5. Target score

0.552832324180412

# 6. Current score

0.73218

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.73218) has done: 'I fixed the model loading path (using a public HuggingFace checkpoint), added a safe device selection, and ensured the tokenizer, model, and separator token are defined before any dataset or training code runs. These changes resolve the NameError and HFValidationError, allowing the full training‑validation loop and prediction to execute and produce a proper `submission.csv` file.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup,
)
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm



## === cell 1
path = Path("../input/us-patent-phrase-to-phrase-matching")
df = pd.read_csv(path / "train.csv")
test_df = pd.read_csv(path / "test.csv")



## === cell 2
anchors = df["anchor"].unique()
np.random.seed(42)
np.random.shuffle(anchors)

val_prop = 0.25
val_sz = int(len(anchors) * val_prop)
val_anchors = anchors[:val_sz]

is_val = df["anchor"].isin(val_anchors)
train_df = df[~is_val].reset_index(drop=True)
val_df = df[is_val].reset_index(drop=True)

print("Train/Val size:", len(train_df), len(val_df))
print("Train/Val score means:", train_df["score"].mean(), val_df["score"].mean())



## === cell 3
model_name = "distilbert-base-uncased"
tokz = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

sep = tokz.sep_token if tokz.sep_token else " "

df["inputs"] = df["context"] + sep + df["anchor"] + sep + df["target"]
df["label"] = df["score"].astype(np.float32)

is_val = df["anchor"].isin(val_anchors)
train_df = df[~is_val].reset_index(drop=True)
val_df = df[is_val].reset_index(drop=True)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 4
class TextPairDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=256):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        inputs = self.tokenizer(
            row["inputs"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in inputs.items()}
        item["label"] = torch.tensor(row["label"], dtype=torch.float)
        return item




## === cell 5
BATCH_SIZE = 16

train_ds = TextPairDataset(train_df, tokz)
val_ds = TextPairDataset(val_df, tokz)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)



## === cell 6
lr = 8e-5
wd = 0.01
epochs = 1
warmup_ratio = 0.1

optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)

num_training_steps = len(train_dl) * epochs
num_warmup_steps = int(num_training_steps * warmup_ratio)

scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
)

loss_fn = nn.MSELoss()




## === cell 7
def pearsonr(x, y):
    return np.corrcoef(x, y)[0, 1]




## === cell 8
for epoch in range(epochs):
    model.train()
    train_loss = 0.0

    for batch in tqdm(train_dl, desc=f"Epoch {epoch+1} Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = outputs.logits

        loss = loss_fn(preds, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_dl)

    model.eval()
    val_loss = 0.0
    preds_all = []
    labels_all = []

    with torch.no_grad():
        for batch in tqdm(val_dl, desc=f"Epoch {epoch+1} Validation"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].unsqueeze(1).to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits

            loss = loss_fn(preds, labels)
            val_loss += loss.item()

            preds_all.append(preds.cpu().numpy())
            labels_all.append(labels.cpu().numpy())

    preds_all = np.concatenate(preds_all).flatten()
    labels_all = np.concatenate(labels_all).flatten()
    pearson = pearsonr(preds_all, labels_all)
    avg_val_loss = val_loss / len(val_dl)

    print(
        f"\nEpoch {epoch+1}: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f} | Pearson = {pearson:.4f}"
    )



## === cell 9
test_df["inputs"] = (
    test_df["context"] + sep + test_df["anchor"] + sep + test_df["target"]
)




## === cell 10
class TestDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=256):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        inputs = self.tokenizer(
            row["inputs"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in inputs.items()}
        item["id"] = row["id"]
        return item


test_ds = TestDataset(test_df, tokz)
test_dl = DataLoader(test_ds, batch_size=128)



## === cell 11
model.eval()
pred_ids = []
pred_scores = []

with torch.no_grad():
    for batch in tqdm(test_dl, desc="Predicting"):
        ids = batch.pop("id")
        inputs = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**inputs)
        preds = outputs.logits.squeeze(-1).cpu().numpy()

        pred_ids.extend(ids)
        pred_scores.extend(preds)



## === cell 12
submission = pd.DataFrame({"id": pred_ids, "score": pred_scores})
submission.to_csv("submission.csv", index=False)



## === cell 13
print(submission.head())
