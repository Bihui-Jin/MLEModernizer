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

No external packages required in the script and installed.

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
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        if filename.endswith((".csv", ".md", ".zip")):
            print(os.path.join(dirname, filename))



## === cell 1
data_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2"
train_file_path = f"{data_path}/train.csv"
test_file_path = f"{data_path}/test.csv"

if not os.path.exists(train_file_path):
    train_file_path = "/kaggle/input/train.csv"
    test_file_path = "/kaggle/input/test.csv"

train_df = pd.read_csv(train_file_path)
test_df = pd.read_csv(test_file_path)

train_df = train_df.rename(columns={"full_text": "text", "score": "labels"})
test_df = test_df.rename(columns={"full_text": "text"})

train_df["labels"] = train_df["labels"].astype(int) - 1  # 0..5

print(train_df[["essay_id", "text", "labels"]].head())
print(test_df[["essay_id", "text"]].head())
print("train shape:", train_df.shape, "test shape:", test_df.shape)



## === cell 2
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader

import sys
import subprocess


def _ensure_working_protobuf():
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver

        print("protobuf version:", pb_ver)
        major = int(pb_ver.split(".", 1)[0])
        if major >= 5:
            raise RuntimeError(
                f"protobuf {pb_ver} is incompatible; need < 5 for this environment."
            )
    except Exception as e:
        print("Adjusting protobuf due to:", repr(e))
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf<5"]
        )
        import importlib

        importlib.invalidate_caches()
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver2

        print("protobuf version (after fix):", pb_ver2)


_ensure_working_protobuf()

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)




## === cell 3
class EssayDataset(Dataset):
    def __init__(self, df, tokenizer, with_labels=True, max_length=512):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.with_labels = with_labels
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc = self.tokenizer(
            row["text"],
            truncation=True,
            max_length=self.max_length,
        )
        if self.with_labels:
            enc["labels"] = int(row["labels"])
        return enc




## === cell 4
id2label = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}
label2id = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}

is_submission = True

model_name = "roberta-large"



## === cell 5
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)




## === cell 6
def _list_local_model_dirs(root="/kaggle/input"):
    dirs = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "config.json" in filenames:
            dirs.append(dirpath)
    return dirs


def _try_load_tokenizer_and_model(primary_name: str):
    last_err = None

    local_candidates = []
    for d in _list_local_model_dirs("/kaggle/input"):
        dl = d.lower()
        if any(
            k in dl
            for k in [
                "essay",
                "aes",
                "kappa",
                "learning-agency",
                "roberta",
                "deberta",
                "automated-essay",
            ]
        ):
            local_candidates.append(d)

    candidates = local_candidates + [
        primary_name,
        "/kaggle/input/roberta-large",
        "/kaggle/input/roberta-large-squad2",
        "/kaggle/input/roberta-base",
    ]

    for local_only in (True, False):
        for cand in candidates:
            try:
                tok = AutoTokenizer.from_pretrained(cand, local_files_only=local_only)
                mdl = AutoModelForSequenceClassification.from_pretrained(
                    cand,
                    num_labels=6,
                    id2label=id2label,
                    label2id=label2id,
                    local_files_only=local_only,
                )
                print(
                    f"Loaded model/tokenizer from: {cand} (local_files_only={local_only})"
                )
                return tok, mdl
            except Exception as e:
                last_err = e
                continue

    raise RuntimeError(
        f"Failed to load model/tokenizer from candidates (n={len(candidates)}). Last error: {last_err}"
    )


tokenizer, model = _try_load_tokenizer_and_model(model_name)
model.to(device)




## === cell 7
def quadratic_weighted_kappa(y_true, y_pred, min_rating=0, max_rating=5):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    n_ratings = max_rating - min_rating + 1
    O = np.zeros((n_ratings, n_ratings), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        O[a - min_rating, b - min_rating] += 1.0

    act_hist = np.bincount(y_true - min_rating, minlength=n_ratings).astype(np.float64)
    pred_hist = np.bincount(y_pred - min_rating, minlength=n_ratings).astype(np.float64)

    E = np.outer(act_hist, pred_hist)
    E = E / E.sum() * O.sum()

    W = np.zeros((n_ratings, n_ratings), dtype=np.float64)
    for i in range(n_ratings):
        for j in range(n_ratings):
            W[i, j] = ((i - j) ** 2) / ((n_ratings - 1) ** 2)

    num = (W * O).sum()
    den = (W * E).sum()
    return 1.0 - num / den if den != 0 else 0.0


train_split_df, valid_split_df = train_test_split(
    train_df,
    train_size=0.98,  # keep most data for training; small valid for sanity QWK
    random_state=seed,
    stratify=train_df["labels"],
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_data = EssayDataset(train_split_df, tokenizer, with_labels=True)
valid_data = EssayDataset(valid_split_df, tokenizer, with_labels=True)

train_loader = DataLoader(
    train_data, batch_size=8, shuffle=True, collate_fn=data_collator
)
valid_loader = DataLoader(
    valid_data, batch_size=16, shuffle=False, collate_fn=data_collator
)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
model.train()

epochs = 1
for ep in range(epochs):
    total_loss = 0.0
    n_steps = 0
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu().item())
        n_steps += 1
    print(f"epoch {ep+1}/{epochs} - train loss: {total_loss/max(n_steps,1):.4f}")

model.eval()
v_true, v_pred = [], []
with torch.no_grad():
    for batch in valid_loader:
        labels = batch["labels"].numpy()
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        v_true.append(labels)
        v_pred.append(preds)
v_true = np.concatenate(v_true)
v_pred = np.concatenate(v_pred)
print("valid QWK (labels 0..5):", quadratic_weighted_kappa(v_true, v_pred))



## === cell 8
print(model.__class__.__name__)



## === cell 9
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

test_dataset = EssayDataset(test_df, tokenizer, with_labels=False)
test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    collate_fn=data_collator,
)

model.eval()
all_preds = []

with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        preds = torch.argmax(outputs.logits, dim=-1).detach().cpu().numpy()
        all_preds.append(preds)

all_preds = np.concatenate(all_preds, axis=0)
scores = (all_preds + 1).astype(int)

print("Pred score range:", int(scores.min()), int(scores.max()), "n=", len(scores))



## === cell 10
submission_df = pd.DataFrame(
    {
        "essay_id": test_df["essay_id"].astype(str).values,
        "score": scores,
    }
)

submission_df["score"] = submission_df["score"].astype(int).clip(1, 6)

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print("Wrote:", submission_path)
print(submission_df.head())
print("submission shape:", submission_df.shape)
