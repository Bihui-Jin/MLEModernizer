# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

# 5. Target score

0.8077619023103844

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the `datasets` dependency that’s crashing (protobuf/MessageFactory) and replace it with a minimal pandas-based pipeline that preserves the same core idea: a RoBERTa sequence classifier over essay text with labels shifted to 0–5. I also fix the invalid `from_pretrained('/kaggle/input/roberta-large')` path usage by loading a real pretrained checkpoint (`roberta-large`) directly (Kaggle has it cached), ensuring `model`/`tokenizer` are defined in submission mode. Finally, I replace the `pipeline()` inference (which is slow and returns string labels) with a batched `model(**tokenized)` forward pass and argmax to produce integer scores 1–6, then write `submission.csv` with the required columns.'
- What this solution (achieved 0.0) has done: 'I fix the crash coming from importing `Trainer/TrainingArguments` (it triggers a protobuf `MessageFactory` error in this Kaggle Python 3.12 environment) by removing those imports and leaving the rest of the RoBERTa inference pipeline intact. Since `is_submission=True`, none of the training code paths are needed; this keeps the core logic (RoBERTa large + argmax over 6 labels, score = pred+1) unchanged while making the notebook run end-to-end. I also ensure the data path is robust (fallback to the non-nested `/kaggle/input/...` files if needed) and that a valid `submission.csv` is always written with the required columns. This should move the score from 0.0 (crash/invalid) to a non-zero valid score without changing modeling semantics.'
- What this solution (achieved 0.0) has done: 'I fix the protobuf/transformers import crash that prevents the notebook from running by pinning protobuf to the compatible pure-Python implementation before importing `transformers`, which resolves the `MessageFactory.GetPrototype` error in many Kaggle Py3.12 images. I also make the data-path selection more robust (still using the same files) and ensure we always write a correctly-formatted `submission.csv` aligned to `test_df` order. These changes are execution/stability fixes and should move the score from 0.0 (failed/invalid run) to a valid non-zero score without changing the model’s core inference logic (RoBERTa-large argmax over 6 labels, mapped to 1–6).'
- What this solution (achieved 0.0) has done: 'I fix the runtime crash in the `transformers` import caused by an incompatible protobuf version in this Kaggle Python 3.12 image by forcing a compatible protobuf configuration and (if needed) safely downgrading protobuf at runtime before importing `transformers`. I also add a small fallback to load the model/tokenizer from a local Kaggle input directory if online/cached resolution fails, without changing the core inference logic (RoBERTa-large, argmax over 6 labels, map to 1–6). Finally, I ensure the submission is always written as a valid `submission.csv` with the correct columns and row alignment to `test_df`.'

# 9. Code solution

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
def _try_load_tokenizer_and_model(primary_name: str):
    last_err = None
    candidates = [
        primary_name,
        "/kaggle/input/roberta-large",
        "/kaggle/input/roberta-large-squad2",
        "/kaggle/input/roberta-base",
    ]
    for cand in candidates:
        try:
            tok = AutoTokenizer.from_pretrained(cand)
            mdl = AutoModelForSequenceClassification.from_pretrained(
                cand,
                num_labels=6,
                id2label=id2label,
                label2id=label2id,
            )
            print("Loaded model/tokenizer from:", cand)
            return tok, mdl
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(
        f"Failed to load model/tokenizer from candidates {candidates}. Last error: {last_err}"
    )


tokenizer, model = _try_load_tokenizer_and_model(model_name)
model.to(device)



## === cell 7
if not is_submission:
    train_split_df, valid_split_df = train_test_split(
        train_df,
        train_size=0.8,
        random_state=seed,
        stratify=train_df["labels"],
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_data = EssayDataset(train_split_df, tokenizer, with_labels=True)
    valid_data = EssayDataset(valid_split_df, tokenizer, with_labels=True)



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
