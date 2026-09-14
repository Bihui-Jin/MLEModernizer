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

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

import pandas as pd
import numpy as np

from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
)

import torch




## === cell 1
MAX_LENGTH = 512


def tokenize(example, tokenizer, max_length=None):
    return tokenizer(
        example["full_text"],
        padding=False,
        truncation=True,
        max_length=max_length,
    )


class _TokenizedTextDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_length, batch_tokenize_size=256):
        self.essay_id = df["essay_id"].to_numpy()
        self.texts = df["full_text"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_tokenize_size = int(batch_tokenize_size)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {"essay_id": self.essay_id[idx], "full_text": self.texts[idx]}


class _BatchTokenizeThenPadCollator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._pad = DataCollatorWithPadding(tokenizer)

    def __call__(self, batch):
        texts = [x["full_text"] for x in batch]
        tok = self.tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
        )
        return self._pad(tok)


def build_dataset(data_path, tokenizer):
    df = pd.read_csv(data_path, usecols=["essay_id", "full_text"])
    return _TokenizedTextDataset(df, tokenizer, MAX_LENGTH)


def _predict_logits_torch(model, ds, collator, batch_size=32, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    from torch.utils.data import DataLoader

    if torch.cuda.is_available():
        num_workers = 2
        pin_memory = True
    else:
        num_workers = 4

        pin_memory = False

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )

    logits_chunks = []
    with torch.inference_mode():
        for features in loader:
            if device.type == "cuda":
                features = {
                    k: v.to(device, non_blocking=True) for k, v in features.items()
                }
            else:
                features = {k: v.to(device) for k, v in features.items()}
            out = model(**features)
            logits_chunks.append(out.logits.detach().cpu())

    return torch.cat(logits_chunks, dim=0).numpy()


def run_inference():
    test_data_path = (
        "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
    )
    sample_sub_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv"
    trained_model_paths = "/kaggle/input/aes-baseline-deberta-v3-base/transformers/v1/1"

    model_id_or_path = (
        trained_model_paths
        if os.path.isdir(trained_model_paths)
        else "microsoft/deberta-v3-base"
    )

    try:
        torch.set_num_threads(min(8, os.cpu_count() or 1))
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_id_or_path)

    collator = _BatchTokenizeThenPadCollator(tokenizer, MAX_LENGTH)
    ds = build_dataset(test_data_path, tokenizer)

    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception:
            pass

    logits = _predict_logits_torch(model, ds, collator, batch_size=64)

    if logits.ndim == 2:
        preds = np.argmax(logits, axis=1)
        preds = preds + 1
    else:
        preds = logits.squeeze()

    preds = np.asarray(preds).astype(int)
    preds = np.clip(preds, 1, 6)

    sub = pd.read_csv(sample_sub_path)
    pred_df = pd.DataFrame({"essay_id": ds.essay_id, "score": preds})
    score_map = dict(zip(pred_df["essay_id"].values, pred_df["score"].values))
    sub["score"] = sub["essay_id"].map(score_map).fillna(3).astype(int).clip(1, 6)

    sub.to_csv("submission.csv", index=False)
    return sub




## === cell 2
_ = run_inference()
print("Wrote submission.csv with", len(_), "rows and columns:", list(_.columns))
print(_.head())
