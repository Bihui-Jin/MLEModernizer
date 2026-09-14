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

import pandas as pd
import numpy as np

from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
)
from datasets import Dataset

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


def build_dataset(data_path, tokenizer):
    data = pd.read_csv(data_path)
    data = Dataset.from_pandas(data, preserve_index=False)

    try:
        import multiprocessing as _mp

        _cpu = _mp.cpu_count()
    except Exception:
        _cpu = 2
    num_proc = max(1, min(4, _cpu))

    data = data.map(
        tokenize,
        batched=True,
        batch_size=512,
        num_proc=num_proc,
        fn_kwargs={"tokenizer": tokenizer, "max_length": MAX_LENGTH},
        remove_columns=["full_text"],
        desc="Tokenizing",
    )
    return data


def _predict_logits_torch(model, ds, collator, batch_size=32, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    needed_cols = [
        c
        for c in ("input_ids", "attention_mask", "token_type_ids")
        if c in ds.column_names
    ]
    ds_t = ds.with_format("torch", columns=needed_cols)

    n = len(ds_t)
    logits_chunks = []

    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = [ds_t[i] for i in range(start, min(start + batch_size, n))]
            features = collator(batch)
            features = {k: v.to(device) for k, v in features.items()}

            out = model(**features)
            logits = out.logits
            logits_chunks.append(logits.detach().cpu())

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

    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_id_or_path)
    collator = DataCollatorWithPadding(tokenizer)

    ds = build_dataset(test_data_path, tokenizer)

    logits = _predict_logits_torch(model, ds, collator, batch_size=32)

    if logits.ndim == 2:
        preds = np.argmax(logits, axis=1)
        preds = preds + 1
    else:
        preds = logits.squeeze()

    preds = np.asarray(preds).astype(int)
    preds = np.clip(preds, 1, 6)

    sub = pd.read_csv(sample_sub_path)
    pred_df = pd.DataFrame({"essay_id": ds["essay_id"], "score": preds})
    sub = sub.drop(columns=["score"]).merge(pred_df, on="essay_id", how="left")

    sub["score"] = sub["score"].fillna(3).astype(int).clip(1, 6)

    sub.to_csv("submission.csv", index=False)
    return sub




## === cell 2
_ = run_inference()
print("Wrote submission.csv with", len(_), "rows and columns:", list(_.columns))
print(_.head())
