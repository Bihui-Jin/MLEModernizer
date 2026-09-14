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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import json
import pandas as pd

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from datasets import Dataset

CANDIDATE_TEST_PATHS = [
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv",
    "/kaggle/data/learning-agency-lab-automated-essay-scoring-2/test.csv",
    "/kaggle/input/test.csv",
    "/kaggle/data/test.csv",
]
TEST_DATA_PATH = next((p for p in CANDIDATE_TEST_PATHS if os.path.exists(p)), None)
if TEST_DATA_PATH is None:
    raise FileNotFoundError(
        f"Could not find test.csv in any of: {CANDIDATE_TEST_PATHS}"
    )

MAX_LENGTH = 2048
MODEL_PATH = "/kaggle/input/training-fold0-aes-deberta-model-starter/deberta-small-fold0/checkpoint-8500"
EVAL_BATCH_SIZE = 1

print("Environment ready. Using TEST_DATA_PATH:", TEST_DATA_PATH)




## === cell 1
def _resolve_local_model_dir(model_path: str) -> str:
    mp = model_path.rstrip("/")
    if os.path.isdir(mp):
        return mp
    raise FileNotFoundError(f"MODEL_PATH directory not found: {mp}")


def _infer_base_model_name(model_dir: str) -> str | None:
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(cfg_path):
        return None
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg.get("_name_or_path") or cfg.get("model_name_or_path")


def _looks_like_hf_model_dir(d: str) -> bool:
    if not os.path.isdir(d):
        return False
    expected_any = [
        "config.json",
        "pytorch_model.bin",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "spiece.model",
    ]
    return any(os.path.exists(os.path.join(d, f)) for f in expected_any)


def _find_fallback_checkpoint_dir(search_roots: list[str]) -> str | None:
    candidates = []
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            base = os.path.basename(dirpath)
            if base.startswith("checkpoint-") and "config.json" in filenames:
                candidates.append(dirpath)

    if not candidates:
        for root in search_roots:
            if not os.path.isdir(root):
                continue
            for dirpath, dirnames, filenames in os.walk(root):
                if "config.json" in filenames and _looks_like_hf_model_dir(dirpath):
                    candidates.append(dirpath)

    if not candidates:
        return None

    def _step_key(p: str) -> int:
        b = os.path.basename(p)
        if b.startswith("checkpoint-"):
            try:
                return int(b.split("-")[-1])
            except Exception:
                return -1
        return -1

    candidates.sort(key=lambda p: (_step_key(p), len(p)))
    return candidates[-1]


try:
    model_dir = _resolve_local_model_dir(MODEL_PATH)
except FileNotFoundError:
    model_dir = _find_fallback_checkpoint_dir(
        search_roots=["/kaggle/input", "/kaggle/data"]
    )

base_name = _infer_base_model_name(model_dir) if model_dir is not None else None
if model_dir is None and base_name is None:
    base_name = "microsoft/deberta-v3-small"

print("Using model_dir:", model_dir)
print("Base model name (from config or fallback):", base_name)



## === cell 2
if model_dir is not None:
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir, local_files_only=True, use_fast=True
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_name, use_fast=True)
else:
    tokenizer = AutoTokenizer.from_pretrained(base_name, use_fast=True)


def tokenize(sample):
    return tokenizer(sample["full_text"], max_length=MAX_LENGTH, truncation=True)


df_test = pd.read_csv(TEST_DATA_PATH).reset_index(drop=True)

ds = Dataset.from_pandas(df_test, preserve_index=False).map(tokenize, batched=False)

to_remove = [c for c in ["essay_id", "full_text"] if c in ds.column_names]
ds = ds.remove_columns(to_remove)

if model_dir is not None:
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, local_files_only=True
        )
    except Exception:
        model = AutoModelForSequenceClassification.from_pretrained(base_name)
else:
    model = AutoModelForSequenceClassification.from_pretrained(base_name)

args = TrainingArguments(
    output_dir=".",
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    tokenizer=tokenizer,
)

pred = trainer.predict(ds).predictions

df_test["score"] = (pred.argmax(-1) + 1).astype(int)
df_test["score"] = df_test["score"].clip(1, 6)

print(df_test[["essay_id", "score"]].head())



## === cell 3
sub = df_test[["essay_id", "score"]].copy()
sub.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
print("submission.csv saved at:", os.path.abspath("submission.csv"))
print("submission.csv exists:", os.path.exists("submission.csv"))
