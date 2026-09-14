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

0.7984215469600132

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import json
import pandas as pd

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from datasets import Dataset

TEST_DATA_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
MAX_LENGTH = 2048

MODEL_PATH = "/kaggle/input/training-fold0-aes-deberta-model-starter/deberta-small-fold0/checkpoint-8500"
EVAL_BATCH_SIZE = 1




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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


def _find_fallback_checkpoint_dir(search_roots: list[str]) -> str:
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
        raise FileNotFoundError(
            "Could not find any HuggingFace model checkpoint directory under /kaggle/input. "
            "Please add the trained model dataset to the notebook or update MODEL_PATH."
        )

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
        search_roots=[
            "/kaggle/input",
            "/kaggle/data",
        ]
    )

base_name = _infer_base_model_name(model_dir)

print("Using model_dir:", model_dir)
print("Base model name from config (if any):", base_name)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1236337517.py in <cell line: 0>()
     74 try:
---> 75     model_dir = _resolve_local_model_dir(MODEL_PATH)
     76 except FileNotFoundError:

/tmp/ipykernel_11/1236337517.py in _resolve_local_model_dir(model_path)
      4         return mp
----> 5     raise FileNotFoundError(f"MODEL_PATH directory not found: {mp}")
      6 

FileNotFoundError: MODEL_PATH directory not found: /kaggle/input/training-fold0-aes-deberta-model-starter/deberta-small-fold0/checkpoint-8500

During handling of the above exception, another exception occurred:

FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1236337517.py in <cell line: 0>()
     76 except FileNotFoundError:
     77     # Minimal, robust fallback: search in /kaggle/input (where Kaggle mounts datasets).
---> 78     model_dir = _find_fallback_checkpoint_dir(
     79         search_roots=[
     80             "/kaggle/input",

/tmp/ipykernel_11/1236337517.py in _find_fallback_checkpoint_dir(search_roots)
     53 
     54     if not candidates:
---> 55         raise FileNotFoundError(
     56             "Could not find any HuggingFace model checkpoint directory under /kaggle/input. "
     57             "Please add the trained model dataset to the notebook or update MODEL_PATH."

FileNotFoundError: Could not find any HuggingFace model checkpoint directory under /kaggle/input. Please add the trained model dataset to the notebook or update MODEL_PATH.

## === cell 2
try:
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, use_fast=True
    )
except Exception:
    if base_name is None:
        raise
    tokenizer = AutoTokenizer.from_pretrained(
        base_name, local_files_only=True, use_fast=True
    )


def tokenize(sample):
    return tokenizer(sample["full_text"], max_length=MAX_LENGTH, truncation=True)


df_test = pd.read_csv(TEST_DATA_PATH)
ds = Dataset.from_pandas(df_test, preserve_index=False).map(tokenize, batched=False)

to_remove = [c for c in ["essay_id", "full_text"] if c in ds.column_names]
ds = ds.remove_columns(to_remove)

model = AutoModelForSequenceClassification.from_pretrained(
    model_dir, local_files_only=True
)

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
df_test["score"] = (pred.argmax(-1) + 1).astype(int).clip(1, 6)

print(df_test[["essay_id", "score"]].head())




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1720978392.py in <cell line: 0>()
      3     tokenizer = AutoTokenizer.from_pretrained(
----> 4         model_dir, local_files_only=True, use_fast=True
      5     )

NameError: name 'model_dir' is not defined

During handling of the above exception, another exception occurred:

NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1720978392.py in <cell line: 0>()
      5     )
      6 except Exception:
----> 7     if base_name is None:
      8         raise
      9     tokenizer = AutoTokenizer.from_pretrained(

NameError: name 'base_name' is not defined

## === cell 3
sub = df_test[["essay_id", "score"]].copy()
sub.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
print("submission.csv saved at:", os.path.abspath("submission.csv"))

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2784831303.py in <cell line: 0>()
      1 # Ensure valid submission file with required columns and .csv suffix
----> 2 sub = df_test[["essay_id", "score"]].copy()
      3 sub.to_csv("submission.csv", index=False)
      4 
      5 print("Wrote submission.csv with shape:", sub.shape)

NameError: name 'df_test' is not defined
