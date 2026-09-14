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
    if not os.path.isdir(mp):
        raise FileNotFoundError(f"MODEL_PATH directory not found: {mp}")
    return mp


def _infer_base_model_name(model_dir: str) -> str:
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(cfg_path):
        return None
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg.get("_name_or_path") or cfg.get("model_name_or_path")


model_dir = _resolve_local_model_dir(MODEL_PATH)
base_name = _infer_base_model_name(model_dir)

try:
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
except Exception:
    if base_name is None:
        raise
    tokenizer = AutoTokenizer.from_pretrained(base_name, local_files_only=True)


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

df_test[["essay_id", "score"]].head()



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2017647581.py in <cell line: 0>()
     18 
     19 
---> 20 model_dir = _resolve_local_model_dir(MODEL_PATH)
     21 base_name = _infer_base_model_name(model_dir)
     22 

/tmp/ipykernel_11/2017647581.py in _resolve_local_model_dir(model_path)
      4     mp = model_path.rstrip("/")
      5     if not os.path.isdir(mp):
----> 6         raise FileNotFoundError(f"MODEL_PATH directory not found: {mp}")
      7     return mp
      8 

FileNotFoundError: MODEL_PATH directory not found: /kaggle/input/training-fold0-aes-deberta-model-starter/deberta-small-fold0/checkpoint-8500

## === cell 2
sub = df_test[["essay_id", "score"]].copy()
sub.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3605381337.py in <cell line: 0>()
      1 # Ensure valid submission file with required columns and .csv suffix
----> 2 sub = df_test[["essay_id", "score"]].copy()
      3 sub.to_csv("submission.csv", index=False)
      4 
      5 print("Wrote submission.csv with shape:", sub.shape)

NameError: name 'df_test' is not defined
