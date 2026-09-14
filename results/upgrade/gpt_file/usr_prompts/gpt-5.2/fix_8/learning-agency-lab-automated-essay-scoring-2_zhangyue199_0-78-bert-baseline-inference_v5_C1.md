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
import re
import numpy as np
import pandas as pd

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

from transformers import AutoTokenizer, AutoModelForSequenceClassification



## === cell 1
COMP_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2"
test_path = os.path.join(COMP_PATH, "test.csv")

candidate_model_paths = [
    "/kaggle/input/bert-baseline-train/output/bert-base",
    "/kaggle/input/bert-baseline-train/bert-base",
    "/kaggle/input/bert-baseline-train",
]
model_path = next((p for p in candidate_model_paths if os.path.isdir(p)), None)
if model_path is None:
    model_path = "bert-base-uncased"

MAX_LEN = 1024

test_df = pd.read_csv(test_path)
assert {"essay_id", "full_text"}.issubset(test_df.columns)



## === cell 2
s = test_df["full_text"].astype(str)
s = s.str.replace(r"\s+", " ", regex=True)
s = s.str.replace(r"[^a-zA-Z0-9]", " ", regex=True)
s = s.str.strip()
test_df["full_text"] = s



## === cell 3
import torch
from datasets import Dataset

torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

is_local_dir = os.path.isdir(model_path)

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=is_local_dir,
    use_fast=True,
)

tokenizer_max = getattr(tokenizer, "model_max_length", 512)
if tokenizer_max is None or tokenizer_max > 100000:
    tokenizer_max = 512
MAX_LEN = int(min(MAX_LEN, tokenizer_max))

model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    local_files_only=is_local_dir,
)

test_ds = Dataset.from_pandas(test_df[["essay_id", "full_text"]], preserve_index=False)


def _tokenize_batch(batch):
    return tokenizer(
        batch["full_text"],
        padding=False,
        truncation=True,
        max_length=MAX_LEN,
    )


num_proc = min(4, (os.cpu_count() or 2))
test_ds = test_ds.map(
    _tokenize_batch,
    batched=True,
    num_proc=num_proc,
    remove_columns=["full_text"],
    desc="Tokenizing test",
)

from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer, padding="longest", return_tensors="pt"
)

model_input_keys = [
    k
    for k in ["input_ids", "attention_mask", "token_type_ids"]
    if k in test_ds.column_names
]


def collate_drop_non_inputs(features):
    features_in = [{k: f[k] for k in model_input_keys} for f in features]
    return data_collator(features_in)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

batch_size = 32
num_workers = min(4, (os.cpu_count() or 2))
pin_memory = torch.cuda.is_available()

loader = DataLoader(
    test_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
    collate_fn=collate_drop_non_inputs,
)

pred_batches = []
with torch.inference_mode():
    for batch in loader:
        batch = {k: v.to(device, non_blocking=pin_memory) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        pred_batches.append(logits.detach().cpu())

preds = torch.cat(pred_batches, dim=0).numpy()

if preds.ndim == 2 and preds.shape[1] > 1:
    scores = np.argmax(preds, axis=1) + 1
else:
    scores = np.ravel(preds)
    scores = np.rint(scores).astype(int)

scores = np.clip(scores, 1, 6).astype(int)

sub = test_df[["essay_id"]].copy()
sub["score"] = scores
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
assert os.path.exists("submission.csv") and sub.shape[1] == 2
assert list(sub.columns) == ["essay_id", "score"]
assert len(sub) == len(test_df)
