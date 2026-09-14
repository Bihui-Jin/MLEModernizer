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

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

import json
import random
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset

from transformers import AutoTokenizer, AutoModelForSequenceClassification

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

EVAL_BATCH_SIZE = 32 if torch.cuda.is_available() else 8

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.set_num_threads(1)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

try:
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
except Exception:
    pass

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


try:
    model_dir = _resolve_local_model_dir(MODEL_PATH)
except FileNotFoundError:
    model_dir = None

base_name = _infer_base_model_name(model_dir) if model_dir is not None else None
if model_dir is None and base_name is None:
    base_name = "microsoft/deberta-v3-small"

print("Using model_dir:", model_dir)
print("Base model name (from config or fallback):", base_name)




## === cell 2
class TextTestDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], idx


if model_dir is not None:
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir, local_files_only=True, use_fast=True
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_name, use_fast=True)
else:
    tokenizer = AutoTokenizer.from_pretrained(base_name, use_fast=True)

df_test = pd.read_csv(TEST_DATA_PATH).reset_index(drop=True)
texts = df_test["full_text"].astype(str).tolist()

ds = TextTestDataset(texts)


def _collate_tokenize(batch):
    batch_texts, batch_indices = zip(*batch)
    enc = tokenizer(
        list(batch_texts),
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt",
    )
    return (
        enc["input_ids"],
        enc["attention_mask"],
        torch.tensor(batch_indices, dtype=torch.int64),
    )


if torch.cuda.is_available():
    num_workers = min(4, (os.cpu_count() or 2))
else:
    num_workers = min(2, (os.cpu_count() or 2))

loader = DataLoader(
    ds,
    batch_size=EVAL_BATCH_SIZE,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    collate_fn=_collate_tokenize,
    persistent_workers=(num_workers > 0),
    prefetch_factor=2 if num_workers > 0 else None,
)

if model_dir is not None:
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, local_files_only=True
        )
    except Exception:
        model = AutoModelForSequenceClassification.from_pretrained(base_name)
else:
    model = AutoModelForSequenceClassification.from_pretrained(base_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

DO_COMPILE = False
if DO_COMPILE and torch.cuda.is_available():
    try:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
        print("torch.compile enabled for inference.")
    except Exception as e:
        print(
            "torch.compile not available/failed; continuing without compile. Reason:",
            repr(e),
        )

pred_scores = np.empty(len(texts), dtype=np.int64)

with torch.inference_mode():
    for input_ids, attention_mask, batch_indices in loader:
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = outputs.logits.argmax(-1).detach().cpu().numpy() + 1  # scores 1..N

        pred_scores[batch_indices.numpy()] = preds

df_test["score"] = pred_scores.astype(int)
df_test["score"] = df_test["score"].clip(1, 6)

print(df_test[["essay_id", "score"]].head())



## === cell 3
sub = df_test[["essay_id", "score"]].copy()
sub.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
print("submission.csv saved at:", os.path.abspath("submission.csv"))
print("submission.csv exists:", os.path.exists("submission.csv"))
