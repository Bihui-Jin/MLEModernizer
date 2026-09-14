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

try:
    from google.protobuf.message_factory import MessageFactory  # type: ignore

    if not hasattr(MessageFactory, "GetPrototype") and hasattr(
        MessageFactory, "GetMessageClass"
    ):
        MessageFactory.GetPrototype = MessageFactory.GetMessageClass  # type: ignore[attr-defined]
except Exception:
    pass

import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)

TEST_DATA_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
MAX_LENGTH = 1024

MODEL_PATH = "/kaggle/input/training-fold0-aes-deberta-model-starter/deberta-small-fold0/checkpoint-8500/"
EVAL_BATCH_SIZE = 16

FALLBACK_LOCAL_MODEL_DIRS = [
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/deberta-small-fold0/checkpoint-8500",
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/deberta-small-fold0",
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2",
]

FALLBACK_HF_MODEL_ID = "microsoft/deberta-v3-small"


def _is_valid_hf_model_dir(path: str) -> bool:
    """Only treat a path as a local HF model if it has a parseable config.json with model_type."""
    if not path or not os.path.isdir(path):
        return False
    cfg = os.path.join(path, "config.json")
    if not os.path.isfile(cfg):
        return False
    try:
        import json

        with open(cfg, "r", encoding="utf-8") as f:
            j = json.load(f)
        return isinstance(j, dict) and ("model_type" in j)
    except Exception:
        return False


candidate_model_path = MODEL_PATH.rstrip("/")

if not _is_valid_hf_model_dir(candidate_model_path):
    for p in FALLBACK_LOCAL_MODEL_DIRS:
        if _is_valid_hf_model_dir(p):
            candidate_model_path = p
            break

use_local_only = _is_valid_hf_model_dir(candidate_model_path)
if not use_local_only:
    candidate_model_path = FALLBACK_HF_MODEL_ID  # may require internet
    use_local_only = False

df_test = pd.read_csv(TEST_DATA_PATH)

baseline_score = int(np.clip(np.round(3.0), 1, 6))

df_test_out = df_test[["essay_id"]].copy()
df_test_out["score"] = baseline_score

try:
    try:
        import torch

        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if torch.cuda.is_available():
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass

        if not torch.cuda.is_available():
            try:
                torch.set_num_threads(min(4, os.cpu_count() or 2))
                torch.set_num_interop_threads(1)
            except Exception:
                pass

    except Exception:
        torch = None  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(
        candidate_model_path,
        local_files_only=use_local_only,
        use_fast=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        candidate_model_path,
        local_files_only=use_local_only,
    )
    model.eval()

    if torch is None:
        raise RuntimeError("PyTorch not available for inference loop.")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    texts = df_test["full_text"].astype(str).tolist()
    n = len(texts)

    collator = DataCollatorWithPadding(
        tokenizer=tokenizer, pad_to_multiple_of=8 if torch.cuda.is_available() else None
    )

    tok_chunk_size = 512  # keep same chunk size for memory stability
    input_ids_list = [None] * n
    attention_mask_list = [None] * n
    for start in range(0, n, tok_chunk_size):
        end = min(n, start + tok_chunk_size)
        enc = tokenizer(
            texts[start:end],
            max_length=MAX_LENGTH,
            truncation=True,
            padding=False,
            return_attention_mask=True,
            return_token_type_ids=False,
        )
        input_ids_chunk = enc["input_ids"]
        attention_mask_chunk = enc["attention_mask"]
        input_ids_list[start:end] = input_ids_chunk
        attention_mask_list[start:end] = attention_mask_chunk

    from torch.utils.data import Dataset, DataLoader

    class _TokDataset(Dataset):
        __slots__ = ("input_ids", "attention_mask")

        def __init__(self, input_ids, attention_mask):
            self.input_ids = input_ids
            self.attention_mask = attention_mask

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            return {
                "input_ids": self.input_ids[idx],
                "attention_mask": self.attention_mask[idx],
            }

    ds = _TokDataset(input_ids_list, attention_mask_list)

    def _collate_features(features):
        return collator(features)

    dl = DataLoader(
        ds,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        collate_fn=_collate_features,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    n = len(ds)
    num_labels = int(getattr(model.config, "num_labels", 1) or 1)
    logits_all = np.empty((n, num_labels), dtype=np.float32)

    offset = 0
    use_cuda_amp = torch.cuda.is_available()
    autocast_ctx = (
        torch.cuda.amp.autocast(dtype=torch.float16)
        if use_cuda_amp
        else torch.autocast("cpu", enabled=False)
    )

    with torch.inference_mode():
        with autocast_ctx:
            for batch in dl:
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                out = model(**batch)
                logits = out.logits
                bs = logits.shape[0]
                logits_all[offset : offset + bs] = logits.detach().float().cpu().numpy()
                offset += bs

    preds = logits_all
    preds = np.asarray(preds)
    if preds.ndim == 2 and preds.shape[1] > 1:
        scores = preds.argmax(-1) + 1  # classes 0-5 -> score 1-6
    else:
        scores = preds.reshape(-1)
        scores = np.rint(scores).astype(int)
        scores = np.clip(scores, 1, 6)

    df_test_out["score"] = scores

except Exception as e:
    print("Inference failed; writing baseline submission. Error was:", repr(e))

df_test_out.head()



## === cell 1
sub = df_test_out[["essay_id", "score"]].copy()
sub["score"] = (
    pd.to_numeric(sub["score"], errors="coerce")
    .fillna(3)
    .round()
    .astype(int)
    .clip(1, 6)
)
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("Model source:", candidate_model_path)
print("local_files_only:", use_local_only)
