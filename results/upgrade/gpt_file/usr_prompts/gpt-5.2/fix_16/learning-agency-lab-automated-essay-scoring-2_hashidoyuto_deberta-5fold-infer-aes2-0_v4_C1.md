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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
tensorflow-datasets==4.9.9
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
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
os.environ.setdefault("TRANSFORMERS_NO_JAX", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import sys

for m in list(sys.modules.keys()):
    if m.startswith("google.protobuf"):
        sys.modules.pop(m, None)



## === cell 1
import random
import glob
import warnings
import numpy as np
import pandas as pd
import torch

warnings.simplefilter("ignore")

from transformers import AutoTokenizer, AutoModelForSequenceClassification




## === cell 2
class PATHS:
    test_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
    test_path_alt = (
        "/kaggle/data/learning-agency-lab-automated-essay-scoring-2/test.csv"
    )
    model_dir = "/kaggle/input/groupkfold-deberta-aes2-0/"




## === cell 3
class CFG:
    max_length = 512
    num_labels = 6




## === cell 4
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


seed_everything(42)




## === cell 5
class Tokenize(object):
    def __init__(self, test, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.test = test

    def __call__(self):
        texts = self.test["full_text"].tolist()
        enc = self.tokenizer(
            texts,
            truncation=True,
            max_length=CFG.max_length,
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors=None,
            padding=False,
        )
        return enc, self.tokenizer




## === cell 6
if os.path.exists(PATHS.test_path):
    test = pd.read_csv(PATHS.test_path)
elif os.path.exists(PATHS.test_path_alt):
    test = pd.read_csv(PATHS.test_path_alt)
else:
    raise FileNotFoundError(
        f"Could not find test.csv at {PATHS.test_path} or {PATHS.test_path_alt}"
    )

model_paths = []
if os.path.exists(PATHS.model_dir):
    cand = glob.glob(os.path.join(PATHS.model_dir, "*fold*"))
    cand = [p for p in cand if os.path.isdir(p)]
    cand.sort()
    model_paths = cand

if len(model_paths) == 0:
    model_paths = ["distilbert-base-uncased"]

model_paths



## === cell 7
from torch.utils.data import DataLoader, Dataset

use_fp16 = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    tokenize = Tokenize(test, model_paths[0])
    tokenized_test, tokenizer = tokenize()
except Exception as e:
    print(
        f"[WARN] Tokenizer load failed for '{model_paths[0]}': {type(e).__name__}: {e}"
    )
    safe_tok = "distilbert-base-uncased"
    tokenize = Tokenize(test, safe_tok)
    tokenized_test, tokenizer = tokenize()

pad_id = tokenizer.pad_token_id
if pad_id is None:
    pad_id = 0

input_ids_list = tokenized_test["input_ids"]
attn_list = tokenized_test["attention_mask"]
n_test = len(input_ids_list)
max_len = CFG.max_length  # fixed to config; tokenization already truncated to this max

input_ids_np = np.full((n_test, max_len), pad_id, dtype=np.int64)
attn_np = np.zeros((n_test, max_len), dtype=np.int64)

for i, (ids, am) in enumerate(zip(input_ids_list, attn_list)):
    l = len(ids)
    if l > max_len:
        l = max_len
    input_ids_np[i, :l] = ids[:l]
    attn_np[i, :l] = am[:l]

input_ids_t = torch.from_numpy(input_ids_np)
attn_t = torch.from_numpy(attn_np)


class EncodedDataset(Dataset):
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }


per_device_eval_bs = 64 if torch.cuda.is_available() else 16  # safe: inference only
num_workers = 2 if torch.cuda.is_available() else 0

dl = DataLoader(
    EncodedDataset(input_ids_t, attn_t),
    batch_size=per_device_eval_bs,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=2 if num_workers > 0 else None,
)


def _maybe_compile(model):
    if hasattr(torch, "compile"):
        try:
            return torch.compile(model, mode="reduce-overhead", fullgraph=False)
        except Exception:
            return model
    return model


def predict_logits(model, dataloader):
    model.eval()
    all_logits = []
    with torch.inference_mode():
        for batch in dataloader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            if use_fp16 and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = model(**batch)
            else:
                out = model(**batch)
            all_logits.append(out.logits.detach().cpu().numpy())
    return np.concatenate(all_logits, axis=0)


final_pred_logits = None
n_models = 0

for i, model_path in enumerate(model_paths):
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=CFG.num_labels
        ).to(device)
    except Exception as e:
        print(f"[WARN] Model load failed for '{model_path}': {type(e).__name__}: {e}")
        continue

    model = _maybe_compile(model)

    pre_preds = predict_logits(model, dl)  # (n_test, num_labels)

    if final_pred_logits is None:
        if (
            pre_preds.ndim != 2
            or pre_preds.shape[0] != n_test
            or pre_preds.shape[1] != CFG.num_labels
        ):
            raise ValueError(
                f"Unexpected prediction shape at model {i}: {pre_preds.shape}"
            )
        final_pred_logits = np.zeros_like(pre_preds, dtype=np.float64)

    final_pred_logits += pre_preds.astype(np.float64, copy=False)
    n_models += 1

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if n_models == 0 or final_pred_logits is None:
    fallback_model_name = "distilbert-base-uncased"
    print(
        f"[WARN] All provided models failed; falling back to '{fallback_model_name}'."
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        fallback_model_name, num_labels=CFG.num_labels
    ).to(device)
    model = _maybe_compile(model)
    final_pred_logits = predict_logits(model, dl).astype(np.float64, copy=False)
    n_models = 1
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

n_models, final_pred_logits.shape



## === cell 8
final_pred_logits /= n_models
final_pred = final_pred_logits.argmax(axis=1) + 1  # class 0-5 -> score 1-6
final_pred = np.clip(final_pred, 1, 6)

submission = pd.DataFrame(
    {"essay_id": test["essay_id"].values, "score": final_pred.astype(int)}
)
submission.to_csv("submission.csv", index=False)

submission.shape, submission.dtypes, submission.head()
