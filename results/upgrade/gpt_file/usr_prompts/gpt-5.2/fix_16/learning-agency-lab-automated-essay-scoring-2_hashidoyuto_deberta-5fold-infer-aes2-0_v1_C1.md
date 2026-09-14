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

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")



## === cell 1
import sys
import subprocess



## === cell 2
import random
import glob
import warnings
import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification

warnings.simplefilter("ignore")


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(42)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True




## === cell 3
class PATHS:
    test_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
    model_dir = "/kaggle/input/debert-v3-base-for-aes2-0/"




## === cell 4
class CFG:
    max_length = 512
    num_labels = 6




## === cell 5
def resolve_model_path(model_dir: str) -> str:
    """
    Speed-preserving change: avoid an expensive recursive scan of /kaggle/input
    unless the requested model_dir is actually missing/invalid.
    Fallback behavior remains identical if model_dir isn't usable.
    """

    def _is_valid_model_dir(p: str) -> bool:
        p = os.path.normpath(p)
        return os.path.isdir(p) and os.path.exists(os.path.join(p, "config.json"))

    def _resolve_under(root_dir: str) -> str:
        root_dir = os.path.normpath(root_dir)

        if _is_valid_model_dir(root_dir):
            return root_dir

        fold_candidates = sorted(
            [p for p in glob.glob(os.path.join(root_dir, "*fold*")) if os.path.isdir(p)]
        )
        for p in fold_candidates:
            if _is_valid_model_dir(p):
                return os.path.normpath(p)

        config_hits = glob.glob(
            os.path.join(root_dir, "**", "config.json"), recursive=True
        )
        if len(config_hits) > 0:
            config_hits.sort(key=lambda x: (x.count(os.sep), x))
            return os.path.normpath(os.path.dirname(config_hits[0]))

        raise FileNotFoundError

    if _is_valid_model_dir(model_dir):
        return os.path.normpath(model_dir)

    try:
        return _resolve_under(model_dir)
    except FileNotFoundError:
        pass

    try:
        return _resolve_under("/kaggle/input")
    except FileNotFoundError:
        pass

    return "microsoft/deberta-v3-base"




## === cell 6
from torch.utils.data import Dataset, DataLoader

test = pd.read_csv(PATHS.test_path)

model_path = resolve_model_path(PATHS.model_dir)
local_only = os.path.isdir(model_path)

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=local_only,
    use_fast=True,
)

texts = test["full_text"].tolist()

model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    num_labels=CFG.num_labels,
    local_files_only=local_only,
    ignore_mismatched_sizes=True,  # keep identical behavior to original
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
torch.set_grad_enabled(False)

if torch.cuda.is_available():
    try:
        model = model.to(memory_format=torch.channels_last)
    except Exception:
        pass

USE_TORCH_COMPILE = False
if USE_TORCH_COMPILE and hasattr(torch, "compile"):
    try:
        model = torch.compile(model, mode="reduce-overhead")
    except Exception:
        pass

per_device_eval_bs = 128 if torch.cuda.is_available() else 64


class TextDataset(Dataset):
    def __init__(self, texts_):
        self.texts = texts_

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


def collate_fn(batch_texts):
    enc = tokenizer(
        batch_texts,
        truncation=True,
        max_length=CFG.max_length,
        padding=True,
        return_attention_mask=True,
        return_token_type_ids=False,
        return_tensors="pt",
    )
    return enc


ds = TextDataset(texts)
dl = DataLoader(
    ds,
    batch_size=per_device_eval_bs,
    shuffle=False,
    num_workers=2 if torch.cuda.is_available() else 0,
    pin_memory=torch.cuda.is_available(),
    collate_fn=collate_fn,
    persistent_workers=True if (torch.cuda.is_available()) else False,
)

n = len(texts)
logits_out = np.empty((n, CFG.num_labels), dtype=np.float32)

offset = 0
with torch.inference_mode():
    for enc in dl:
        input_ids = enc["input_ids"].to(device, non_blocking=True)
        attention_mask = enc["attention_mask"].to(device, non_blocking=True)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        bs = outputs.logits.size(0)
        logits_out[offset : offset + bs] = outputs.logits.detach().float().cpu().numpy()
        offset += bs

pred_scores = logits_out.argmax(axis=1).astype(np.int64) + 1
pred_scores = np.clip(pred_scores, 1, 6)

submission = pd.DataFrame({"essay_id": test["essay_id"].values, "score": pred_scores})
submission.to_csv("submission.csv", index=False)

submission
