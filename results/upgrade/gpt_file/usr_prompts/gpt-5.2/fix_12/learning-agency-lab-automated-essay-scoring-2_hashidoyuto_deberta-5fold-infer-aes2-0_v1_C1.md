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
    Bugfix (runtime): ensure we can load a model even if the expected Kaggle Dataset model_dir isn't attached.
    Priority:
      1) Use a valid local directory containing config.json (original intent).
      2) Search under /kaggle/input for any attached HF model directory.
      3) Fall back to a downloadable HF hub model id (keeps inference pipeline intact).
    """

    def _resolve_under(root_dir: str) -> str:
        root_dir = os.path.normpath(root_dir)

        if os.path.isdir(root_dir) and os.path.exists(
            os.path.join(root_dir, "config.json")
        ):
            return root_dir

        fold_candidates = sorted(
            [p for p in glob.glob(os.path.join(root_dir, "*fold*")) if os.path.isdir(p)]
        )
        for p in fold_candidates:
            if os.path.exists(os.path.join(p, "config.json")):
                return os.path.normpath(p)

        config_hits = glob.glob(
            os.path.join(root_dir, "**", "config.json"), recursive=True
        )
        if len(config_hits) > 0:
            config_hits.sort(key=lambda x: (x.count(os.sep), x))
            return os.path.normpath(os.path.dirname(config_hits[0]))

        raise FileNotFoundError

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
test = pd.read_csv(PATHS.test_path)

model_path = resolve_model_path(PATHS.model_dir)
local_only = os.path.isdir(model_path)

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=local_only,
    use_fast=True,
)

texts = test["full_text"].tolist()

enc = tokenizer(
    texts,
    truncation=True,
    max_length=CFG.max_length,
    padding="max_length",
    return_attention_mask=True,
    return_token_type_ids=False,
)

input_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
attention_mask = torch.tensor(enc["attention_mask"], dtype=torch.long)


class PreTokenizedDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx: int):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }


tokenized_test = PreTokenizedDataset(input_ids, attention_mask)




## === cell 7
model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    num_labels=CFG.num_labels,
    local_files_only=local_only,
    ignore_mismatched_sizes=True,  # keep identical behavior to original
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

USE_TORCH_COMPILE = False
if USE_TORCH_COMPILE and hasattr(torch, "compile"):
    try:
        model = torch.compile(model, mode="reduce-overhead")
    except Exception:
        pass

per_device_eval_bs = 64 if torch.cuda.is_available() else 32

num_workers = 0

loader = torch.utils.data.DataLoader(
    tokenized_test,
    batch_size=per_device_eval_bs,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
)

all_logits = []
with torch.inference_mode():
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        outputs = model(**batch)
        all_logits.append(outputs.logits.detach().cpu())

predictions = torch.cat(all_logits, dim=0).numpy()

pred_scores = predictions.argmax(axis=1).astype(np.int64) + 1
pred_scores = np.clip(pred_scores, 1, 6)

if len(pred_scores) != len(test):
    raise RuntimeError(
        f"Prediction length mismatch: {len(pred_scores)} vs test {len(test)}"
    )

submission = pd.DataFrame({"essay_id": test["essay_id"].values, "score": pred_scores})
submission.to_csv("submission.csv", index=False)

submission
