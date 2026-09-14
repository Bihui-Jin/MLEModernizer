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

# 5. Target score

0.741923748724992

# 6. Current score

-0.00633

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -0.03275) has done: 'The timeout is dominated by tokenizing all 15k essays up front into Python lists (very slow and memory-heavy) and by padding/batching overhead in the DataLoader. I keep the same model and inference logic, but switch to on-the-fly tokenization per batch using a Dataset that stores raw texts and a collator that tokenizes batches (fast tokenizer + Rust parallelism) to avoid building a huge `encodings` object. I also enable `model.eval()` + `torch.inference_mode()` as before, keep determinism settings, and use efficient DataLoader settings (workers, prefetch, pinned memory) while preserving identical truncation/max_length/padding semantics and argmax-to-score mapping.'
- What this solution (achieved -0.0074) has done: 'I fix two execution blockers: the offline HuggingFace model loading failure (by loading a compatible transformer checkpoint that is already available offline in the Kaggle environment) and the DataLoader collator error caused by passing `padding` twice. These changes keep the same inference-only, argmax-classification-to-1..6 mapping core logic, but make the notebook run end-to-end and reliably write `submission.csv`. Since your current score is extremely low, switching to a locally available pretrained sequence-classification checkpoint should also move the score substantially toward the target band (while still preserving the same evaluation semantics).'
- What this solution (achieved -0.00316) has done: 'I fix the execution blockers by (1) loading a transformer checkpoint from the local Kaggle filesystem instead of relying on an offline HuggingFace cache, and (2) ensuring the `model`/`tokenizer` are always defined so later cells don’t crash. To move the score up toward your target, I keep the same inference-only argmax-to-1..6 mapping, but switch the locally loaded checkpoint to a model that is actually trained for essay scoring (stored under the competition input folder). I also make the model’s output head size match 6 labels without re-downloading anything, keeping the existing evaluation semantics. Finally, I ensure `submission.csv` is written with the required columns and row count.'
- What this solution (achieved -0.00633) has done: 'I fix the root execution blocker: `transformers` is treating your local filesystem path as an invalid Hub repo id, so the tokenizer/model never load and downstream cells crash with `model is not defined`. The minimal robust fix is to load from the local directory via `pathlib.Path` (which `transformers` accepts as a local path) and add a tiny fallback to a known-available Kaggle transformer cache only if the expected folder is missing. I also add an explicit post-load check that the tokenizer/model actually exist before proceeding, keeping all inference logic (argmax over 6 labels, +1 mapping) unchanged. This should both run end-to-end and move score upward substantially versus the broken/near-random outputs.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

torch.set_grad_enabled(False)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True



## === cell 1
test_data = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
)
test_data



## === cell 2
MODEL_DIR = Path(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/learning-agency-lab-automated-essay-scoring-2"
)
MODEL_NAME = MODEL_DIR / "deberta-v3-base"  # local folder path

if not MODEL_NAME.exists():
    candidates = [
        Path("/kaggle/input") / "deberta-v3-base",
        Path("/kaggle/input") / "microsoft-deberta-v3-base",
        Path("/kaggle/working") / "deberta-v3-base",
    ]
    found = None
    for c in candidates:
        if c.exists():
            found = c
            break
    if found is None:
        raise FileNotFoundError(
            f"Expected local model directory not found: {MODEL_NAME}. "
            f"Also tried: {candidates}"
        )
    MODEL_NAME = found

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, local_files_only=True, use_fast=True
)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, local_files_only=True
)

if getattr(model.config, "num_labels", None) != 6:
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=6,
        ignore_mismatched_sizes=True,
        local_files_only=True,
    )

model.eval()



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3878572333.py in <cell line: 0>()
     20             break
     21     if found is None:
---> 22         raise FileNotFoundError(
     23             f"Expected local model directory not found: {MODEL_NAME}. "
     24             f"Also tried: {candidates}"

FileNotFoundError: Expected local model directory not found: /kaggle/input/learning-agency-lab-automated-essay-scoring-2/learning-agency-lab-automated-essay-scoring-2/deberta-v3-base. Also tried: [PosixPath('/kaggle/input/deberta-v3-base'), PosixPath('/kaggle/input/microsoft-deberta-v3-base'), PosixPath('/kaggle/working/deberta-v3-base')]

## === cell 3
MAX_LEN = 1536

tok_kwargs = dict(
    truncation=True,
    max_length=MAX_LEN,
    padding=False,  # collator will pad dynamically
    return_attention_mask=True,
    return_token_type_ids=False,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")


class RawTextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


test_dataset = RawTextDataset(test_data["full_text"].tolist())


class TokenizeCollator:
    def __init__(self, tokenizer, tok_kwargs, pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.tok_kwargs = {k: v for k, v in tok_kwargs.items() if k != "padding"}
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch_texts):
        enc = self.tokenizer(
            batch_texts,
            **self.tok_kwargs,
            padding="longest",
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}




## === cell 4
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model.to(device)

if hasattr(model, "config"):
    try:
        model.config.use_cache = False
    except Exception:
        pass

if use_cuda and os.environ.get("DISABLE_TORCH_COMPILE", "0") != "1":
    try:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
    except Exception:
        pass

pad_to_multiple_of = 8 if use_cuda else None
collate_fn = TokenizeCollator(
    tokenizer=tokenizer, tok_kwargs=tok_kwargs, pad_to_multiple_of=pad_to_multiple_of
)

if use_cuda:
    num_workers = min(4, os.cpu_count() or 2)
    prefetch_factor = 4
    persistent_workers = True
else:
    num_workers = 0
    prefetch_factor = None
    persistent_workers = False

dl_kwargs = dict(
    dataset=test_dataset,
    batch_size=(64 if use_cuda else 8),
    shuffle=False,
    num_workers=num_workers,
    pin_memory=use_cuda,
    collate_fn=collate_fn,
    persistent_workers=persistent_workers,
)
if num_workers > 0:
    dl_kwargs["prefetch_factor"] = prefetch_factor

test_loader = DataLoader(**dl_kwargs)

if use_cuda:
    try:
        model = model.to(memory_format=torch.channels_last)
    except Exception:
        pass



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2512762249.py in <cell line: 0>()
      1 use_cuda = torch.cuda.is_available()
      2 device = torch.device("cuda" if use_cuda else "cpu")
----> 3 model.to(device)
      4 
      5 if hasattr(model, "config"):

NameError: name 'model' is not defined

## === cell 5
pred_classes = np.empty(len(test_dataset), dtype=np.int64)

model.eval()
offset = 0

with torch.inference_mode():
    for batch in test_loader:
        if use_cuda:
            batch["input_ids"] = batch["input_ids"].to(device, non_blocking=True)
            batch["attention_mask"] = batch["attention_mask"].to(
                device, non_blocking=True
            )
        else:
            batch["input_ids"] = batch["input_ids"].to(device)
            batch["attention_mask"] = batch["attention_mask"].to(device)

        outputs = model(**batch)
        batch_pred = torch.argmax(outputs.logits, dim=-1)
        batch_pred_np = batch_pred.cpu().numpy().astype(np.int64, copy=False)

        bs = batch_pred_np.shape[0]
        pred_classes[offset : offset + bs] = batch_pred_np
        offset += bs

assert offset == len(test_dataset)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3355896025.py in <cell line: 0>()
      1 pred_classes = np.empty(len(test_dataset), dtype=np.int64)
      2 
----> 3 model.eval()
      4 offset = 0
      5 

NameError: name 'model' is not defined

## === cell 6
predicted_scores = (pred_classes + 1).clip(1, 6).astype("int32")



## === cell 7
submission = pd.DataFrame(
    {"essay_id": test_data["essay_id"].values, "score": predicted_scores}
)

submission.to_csv("submission.csv", index=False)
submission



## === cell 8
assert submission.shape[0] == test_data.shape[0]
assert list(submission.columns) == ["essay_id", "score"]
assert submission["score"].between(1, 6).all()
print("Wrote submission.csv with shape:", submission.shape)
