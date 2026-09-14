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
Given questions and answers from various StackExchange properties, predict target values of 30 labels for each question-answer pair.

## Metric
Mean column-wise Spearman's correlation coefficient. The Spearman's rank correlation is computed for each target column, and the mean of these values is calculated for the submission score.

## Submission Format
For each qa_id in the test set, you must predict a probability for each target variable. The predictions should be in the range [0,1]. The file should contain a header and have the following format:

```
qa_id,question_asker_intent_understanding,...,answer_well_written
6,0.0,...,0.5
8,0.5,...,0.1
18,1.0,...,0.0
etc.
```

## Dataset
The list of 30 target labels are the same as the column names in the `sample_submission.csv` file. Target labels with the prefix `question_` relate to the `question_title` and/or `question_body` features in the data. Target labels with the prefix `answer_` relate to the `answer` feature.

Target labels are aggregated from multiple raters, and can have continuous values in the range `[0,1]`. Therefore, predictions must also be in that range.

- **train.csv** - the training data (target labels are the last 30 columns)
- **test.csv** - the test set (you must predict 30 labels for each test set row)
- **sample_submission.csv** - a sample submission file in the correct format; column names are the 30 target labels

# 2. Python version

3.8

# 3. Installed packages

fastai==2.8.5
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
scipy==1.15.3
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (83 lines)
            sample_submission.csv (609 lines)
            sample_submission.csv.zip (8.9 kB)
            test.csv (19551 lines)
            test.csv.zip (471.7 kB)
            train.csv (159837 lines)
            train.csv.zip (4.2 MB)
            google-quest-challenge/
                description.md (83 lines)
                sample_submission.csv (609 lines)
                ... and 5 other files
                google-quest-challenge/
        input/
            description.md (83 lines)
            sample_submission.csv (609 lines)
            sample_submission.csv.zip (8.9 kB)
            test.csv (19551 lines)
            test.csv.zip (471.7 kB)
            train.csv (159837 lines)
            train.csv.zip (4.2 MB)
            google-quest-challenge/
                description.md (83 lines)
                sample_submission.csv (609 lines)
                ... and 5 other files
                google-quest-challenge/
        working/
            google-quest-challenge/
                description.md (83 lines)
                sample_submission.csv (609 lines)
                ... and 5 other files
                google-quest-challenge/
```

-> data/google-quest-challenge/sample_submission.csv has 608 rows and 31 columns.
The columns are: qa_id, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer, question_fact_seeking, question_has_commonly_accepted_answer, question_interestingness_others, question_interestingness_self, question_multi_intent, question_not_really_a_question, question_opinion_seeking, question_type_choice, question_type_compare, question_type_consequence... and 16 more columns

-> data/google-quest-challenge/test.csv has 19550 rows and 11 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host

-> data/google-quest-challenge/train.csv has 159836 rows and 41 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer... and 26 more columns

-> data/sample_submission.csv has 608 rows and 31 columns.
The columns are: qa_id, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer, question_fact_seeking, question_has_commonly_accepted_answer, question_interestingness_others, question_interestingness_self, question_multi_intent, question_not_really_a_question, question_opinion_seeking, question_type_choice, question_type_compare, question_type_consequence... and 16 more columns

-> data/test.csv has 19550 rows and 11 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host

-> data/train.csv has 159836 rows and 41 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer... and 26 more columns

-> (stopped after 10 files for performance)

# 5. Target score

0.2415212217454294

# 6. Current score

0.36448

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved nan) has done: 'I make the notebook compatible with the installed fastai v2 by removing fastai v1-only imports/classes (e.g., `fastai.callbacks`, `TextList`, `BaseTokenizer`, `Vocab`, `DatasetType`) and replacing them with a minimal Hugging Face + PyTorch training/inference loop that preserves the core idea: fine-tune `roberta-base` for 30 continuous targets. I also fix the invalid model path usage (`MODEL_ROOT` doesn’t exist) by loading `roberta-base` from the local Hugging Face cache in the Kaggle image, and ensure predictions are clipped to `[0,1]` as required. Finally, I guarantee `submission.csv` is written with the exact `sample_submission.csv` columns and correct row alignment with `test.csv`.'
- What this solution (achieved 0.36448) has done: 'I fix the import-time crash coming from an incompatible `protobuf`/`sentencepiece` stack pulled in by `transformers` by forcing the pure-Python protobuf implementation before any HF imports. Then I fix the `AdamW` NameError by importing it from `torch.optim` (Transformers no longer reliably exposes it the old way), which also restore `num_epochs`/scheduler initialization so training runs and the submission is produced. I keep the model/loop/feature construction unchanged and only make these compatibility fixes so you get a valid `submission.csv` end-to-end. Finally, I add a small safety fallback to load tokenizer/model locally if online access is disabled.'
- What this solution (achieved 0.36448) has done: 'I fix the import-time crash caused by an incompatible protobuf C++ runtime by forcing the pure-Python protobuf implementation *and* disabling the C++ descriptor pool before any `transformers`-related imports occur. This is a correctness/stability fix only; it keeps the same Roberta regression model, data construction, training loop, and prediction clipping, so it should not materially change your achieved score aside from negligible nondeterminism effects. I also add a small tokenizer/model fallback to avoid sentencepiece usage and ensure everything loads offline in Kaggle. Finally, I keep the submission writing logic identical while ensuring the run completes end-to-end and produces `submission.csv`.'
- What this solution (achieved 0.36448) has done: 'I fix the import-time crash (`MessageFactory.GetPrototype` protobuf mismatch) by forcing a compatible protobuf runtime path before any `transformers` import and by avoiding optional protobuf-dependent parts during import. This is a stability/correctness change only and keeps the same RoBERTa regression model, dataset construction, training loop, and prediction post-processing, so it should not materially change your score (and you’re already above the target band). I also add a safe fallback to `AutoTokenizer/AutoModelForSequenceClassification` in case a specific Roberta class import triggers the same protobuf path on this image, while keeping the underlying model identical (`roberta-base`). Finally, I keep the submission writing exactly aligned to `sample_submission.csv` and ensure `submission.csv` is always produced.'
- What this solution (achieved 0.36448) has done: 'I fix the protobuf-related crash by forcing the pure-Python protobuf backend earlier and more robustly (before any Transformers sub-imports can trigger the incompatible C++ descriptor path), and by using `transformers.utils.import_utils` settings to avoid optional protobuf-dependent imports. This is a runtime/stability fix only and does not change your model architecture, data, loss, or training loop, so the score should remain in the same neighborhood (already above the target band). I also keep the offline-safe `local_files_only` loading logic intact and ensure `submission.csv` is always written with the exact `sample_submission.csv` columns and test-row alignment. No score-chasing changes are introduced.'
- What this solution (achieved 0.36448) has done: 'I fix the protobuf/transformers import crash that triggers `MessageFactory.GetPrototype` by setting the protobuf environment variables *before any protobuf-related modules can load* and by forcing a safe import order; this is a runtime/stability fix and keeps your model/training logic unchanged. I also make the HF offline loading more robust by preferring the competition dataset’s bundled HF files if present, otherwise falling back to the standard cache/local resolution. Finally, I keep submission writing identical but ensure it always uses the test `qa_id` ordering and the exact `sample_submission.csv` label columns, with predictions clipped to `[0,1]`.'
- What this solution (achieved 0.36448) has done: 'We fix the runtime crash (`MessageFactory.GetPrototype`) by forcing a compatible protobuf runtime *before* any Transformers import and by additionally disabling TF/protobuf heavy optional imports via environment flags; this is a stability fix that doesn’t change your model/training logic. To keep execution robust in Kaggle, we also add a safe fallback to load the model/tokenizer from the common Kaggle HF cache if `local_files_only=True` can’t resolve directly. Since your current score (0.36448) is already well above the target (0.2415) and within the allowed tolerance band, we avoid any score-tuning changes and focus strictly on correctness/end-to-end submission generation. The rest of the pipeline (RoBERTa regression head, MSE loss, epochs, max_len, clipping, submission columns) remains unchanged.'
- What this solution (achieved 0.36448) has done: 'We fix the protobuf/transformers crash (`MessageFactory.GetPrototype`) by ensuring protobuf is never imported before we force the pure-Python implementation, and by explicitly clearing any already-loaded protobuf modules from `sys.modules` before importing `transformers`. We also switch model/config/tokenizer instantiation to use `local_files_only=True` first (as you already intended) while keeping the same RoBERTa regression model, training loop, loss, epochs, max_len, and clipping unchanged (so score impact should be negligible). Finally, we keep the exact submission formatting and always write `submission.csv` with the sample’s columns and test alignment.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "3"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_DISABLE_CDESCRIPTORS"] = "1"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys

for m in list(sys.modules.keys()):
    if m.startswith("google.protobuf"):
        del sys.modules[m]

import random
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from torch.optim import AdamW




## === cell 1
def seed_all(seed_value: int):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


seed = 42
seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device



## === cell 2
DATA_ROOT = Path("/kaggle/input/google-quest-challenge")
train = pd.read_csv(DATA_ROOT / "train.csv")
test = pd.read_csv(DATA_ROOT / "test.csv")
sample_sub = pd.read_csv(DATA_ROOT / "sample_submission.csv")

print("train:", train.shape, "test:", test.shape, "sample:", sample_sub.shape)

labels = list(sample_sub.columns[1:].values)
assert len(labels) == 30
labels[:5], labels[-3:]



## === cell 3
TEXT_COLS = ["question_title", "question_body", "answer"]


def build_text(df: pd.DataFrame) -> pd.Series:
    return (
        df["question_title"].fillna("").astype(str)
        + " </s></s> "
        + df["question_body"].fillna("").astype(str)
        + " </s></s> "
        + df["answer"].fillna("").astype(str)
    )


train_text = build_text(train)
test_text = build_text(test)

y = train[labels].astype(np.float32).values



## === cell 4
n = len(train)
idx = np.arange(n)
rng = np.random.RandomState(seed)
rng.shuffle(idx)
val_size = int(0.1 * n)
val_idx = idx[:val_size]
trn_idx = idx[val_size:]

len(trn_idx), len(val_idx)



## === cell 5
pretrained_model_name = "roberta-base"

HF_CACHE_CANDIDATES = [
    Path("/kaggle/input/huggingface-models"),
    Path("/kaggle/working/hf_cache"),
    Path("/kaggle/working/.cache/huggingface"),
    Path("/root/.cache/huggingface"),
]


def _try_from_pretrained(obj, name_or_path, **kwargs):
    last_e = None
    try:
        return obj.from_pretrained(name_or_path, **kwargs)
    except Exception as e:
        last_e = e
    for base in HF_CACHE_CANDIDATES:
        hub = base / "hub"
        if hub.exists():
            pattern = f"models--{name_or_path.replace('/', '--')}"
            for mdir in hub.glob(pattern):
                snaps = mdir / "snapshots"
                if snaps.exists():
                    for snap in snaps.iterdir():
                        if snap.is_dir():
                            try:
                                return obj.from_pretrained(snap.as_posix(), **kwargs)
                            except Exception as e:
                                last_e = e
    raise last_e


try:
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name, local_files_only=True, use_fast=True
    )
except Exception:
    try:
        tokenizer = _try_from_pretrained(
            AutoTokenizer, pretrained_model_name, local_files_only=True, use_fast=True
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, use_fast=True)

max_len = 256  # preserve previous setting
bs = 16
use_fp16 = False


class QuestDataset(Dataset):
    def __init__(self, texts, targets=None):
        self.texts = list(texts)
        self.targets = targets

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = tokenizer(
            self.texts[i],
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        if self.targets is not None:
            item["labels"] = torch.tensor(self.targets[i], dtype=torch.float32)
        return item


train_ds = QuestDataset(train_text.iloc[trn_idx].values, y[trn_idx])
val_ds = QuestDataset(train_text.iloc[val_idx].values, y[val_idx])
test_ds = QuestDataset(test_text.values, targets=None)

train_dl = DataLoader(
    train_ds,
    batch_size=bs,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)
val_dl = DataLoader(
    val_ds,
    batch_size=bs,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)
test_dl = DataLoader(
    test_ds,
    batch_size=bs,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)



## === cell 6
try:
    config = AutoConfig.from_pretrained(pretrained_model_name, local_files_only=True)
except Exception:
    try:
        config = _try_from_pretrained(
            AutoConfig, pretrained_model_name, local_files_only=True
        )
    except Exception:
        config = AutoConfig.from_pretrained(pretrained_model_name)

config.num_labels = 30
config.problem_type = "regression"

try:
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name, config=config, local_files_only=True
    ).to(device)
except Exception:
    try:
        model = _try_from_pretrained(
            AutoModelForSequenceClassification,
            pretrained_model_name,
            config=config,
            local_files_only=True,
        ).to(device)
    except Exception:
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name, config=config
        ).to(device)

criterion = nn.MSELoss()
optimizer = AdamW(model.parameters(), lr=2e-5)

num_epochs = 3  # preserve previous setting
total_steps = num_epochs * len(train_dl)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
)

scaler = torch.cuda.amp.GradScaler(enabled=(use_fp16 and device.type == "cuda"))




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 7
def run_eval(model, dl):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in dl:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, targets)
            losses.append(loss.item())
    return float(np.mean(losses)) if losses else np.nan


def train_one_epoch(model, dl):
    model.train()
    losses = []
    for batch in dl:
        optimizer.zero_grad(set_to_none=True)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["labels"].to(device)

        with torch.cuda.amp.autocast(enabled=(use_fp16 and device.type == "cuda")):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        losses.append(loss.item())
    return float(np.mean(losses)) if losses else np.nan




## === cell 8
seed_all(seed)
best_val = float("inf")
best_state = None

for epoch in range(1, num_epochs + 1):
    tr_loss = train_one_epoch(model, train_dl)
    va_loss = run_eval(model, val_dl)
    print(
        f"epoch {epoch}/{num_epochs} - train_loss: {tr_loss:.5f} - val_loss: {va_loss:.5f}"
    )
    if va_loss < best_val:
        best_val = va_loss
        best_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }

if best_state is not None:
    model.load_state_dict(best_state)



## === cell 9
model.eval()
all_preds = []

with torch.no_grad():
    for batch in test_dl:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        all_preds.append(logits.detach().cpu().numpy())

test_preds = np.concatenate(all_preds, axis=0)
test_preds = np.clip(test_preds, 0.0, 1.0)

print("test_preds shape:", test_preds.shape)

submission = pd.DataFrame({"qa_id": test["qa_id"].values})
for i, c in enumerate(labels):
    submission[c] = test_preds[:, i]

submission = submission[["qa_id"] + labels]
submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())



## === cell 10
assert os.path.exists("submission.csv")
sub_chk = pd.read_csv("submission.csv")
assert list(sub_chk.columns) == ["qa_id"] + labels
assert len(sub_chk) == len(test)
assert (
    sub_chk[labels].min().min() >= 0.0 - 1e-6
    and sub_chk[labels].max().max() <= 1.0 + 1e-6
)
sub_chk.shape, sub_chk.iloc[:2, :5]
