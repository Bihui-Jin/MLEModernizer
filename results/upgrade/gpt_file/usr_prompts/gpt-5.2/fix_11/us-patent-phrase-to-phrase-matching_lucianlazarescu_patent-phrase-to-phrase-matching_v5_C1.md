# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Given pairs of phrases (an `anchor` and a `target` phrase), build a model to rate how similar they are.  

## Metric
Pearson correlation coefficient.

## Submission Format
For each `id` (representing a pair of phrases) in the test set, you must predict the similarity `score`. The file should contain a header and have the following format:

```
id,score
4112d61851461f60,0
09e418c93a776564,0.25
36baf228038e314b,1
etc.

```

## Dataset
The scores are in the 0-1 range with increments of 0.25 with the following meanings:

- **1.0** - Very close match. This is typically an exact match except possibly for differences in conjugation, quantity (e.g. singular vs. plural), and addition or removal of stopwords (e.g. "the", "and", "or").
- **0.75** - Close synonym, e.g. "mobile phone" vs. "cellphone". This also includes abbreviations, e.g. "TCP" -> "transmission control protocol".
- **0.5** - Synonyms which don't have the same meaning (same function, same properties). This includes broad-narrow (hyponym) and narrow-broad (hypernym) matches.
- **0.25** - Somewhat related, e.g. the two phrases are in the same high level domain but are not synonyms. This also includes antonyms.
- **0.0** - Unrelated.

Files
-----

- **train.csv** - the training set, containing phrases, contexts, and their similarity scores
- **test.csv** - the test set set, identical in structure to the training set but without the score
- **sample_submission.csv** - a sample submission file in the correct format

Columns
-------

- `id` - a unique identifier for a pair of phrases
- `anchor` - the first phrase
- `target` - the second phrase
- `context` - the [CPC classification (version 2021.05)](https://en.wikipedia.org/wiki/Cooperative_Patent_Classification), which indicates the subject within which the similarity is to be scored
- `score` - the similarity. This is sourced from a combination of one or more manual expert ratings.

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
            description.md (118 lines)
            sample_submission.csv (3649 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (3649 lines)
            test.csv.zip (86.4 kB)
            train.csv (32826 lines)
            train.csv.zip (790.5 kB)
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
        input/
            description.md (118 lines)
            sample_submission.csv (3649 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (3649 lines)
            test.csv.zip (86.4 kB)
            train.csv (32826 lines)
            train.csv.zip (790.5 kB)
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
        working/
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
```

-> data/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> data/test.csv has 3648 rows and 4 columns.
The columns are: id, anchor, target, context

-> data/train.csv has 32825 rows and 5 columns.
The columns are: id, anchor, target, context, score

-> data/us-patent-phrase-to-phrase-matching/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> data/us-patent-phrase-to-phrase-matching/test.csv has 3648 rows and 4 columns.
The columns are: id, anchor, target, context

-> data/us-patent-phrase-to-phrase-matching/train.csv has 32825 rows and 5 columns.
The columns are: id, anchor, target, context, score

-> input/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import random

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


set_seed(42)



## === cell 1
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from transformers.trainer_utils import EvalPrediction
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

from transformers import __version__ as transformers_version

print("transformers:", transformers_version)

import torch

print("torch:", torch.__version__, "cuda_available:", torch.cuda.is_available())

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass
torch.backends.cudnn.benchmark = False

try:
    torch.set_num_threads(min(4, os.cpu_count() or 2))
except Exception:
    pass

try:
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
except Exception:
    pass




## === cell 2
def corr(x, y):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = x.size
    if n == 0 or y.size == 0:
        return 0.0
    xm = x.mean()
    ym = y.mean()
    xv = x - xm
    yv = y - ym
    denom = np.sqrt(np.dot(xv, xv) * np.dot(yv, yv))
    if denom == 0.0:
        return 0.0
    c = float(np.dot(xv, yv) / denom)
    if np.isnan(c):
        return 0.0
    return c




## === cell 3
def corr_d(eval_pred: EvalPrediction):
    preds = eval_pred.predictions
    labels = eval_pred.label_ids
    preds = np.asarray(preds).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    return {"pearson": corr(preds, labels)}




## === cell 4
path = "/kaggle/input/us-patent-phrase-to-phrase-matching/"
train_data = pd.read_csv(path + "train.csv")
test_data = pd.read_csv(path + "test.csv")

print(train_data.head())
print(test_data.head())
print("train shape:", train_data.shape, "test shape:", test_data.shape)



## === cell 5
train_data["section"] = train_data.context.str[0]
test_data["section"] = test_data.context.str[0]
print(train_data.section.value_counts().head())



## === cell 6
model_nm = "microsoft/deberta-v3-small"


def load_tokenizer_and_model(model_id: str):
    candidate_paths = [model_id]
    for base in ["/kaggle/input", "/kaggle/working"]:
        cand = os.path.join(base, model_id.replace("/", "_"))
        candidate_paths.append(cand)

    last_err = None
    for p in candidate_paths:
        try:
            tok = AutoTokenizer.from_pretrained(p, use_fast=True)
            mdl = AutoModelForSequenceClassification.from_pretrained(p, num_labels=1)
            return tok, mdl
        except Exception as e:
            last_err = e
    raise RuntimeError(
        f"Could not load tokenizer/model for '{model_id}'. Last error: {last_err}"
    )


tokenizer, model = load_tokenizer_and_model(model_nm)

sep = tokenizer.sep_token if tokenizer.sep_token is not None else "[SEP]"
print("sep token:", sep)

ENABLE_TORCH_COMPILE = False
if ENABLE_TORCH_COMPILE:
    try:
        if hasattr(torch, "compile"):
            model = torch.compile(model)
            print("torch.compile enabled")
    except Exception as e:
        print("torch.compile not enabled:", repr(e))




## === cell 7
def prepare_data(df: pd.DataFrame):
    ctx = df["context"].astype(str).to_numpy()
    anc = df["anchor"].astype(str).to_numpy()
    tgt = df["target"].astype(str).to_numpy()
    df["input"] = (
        pd.Series(ctx) + sep + pd.Series(anc) + sep + pd.Series(tgt)
    ).to_numpy()


prepare_data(train_data)
prepare_data(test_data)

print(train_data[["input", "score"]].head())



## === cell 8
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


def tokenize_texts(texts, tokenizer, max_length: int):
    enc = tokenizer(
        list(texts),
        truncation=True,
        max_length=max_length,
        padding=False,  # dynamic padding via collator
        return_attention_mask=True,
    )
    input_ids = enc["input_ids"]
    attn = enc["attention_mask"]
    return input_ids, attn


class TextRegressionTokenizedDataset(torch.utils.data.Dataset):
    def __init__(
        self, input_ids, attention_mask, labels=None, with_labels: bool = True
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.with_labels = with_labels
        self.labels = None
        if with_labels:
            self.labels = np.asarray(labels, dtype=np.float32)

    def __len__(self):
        return int(len(self.input_ids))

    def __getitem__(self, idx: int):
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }
        if self.with_labels:
            item["labels"] = self.labels[idx]
        return item


@dataclass
class DynamicPaddingRegressionCollator:
    tokenizer: Any
    pad_to_multiple_of: Optional[int] = 8  # faster tensor cores on GPU, harmless on CPU

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        labels = None
        if "labels" in features[0]:
            labels = torch.tensor([f["labels"] for f in features], dtype=torch.float32)
            features = [{k: v for k, v in f.items() if k != "labels"} for f in features]

        batch = self.tokenizer.pad(
            features,
            padding=True,
            max_length=None,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        if labels is not None:
            batch["labels"] = labels
        return batch


rng = np.random.RandomState(42)
idx = np.arange(len(train_data))
rng.shuffle(idx)
cut = int(len(idx) * (1 - 0.25))
train_idx, valid_idx = idx[:cut], idx[cut:]

train_df = train_data.iloc[train_idx].copy()
valid_df = train_data.iloc[valid_idx].copy()

max_length = 256
train_input_ids, train_attn = tokenize_texts(
    train_df["input"].values, tokenizer, max_length
)
valid_input_ids, valid_attn = tokenize_texts(
    valid_df["input"].values, tokenizer, max_length
)
test_input_ids, test_attn = tokenize_texts(
    test_data["input"].values, tokenizer, max_length
)

train_labels = train_df["score"].to_numpy(dtype=np.float32, copy=True)
valid_labels = valid_df["score"].to_numpy(dtype=np.float32, copy=True)

train_ds = TextRegressionTokenizedDataset(
    train_input_ids,
    train_attn,
    labels=train_labels,
    with_labels=True,
)
valid_ds = TextRegressionTokenizedDataset(
    valid_input_ids,
    valid_attn,
    labels=valid_labels,
    with_labels=True,
)
eval_ds = TextRegressionTokenizedDataset(
    test_input_ids, test_attn, labels=None, with_labels=False
)

data_collator = DynamicPaddingRegressionCollator(
    tokenizer=tokenizer, pad_to_multiple_of=8
)

print("train/valid sizes:", len(train_ds), len(valid_ds), "test:", len(eval_ds))




## === cell 9
class CompatTrainer(Trainer):
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        inputs.pop("num_items_in_batch", None)
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        return (loss, outputs) if return_outputs else loss


bs = 32
epochs = 4
lr = 8e-5

use_fp16 = torch.cuda.is_available()
num_workers = min(4, (os.cpu_count() or 2))

args = TrainingArguments(
    output_dir="outputs",
    learning_rate=lr,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    fp16=use_fp16,
    eval_strategy="epoch",
    per_device_train_batch_size=bs,
    per_device_eval_batch_size=bs * 2,
    num_train_epochs=epochs,
    weight_decay=0.01,
    report_to="none",
    save_strategy="no",
    logging_steps=50,
    remove_unused_columns=False,
    dataloader_num_workers=num_workers,
    dataloader_pin_memory=torch.cuda.is_available(),
    dataloader_persistent_workers=(num_workers > 0),
    group_by_length=False,
    optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
    eval_accumulation_steps=32,
    prediction_loss_only=False,
)

trainer = CompatTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=corr_d,
)



## === cell 10
trainer.train()



## === cell 11
with torch.no_grad():
    pred_out = trainer.predict(eval_ds)
preds = np.asarray(pred_out.predictions, dtype=float).reshape(-1)
preds = np.clip(preds, 0.0, 1.0)

score = np.select(
    [
        preds >= 0.875,
        preds >= 0.625,
        preds >= 0.375,
        preds >= 0.125,
    ],
    [1.0, 0.75, 0.5, 0.25],
    default=0.0,
).astype(float)

print("preds range:", float(np.min(preds)), float(np.max(preds)))
print("unique snapped scores:", sorted(set(score.tolist())))



## === cell 12
submission = pd.DataFrame({"id": test_data["id"].values, "score": score})
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Saved to:", os.path.abspath("submission.csv"))
