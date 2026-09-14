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

3.10

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

# 5. Target score

0.7954764062840406

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved nan) has done: 'I fix the import-time protobuf/TF conflict causing the `MessageFactory.GetPrototype` crash by setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` before importing `transformers/datasets`. Then I fix the dataset formatting/collation issues by using the expected `labels` field (not `label`) and by disabling `remove_unused_columns` so the Trainer keeps `input_ids`/`attention_mask`. Finally, I keep the same model and training loop but ensure prediction runs and writes a valid `submission.csv` with `id,score` clipped to `[0,1]`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd

print("Kaggle input root exists:", os.path.exists("/kaggle/input"))
print("Kaggle data root exists:", os.path.exists("/kaggle/data"))



## === cell 1
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed,
)
from datasets import Dataset as HFDataset

set_seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

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


class CompatTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if "num_items_in_batch" in inputs:
            inputs = dict(inputs)
            inputs.pop("num_items_in_batch", None)
        return super().compute_loss(
            model, inputs, return_outputs=return_outputs, **kwargs
        )




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
def prepare_df(df, tokenizer):
    df = df.rename(columns={"score": "label"}).copy()
    sep = " " + (tokenizer.sep_token or "[SEP]") + " "

    ctx = df["context"].astype(str)
    anc = df["anchor"].astype(str).str.lower()
    tgt = df["target"].astype(str).str.lower()

    section = ctx.str.strip().str[0]
    sec_tok = "[" + section + "]"

    a0 = sec_tok.to_numpy(dtype=object)
    a1 = ctx.to_numpy(dtype=object)
    a2 = anc.to_numpy(dtype=object)
    a3 = tgt.to_numpy(dtype=object)

    s = np.char.add(a0, sep)
    s = np.char.add(s, a1)
    s = np.char.add(s, sep)
    s = np.char.add(s, a2)
    s = np.char.add(s, sep)
    s = np.char.add(s, a3)

    df["inputs"] = s
    return df


def tokenize_dataset(df, tokenizer, with_labels: bool):
    texts = df["inputs"].tolist()

    enc = tokenizer(
        texts,
        padding=False,  # dynamic padding via data collator
        truncation=True,
        max_length=128,
        return_attention_mask=True,
    )

    data = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
    }
    if with_labels:
        data["labels"] = df["label"].astype(np.float32).to_numpy()

    ds = HFDataset.from_dict(data)
    ds.set_format(type="torch", columns=list(data.keys()))
    return ds




## === cell 3
MODEL_NAME = "microsoft/deberta-v3-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=1,
    problem_type="regression",
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)



## === cell 4
train_path = "/kaggle/input/us-patent-phrase-to-phrase-matching/train.csv"
test_path = "/kaggle/input/us-patent-phrase-to-phrase-matching/test.csv"
if not os.path.exists(train_path):
    train_path = "/kaggle/data/us-patent-phrase-to-phrase-matching/train.csv"
if not os.path.exists(test_path):
    test_path = "/kaggle/data/us-patent-phrase-to-phrase-matching/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

train_df = prepare_df(train_df, tokenizer)
test_df = prepare_df(test_df, tokenizer)

train_ds = tokenize_dataset(train_df, tokenizer, with_labels=True)
test_ds = tokenize_dataset(test_df, tokenizer, with_labels=False)

cpu_cnt = os.cpu_count() or 1
dl_workers = min(4, cpu_cnt)

training_args = TrainingArguments(
    output_dir="out",
    overwrite_output_dir=True,
    do_train=True,
    do_eval=False,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    num_train_epochs=1.0,
    weight_decay=0.01,
    logging_steps=200,
    save_strategy="no",
    report_to=[],
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=dl_workers,
    dataloader_pin_memory=torch.cuda.is_available(),
    dataloader_persistent_workers=(dl_workers > 0),
    dataloader_prefetch_factor=2 if dl_workers > 0 else None,
    remove_unused_columns=False,
    optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
    group_by_length=True,
    length_column_name=None,  # let Trainer infer from input_ids
    prediction_loss_only=True,
)

trainer = CompatTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_ds,
)

trainer.train()

pred_out = trainer.predict(test_ds)
logits = pred_out.predictions

if isinstance(logits, (tuple, list)):
    logits = logits[0]
logits = np.asarray(logits)

if logits.ndim == 2 and logits.shape[1] == 1:
    preds = logits[:, 0]
elif logits.ndim == 1:
    preds = logits
else:
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    idx = np.arange(probs.shape[1], dtype=np.float32)
    if probs.shape[1] > 1:
        idx = idx / (probs.shape[1] - 1)
    preds = (probs * idx[None, :]).sum(axis=1)

preds = preds.astype(float)
preds = np.clip(preds, 0.0, 1.0).reshape(-1)

sub_df = pd.DataFrame({"id": test_df["id"].values, "score": preds})
sub_df.to_csv("submission.csv", index=False)

print(sub_df.head())
print("Wrote submission.csv with shape:", sub_df.shape)
print("submission.csv exists:", os.path.exists("submission.csv"))

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2355673915.py in <cell line: 0>()
      9 test_df = pd.read_csv(test_path)
     10 
---> 11 train_df = prepare_df(train_df, tokenizer)
     12 test_df = prepare_df(test_df, tokenizer)
     13 

/tmp/ipykernel_11/189668123.py in prepare_df(df, tokenizer)
     18     a3 = tgt.to_numpy(dtype=object)
     19 
---> 20     s = np.char.add(a0, sep)
     21     s = np.char.add(s, a1)
     22     s = np.char.add(s, sep)

/usr/local/lib/python3.11/dist-packages/numpy/core/defchararray.py in add(x1, x2)
    330         # object dtype itemsize as num chars (worked on short strings).
    331         # bytes + void worked but promoting void->bytes is dubious also.
--> 332         raise TypeError(
    333             "np.char.add() requires both arrays of the same dtype kind, but "
    334             f"got dtypes: '{arr1.dtype}' and '{arr2.dtype}' (the few cases "

TypeError: np.char.add() requires both arrays of the same dtype kind, but got dtypes: 'object' and '<U7' (the few cases where this used to work often lead to incorrect results).
