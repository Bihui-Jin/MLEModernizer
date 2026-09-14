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

3.11

# 3. Installed packages

No external packages required in the script and installed.

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

0.8066012823938029

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
from pathlib import Path
import os

path = Path("/kaggle/input/us-patent-phrase-to-phrase-matching")
assert (path / "train.csv").exists(), f"Missing train.csv at {path}"



## === cell 1
print("Listing:", path)
print(sorted([p.name for p in path.iterdir()])[:50])



## === cell 2
import pandas as pd

df = pd.read_csv(path / "train.csv")
print(df.shape)
df.head()



## === cell 3
df.tail()



## === cell 4
df.describe(include=object)



## === cell 5
df["input"] = "TEXT1: " + df.context + "; TEXT2: " + df.target + "; ANC: " + df.anchor
df.input.head()



## === cell 6
from datasets import Dataset, DatasetDict

ds = Dataset.from_pandas(df, preserve_index=False)



## === cell 7
ds



## === cell 8
_candidate_model_dirs = [
    Path("/kaggle/input/deberta-v3-small"),
    Path("/kaggle/input/microsoft-deberta-v3-small"),
    Path("/kaggle/input/deberta-v3-small-hf"),
    Path("/kaggle/input/deberta-v3-base"),
]
model_name = None
for p in _candidate_model_dirs:
    if p.exists():
        model_name = str(p)
        break

if model_name is None:
    model_name = "microsoft/deberta-v3-small"

print("Using model:", model_name)



## === cell 9
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokz = AutoTokenizer.from_pretrained(model_name, use_fast=True)
type(tokz)



## === cell 10
tokz.tokenize('This is a piece of text that the "tokenizer" is going to tokenize!')[:30]




## === cell 11
def tok_func(x):
    return tokz(x["input"], truncation=True)




## === cell 12
import os as _os
from datasets import set_caching_enabled

set_caching_enabled(True)

_num_proc = min(4, (_os.cpu_count() or 2))

tok_ds = ds.map(
    tok_func,
    batched=True,
    num_proc=_num_proc,
    load_from_cache_file=True,  # was False; safe because tok_func is deterministic
    desc="Tokenizing train",
)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/4283074871.py in <cell line: 0>()
      2 # so reruns and downstream operations don't re-tokenize; keep deterministic num_proc.
      3 import os as _os
----> 4 from datasets import set_caching_enabled
      5 
      6 set_caching_enabled(True)

ImportError: cannot import name 'set_caching_enabled' from 'datasets' (/usr/local/lib/python3.11/dist-packages/datasets/__init__.py)

## === cell 13
tok_ds[0]["input_ids"][:20], len(tok_ds[0]["input_ids"])



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/143199169.py in <cell line: 0>()
----> 1 tok_ds[0]["input_ids"][:20], len(tok_ds[0]["input_ids"])
      2 

NameError: name 'tok_ds' is not defined

## === cell 14
if hasattr(tokz, "vocab") and isinstance(tokz.vocab, dict) and "▁going" in tokz.vocab:
    print(tokz.vocab["▁going"])
else:
    print("Tokenizer vocab lookup skipped (not supported for this tokenizer).")



## === cell 15
tok_ds = tok_ds.rename_columns({"score": "labels"})



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3717749454.py in <cell line: 0>()
----> 1 tok_ds = tok_ds.rename_columns({"score": "labels"})
      2 

NameError: name 'tok_ds' is not defined

## === cell 16
tok_ds[0]



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3206505560.py in <cell line: 0>()
----> 1 tok_ds[0]
      2 

NameError: name 'tok_ds' is not defined

## === cell 17
eval_df = pd.read_csv(path / "test.csv")
eval_df.describe(include=object)



## === cell 18
dds = tok_ds.train_test_split(0.25, seed=42)
dds



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1176784158.py in <cell line: 0>()
----> 1 dds = tok_ds.train_test_split(0.25, seed=42)
      2 dds
      3 

NameError: name 'tok_ds' is not defined

## === cell 19
eval_df["input"] = (
    "TEXT1: "
    + eval_df.context
    + "; TEXT2: "
    + eval_df.target
    + "; ANC: "
    + eval_df.anchor
)
eval_df.input.head()



## === cell 20
from datasets import Dataset as _Dataset

eval_ds = _Dataset.from_pandas(eval_df, preserve_index=False).map(
    tok_func,
    batched=True,
    num_proc=_num_proc,
    load_from_cache_file=True,  # was False
    desc="Tokenizing test",
)



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1622426825.py in <cell line: 0>()
      5     tok_func,
      6     batched=True,
----> 7     num_proc=_num_proc,
      8     load_from_cache_file=True,  # was False
      9     desc="Tokenizing test",

NameError: name '_num_proc' is not defined

## === cell 21
_keep_cols = {"input_ids", "attention_mask", "token_type_ids", "labels", "id"}


def _prune_columns(dset):
    cols = set(dset.column_names)
    remove = [c for c in cols if c not in _keep_cols]
    if remove:
        dset = dset.remove_columns(remove)
    return dset


dds = DatasetDict({k: _prune_columns(v) for k, v in dds.items()})
eval_ds = _prune_columns(eval_ds)

for k in dds:
    dds[k].set_format(
        type="torch",
        columns=[
            c
            for c in ["input_ids", "attention_mask", "token_type_ids", "labels"]
            if c in dds[k].column_names
        ],
    )
eval_ds.set_format(
    type="torch",
    columns=[
        c
        for c in ["input_ids", "attention_mask", "token_type_ids"]
        if c in eval_ds.column_names
    ],
)



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1832936086.py in <cell line: 0>()
     12 
     13 
---> 14 dds = DatasetDict({k: _prune_columns(v) for k, v in dds.items()})
     15 eval_ds = _prune_columns(eval_ds)
     16 

NameError: name 'dds' is not defined

## === cell 22
import numpy as np


def corr(x, y):
    return np.corrcoef(x, y)[0][1]


def corr_d(eval_pred):
    preds, labels = eval_pred
    preds = np.asarray(preds).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    return {"pearson": corr(preds, labels)}




## === cell 23
from transformers import TrainingArguments, Trainer
import transformers

print("transformers version:", transformers.__version__)



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 24
bs = 128
epochs = 4
lr = 8e-5



## === cell 25
import torch

use_fp16 = torch.cuda.is_available()

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
    logging_steps=50,
    remove_unused_columns=True,
    disable_tqdm=True,
    dataloader_num_workers=min(4, (_os.cpu_count() or 2)),
    dataloader_pin_memory=torch.cuda.is_available(),
)



## === cell 26
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokz)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dds["train"],
    eval_dataset=dds["test"],
    tokenizer=tokz,
    data_collator=data_collator,
    compute_metrics=corr_d,
)



## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/839946436.py in <cell line: 0>()
      8     model=model,
      9     args=args,
---> 10     train_dataset=dds["train"],
     11     eval_dataset=dds["test"],
     12     tokenizer=tokz,

NameError: name 'dds' is not defined

## === cell 27
trainer.train()



## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3352579090.py in <cell line: 0>()
----> 1 trainer.train()
      2 

NameError: name 'trainer' is not defined

## === cell 28
pred_out = trainer.predict(eval_ds)
preds = np.asarray(pred_out.predictions).astype(float).reshape(-1)

preds = np.clip(preds, 0, 1)
preds.min(), preds.max(), preds[:5]



## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3816707942.py in <cell line: 0>()
----> 1 pred_out = trainer.predict(eval_ds)
      2 preds = np.asarray(pred_out.predictions).astype(float).reshape(-1)
      3 
      4 preds = np.clip(preds, 0, 1)
      5 preds.min(), preds.max(), preds[:5]

NameError: name 'trainer' is not defined

## === cell 29
submission = pd.DataFrame({"id": eval_df["id"].values, "score": preds})
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Saved at:", os.path.abspath("submission.csv"))

## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/749373618.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({"id": eval_df["id"].values, "score": preds})
      2 submission.to_csv("submission.csv", index=False)
      3 
      4 print(submission.head())
      5 print("Wrote submission.csv with shape:", submission.shape)

NameError: name 'preds' is not defined
