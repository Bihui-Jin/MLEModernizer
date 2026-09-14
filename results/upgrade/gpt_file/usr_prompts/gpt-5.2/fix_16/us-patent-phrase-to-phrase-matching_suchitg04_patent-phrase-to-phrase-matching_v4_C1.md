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
print("Using data path:", path)




## === cell 2
import pandas as pd

df = pd.read_csv(
    path / "train.csv", usecols=["id", "anchor", "target", "context", "score"]
)
print("train shape:", df.shape)




## === cell 3
pass




## === cell 4
pass




## === cell 5
df["input"] = (
    "TEXT1: " + df["context"] + "; TEXT2: " + df["target"] + "; ANC: " + df["anchor"]
)




## === cell 6
from datasets import Dataset, DatasetDict
import datasets as _datasets
import os as _os

_cache_dir = _os.environ.get("HF_DATASETS_CACHE", "/kaggle/working/hf_datasets_cache")
_os.environ["HF_DATASETS_CACHE"] = _cache_dir
_datasets.config.HF_DATASETS_CACHE = _cache_dir

ds = Dataset.from_pandas(df, preserve_index=False)




## === cell 7
_ = ds




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
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokz = AutoTokenizer.from_pretrained(model_name, use_fast=True)




## === cell 10
pass




## === cell 11
_MAX_LEN = min(getattr(tokz, "model_max_length", 512) or 512, 512)

import numpy as _np


def tok_func(batch):
    out = tokz(
        batch["input"],
        truncation=True,
        max_length=_MAX_LEN,
    )
    am = out.get("attention_mask", None)
    if am is None:
        out["length"] = [0] * len(out["input_ids"])
    else:
        out["length"] = _np.asarray(am, dtype=_np.int32).sum(axis=1).tolist()
    return out




## === cell 12
_num_proc = 1

tok_ds = ds.map(
    tok_func,
    batched=True,
    num_proc=_num_proc,
    load_from_cache_file=True,
    keep_in_memory=True,
    desc="Tokenizing train (+length)",
)




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
RemoteTraceback                           Traceback (most recent call last)
RemoteTraceback: 
"""
Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/multiprocess/pool.py", line 125, in worker
    result = (True, func(*args, **kwds))
                    ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/datasets/utils/py_utils.py", line 586, in _write_generator_to_queue
    for i, result in enumerate(func(**kwargs)):
  File "/usr/local/lib/python3.11/dist-packages/datasets/arrow_dataset.py", line 3697, in _map_single
    for i, batch in iter_outputs(shard_iterable):
  File "/usr/local/lib/python3.11/dist-packages/datasets/arrow_dataset.py", line 3647, in iter_outputs
    yield i, apply_function(example, i, offset=offset)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/datasets/arrow_dataset.py", line 3570, in apply_function
    processed_inputs = function(*fn_args, *additional_args, **fn_kwargs)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_11/510212334.py", line 18, in tok_func
    out["length"] = _np.asarray(am, dtype=_np.int32).sum(axis=1).tolist()
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (1000,) + inhomogeneous part.
"""

The above exception was the direct cause of the following exception:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3795820793.py in <cell line: 0>()
      4 _num_proc = 1
      5 
----> 6 tok_ds = ds.map(
      7     tok_func,
      8     batched=True,

/usr/local/lib/python3.11/dist-packages/datasets/arrow_dataset.py in wrapper(*args, **kwargs)
    560         }
    561         # apply actual function
--> 562         out: Union["Dataset", "DatasetDict"] = func(self, *args, **kwargs)
    563         datasets: list["Dataset"] = list(out.values()) if isinstance(out, dict) else [out]
    564         # re-apply format to the output

/usr/local/lib/python3.11/dist-packages/datasets/arrow_dataset.py in map(self, function, with_indices, with_rank, input_columns, batched, batch_size, drop_last_batch, remove_columns, keep_in_memory, load_from_cache_file, cache_file_name, writer_batch_size, features, disable_nullable, fn_kwargs, num_proc, suffix_template, new_fingerprint, desc, try_original_type)
   3330                         logger.info(f"Spawning {num_proc} processes")
   3331 
-> 3332                         for rank, done, content in iflatmap_unordered(
   3333                             pool, Dataset._map_single, kwargs_iterable=unprocessed_kwargs_per_job
   3334                         ):

/usr/local/lib/python3.11/dist-packages/datasets/utils/py_utils.py in iflatmap_unordered(pool, func, kwargs_iterable)
    624             if not pool_changed:
    625                 # we get the result in case there's an error to raise
--> 626                 [async_result.get(timeout=0.05) for async_result in async_results]
    627 
    628 

/usr/local/lib/python3.11/dist-packages/datasets/utils/py_utils.py in <listcomp>(.0)
    624             if not pool_changed:
    625                 # we get the result in case there's an error to raise
--> 626                 [async_result.get(timeout=0.05) for async_result in async_results]
    627 
    628 

/usr/local/lib/python3.11/dist-packages/multiprocess/pool.py in get(self, timeout)
    772             return self._value
    773         else:
--> 774             raise self._value
    775 
    776     def _set(self, i, obj):

ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (1000,) + inhomogeneous part.

## === cell 13
pass




## === cell 14
pass




## === cell 15
tok_ds = tok_ds.rename_columns({"score": "labels"})




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3453749365.py in <cell line: 0>()
----> 1 tok_ds = tok_ds.rename_columns({"score": "labels"})
      2 
      3 

NameError: name 'tok_ds' is not defined

## === cell 16
_ = tok_ds[0]




## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3495506986.py in <cell line: 0>()
----> 1 _ = tok_ds[0]
      2 
      3 

NameError: name 'tok_ds' is not defined

## === cell 17
eval_df = pd.read_csv(path / "test.csv", usecols=["id", "anchor", "target", "context"])




## === cell 18
pass




## === cell 19
dds = tok_ds.train_test_split(0.25, seed=42)




## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/565878239.py in <cell line: 0>()
----> 1 dds = tok_ds.train_test_split(0.25, seed=42)
      2 
      3 

NameError: name 'tok_ds' is not defined

## === cell 20
eval_df["input"] = (
    "TEXT1: "
    + eval_df["context"]
    + "; TEXT2: "
    + eval_df["target"]
    + "; ANC: "
    + eval_df["anchor"]
)

from datasets import Dataset as _Dataset

eval_ds = _Dataset.from_pandas(eval_df, preserve_index=False).map(
    tok_func,
    batched=True,
    num_proc=_num_proc,
    load_from_cache_file=True,
    keep_in_memory=True,
    desc="Tokenizing test (+length)",
)




## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
RemoteTraceback                           Traceback (most recent call last)
RemoteTraceback: 
"""
Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/multiprocess/pool.py", line 125, in worker
    result = (True, func(*args, **kwds))
                    ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/datasets/utils/py_utils.py", line 586, in _write_generator_to_queue
    for i, result in enumerate(func(**kwargs)):
  File "/usr/local/lib/python3.11/dist-packages/datasets/arrow_dataset.py", line 3697, in _map_single
    for i, batch in iter_outputs(shard_iterable):
  File "/usr/local/lib/python3.11/dist-packages/datasets/arrow_dataset.py", line 3647, in iter_outputs
    yield i, apply_function(example, i, offset=offset)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/datasets/arrow_dataset.py", line 3570, in apply_function
    processed_inputs = function(*fn_args, *additional_args, **fn_kwargs)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_11/510212334.py", line 18, in tok_func
    out["length"] = _np.asarray(am, dtype=_np.int32).sum(axis=1).tolist()
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (1000,) + inhomogeneous part.
"""

The above exception was the direct cause of the following exception:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/988546178.py in <cell line: 0>()
     10 from datasets import Dataset as _Dataset
     11 
---> 12 eval_ds = _Dataset.from_pandas(eval_df, preserve_index=False).map(
     13     tok_func,
     14     batched=True,

/usr/local/lib/python3.11/dist-packages/datasets/arrow_dataset.py in wrapper(*args, **kwargs)
    560         }
    561         # apply actual function
--> 562         out: Union["Dataset", "DatasetDict"] = func(self, *args, **kwargs)
    563         datasets: list["Dataset"] = list(out.values()) if isinstance(out, dict) else [out]
    564         # re-apply format to the output

/usr/local/lib/python3.11/dist-packages/datasets/arrow_dataset.py in map(self, function, with_indices, with_rank, input_columns, batched, batch_size, drop_last_batch, remove_columns, keep_in_memory, load_from_cache_file, cache_file_name, writer_batch_size, features, disable_nullable, fn_kwargs, num_proc, suffix_template, new_fingerprint, desc, try_original_type)
   3330                         logger.info(f"Spawning {num_proc} processes")
   3331 
-> 3332                         for rank, done, content in iflatmap_unordered(
   3333                             pool, Dataset._map_single, kwargs_iterable=unprocessed_kwargs_per_job
   3334                         ):

/usr/local/lib/python3.11/dist-packages/datasets/utils/py_utils.py in iflatmap_unordered(pool, func, kwargs_iterable)
    624             if not pool_changed:
    625                 # we get the result in case there's an error to raise
--> 626                 [async_result.get(timeout=0.05) for async_result in async_results]
    627 
    628 

/usr/local/lib/python3.11/dist-packages/datasets/utils/py_utils.py in <listcomp>(.0)
    624             if not pool_changed:
    625                 # we get the result in case there's an error to raise
--> 626                 [async_result.get(timeout=0.05) for async_result in async_results]
    627 
    628 

/usr/local/lib/python3.11/dist-packages/multiprocess/pool.py in get(self, timeout)
    772             return self._value
    773         else:
--> 774             raise self._value
    775 
    776     def _set(self, i, obj):

ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (1000,) + inhomogeneous part.

## === cell 21
_keep_cols = {"input_ids", "attention_mask", "token_type_ids", "labels", "id", "length"}


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
/tmp/ipykernel_11/2834964068.py in <cell line: 0>()
     10 
     11 
---> 12 dds = DatasetDict({k: _prune_columns(v) for k, v in dds.items()})
     13 eval_ds = _prune_columns(eval_ds)
     14 

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


def make_training_args(**kwargs):
    try:
        return TrainingArguments(**kwargs)
    except TypeError as e:
        if (
            "evaluation_strategy" in kwargs
            and "unexpected keyword argument 'evaluation_strategy'" in str(e)
        ):
            kwargs2 = dict(kwargs)
            kwargs2["eval_strategy"] = kwargs2.pop("evaluation_strategy")
            return TrainingArguments(**kwargs2)
        raise




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
import random

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True  # SPEED: faster conv/autotune; deterministic is not guaranteed but DeBERTa uses no convs

use_fp16 = torch.cuda.is_available()

_num_workers = 0

args = make_training_args(
    output_dir="outputs",
    learning_rate=lr,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    fp16=use_fp16,
    evaluation_strategy="no",
    per_device_train_batch_size=bs,
    per_device_eval_batch_size=bs * 2,
    num_train_epochs=epochs,
    weight_decay=0.01,
    report_to="none",
    logging_steps=200,
    remove_unused_columns=True,
    disable_tqdm=True,
    dataloader_num_workers=_num_workers,
    dataloader_pin_memory=torch.cuda.is_available(),
    dataloader_persistent_workers=False,
    group_by_length=True,
    length_column_name="length",
    seed=seed,
    data_seed=seed,
)




## === cell 26
from transformers import DataCollatorWithPadding, AutoConfig

data_collator = DataCollatorWithPadding(
    tokenizer=tokz, pad_to_multiple_of=8 if torch.cuda.is_available() else None
)

cfg = AutoConfig.from_pretrained(model_name)
cfg.num_labels = 1
cfg.problem_type = "regression"

model = AutoModelForSequenceClassification.from_pretrained(model_name, config=cfg)

if hasattr(model, "gradient_checkpointing_disable"):
    model.gradient_checkpointing_disable()
if hasattr(model.config, "use_cache"):
    model.config.use_cache = False

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dds["train"],
    eval_dataset=dds["test"],  # preserved structure (even though eval is disabled)
    tokenizer=tokz,
    data_collator=data_collator,
    compute_metrics=corr_d,
)

trainer.train()

pred_out = trainer.predict(eval_ds)
preds = np.asarray(pred_out.predictions).astype(float).reshape(-1)

preds = np.clip(preds, 0, 1)

submission = pd.DataFrame({"id": eval_df["id"].values, "score": preds})
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Saved at:", os.path.abspath("submission.csv"))

## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/734225837.py in <cell line: 0>()
     19     model=model,
     20     args=args,
---> 21     train_dataset=dds["train"],
     22     eval_dataset=dds["test"],  # preserved structure (even though eval is disabled)
     23     tokenizer=tokz,

NameError: name 'dds' is not defined
