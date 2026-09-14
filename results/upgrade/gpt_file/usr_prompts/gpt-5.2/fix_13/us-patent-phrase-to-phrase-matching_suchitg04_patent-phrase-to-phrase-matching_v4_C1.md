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
import os as _os2

_os2.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokz = AutoTokenizer.from_pretrained(model_name, use_fast=True)



## === cell 10
pass



## === cell 11
_MAX_LEN = min(getattr(tokz, "model_max_length", 512) or 512, 512)


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
        out["length"] = [sum(x) for x in am]
    return out




## === cell 12
import os as _os3

_num_proc = min(4, (_os3.cpu_count() or 2))
if _num_proc < 1:
    _num_proc = 1

tok_ds = ds.map(
    tok_func,
    batched=True,
    num_proc=_num_proc,
    load_from_cache_file=True,
    keep_in_memory=True,  # reduces disk/cache overhead during subsequent operations; no semantic change
    desc="Tokenizing train (+length)",
)



## === cell 13
pass



## === cell 14
pass



## === cell 15
tok_ds = tok_ds.rename_columns({"score": "labels"})



## === cell 16
_ = tok_ds[0]



## === cell 17
eval_df = pd.read_csv(path / "test.csv", usecols=["id", "anchor", "target", "context"])



## === cell 18
pass



## === cell 19
dds = tok_ds.train_test_split(0.25, seed=42)



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

use_fp16 = torch.cuda.is_available()

_num_workers = min(4, (os.cpu_count() or 2))

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
    dataloader_persistent_workers=(_num_workers > 0),
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

if hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

try:
    if hasattr(torch, "compile"):
        model = torch.compile(model)
except Exception:
    pass

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dds["train"],
    eval_dataset=dds[
        "test"
    ],  # kept to preserve semantics/structure even if eval is disabled
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
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1494103465.py in <cell line: 0>()
     39 )
     40 
---> 41 trainer.train()
     42 
     43 pred_out = trainer.predict(eval_ds)

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in train(self, resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)
   2204                 hf_hub_utils.enable_progress_bars()
   2205         else:
-> 2206             return inner_training_loop(
   2207                 args=args,
   2208                 resume_from_checkpoint=resume_from_checkpoint,

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in _inner_training_loop(self, batch_size, args, resume_from_checkpoint, trial, ignore_keys_for_eval)
   2254         logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
   2255         # Data loader and number of training steps
-> 2256         train_dataloader = self.get_train_dataloader()
   2257         if self.is_fsdp_xla_v2_enabled:
   2258             train_dataloader = tpu_spmd_dataloader(train_dataloader)

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in get_train_dataloader(self)
   1051             raise ValueError("Trainer: training requires a train_dataset.")
   1052 
-> 1053         return self._get_dataloader(
   1054             dataset=self.train_dataset,
   1055             description="Training",

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in _get_dataloader(self, dataset, description, batch_size, sampler_fn, is_training, dataloader_key)
   1005         data_collator = self.data_collator
   1006         if is_datasets_available() and isinstance(dataset, datasets.Dataset):
-> 1007             dataset = self._remove_unused_columns(dataset, description=description)
   1008         else:
   1009             data_collator = self._get_collator_with_removed_columns(self.data_collator, description=description)

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in _remove_unused_columns(self, dataset, description)
    931         columns = [k for k in signature_columns if k in dataset.column_names]
    932         if len(columns) == 0:
--> 933             raise ValueError(
    934                 f"No columns in the dataset match the model's forward method signature: ({', '.join(signature_columns)}). "
    935                 f"The following columns have been ignored: [{', '.join(ignored_columns)}]. "

ValueError: No columns in the dataset match the model's forward method signature: (args, kwargs, label, label_ids). The following columns have been ignored: [attention_mask, token_type_ids, id, length, input_ids, labels]. Please check the dataset and model. You may need to set `remove_unused_columns=False` in `TrainingArguments`.
