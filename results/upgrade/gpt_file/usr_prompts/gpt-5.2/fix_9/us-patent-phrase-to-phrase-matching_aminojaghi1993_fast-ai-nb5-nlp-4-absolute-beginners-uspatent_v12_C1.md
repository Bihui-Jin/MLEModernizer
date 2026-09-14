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
from pathlib import Path

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
import pandas as pd



## === cell 1
import warnings, logging

warnings.simplefilter("ignore")
logging.disable(logging.WARNING)



## === cell 2
try:
    import google.protobuf.internal.api_implementation as _api_impl

    if _api_impl.Type() != "python":
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
except Exception:
    pass

import datasets
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)




## === cell 3
def corr(x, y):
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    if x.size == 0 or y.size == 0:
        return 0.0
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0][1])


def corr_d(eval_pred):
    preds, labels = eval_pred
    preds = np.asarray(preds).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    return {"pearson": corr(preds, labels)}




## === cell 4
candidates = [
    Path("/kaggle/input/us-patent-phrase-to-phrase-matching"),
    Path("../input/us-patent-phrase-to-phrase-matching"),
    Path("../input") / "us-patent-phrase-to-phrase-matching",
    Path("/kaggle/data/us-patent-phrase-to-phrase-matching"),
    Path("/kaggle/input"),
    Path("../input"),
    Path("/kaggle/data"),
]
path = None
for c in candidates:
    if (c / "train.csv").exists() and (c / "test.csv").exists():
        path = c
        break
if path is None:
    raise FileNotFoundError(f"Could not find dataset folder in any of: {candidates}")

input_file_list = [o.name for o in path.iterdir() if o.is_file()]
print(f"Using path: {path}")
print(f"input_file_list: {input_file_list[:20]}")



## === cell 5
df = pd.read_csv(path / "train.csv")
eval_df = pd.read_csv(path / "test.csv")
print(df.shape, eval_df.shape)
print(df.columns)



## === cell 6
model_path = "microsoft/deberta-v3-small"

tokz = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=False)


def tok_func(batch):
    return tokz(batch["inputs"], truncation=True)




## === cell 7
anchors = df.anchor.unique()
np.random.seed(42)
np.random.shuffle(anchors)

val_prop = 0.25
val_sz = int(len(anchors) * val_prop)
val_anchors = anchors[:val_sz]

is_val = np.isin(df.anchor, val_anchors)
idxs = np.arange(len(df))
val_idxs = idxs[is_val]
trn_idxs = idxs[~is_val]

print(
    f"train rows: {len(trn_idxs)}  valid rows: {len(val_idxs)}  unique anchors: {len(anchors)}"
)



## === cell 8
bs = 256
epochs = 4
lr = 8e-5
wd = 0.01


def _num_proc_for_map():
    return 1


import hashlib
import json


def _hash_series_strings(s: pd.Series) -> str:
    h = pd.util.hash_pandas_object(s.astype(str), index=False).values
    return hashlib.sha256(h.tobytes()).hexdigest()


def _tokenizer_fingerprint(tokz) -> str:
    cfg = {
        "name_or_path": getattr(tokz, "name_or_path", None),
        "vocab_size": getattr(tokz, "vocab_size", None),
        "model_max_length": getattr(tokz, "model_max_length", None),
        "added_tokens": sorted(list(getattr(tokz, "added_tokens_encoder", {}).keys())),
        "special_tokens_map": getattr(tokz, "special_tokens_map", None),
    }
    blob = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _cache_file(cache_dir: Path, prefix: str, key: str) -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir / f"{prefix}_{key}.arrow")


def get_dds(df_):
    _df = df_.loc[:, ["inputs", "score"]]
    ds = Dataset.from_pandas(_df, preserve_index=False).rename_column("score", "label")

    cache_dir = Path("./hf_cache")
    key = hashlib.sha256(
        (
            _hash_series_strings(df_["inputs"])
            + "|"
            + _tokenizer_fingerprint(tokz)
            + "|trunc=True"
        ).encode("utf-8")
    ).hexdigest()

    tok_ds = ds.map(
        tok_func,
        batched=True,
        num_proc=_num_proc_for_map(),
        load_from_cache_file=True,
        cache_file_name=_cache_file(cache_dir, "train_valid_tok", key),
        desc="Tokenizing train/valid",
    )

    if "inputs" in tok_ds.column_names:
        tok_ds = tok_ds.remove_columns(["inputs"])

    return DatasetDict(
        {"train": tok_ds.select(trn_idxs), "validation": tok_ds.select(val_idxs)}
    )


def get_model():
    return AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=1, local_files_only=False
    )


def get_trainer(dds, model=None):
    if model is None:
        model = get_model()

    use_fp16 = False
    try:
        import torch

        use_fp16 = torch.cuda.is_available()
    except Exception:
        use_fp16 = False

    from transformers import DataCollatorWithPadding

    data_collator = DataCollatorWithPadding(tokenizer=tokz, pad_to_multiple_of=8)

    dl_workers = 2 if (os.cpu_count() or 1) >= 4 else 0

    try:
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    except Exception:
        pass

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
        weight_decay=wd,
        report_to="none",
        logging_steps=50,
        save_strategy="no",
        dataloader_num_workers=dl_workers,
        dataloader_pin_memory=True,
        remove_unused_columns=True,
    )
    return Trainer(
        model=model,
        args=args,
        train_dataset=dds["train"],
        eval_dataset=dds["validation"],
        tokenizer=tokz,
        compute_metrics=corr_d,
        data_collator=data_collator,
    )




## === cell 9
sep = " [s] "

df["section"] = df["context"].str[0]
df["sectok"] = "[" + df["section"] + "]"
sectoks = list(df["sectok"].unique())
tokz.add_special_tokens({"additional_special_tokens": sectoks})

df["inputs"] = (
    df["sectok"]
    + sep
    + df["context"]
    + sep
    + df["anchor"].str.lower()
    + sep
    + df["target"]
)

dds = get_dds(df)
print(dds)



## === cell 10
model = get_model()
model.resize_token_embeddings(len(tokz))
trainer = get_trainer(dds, model=model)
trainer.train()



## === cell 11
eval_df["section"] = eval_df["context"].str[0]
eval_df["sectok"] = "[" + eval_df["section"] + "]"
eval_df["inputs"] = (
    eval_df["sectok"]
    + sep
    + eval_df["context"]
    + sep
    + eval_df["anchor"].str.lower()
    + sep
    + eval_df["target"]
)

_eval_keep = eval_df.loc[:, ["id", "inputs"]]

cache_dir = Path("./hf_cache")
key_test = hashlib.sha256(
    (
        _hash_series_strings(eval_df["inputs"])
        + "|"
        + _tokenizer_fingerprint(tokz)
        + "|trunc=True"
    ).encode("utf-8")
).hexdigest()

eval_ds = Dataset.from_pandas(_eval_keep, preserve_index=False).map(
    tok_func,
    batched=True,
    num_proc=_num_proc_for_map(),
    load_from_cache_file=True,
    cache_file_name=_cache_file(cache_dir, "test_tok", key_test),
    desc="Tokenizing test",
)

rm_cols = [c for c in ["inputs", "id"] if c in eval_ds.column_names]
eval_ds = eval_ds.remove_columns(rm_cols)

preds = trainer.predict(eval_ds).predictions.astype(float).reshape(-1)
preds = np.clip(preds, 0, 1)

submission = pd.DataFrame({"id": eval_df["id"].values, "score": preds})
submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv", submission.shape)
print(submission.head())

sam_sub = pd.read_csv(path / "sample_submission.csv")
print("sample cols:", list(sam_sub.columns), "our cols:", list(submission.columns))
print("sample rows:", len(sam_sub), "our rows:", len(submission))



## === cell 12
import shutil

print(os.getcwd())
print("----")
print("Files in cwd:", sorted(os.listdir("."))[:50])

if os.path.exists("outputs"):
    shutil.rmtree("outputs", ignore_errors=True)

print("----")
print("Files after cleanup:", sorted(os.listdir("."))[:50])
