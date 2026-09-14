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

# 5. Target score

0.8038069821115165

# 6. Current score

-0.00135

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved nan) has done: 'The protobuf/Datasets import error is caused by an incompatible protobuf implementation in this environment; setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` before importing `datasets/transformers` fixes it. Your `TrainingArguments` error indicates this Transformers build expects the newer argument name `eval_strategy` rather than `evaluation_strategy`, so I switch to the compatible name (keeping semantics identical). Because training never ran, `trainer` was undefined and inference crashed; fixing training unblocks submission generation. I also make the input path resolution robust for both `/kaggle/input/...` and `../input/...` so the notebook runs end-to-end and writes `submission.csv` with the required columns.'
- What this solution (achieved -0.00135) has done: 'I fix the crash in the `datasets/transformers` import by pinning protobuf to the pure-Python implementation early and adding a safe fallback to avoid the `MessageFactory.GetPrototype` breakage. Then I fix the training DataLoader error by ensuring tokenization produces `input_ids/attention_mask` columns (your current `tok_func` reads a non-existent `batch["inputs"]` key), and I fix prediction by setting `remove_unused_columns=False` so the Trainer won’t drop the token columns at inference time. Finally, I keep your model/loop/metric intact and ensure `submission.csv` is written with exactly `id,score` and aligned row order.'

# 9. Code solution

## === cell 0
import os
from pathlib import Path

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd



## === cell 1
import warnings, logging

warnings.simplefilter("ignore")
logging.disable(logging.WARNING)



## === cell 2
SEED = 42
np.random.seed(SEED)

try:
    import torch

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass



## === cell 3
try:
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
        set_seed,
    )

    set_seed(SEED)
except Exception as e:
    raise RuntimeError(
        "Failed to import datasets/transformers. "
        "This notebook expects them to be available in the Kaggle environment."
    ) from e




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 4
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




## === cell 5
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



## === cell 6
df = pd.read_csv(path / "train.csv")
eval_df = pd.read_csv(path / "test.csv")
print(df.shape, eval_df.shape)
print(df.columns)



## === cell 7
model_path = "microsoft/deberta-v3-small"

tokz = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=False)


def tok_func(batch):
    return tokz(batch["inputs"], truncation=True)




## === cell 8
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



## === cell 9
bs = 256
epochs = 4
lr = 8e-5
wd = 0.01


def _num_proc_for_map():
    n = os.cpu_count() or 1
    return min(4, n) if n >= 2 else 1


import hashlib
import json


def _hash_series_strings(s: pd.Series) -> str:
    arr = s.astype(str).to_numpy()
    h = hashlib.sha256()
    h.update(str(len(arr)).encode("utf-8"))
    for v in arr:
        h.update(b"\x1f")
        h.update(v.encode("utf-8"))
    return h.hexdigest()


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
    cache_dir = Path("./hf_cache")
    tok_fp = _tokenizer_fingerprint(tokz)

    trn_df = df_.iloc[trn_idxs][["inputs", "score"]].reset_index(drop=True)
    val_df = df_.iloc[val_idxs][["inputs", "score"]].reset_index(drop=True)

    trn_key = hashlib.sha256(
        (_hash_series_strings(trn_df["inputs"]) + "|" + tok_fp + "|trunc=True").encode(
            "utf-8"
        )
    ).hexdigest()
    val_key = hashlib.sha256(
        (_hash_series_strings(val_df["inputs"]) + "|" + tok_fp + "|trunc=True").encode(
            "utf-8"
        )
    ).hexdigest()

    trn_ds = Dataset.from_pandas(trn_df, preserve_index=False).rename_column(
        "score", "label"
    )
    val_ds = Dataset.from_pandas(val_df, preserve_index=False).rename_column(
        "score", "label"
    )

    trn_tok = trn_ds.map(
        tok_func,
        batched=True,
        num_proc=_num_proc_for_map(),
        load_from_cache_file=True,
        cache_file_name=_cache_file(cache_dir, "train_tok", trn_key),
        desc="Tokenizing train",
    )
    val_tok = val_ds.map(
        tok_func,
        batched=True,
        num_proc=_num_proc_for_map(),
        load_from_cache_file=True,
        cache_file_name=_cache_file(cache_dir, "valid_tok", val_key),
        desc="Tokenizing valid",
    )

    if "inputs" in trn_tok.column_names:
        trn_tok = trn_tok.remove_columns(["inputs"])
    if "inputs" in val_tok.column_names:
        val_tok = val_tok.remove_columns(["inputs"])

    return DatasetDict({"train": trn_tok, "validation": val_tok})


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

    cpu = os.cpu_count() or 1
    dl_workers = 4 if cpu >= 8 else (2 if cpu >= 4 else 0)

    try:
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    except Exception:
        pass

    try:
        import torch

        if hasattr(torch, "compile"):
            model = torch.compile(model)
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
        remove_unused_columns=False,
        seed=SEED,
        data_seed=SEED,
        dataloader_prefetch_factor=2 if dl_workers > 0 else None,
        dataloader_persistent_workers=True if dl_workers > 0 else False,
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




## === cell 10
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



## === cell 11
model = get_model()
model.resize_token_embeddings(len(tokz))
trainer = get_trainer(dds, model=model)
trainer.train()



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4072214710.py in <cell line: 0>()
      2 model.resize_token_embeddings(len(tokz))
      3 trainer = get_trainer(dds, model=model)
----> 4 trainer.train()
      5 

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in train(self, resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)
   2204                 hf_hub_utils.enable_progress_bars()
   2205         else:
-> 2206             return inner_training_loop(
   2207                 args=args,
   2208                 resume_from_checkpoint=resume_from_checkpoint,

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in _inner_training_loop(self, batch_size, args, resume_from_checkpoint, trial, ignore_keys_for_eval)
   2546                     )
   2547                     with context():
-> 2548                         tr_loss_step = self.training_step(model, inputs, num_items_in_batch)
   2549 
   2550                     if (

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in training_step(self, model, inputs, num_items_in_batch)
   3747 
   3748         with self.compute_loss_context_manager():
-> 3749             loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
   3750 
   3751         del inputs

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in compute_loss(self, model, inputs, return_outputs, num_items_in_batch)
   3834                 loss_kwargs["num_items_in_batch"] = num_items_in_batch
   3835             inputs = {**inputs, **loss_kwargs}
-> 3836         outputs = model(**inputs)
   3837         # Save past state if it exists
   3838         # TODO: this needs to be fixed and made cleaner later.

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/eval_frame.py in _fn(*args, **kwargs)
    572 
    573             try:
--> 574                 return fn(*args, **kwargs)
    575             finally:
    576                 # Restore the dynamic layer stack depth if necessary.

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

TypeError: DebertaV2ForSequenceClassification.forward() got an unexpected keyword argument 'num_items_in_batch'

## === cell 12
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
tok_fp = _tokenizer_fingerprint(tokz)
key_test = hashlib.sha256(
    (_hash_series_strings(eval_df["inputs"]) + "|" + tok_fp + "|trunc=True").encode(
        "utf-8"
    )
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

preds = trainer.predict(eval_ds).predictions
preds = np.asarray(preds, dtype=float).reshape(-1)
preds = np.clip(preds, 0, 1)

submission = pd.DataFrame({"id": eval_df["id"].values, "score": preds})
submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv", submission.shape)
print(submission.head())

sam_sub = pd.read_csv(path / "sample_submission.csv")
print("sample cols:", list(sam_sub.columns), "our cols:", list(submission.columns))
print("sample rows:", len(sam_sub), "our rows:", len(submission))



## === cell 13
import shutil

print(os.getcwd())
print("----")
print("Files in cwd:", sorted(os.listdir("."))[:50])

if os.path.exists("outputs"):
    shutil.rmtree("outputs", ignore_errors=True)

print("----")
print("Files after cleanup:", sorted(os.listdir("."))[:50])
