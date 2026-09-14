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

0.8039326304000577

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault(
    "CUBLAS_WORKSPACE_CONFIG", ":4096:8"
)  # deterministic cublas where applicable

from pathlib import Path
import random
import numpy as np
import pandas as pd

import datasets
from datasets import Dataset

import torch

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

torch.use_deterministic_algorithms(False)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
path = Path("../input/us-patent-phrase-to-phrase-matching")
if not path.exists():
    alt = Path("/kaggle/input/us-patent-phrase-to-phrase-matching")
    if alt.exists():
        path = alt

assert (path / "train.csv").exists(), f"train.csv not found under: {path}"
assert (path / "test.csv").exists(), f"test.csv not found under: {path}"
assert (
    path / "sample_submission.csv"
).exists(), f"sample_submission.csv not found under: {path}"



## === cell 2
df = pd.read_csv(path / "train.csv")

ctx = df["context"].astype(str).to_numpy()
tgt = df["target"].astype(str).to_numpy()
anc = df["anchor"].astype(str).to_numpy()
df["input"] = np.char.add(
    np.char.add(np.char.add("TEXT1: ", ctx), "; TEXT2: "),
    np.char.add(np.char.add(tgt, "; ANC1: "), anc),
)

ds = Dataset.from_pandas(df, preserve_index=False)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2033570159.py in <cell line: 0>()
      7 anc = df["anchor"].astype(str).to_numpy()
      8 df["input"] = np.char.add(
----> 9     np.char.add(np.char.add("TEXT1: ", ctx), "; TEXT2: "),
     10     np.char.add(np.char.add(tgt, "; ANC1: "), anc),
     11 )

/usr/local/lib/python3.11/dist-packages/numpy/core/defchararray.py in add(x1, x2)
    330         # object dtype itemsize as num chars (worked on short strings).
    331         # bytes + void worked but promoting void->bytes is dubious also.
--> 332         raise TypeError(
    333             "np.char.add() requires both arrays of the same dtype kind, but "
    334             f"got dtypes: '{arr1.dtype}' and '{arr2.dtype}' (the few cases "

TypeError: np.char.add() requires both arrays of the same dtype kind, but got dtypes: '<U7' and 'object' (the few cases where this used to work often lead to incorrect results).

## === cell 3
candidate_paths = [
    Path("../input/ms-deberta-v3-small/pytorch/small/1/ms-deberta-v3-small-local"),
    Path("/kaggle/input/ms-deberta-v3-small/pytorch/small/1/ms-deberta-v3-small-local"),
]
model_path = None
for p in candidate_paths:
    if p.exists():
        model_path = str(p)
        break

if model_path is None:
    model_path = "microsoft/deberta-v3-small"

tokz = AutoTokenizer.from_pretrained(
    model_path, local_files_only=(model_path != "microsoft/deberta-v3-small")
)



## === cell 4
MAX_LEN = int(getattr(tokz, "model_max_length", 512))
if MAX_LEN > 512:
    MAX_LEN = 512


def tok_func(batch):
    return tokz(batch["input"], truncation=True, max_length=MAX_LEN)


num_proc = 1
datasets.set_caching_enabled(True)

tok_ds = ds.map(
    tok_func,
    batched=True,
    num_proc=num_proc,
    desc="Tokenizing train",
    load_from_cache_file=True,
)
tok_ds = tok_ds.rename_columns({"score": "labels"})
keep_cols = {"input_ids", "attention_mask", "token_type_ids", "labels"}
remove_cols = [c for c in tok_ds.column_names if c not in keep_cols]
if remove_cols:
    tok_ds = tok_ds.remove_columns(remove_cols)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/256948837.py in <cell line: 0>()
     12 # Speed: enable dataset caching so repeated runs don't redo tokenization work.
     13 num_proc = 1
---> 14 datasets.set_caching_enabled(True)
     15 
     16 tok_ds = ds.map(

AttributeError: module 'datasets' has no attribute 'set_caching_enabled'

## === cell 5
eval_df = pd.read_csv(path / "test.csv")



## === cell 6
dds = tok_ds.train_test_split(0.2, seed=42)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4252010794.py in <cell line: 0>()
----> 1 dds = tok_ds.train_test_split(0.2, seed=42)
      2 

NameError: name 'tok_ds' is not defined

## === cell 7
ctx = eval_df["context"].astype(str).to_numpy()
tgt = eval_df["target"].astype(str).to_numpy()
anc = eval_df["anchor"].astype(str).to_numpy()
eval_df["input"] = np.char.add(
    np.char.add(np.char.add("TEXT1: ", ctx), "; TEXT2: "),
    np.char.add(np.char.add(tgt, "; ANC1: "), anc),
)

eval_ds = Dataset.from_pandas(eval_df, preserve_index=False)
eval_ds = eval_ds.map(
    tok_func,
    batched=True,
    num_proc=num_proc,
    desc="Tokenizing test",
    load_from_cache_file=True,
)
keep_cols_test = {"input_ids", "attention_mask", "token_type_ids"}
remove_cols_test = [c for c in eval_ds.column_names if c not in keep_cols_test]
if remove_cols_test:
    eval_ds = eval_ds.remove_columns(remove_cols_test)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/20557931.py in <cell line: 0>()
      4 anc = eval_df["anchor"].astype(str).to_numpy()
      5 eval_df["input"] = np.char.add(
----> 6     np.char.add(np.char.add("TEXT1: ", ctx), "; TEXT2: "),
      7     np.char.add(np.char.add(tgt, "; ANC1: "), anc),
      8 )

/usr/local/lib/python3.11/dist-packages/numpy/core/defchararray.py in add(x1, x2)
    330         # object dtype itemsize as num chars (worked on short strings).
    331         # bytes + void worked but promoting void->bytes is dubious also.
--> 332         raise TypeError(
    333             "np.char.add() requires both arrays of the same dtype kind, but "
    334             f"got dtypes: '{arr1.dtype}' and '{arr2.dtype}' (the few cases "

TypeError: np.char.add() requires both arrays of the same dtype kind, but got dtypes: '<U7' and 'object' (the few cases where this used to work often lead to incorrect results).

## === cell 8
def corr_d(eval_pred):
    preds, labels = eval_pred
    preds = np.asarray(preds, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    preds -= preds.mean()
    labels -= labels.mean()
    denom = np.sqrt((preds * preds).sum()) * np.sqrt((labels * labels).sum())
    pearson = float((preds * labels).sum() / denom) if denom != 0 else 0.0
    return {"pearson": pearson}




## === cell 9
bs = 128
epochs = 2
lr = 8e-5

args = TrainingArguments(
    output_dir="outputs",
    learning_rate=lr,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    fp16=True,
    eval_strategy="epoch",
    per_device_train_batch_size=bs,
    per_device_eval_batch_size=bs * 2,
    num_train_epochs=epochs,
    weight_decay=0.01,
    report_to="none",
    logging_strategy="steps",
    logging_steps=50,
    save_strategy="no",
    remove_unused_columns=False,
    dataloader_num_workers=0,
    dataloader_pin_memory=True,
    dataloader_prefetch_factor=None,  # must be None when num_workers=0
    seed=42,
    data_seed=42,
)



## === cell 10
model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    num_labels=1,
    local_files_only=(model_path != "microsoft/deberta-v3-small"),
)

data_collator = DataCollatorWithPadding(tokenizer=tokz, pad_to_multiple_of=8)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dds["train"],
    eval_dataset=dds["test"],
    tokenizer=tokz,
    data_collator=data_collator,
    compute_metrics=corr_d,
)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1544814968.py in <cell line: 0>()
     12     model=model,
     13     args=args,
---> 14     train_dataset=dds["train"],
     15     eval_dataset=dds["test"],
     16     tokenizer=tokz,

NameError: name 'dds' is not defined

## === cell 11
trainer.train()



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3352579090.py in <cell line: 0>()
----> 1 trainer.train()
      2 

NameError: name 'trainer' is not defined

## === cell 12
pred_out = trainer.predict(eval_ds)
preds = np.asarray(pred_out.predictions, dtype=np.float64).reshape(-1)
preds = np.clip(preds, 0.0, 1.0)

submission = pd.DataFrame({"id": eval_df["id"].values, "score": preds})
submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2128262932.py in <cell line: 0>()
----> 1 pred_out = trainer.predict(eval_ds)
      2 preds = np.asarray(pred_out.predictions, dtype=np.float64).reshape(-1)
      3 preds = np.clip(preds, 0.0, 1.0)
      4 
      5 submission = pd.DataFrame({"id": eval_df["id"].values, "score": preds})

NameError: name 'trainer' is not defined

## === cell 13
print("samples submission----------------")
sam_sub = pd.read_csv(path / "sample_submission.csv")
print(sam_sub.head(3))
print(sam_sub.dtypes)

print("our submission----------------")
our_sub = pd.read_csv("submission.csv")
print(our_sub.head(3))
print(our_sub.dtypes)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1852432438.py in <cell line: 0>()
      5 
      6 print("our submission----------------")
----> 7 our_sub = pd.read_csv("submission.csv")
      8 print(our_sub.head(3))
      9 print(our_sub.dtypes)

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: 'submission.csv'

## === cell 14
import os as _os

print(_os.getcwd())
print("----")
print("Files in CWD:")
print("\n".join(sorted(_os.listdir("."))))
import shutil as _shutil

_shutil.rmtree("outputs", ignore_errors=True)
print("----")
print("Files in CWD after removing outputs (if existed):")
print("\n".join(sorted(_os.listdir("."))))
