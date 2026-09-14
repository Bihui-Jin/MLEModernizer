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

0.8069452756275215

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
from pathlib import Path

iskaggle = os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "") != ""
print("iskaggle:", iskaggle)



## === cell 1
creds = ""



## === cell 2
cred_path = Path("~/.kaggle/kaggle.json").expanduser()
if not cred_path.exists() and creds:
    cred_path.parent.mkdir(exist_ok=True)
    cred_path.write_text(creds)
    cred_path.chmod(0o600)



## === cell 3
path = Path("us-patent-phrase-to-phrase-matching")



## === cell 4
from zipfile import ZipFile

if iskaggle:
    path = Path("../input/us-patent-phrase-to-phrase-matching")
else:
    if not path.exists():
        import zipfile, kaggle  # type: ignore

        kaggle.api.competition_download_cli(str(path))
        zipfile.ZipFile(f"{path}.zip").extractall(path)

print("Using data path:", path)
print("Train exists:", (path / "train.csv").exists())
print("Test exists:", (path / "test.csv").exists())



## === cell 5
try:
    import datasets  # noqa: F401
except Exception:
    if iskaggle:
        os.system(
            "pip install --no-index --find-links ../input/huggingface-datasets/huggingface-datasets datasets -q"
        )



## === cell 6
import pandas as pd

df = pd.read_csv(path / "train.csv")
print(df.shape)
df.head()



## === cell 7
df["input"] = "TEXT1: " + df.context + "; TEXT2: " + df.target + "; ANC1: " + df.anchor
df["input"].head()



## === cell 8
from datasets import Dataset

ds = Dataset.from_pandas(df)
ds




## === cell 9
def resolve_local_model_dir():
    candidates = [
        Path("../input/debertav3small/debertav3small"),
        Path("../input/debertav3small"),
        Path("../input") / "deberta-v3-small",  # alternative dataset naming
        Path("../input") / "microsoft-deberta-v3-small",
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return str(c)

    base = Path("../input")
    if base.exists():
        for cfg in base.rglob("config.json"):
            return str(cfg.parent)

    raise FileNotFoundError(
        "Could not find a local transformers model directory under ../input. "
        "Please add a dataset containing a DeBERTa model (with config.json)."
    )


model_nm = resolve_local_model_dir()
print("Resolved model directory:", model_nm)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/471905192.py in <cell line: 0>()
     25 
     26 
---> 27 model_nm = resolve_local_model_dir()
     28 print("Resolved model directory:", model_nm)
     29 

/tmp/ipykernel_11/471905192.py in resolve_local_model_dir()
     19             return str(cfg.parent)
     20 
---> 21     raise FileNotFoundError(
     22         "Could not find a local transformers model directory under ../input. "
     23         "Please add a dataset containing a DeBERTa model (with config.json)."

FileNotFoundError: Could not find a local transformers model directory under ../input. Please add a dataset containing a DeBERTa model (with config.json).

## === cell 10
from transformers import AutoTokenizer

tokz = AutoTokenizer.from_pretrained(model_nm, local_files_only=True)
tokz.tokenize("G'day folks, I'm Jeremy from fast.ai!")[:20]




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4173849416.py in <cell line: 0>()
      2 
      3 # Fix: treat model as local files to avoid any Hub validation/network calls
----> 4 tokz = AutoTokenizer.from_pretrained(model_nm, local_files_only=True)
      5 tokz.tokenize("G'day folks, I'm Jeremy from fast.ai!")[:20]
      6 

NameError: name 'model_nm' is not defined

## === cell 11
def tok_func(x):
    return tokz(x["input"], truncation=True)


tok_ds = ds.map(tok_func, batched=True)

row = tok_ds[0]
row["input"], row["input_ids"][:20]



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3206277993.py in <cell line: 0>()
      3 
      4 
----> 5 tok_ds = ds.map(tok_func, batched=True)
      6 
      7 row = tok_ds[0]

/usr/local/lib/python3.11/dist-packages/datasets/arrow_dataset.py in wrapper(*args, **kwargs)
    560         }
    561         # apply actual function
--> 562         out: Union["Dataset", "DatasetDict"] = func(self, *args, **kwargs)
    563         datasets: list["Dataset"] = list(out.values()) if isinstance(out, dict) else [out]
    564         # re-apply format to the output

/usr/local/lib/python3.11/dist-packages/datasets/arrow_dataset.py in map(self, function, with_indices, with_rank, input_columns, batched, batch_size, drop_last_batch, remove_columns, keep_in_memory, load_from_cache_file, cache_file_name, writer_batch_size, features, disable_nullable, fn_kwargs, num_proc, suffix_template, new_fingerprint, desc, try_original_type)
   3339                 else:
   3340                     for unprocessed_kwargs in unprocessed_kwargs_per_job:
-> 3341                         for rank, done, content in Dataset._map_single(**unprocessed_kwargs):
   3342                             check_if_shard_done(rank, done, content)
   3343 

/usr/local/lib/python3.11/dist-packages/datasets/arrow_dataset.py in _map_single(shard, function, with_indices, with_rank, input_columns, batched, batch_size, drop_last_batch, remove_columns, keep_in_memory, cache_file_name, writer_batch_size, features, disable_nullable, fn_kwargs, new_fingerprint, rank, offset, try_original_type)
   3695                 else:
   3696                     _time = time.time()
-> 3697                     for i, batch in iter_outputs(shard_iterable):
   3698                         num_examples_in_batch = len(i)
   3699                         if update_data:

/usr/local/lib/python3.11/dist-packages/datasets/arrow_dataset.py in iter_outputs(shard_iterable)
   3645             else:
   3646                 for i, example in shard_iterable:
-> 3647                     yield i, apply_function(example, i, offset=offset)
   3648 
   3649         num_examples_progress_update = 0

/usr/local/lib/python3.11/dist-packages/datasets/arrow_dataset.py in apply_function(pa_inputs, indices, offset)
   3568             """Utility to apply the function on a selection of columns."""
   3569             inputs, fn_args, additional_args, fn_kwargs = prepare_inputs(pa_inputs, indices, offset=offset)
-> 3570             processed_inputs = function(*fn_args, *additional_args, **fn_kwargs)
   3571             return prepare_outputs(pa_inputs, inputs, processed_inputs)
   3572 

/tmp/ipykernel_11/3206277993.py in tok_func(x)
      1 def tok_func(x):
----> 2     return tokz(x["input"], truncation=True)
      3 
      4 
      5 tok_ds = ds.map(tok_func, batched=True)

NameError: name 'tokz' is not defined

## === cell 12
tok_ds = tok_ds.rename_columns({"score": "labels"})
tok_ds



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4078204769.py in <cell line: 0>()
      1 # Prepare labels column as expected by Trainer
----> 2 tok_ds = tok_ds.rename_columns({"score": "labels"})
      3 tok_ds
      4 

NameError: name 'tok_ds' is not defined

## === cell 13
dds = tok_ds.train_test_split(0.25, seed=42)
dds



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3750529596.py in <cell line: 0>()
      1 # Train/validation split (preserve original)
----> 2 dds = tok_ds.train_test_split(0.25, seed=42)
      3 dds
      4 

NameError: name 'tok_ds' is not defined

## === cell 14
eval_df = pd.read_csv(path / "test.csv")
eval_df["input"] = (
    "TEXT1: "
    + eval_df.context
    + "; TEXT2: "
    + eval_df.target
    + "; ANC1: "
    + eval_df.anchor
)
eval_ds = Dataset.from_pandas(eval_df).map(tok_func, batched=True)
eval_ds



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1937814623.py in <cell line: 0>()
      9     + eval_df.anchor
     10 )
---> 11 eval_ds = Dataset.from_pandas(eval_df).map(tok_func, batched=True)
     12 eval_ds
     13 

/usr/local/lib/python3.11/dist-packages/datasets/arrow_dataset.py in wrapper(*args, **kwargs)
    560         }
    561         # apply actual function
--> 562         out: Union["Dataset", "DatasetDict"] = func(self, *args, **kwargs)
    563         datasets: list["Dataset"] = list(out.values()) if isinstance(out, dict) else [out]
    564         # re-apply format to the output

/usr/local/lib/python3.11/dist-packages/datasets/arrow_dataset.py in map(self, function, with_indices, with_rank, input_columns, batched, batch_size, drop_last_batch, remove_columns, keep_in_memory, load_from_cache_file, cache_file_name, writer_batch_size, features, disable_nullable, fn_kwargs, num_proc, suffix_template, new_fingerprint, desc, try_original_type)
   3339                 else:
   3340                     for unprocessed_kwargs in unprocessed_kwargs_per_job:
-> 3341                         for rank, done, content in Dataset._map_single(**unprocessed_kwargs):
   3342                             check_if_shard_done(rank, done, content)
   3343 

/usr/local/lib/python3.11/dist-packages/datasets/arrow_dataset.py in _map_single(shard, function, with_indices, with_rank, input_columns, batched, batch_size, drop_last_batch, remove_columns, keep_in_memory, cache_file_name, writer_batch_size, features, disable_nullable, fn_kwargs, new_fingerprint, rank, offset, try_original_type)
   3695                 else:
   3696                     _time = time.time()
-> 3697                     for i, batch in iter_outputs(shard_iterable):
   3698                         num_examples_in_batch = len(i)
   3699                         if update_data:

/usr/local/lib/python3.11/dist-packages/datasets/arrow_dataset.py in iter_outputs(shard_iterable)
   3645             else:
   3646                 for i, example in shard_iterable:
-> 3647                     yield i, apply_function(example, i, offset=offset)
   3648 
   3649         num_examples_progress_update = 0

/usr/local/lib/python3.11/dist-packages/datasets/arrow_dataset.py in apply_function(pa_inputs, indices, offset)
   3568             """Utility to apply the function on a selection of columns."""
   3569             inputs, fn_args, additional_args, fn_kwargs = prepare_inputs(pa_inputs, indices, offset=offset)
-> 3570             processed_inputs = function(*fn_args, *additional_args, **fn_kwargs)
   3571             return prepare_outputs(pa_inputs, inputs, processed_inputs)
   3572 

/tmp/ipykernel_11/3206277993.py in tok_func(x)
      1 def tok_func(x):
----> 2     return tokz(x["input"], truncation=True)
      3 
      4 
      5 tok_ds = ds.map(tok_func, batched=True)

NameError: name 'tokz' is not defined

## === cell 15
import numpy as np


def corr(x, y):
    return np.corrcoef(x, y)[0][1]


def corr_d(eval_pred):
    preds, labels = eval_pred
    preds = np.asarray(preds).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    return {"pearson": corr(preds, labels)}




## === cell 16
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 17
bs = 128
epochs = 4
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
)



## === cell 18
model = AutoModelForSequenceClassification.from_pretrained(
    model_nm, num_labels=1, local_files_only=True
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dds["train"],
    eval_dataset=dds["test"],
    tokenizer=tokz,
    compute_metrics=corr_d,
)

trainer.train()



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3081062089.py in <cell line: 0>()
      1 model = AutoModelForSequenceClassification.from_pretrained(
----> 2     model_nm, num_labels=1, local_files_only=True
      3 )
      4 
      5 trainer = Trainer(

NameError: name 'model_nm' is not defined

## === cell 19
pred_out = trainer.predict(eval_ds)
preds = np.asarray(pred_out.predictions, dtype=float).reshape(-1)

preds = np.clip(preds, 0.0, 1.0)
preds[:10], preds.shape



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1891016525.py in <cell line: 0>()
----> 1 pred_out = trainer.predict(eval_ds)
      2 preds = np.asarray(pred_out.predictions, dtype=float).reshape(-1)
      3 
      4 # Preserve original post-processing
      5 preds = np.clip(preds, 0.0, 1.0)

NameError: name 'trainer' is not defined

## === cell 20
submission = pd.DataFrame({"id": eval_df["id"].values, "score": preds})
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv:", submission.shape)
submission.head()

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3108933166.py in <cell line: 0>()
      1 # Write valid submission.csv with required columns
----> 2 submission = pd.DataFrame({"id": eval_df["id"].values, "score": preds})
      3 submission.to_csv("submission.csv", index=False)
      4 print("Wrote submission.csv:", submission.shape)
      5 submission.head()

NameError: name 'preds' is not defined
