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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from pathlib import Path
import pandas as pd, numpy as np, datasets
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
import warnings, logging

warnings.simplefilter("ignore")
logging.disable(logging.WARNING)




## === cell 2
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




## === cell 3
path = Path("../input/us-patent-phrase-to-phrase-matching")
if not path.exists():
    path = Path("../input") / "us-patent-phrase-to-phrase-matching"
input_file_list = [o.name for o in path.iterdir()]
print(f"Using path: {path}")
print(f"input_file_list: {input_file_list[:20]}")



## === cell 4
df = pd.read_csv(path / "train.csv")
eval_df = pd.read_csv(path / "test.csv")
print(df.shape, eval_df.shape)
print(df.columns)



## === cell 5
model_path = "microsoft/deberta-v3-small"

tokz = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=False)


def tok_func(batch):
    return tokz(batch["inputs"], truncation=True)




## === cell 6
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



## === cell 7
bs = 256
epochs = 4
lr = 8e-5
wd = 0.01


def get_dds(df_):
    ds = Dataset.from_pandas(df_, preserve_index=False).rename_column("score", "label")
    tok_ds = ds.map(tok_func, batched=True)

    remove_cols = [
        c
        for c in ["anchor", "target", "context", "inputs", "id", "section", "sectok"]
        if c in tok_ds.column_names
    ]
    tok_ds = tok_ds.remove_columns(remove_cols)

    return DatasetDict(
        {"train": tok_ds.select(trn_idxs), "test": tok_ds.select(val_idxs)}
    )


def get_model():
    return AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=1, local_files_only=False
    )


def get_trainer(dds, model=None):
    if model is None:
        model = get_model()
    args = TrainingArguments(
        output_dir="outputs",
        learning_rate=lr,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        fp16=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs * 2,
        num_train_epochs=epochs,
        weight_decay=wd,
        report_to="none",
        logging_steps=50,
        save_strategy="no",
    )
    return Trainer(
        model=model,
        args=args,
        train_dataset=dds["train"],
        eval_dataset=dds["test"],
        tokenizer=tokz,
        compute_metrics=corr_d,
    )




## === cell 8
sep = " [s] "
df["section"] = df.context.str[0]
df["inputs"] = df.context + sep + df.anchor + sep + df.target
df["inputs"] = df.inputs.str.lower()

df["sectok"] = "[" + df.section + "]"
sectoks = list(df.sectok.unique())
tokz.add_special_tokens({"additional_special_tokens": sectoks})

df["inputs"] = (
    df.sectok + sep + df.context + sep + df.anchor.str.lower() + sep + df.target
)
dds = get_dds(df)
print(dds)



## === cell 9
model = get_model()
model.resize_token_embeddings(len(tokz))
trainer = get_trainer(dds, model=model)
trainer.train()



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4072214710.py in <cell line: 0>()
      1 model = get_model()
      2 model.resize_token_embeddings(len(tokz))
----> 3 trainer = get_trainer(dds, model=model)
      4 trainer.train()
      5 

/tmp/ipykernel_11/1821152200.py in get_trainer(dds, model)
     31     if model is None:
     32         model = get_model()
---> 33     args = TrainingArguments(
     34         output_dir="outputs",
     35         learning_rate=lr,

TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'evaluation_strategy'

## === cell 10
eval_df = eval_df.copy()
eval_df["section"] = eval_df.context.str[0]
eval_df["sectok"] = "[" + eval_df.section + "]"
eval_df["inputs"] = (
    eval_df.sectok
    + sep
    + eval_df.context
    + sep
    + eval_df.anchor.str.lower()
    + sep
    + eval_df.target
)

eval_ds = Dataset.from_pandas(eval_df, preserve_index=False).map(tok_func, batched=True)

preds = trainer.predict(eval_ds).predictions.astype(float).reshape(-1)
preds = np.clip(preds, 0, 1)

submission = pd.DataFrame({"id": eval_df["id"].values, "score": preds})
submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv", submission.shape)
print(submission.head())

sam_sub = pd.read_csv(path / "sample_submission.csv")
print("sample cols:", list(sam_sub.columns), "our cols:", list(submission.columns))
print("sample rows:", len(sam_sub), "our rows:", len(submission))



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1621423891.py in <cell line: 0>()
     15 eval_ds = Dataset.from_pandas(eval_df, preserve_index=False).map(tok_func, batched=True)
     16 
---> 17 preds = trainer.predict(eval_ds).predictions.astype(float).reshape(-1)
     18 preds = np.clip(preds, 0, 1)
     19 

NameError: name 'trainer' is not defined

## === cell 11
import shutil, os

print(os.getcwd())
print("----")
print("Files in cwd:", sorted(os.listdir("."))[:50])

if os.path.exists("outputs"):
    shutil.rmtree("outputs", ignore_errors=True)

print("----")
print("Files after cleanup:", sorted(os.listdir("."))[:50])
