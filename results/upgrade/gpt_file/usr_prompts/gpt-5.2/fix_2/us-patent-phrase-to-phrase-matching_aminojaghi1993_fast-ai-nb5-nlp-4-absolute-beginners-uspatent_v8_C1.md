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

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from pathlib import Path
import numpy as np
import pandas as pd

import datasets
from datasets import Dataset

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)



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
df["input"] = (
    "TEXT1: "
    + df.context.astype(str)
    + "; TEXT2: "
    + df.target.astype(str)
    + "; ANC1: "
    + df.anchor.astype(str)
)
ds = Dataset.from_pandas(df, preserve_index=False)



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
def tok_func(x):
    return tokz(x["input"], truncation=True)


tok_ds = ds.map(tok_func, batched=True)
tok_ds = tok_ds.rename_columns({"score": "labels"})



## === cell 5
eval_df = pd.read_csv(path / "test.csv")



## === cell 6
dds = tok_ds.train_test_split(0.2, seed=42)



## === cell 7
eval_df["input"] = (
    "TEXT1: "
    + eval_df.context.astype(str)
    + "; TEXT2: "
    + eval_df.target.astype(str)
    + "; ANC1: "
    + eval_df.anchor.astype(str)
)
eval_ds = Dataset.from_pandas(eval_df, preserve_index=False).map(tok_func, batched=True)




## === cell 8
def corr(x, y):
    return np.corrcoef(x, y)[0][1]


def corr_d(eval_pred):
    preds, labels = eval_pred
    preds = np.asarray(preds).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    return {"pearson": corr(preds, labels)}




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
)



## === cell 10
model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    num_labels=1,
    local_files_only=(model_path != "microsoft/deberta-v3-small"),
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dds["train"],
    eval_dataset=dds["test"],
    tokenizer=tokz,
    compute_metrics=corr_d,
)



## === cell 11
trainer.train()



## === cell 12
pred_out = trainer.predict(eval_ds)
preds = np.asarray(pred_out.predictions, dtype=np.float64).reshape(-1)

preds = np.clip(preds, 0.0, 1.0)

submission = pd.DataFrame({"id": eval_df["id"].values, "score": preds})
submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())



## === cell 13
print("samples submission----------------")
sam_sub = pd.read_csv(path / "sample_submission.csv")
print(sam_sub.head(3))
print(sam_sub.dtypes)

print("our submission----------------")
our_sub = pd.read_csv("submission.csv")
print(our_sub.head(3))
print(our_sub.dtypes)



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
