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

0.7853653726936117

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
import torch

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(42)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def corr(x, y):
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    return np.corrcoef(x, y)[0][1]




## === cell 2
def corr_d(eval_pred):
    preds, labels = eval_pred
    preds = np.asarray(preds).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    return {"pearson": corr(preds, labels)}




## === cell 3
path = "/kaggle/input/us-patent-phrase-to-phrase-matching/"
train_data = pd.read_csv(path + "train.csv")
print(train_data.head())
print(train_data.shape)



## === cell 4
_ = train_data.target.value_counts().head()
_



## === cell 5
_ = train_data.anchor.value_counts().head()
_



## === cell 6
_ = train_data.score.describe()
_



## === cell 7
train_data["section"] = train_data.context.str[0]
_ = train_data.section.value_counts()
_.head()



## === cell 8
train_data["sectok"] = "[" + train_data.section + "]"
sectoks = list(train_data.sectok.unique())
sectoks[:10], len(sectoks)



## === cell 9
test_data = pd.read_csv(path + "test.csv")
print(test_data.head())
print(test_data.shape)



## === cell 10
test_data["section"] = test_data.context.str[0]
_ = test_data.section.value_counts()
_.head()



## === cell 11
model_nm = "microsoft/deberta-v3-small"

tokenizer = AutoTokenizer.from_pretrained(model_nm, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(model_nm, num_labels=1)



## === cell 12
sep = " [s] "




## === cell 13
def prepare_data(df: pd.DataFrame) -> pd.Series:
    out = (
        df["sectok"]
        + sep
        + df["context"].astype(str)
        + sep
        + df["anchor"].astype(str).str.lower()
        + sep
        + df["target"].astype(str)
    )
    return out




## === cell 14
tokenizer.add_special_tokens({"additional_special_tokens": sectoks})
train_data["input"] = prepare_data(train_data)



## === cell 15
model.resize_token_embeddings(len(tokenizer))




## === cell 16
def tok_func_texts(texts):
    return tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=128,
    )




## === cell 17
from torch.utils.data import Dataset


class PatentPairsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, has_labels: bool = True):
        self.df = df.reset_index(drop=True)
        self.has_labels = has_labels
        self.enc = tok_func_texts(self.df["input"].tolist())
        self.labels = None
        if has_labels:
            self.labels = self.df["score"].astype(np.float32).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
        if self.has_labels:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item




## === cell 18
tmp_ds = PatentPairsDataset(train_data.iloc[:5], tokenizer, has_labels=True)
for i in range(2):
    print({k: (v.shape, v.dtype) for k, v in tmp_ds[i].items()})



## === cell 19
val_frac = 0.25
train_df = train_data.sample(frac=1.0, random_state=42).reset_index(drop=True)
n_val = int(len(train_df) * val_frac)
valid_df = train_df.iloc[:n_val].reset_index(drop=True)
train_df2 = train_df.iloc[n_val:].reset_index(drop=True)

train_ds = PatentPairsDataset(train_df2, tokenizer, has_labels=True)
valid_ds = PatentPairsDataset(valid_df, tokenizer, has_labels=True)

len(train_ds), len(valid_ds)



## === cell 20
eval_df = pd.read_csv(path + "test.csv")
eval_df["section"] = eval_df.context.str[0]
eval_df["sectok"] = "[" + eval_df.section + "]"
eval_df["input"] = prepare_data(eval_df)
eval_ds = PatentPairsDataset(eval_df, tokenizer, has_labels=False)



## === cell 21
bs = 32
epochs = 4
lr = 8e-5



## === cell 22
use_fp16 = torch.cuda.is_available()

args = TrainingArguments(
    output_dir="outputs",
    learning_rate=lr,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    fp16=use_fp16,
    evaluation_strategy="epoch",
    per_device_train_batch_size=bs,
    per_device_eval_batch_size=bs * 2,
    num_train_epochs=epochs,
    weight_decay=0.01,
    report_to="none",
    save_strategy="no",
    logging_strategy="steps",
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    tokenizer=tokenizer,
    compute_metrics=corr_d,
)



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/383794595.py in <cell line: 0>()
      4 use_fp16 = torch.cuda.is_available()
      5 
----> 6 args = TrainingArguments(
      7     output_dir="outputs",
      8     learning_rate=lr,

TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'evaluation_strategy'

## === cell 23
trainer.train()



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3352579090.py in <cell line: 0>()
----> 1 trainer.train()
      2 

NameError: name 'trainer' is not defined

## === cell 24
preds = trainer.predict(eval_ds).predictions.astype(float).reshape(-1)
preds = np.clip(preds, 0, 1)



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/917827771.py in <cell line: 0>()
----> 1 preds = trainer.predict(eval_ds).predictions.astype(float).reshape(-1)
      2 preds = np.clip(preds, 0, 1)
      3 

NameError: name 'trainer' is not defined

## === cell 25
score = []
for value in preds:
    if value >= 0.875:
        score.append(1.0)
        continue
    if value >= 0.625:
        score.append(0.75)
        continue
    if value >= 0.375:
        score.append(0.5)
        continue
    if value >= 0.125:
        score.append(0.25)
        continue
    score.append(0.0)



## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1931219075.py in <cell line: 0>()
      1 # Keep original post-processing: snap to {0,0.25,0.5,0.75,1} using fixed thresholds.
      2 score = []
----> 3 for value in preds:
      4     if value >= 0.875:
      5         score.append(1.0)

NameError: name 'preds' is not defined

## === cell 26
submission = pd.DataFrame(
    {"id": eval_df["id"].values, "score": np.array(score, dtype=float)}
)
assert submission.shape[0] == eval_df.shape[0]
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)

## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/80521972.py in <cell line: 0>()
      1 # Bugfix: write a valid CSV submission with correct columns and length.
----> 2 submission = pd.DataFrame(
      3     {"id": eval_df["id"].values, "score": np.array(score, dtype=float)}
      4 )
      5 assert submission.shape[0] == eval_df.shape[0]

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __init__(self, data, index, columns, dtype, copy)
    776         elif isinstance(data, dict):
    777             # GH#38939 de facto copy defaults to False only in non-dict cases
--> 778             mgr = dict_to_mgr(data, index, columns, dtype=dtype, copy=copy, typ=manager)
    779         elif isinstance(data, ma.MaskedArray):
    780             from numpy.ma import mrecords

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in dict_to_mgr(data, index, columns, dtype, typ, copy)
    501             arrays = [x.copy() if hasattr(x, "dtype") else x for x in arrays]
    502 
--> 503     return arrays_to_mgr(arrays, columns, index, dtype=dtype, typ=typ, consolidate=copy)
    504 
    505 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in arrays_to_mgr(arrays, columns, index, dtype, verify_integrity, typ, consolidate)
    112         # figure out the index, if necessary
    113         if index is None:
--> 114             index = _extract_index(arrays)
    115         else:
    116             index = ensure_index(index)

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in _extract_index(data)
    675         lengths = list(set(raw_lengths))
    676         if len(lengths) > 1:
--> 677             raise ValueError("All arrays must be of the same length")
    678 
    679         if have_dicts:

ValueError: All arrays must be of the same length
