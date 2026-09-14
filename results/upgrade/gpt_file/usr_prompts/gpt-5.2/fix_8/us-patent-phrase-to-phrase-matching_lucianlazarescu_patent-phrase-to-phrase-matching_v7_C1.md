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

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass


seed_everything(42)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True




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

tokenizer = AutoTokenizer.from_pretrained(model_nm, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_nm, num_labels=1)



## === cell 12
sep = " [s] "




## === cell 13
def prepare_data(df: pd.DataFrame) -> pd.Series:
    sectok = df["sectok"]
    context = df["context"].astype(str)
    anchor = df["anchor"].astype(str).str.lower()
    target = df["target"].astype(str)
    return sectok + sep + context + sep + anchor + sep + target




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
        if has_labels:
            self.df = df.loc[:, ["input", "score"]].reset_index(drop=True)
        else:
            self.df = df.loc[:, ["input"]].reset_index(drop=True)

        self.has_labels = has_labels

        texts = self.df["input"].to_numpy(dtype=object, copy=False).tolist()
        enc = tok_func_texts(texts)

        input_ids = np.asarray(enc["input_ids"], dtype=np.int64)
        attention_mask = np.asarray(enc["attention_mask"], dtype=np.int64)

        self.input_ids = torch.from_numpy(input_ids)
        self.attention_mask = torch.from_numpy(attention_mask)

        self.token_type_ids = None
        if "token_type_ids" in enc:
            token_type_ids = np.asarray(enc["token_type_ids"], dtype=np.int64)
            self.token_type_ids = torch.from_numpy(token_type_ids)

        self.labels = None
        if has_labels:
            labels = self.df["score"].to_numpy(dtype=np.float32, copy=False)
            self.labels = torch.from_numpy(labels)

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx: int):
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }
        if self.token_type_ids is not None:
            item["token_type_ids"] = self.token_type_ids[idx]
        if self.has_labels:
            item["labels"] = self.labels[idx]
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
eval_df = test_data.copy()
eval_df["sectok"] = "[" + eval_df["section"] + "]"
eval_df["input"] = prepare_data(eval_df)
eval_ds = PatentPairsDataset(eval_df, tokenizer, has_labels=False)



## === cell 21
bs = 32
epochs = 4
lr = 8e-5



## === cell 22
use_fp16 = torch.cuda.is_available()

_num_workers = min(4, (os.cpu_count() or 2))

args = TrainingArguments(
    output_dir="outputs",
    learning_rate=lr,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    fp16=use_fp16,
    eval_strategy="no",  # keep no-eval during training
    per_device_train_batch_size=bs,
    per_device_eval_batch_size=bs * 2,
    num_train_epochs=epochs,
    weight_decay=0.01,
    report_to="none",
    save_strategy="no",
    logging_strategy="steps",
    logging_steps=50,
    dataloader_num_workers=_num_workers,
    dataloader_pin_memory=torch.cuda.is_available(),
    group_by_length=True,
    length_column_name=None,  # let Trainer infer lengths from input_ids/attention_mask
    seed=42,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    compute_metrics=corr_d,
)



## === cell 23
trainer.train()



## === cell 24
_ = trainer.evaluate(eval_dataset=valid_ds)



## === cell 25
preds = trainer.predict(eval_ds).predictions.astype(float).reshape(-1)
preds = np.clip(preds, 0, 1)
preds[:10], preds.min(), preds.max(), len(preds)



## === cell 26
score = np.asarray(preds, dtype=float).reshape(-1)
assert len(score) == len(eval_df), (len(score), len(eval_df))

submission = pd.DataFrame({"id": eval_df["id"].values, "score": score})
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("submission.csv path:", os.path.abspath("submission.csv"))
