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
import numpy as np
import pandas as pd
import random

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

random.seed(42)
np.random.seed(42)

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    set_seed,
    DataCollatorWithPadding,
)

set_seed(42)

try:
    ncpu = os.cpu_count() or 1
    torch.set_num_threads(min(8, ncpu))
    torch.set_num_interop_threads(1)
except Exception:
    pass




## === cell 1
def corr(x, y):
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    if len(x) < 2:
        return 0.0
    return float(np.corrcoef(x, y)[0][1])




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
test_data = pd.read_csv(path + "test.csv")
print(test_data.head())
print(test_data.shape)



## === cell 5
train_data["input"] = (
    "TEXT1: "
    + train_data["context"].astype(str)
    + "; TEXT2: "
    + train_data["target"].astype(str)
    + "; ANC1: "
    + train_data["anchor"].astype(str)
)



## === cell 6
model_nm = "microsoft/deberta-v3-small"
tokenizer = AutoTokenizer.from_pretrained(model_nm, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_nm, num_labels=1)



## === cell 7
_ = model.config




## === cell 8
def tok_func_texts(texts):
    return tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=256,
    )




## === cell 9
from torch.utils.data import (
    Dataset,
)  # kept to preserve original imports if referenced elsewhere



## === cell 10
from sklearn.model_selection import train_test_split

tr_df, va_df = train_test_split(train_data, test_size=0.25, random_state=42)



## === cell 11
from datasets import Dataset as HFDataset


def _tokenize_batch(batch):
    out = tokenizer(
        batch["input"],
        truncation=True,
        max_length=256,
        padding=False,
    )
    out["length"] = [len(ids) for ids in out["input_ids"]]
    return out


tr_hf = HFDataset.from_pandas(tr_df.reset_index(drop=True), preserve_index=False)
va_hf = HFDataset.from_pandas(va_df.reset_index(drop=True), preserve_index=False)

keep_tr = {"input", "score"}
keep_va = {"input", "score"}
tr_hf = tr_hf.remove_columns([c for c in tr_hf.column_names if c not in keep_tr])
va_hf = va_hf.remove_columns([c for c in va_hf.column_names if c not in keep_va])

cache_dir = "/kaggle/working/hf_cache_deberta_v3_small_maxlen256"
os.makedirs(cache_dir, exist_ok=True)

num_proc = min(4, os.cpu_count() or 1)

tr_hf = tr_hf.map(
    _tokenize_batch,
    batched=True,
    num_proc=num_proc,
    cache_file_name=os.path.join(cache_dir, "tr_tok.arrow"),
    remove_columns=["input"],  # safe after tokenization
)
va_hf = va_hf.map(
    _tokenize_batch,
    batched=True,
    num_proc=num_proc,
    cache_file_name=os.path.join(cache_dir, "va_tok.arrow"),
    remove_columns=["input"],  # safe after tokenization
)

tr_hf = tr_hf.rename_column("score", "labels")
va_hf = va_hf.rename_column("score", "labels")

tr_hf.set_format(
    type="torch", columns=["input_ids", "attention_mask", "labels", "length"]
)
va_hf.set_format(
    type="torch", columns=["input_ids", "attention_mask", "labels", "length"]
)

train_ds = tr_hf
valid_ds = va_hf
print("Train/valid sizes:", len(train_ds), len(valid_ds))



## === cell 12
eval_df = test_data.copy()

eval_df["input"] = (
    "TEXT1: "
    + eval_df["context"].astype(str)
    + "; TEXT2: "
    + eval_df["target"].astype(str)
    + "; ANC1: "
    + eval_df["anchor"].astype(str)
)

eval_hf = HFDataset.from_pandas(eval_df.reset_index(drop=True), preserve_index=False)
eval_hf = eval_hf.remove_columns(
    [c for c in eval_hf.column_names if c not in ("input", "id")]
)

eval_hf = eval_hf.map(
    _tokenize_batch,
    batched=True,
    num_proc=num_proc,
    cache_file_name=os.path.join(cache_dir, "te_tok.arrow"),
    remove_columns=["input"],
)
eval_hf.set_format(type="torch", columns=["input_ids", "attention_mask", "length"])

eval_ds = eval_hf
print(eval_df.head())



## === cell 13
bs = 32
epochs = 4
lr = 8e-5



## === cell 14
use_fp16 = torch.cuda.is_available()

if torch.cuda.is_available():
    num_workers = 0
else:
    num_workers = min(4, max(0, (os.cpu_count() or 1) // 2))

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer, pad_to_multiple_of=8 if use_fp16 else None
)

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
    weight_decay=0.01,
    report_to="none",
    save_strategy="no",
    logging_strategy="steps",
    logging_steps=50,
    dataloader_num_workers=num_workers,
    dataloader_pin_memory=torch.cuda.is_available(),
    group_by_length=True,
    length_column_name="length",
    remove_unused_columns=True,
    disable_tqdm=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=corr_d,
)



## === cell 15
trainer.train()



## === cell 16
preds = trainer.predict(eval_ds).predictions
preds = np.asarray(preds, dtype=np.float32).reshape(-1)
preds = np.clip(preds, 0, 1)  # keep in [0,1]

score = np.where(
    preds >= 0.875,
    1.0,
    np.where(
        preds >= 0.625,
        0.75,
        np.where(
            preds >= 0.375,
            0.5,
            np.where(preds >= 0.125, 0.25, 0.0),
        ),
    ),
).astype(np.float32)

submission = pd.DataFrame({"id": eval_df["id"].values, "score": score})
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Columns:", submission.columns.tolist())
