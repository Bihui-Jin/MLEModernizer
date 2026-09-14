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
import numpy as np
import pandas as pd
import random, os, torch
from datasets import Dataset, DatasetDict
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
os.environ["PYTHONHASHSEED"] = "42"




## === cell 1
def corr(x, y):
    return np.corrcoef(x, y)[0, 1]


def corr_d(eval_pred):
    return {"pearson": corr(eval_pred.predictions.squeeze(), eval_pred.label_ids)}




## === cell 2
path = "/kaggle/input/us-patent-phrase-to-phrase-matching/"

train_data = pd.read_csv(path + "train.csv")
test_data = pd.read_csv(path + "test.csv")




## === cell 3
train_data["section"] = train_data.context.str[0]
train_data["sectok"] = "[" + train_data.section + "]"

test_data["section"] = test_data.context.str[0]
test_data["sectok"] = "[" + test_data.section + "]"




## === cell 4
sectoks = list(train_data.sectok.unique())




## === cell 5
model_nm = "microsoft/deberta-v3-small"
tokenizer = AutoTokenizer.from_pretrained(model_nm, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_nm, num_labels=1)




## === cell 6
sep = " [s] "




## === cell 7
def prepare_data(df):
    return df.sectok + sep + df.context + sep + df.anchor.str.lower() + sep + df.target




## === cell 8
tokenizer.add_special_tokens({"additional_special_tokens": sectoks})
model.resize_token_embeddings(len(tokenizer))




## === cell 9
train_data["input"] = prepare_data(train_data)
test_data["input"] = prepare_data(test_data)

train_ds = Dataset.from_pandas(train_data[["input", "score"]])
test_ds = Dataset.from_pandas(test_data[["id", "input"]])




## === cell 10
def tokenize_batch(batch):
    return tokenizer(
        batch["input"], truncation=True, padding="max_length", max_length=128
    )


train_ds = train_ds.map(tokenize_batch, batched=True, num_proc=4)
test_ds = test_ds.map(tokenize_batch, batched=True, num_proc=4)

train_ds = train_ds.rename_column("score", "labels")
train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])




## === cell 11
split_ds = train_ds.train_test_split(test_size=0.25, seed=42)
train_split = split_ds["train"]
val_split = split_ds["test"]




## === cell 12
bs = 32
epochs = 4
lr = 8e-5

args = TrainingArguments(
    output_dir="outputs",
    learning_rate=lr,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    fp16=True,  # enable mixed‑precision for faster training
    eval_strategy="epoch",
    per_device_train_batch_size=bs,
    per_device_eval_batch_size=bs * 2,
    num_train_epochs=epochs,
    weight_decay=0.01,
    report_to="none",
    save_strategy="no",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_split,
    eval_dataset=val_split,
    tokenizer=tokenizer,
    compute_metrics=corr_d,
)




## === cell 13
trainer.train()




## === cell 14
preds = trainer.predict(test_ds).predictions.squeeze()
preds = np.clip(preds, 0, 1)




## === cell 15
bins = np.array([0.125, 0.375, 0.625, 0.875])
scores = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
indices = np.digitize(preds, bins, right=False)
score = scores[indices].tolist()




## === cell 16
submission = pd.DataFrame({"id": test_data["id"], "score": score})
submission.to_csv("submission.csv", index=False)
