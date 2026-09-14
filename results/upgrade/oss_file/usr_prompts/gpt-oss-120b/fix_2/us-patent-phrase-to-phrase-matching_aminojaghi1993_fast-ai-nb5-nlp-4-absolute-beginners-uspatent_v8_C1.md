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
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
base_path = Path("../input/us-patent-phrase-to-phrase-matching")
train_path = base_path / "train.csv"
test_path = base_path / "test.csv"
sample_sub_path = base_path / "sample_submission.csv"



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2248947321.py in <cell line: 0>()
      1 # paths
----> 2 base_path = Path("../input/us-patent-phrase-to-phrase-matching")
      3 train_path = base_path / "train.csv"
      4 test_path = base_path / "test.csv"
      5 sample_sub_path = base_path / "sample_submission.csv"

NameError: name 'Path' is not defined

## === cell 2
train_df = pd.read_csv(train_path)
train_df["input"] = (
    "TEXT1: "
    + train_df["context"]
    + "; TEXT2: "
    + train_df["target"]
    + "; ANC1: "
    + train_df["anchor"]
)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/556217522.py in <cell line: 0>()
      1 # load data and build a single input string
----> 2 train_df = pd.read_csv(train_path)
      3 train_df["input"] = (
      4     "TEXT1: "
      5     + train_df["context"]

NameError: name 'train_path' is not defined

## === cell 3
model_name = "microsoft/deberta-v3-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)



## === cell 4
train_split, val_split = train_test_split(
    train_df, test_size=0.2, random_state=42, stratify=train_df["score"]
)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/674318958.py in <cell line: 0>()
      1 # split train/validation
      2 train_split, val_split = train_test_split(
----> 3     train_df, test_size=0.2, random_state=42, stratify=train_df["score"]
      4 )
      5 

NameError: name 'train_df' is not defined

## === cell 5
class PhraseDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, is_train=True):
        self.texts = df["input"].tolist()
        self.is_train = is_train
        if is_train:
            self.labels = df["score"].astype(np.float32).values
        else:
            self.labels = None
        self.encodings = tokenizer(
            self.texts,
            truncation=True,
            padding=True,
            max_length=256,
        )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        if self.is_train:
            item["labels"] = torch.tensor(self.labels[idx])
        return item


train_dataset = PhraseDataset(train_split, tokenizer, is_train=True)
val_dataset = PhraseDataset(val_split, tokenizer, is_train=True)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2397185982.py in <cell line: 0>()
     24 
     25 
---> 26 train_dataset = PhraseDataset(train_split, tokenizer, is_train=True)
     27 val_dataset = PhraseDataset(val_split, tokenizer, is_train=True)
     28 

NameError: name 'train_split' is not defined

## === cell 6
def pearson_corr(pred):
    preds = pred.predictions.squeeze()
    labels = pred.label_ids.squeeze()
    return {"pearson": np.corrcoef(preds, labels)[0, 1]}




## === cell 7
training_args = TrainingArguments(
    output_dir="outputs",
    learning_rate=8e-5,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=256,
    num_train_epochs=2,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    fp16=True,
    eval_strategy="epoch",
    report_to="none",
    seed=42,
)



## === cell 8
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=1, problem_type="regression"
)



## === cell 9
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=pearson_corr,
)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2102185936.py in <cell line: 0>()
      2     model=model,
      3     args=training_args,
----> 4     train_dataset=train_dataset,
      5     eval_dataset=val_dataset,
      6     tokenizer=tokenizer,

NameError: name 'train_dataset' is not defined

## === cell 10
trainer.train()



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3352579090.py in <cell line: 0>()
----> 1 trainer.train()
      2 

NameError: name 'trainer' is not defined

## === cell 11
test_df = pd.read_csv(test_path)
test_df["input"] = (
    "TEXT1: "
    + test_df["context"]
    + "; TEXT2: "
    + test_df["target"]
    + "; ANC1: "
    + test_df["anchor"]
)
test_dataset = PhraseDataset(test_df, tokenizer, is_train=False)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1155302981.py in <cell line: 0>()
      1 # prepare test data
----> 2 test_df = pd.read_csv(test_path)
      3 test_df["input"] = (
      4     "TEXT1: "
      5     + test_df["context"]

NameError: name 'test_path' is not defined

## === cell 12
preds = trainer.predict(test_dataset).predictions.squeeze()
preds = np.clip(preds, 0, 1)

submission = pd.DataFrame({"id": test_df["id"], "score": preds})
submission.to_csv("submission.csv", index=False)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/136298915.py in <cell line: 0>()
----> 1 preds = trainer.predict(test_dataset).predictions.squeeze()
      2 preds = np.clip(preds, 0, 1)
      3 
      4 submission = pd.DataFrame({"id": test_df["id"], "score": preds})
      5 submission.to_csv("submission.csv", index=False)

NameError: name 'trainer' is not defined

## === cell 13
print("Saved submission.csv")
print(submission.head())

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4178028048.py in <cell line: 0>()
      1 print("Saved submission.csv")
----> 2 print(submission.head())

NameError: name 'submission' is not defined
