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

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
transformers==4.53.3

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

0.8198699774283027

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed,
)

set_seed(42)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def resolve_competition_dir() -> str:
    """
    Kaggle sometimes exposes the dataset under /kaggle/input/<slug>/...
    and your file tree also shows /kaggle/data/... mirrors.
    """
    candidates = [
        "/kaggle/input/us-patent-phrase-to-phrase-matching",
        "/kaggle/data/us-patent-phrase-to-phrase-matching",
        "/kaggle/input",
        "/kaggle/data",
    ]
    for base in candidates:
        if os.path.isdir(base) and (
            os.path.exists(os.path.join(base, "train.csv"))
            or os.path.exists(
                os.path.join(base, "us-patent-phrase-to-phrase-matching", "train.csv")
            )
        ):
            if os.path.exists(os.path.join(base, "train.csv")):
                return base
            nested = os.path.join(base, "us-patent-phrase-to-phrase-matching")
            if os.path.exists(os.path.join(nested, "train.csv")):
                return nested
    return "/kaggle/input/us-patent-phrase-to-phrase-matching"


DATA_DIR = resolve_competition_dir()
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SAMPLE_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

print("Resolved DATA_DIR:", DATA_DIR)
print("Train exists:", os.path.exists(TRAIN_PATH), TRAIN_PATH)
print("Test exists :", os.path.exists(TEST_PATH), TEST_PATH)



## === cell 2
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

MODEL_CANDIDATES = [
    "distilbert-base-uncased",
    "bert-base-uncased",
    "roberta-base",
    "microsoft/deberta-v3-small",
]


def try_load_model_and_tokenizer(name_or_path: str):
    tok = AutoTokenizer.from_pretrained(
        name_or_path, local_files_only=True, use_fast=True
    )
    mdl = AutoModelForSequenceClassification.from_pretrained(
        name_or_path,
        num_labels=5,
        local_files_only=True,
    )
    return mdl, tok


model = None
tokenizer = None
backbone_name = None
last_err = None

local_model_dir = os.path.join(DATA_DIR, "model")
if os.path.isdir(local_model_dir):
    try:
        model, tokenizer = try_load_model_and_tokenizer(local_model_dir)
        backbone_name = local_model_dir
    except Exception as e:
        last_err = e
        model = None
        tokenizer = None

if model is None or tokenizer is None:
    for name in MODEL_CANDIDATES:
        try:
            model, tokenizer = try_load_model_and_tokenizer(name)
            backbone_name = name
            break
        except Exception as e:
            last_err = e
            model = None
            tokenizer = None

if model is None or tokenizer is None:
    raise RuntimeError(
        "Could not load any transformer model locally (offline Kaggle). "
        f"Tried: local_model_dir={local_model_dir!r}, candidates={MODEL_CANDIDATES}. "
        f"Last error: {last_err!r}"
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Loaded backbone:", backbone_name)
print("Device:", device)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/3921798725.py in <cell line: 0>()
     56 
     57 if model is None or tokenizer is None:
---> 58     raise RuntimeError(
     59         "Could not load any transformer model locally (offline Kaggle). "
     60         f"Tried: local_model_dir={local_model_dir!r}, candidates={MODEL_CANDIDATES}. "

RuntimeError: Could not load any transformer model locally (offline Kaggle). Tried: local_model_dir='/kaggle/input/us-patent-phrase-to-phrase-matching/model', candidates=['distilbert-base-uncased', 'bert-base-uncased', 'roberta-base', 'microsoft/deberta-v3-small']. Last error: OSError("We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.\nCheck your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.")

## === cell 3
class MyDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        return item




## === cell 4
LABEL_VALUES = np.linspace(0, 1, 5)  # [0, 0.25, 0.5, 0.75, 1.0]


def score_to_label(y):
    return int(np.clip(np.round(float(y) * 4), 0, 4))


def build_pair_text(df):
    c = df["context"].astype(str).str[0].fillna("")
    t1 = (c + " " + df["anchor"].astype(str)).tolist()
    t2 = df["target"].astype(str).tolist()
    return t1, t2


train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

train_text1, train_text2 = build_pair_text(train_df)
test_text1, test_text2 = build_pair_text(test_df)

train_labels = [score_to_label(y) for y in train_df["score"].values]

train_enc = tokenizer(
    train_text1,
    train_text2,
    truncation=True,
    max_length=128,
)
train_enc["labels"] = train_labels

test_enc = tokenizer(
    test_text1,
    test_text2,
    truncation=True,
    max_length=128,
)

trainset = MyDataset(train_enc)
testset = MyDataset(test_enc)

print("Train examples:", len(trainset), "Test examples:", len(testset))



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1405079940.py in <cell line: 0>()
     22 train_labels = [score_to_label(y) for y in train_df["score"].values]
     23 
---> 24 train_enc = tokenizer(
     25     train_text1,
     26     train_text2,

TypeError: 'NoneType' object is not callable

## === cell 5
args = TrainingArguments(
    output_dir="/kaggle/working/tmp_model",
    per_device_train_batch_size=16,
    num_train_epochs=1,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=200,
    save_strategy="no",
    eval_strategy="no",
    report_to=[],
    fp16=torch.cuda.is_available(),
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=trainset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4077571176.py in <cell line: 0>()
     17     model=model,
     18     args=args,
---> 19     train_dataset=trainset,
     20     tokenizer=tokenizer,
     21     data_collator=data_collator,

NameError: name 'trainset' is not defined

## === cell 6
outputs = trainer.predict(testset)

logits = np.asarray(outputs.predictions)

logits = logits - logits.max(axis=1, keepdims=True)
prob = np.exp(logits)
prob = prob / np.sum(prob, axis=1, keepdims=True)

pred = (prob * LABEL_VALUES.reshape(1, -1)).sum(axis=1)

submit = pd.DataFrame({"id": test_df["id"].values, "score": pred.astype(np.float32)})
submit.to_csv("submission.csv", index=False)

print(submit.head())
print("Wrote submission.csv with shape:", submit.shape)
print("Submission columns:", submit.columns.tolist())

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1434213529.py in <cell line: 0>()
----> 1 outputs = trainer.predict(testset)
      2 
      3 logits = np.asarray(outputs.predictions)
      4 
      5 # Numerically stable softmax

NameError: name 'trainer' is not defined
