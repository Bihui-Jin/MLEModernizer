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
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
tensorflow-datasets==4.9.9
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1
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



## === cell 1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, logging, gc, time
from tqdm.auto import tqdm

import torch
from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import Dataset, DatasetDict

warnings.simplefilter("ignore")
logging.disable(logging.WARNING)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device



## === cell 2
train_df = pd.read_csv("/kaggle/input/us-patent-phrase-to-phrase-matching/train.csv")
test_df = pd.read_csv("/kaggle/input/us-patent-phrase-to-phrase-matching/test.csv")

print(train_df.head())
print(train_df.info())



## === cell 3
missing_values = train_df.isnull().sum()
dataset_summary = train_df.describe(include="all")
missing_values, dataset_summary



## === cell 4
sns.set(style="whitegrid")

plt.figure(figsize=(10, 6))
sns.histplot(train_df["score"], bins=20, kde=True)
plt.title("Distribution of Similarity Scores")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.show()



## === cell 5
common_anchors = train_df["anchor"].value_counts().head(10)
common_targets = train_df["target"].value_counts().head(10)
common_anchors, common_targets



## === cell 6
common_contexts = train_df["context"].value_counts().head(10)
common_contexts



## === cell 7
train_df["anchor_length"] = train_df["anchor"].apply(len)
train_df["target_length"] = train_df["target"].apply(len)

plt.figure(figsize=(14, 6))

max_length = max(train_df["anchor_length"].max(), train_df["target_length"].max())

plt.subplot(1, 2, 1)
sns.histplot(train_df["anchor_length"], bins=30, kde=True)
plt.title("Distribution of Anchor Phrase Lengths")
plt.xlabel("Length of Anchor Phrase")
plt.ylabel("Frequency")
plt.xlim(0, max_length)

plt.subplot(1, 2, 2)
sns.histplot(train_df["target_length"], bins=30, kde=True)
plt.title("Distribution of Target Phrase Lengths")
plt.xlabel("Length of Target Phrase")
plt.xlim(0, max_length)

plt.tight_layout()
plt.show()



## === cell 8
high_similarity_examples = train_df[train_df["score"] >= 0.9].sample(5, random_state=1)
low_similarity_examples = train_df[train_df["score"] <= 0.1].sample(5, random_state=1)

high_similarity_examples, low_similarity_examples



## === cell 9
model_path = "microsoft/deberta-v3-small"
tokenizer_deberta = AutoTokenizer.from_pretrained(model_path)



## === cell 10
sep = tokenizer_deberta.sep_token
train_df["inputs"] = (
    train_df["context"].astype(str)
    + sep
    + train_df["anchor"].astype(str)
    + sep
    + train_df["target"].astype(str)
)
test_df["inputs"] = (
    test_df["context"].astype(str)
    + sep
    + test_df["anchor"].astype(str)
    + sep
    + test_df["target"].astype(str)
)

train_df[["inputs"]].head()



## === cell 11
train_ds = Dataset.from_pandas(
    train_df.rename(columns={"score": "label"}), preserve_index=False
)
test_ds = Dataset.from_pandas(test_df, preserve_index=False)

train_ds, test_ds




## === cell 12
def token_func(examples):
    return tokenizer_deberta(
        examples["inputs"], padding="max_length", truncation=True, max_length=48
    )




## === cell 13
tokenized_train_ds = train_ds.map(token_func, batched=True)
tokenized_test_ds = test_ds.map(token_func, batched=True)

tokenized_train_ds[0]



## === cell 14
columns_to_remove = [
    "anchor",
    "target",
    "anchor_length",
    "target_length",
    "context",
    "inputs",
    "id",
]
tokenized_train_ds = tokenized_train_ds.remove_columns(
    [c for c in columns_to_remove if c in tokenized_train_ds.column_names]
)

test_ids = test_df["id"].copy()
tokenized_test_ds = tokenized_test_ds.remove_columns(
    [c for c in columns_to_remove if c in tokenized_test_ds.column_names]
)

tokenized_train_ds = tokenized_train_ds.with_format(
    "torch",
    columns=["input_ids", "attention_mask", "label"]
    + (
        ["token_type_ids"]
        if "token_type_ids" in tokenized_train_ds.column_names
        else []
    ),
)
tokenized_test_ds = tokenized_test_ds.with_format(
    "torch",
    columns=["input_ids", "attention_mask"]
    + (
        ["token_type_ids"] if "token_type_ids" in tokenized_test_ds.column_names else []
    ),
)

tokenized_train_ds[0]



## === cell 15
train_indices, val_indices = train_test_split(
    range(len(tokenized_train_ds)), test_size=0.2, random_state=SEED
)

dataset_split = DatasetDict(
    {
        "train": tokenized_train_ds.select(train_indices),
        "test": tokenized_train_ds.select(val_indices),
    }
)

dataset_split




## === cell 16
def corr(eval_pred):
    preds, labels = eval_pred
    preds = np.asarray(preds).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    if preds.std() == 0 or labels.std() == 0:
        return {"pearson": 0.0}
    return {"pearson": float(np.corrcoef(preds, labels)[0][1])}




## === cell 17
args = TrainingArguments(
    output_dir="outputs",
    learning_rate=8e-5,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    fp16=False,
    eval_strategy="epoch",
    per_device_train_batch_size=256,
    per_device_eval_batch_size=256,
    num_train_epochs=5,
    weight_decay=0.01,
    report_to="none",
    logging_steps=50,
    seed=SEED,
)

deberta_model = AutoModelForSequenceClassification.from_pretrained(
    model_path, num_labels=1
)
deberta_trainer = Trainer(
    model=deberta_model,
    args=args,
    train_dataset=dataset_split["train"],
    eval_dataset=dataset_split["test"],
    tokenizer=tokenizer_deberta,
    compute_metrics=corr,
)

deberta_model.to(device)



## === cell 18
training_outcome = deberta_trainer.train()
training_outcome



## === cell 19
test_prediction = (
    deberta_trainer.predict(tokenized_test_ds)
    .predictions.astype(np.float32)
    .reshape(-1)
)

test_prediction = np.clip(test_prediction, 0.0, 1.0)

test_prediction[:10], test_prediction.min(), test_prediction.max()



## === cell 20
submission = pd.DataFrame(
    {
        "id": test_ids.values,
        "score": test_prediction,
    }
)

assert submission.shape[0] == len(test_df), "Submission row count mismatch."
assert list(submission.columns) == [
    "id",
    "score",
], "Submission columns must be: id, score"
submission.head(10)



## === cell 21
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
