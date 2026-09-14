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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

try:
    from google.protobuf import message_factory

    if not hasattr(message_factory.MessageFactory, "GetPrototype"):

        def _get_prototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        message_factory.MessageFactory.GetPrototype = _get_prototype
except Exception:
    pass

from pathlib import Path
import pandas as pd, numpy as np, warnings, logging
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)

warnings.simplefilter("ignore")
logging.disable(logging.WARNING)

cpu_cnt = os.cpu_count() or 4
torch.set_num_threads(cpu_cnt)

torch.backends.cudnn.benchmark = True

_fp16 = torch.cuda.is_available()

np.random.seed(42)
torch.manual_seed(42)




## === cell 1
def corr(x, y):
    return np.corrcoef(x, y)[0][1]


def corr_d(eval_pred):
    return {"pearson": corr(*eval_pred)}




## === cell 2
path = Path("../input/us-patent-phrase-to-phrase-matching")
print(f"data path: {path}")



## === cell 3
df = pd.read_csv(path / "train.csv")
eval_df = pd.read_csv(path / "test.csv")



## === cell 4
model_path = "microsoft/deberta-v3-small"
tokz = AutoTokenizer.from_pretrained(model_path)


def tok_func(x):
    return tokz(
        x["inputs"],
        truncation=True,
        padding=False,
        max_length=256,
    )




## === cell 5
anchors = df.anchor.unique()
np.random.shuffle(anchors)

val_prop = 0.25
val_sz = int(len(anchors) * val_prop)
val_anchors = anchors[:val_sz]

is_val = np.isin(df.anchor, val_anchors)
idxs = np.arange(len(df))
val_idxs = idxs[is_val]
trn_idxs = idxs[~is_val]



## === cell 6
bs = 256
epochs = 4
lr = 8e-5
wd = 0.01


def get_dds(df):
    ds = Dataset.from_pandas(df, preserve_index=False).rename_column("score", "labels")
    proc = min(8, os.cpu_count() if hasattr(os, "cpu_count") else 2)
    tok_ds = ds.map(
        tok_func,
        batched=True,
        batch_size=2000,
        num_proc=proc,
        remove_columns=(
            "anchor",
            "target",
            "context",
            "id",
            "section",
            "inputs",
            "sectok",
        ),
    )
    return DatasetDict(
        {"train": tok_ds.select(trn_idxs), "test": tok_ds.select(val_idxs)}
    )


def get_model():
    return AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=1)


class DebertaWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, **kwargs):
        kwargs.pop("num_items_in_batch", None)  # remove the offending argument
        return self.model(**kwargs)


def get_trainer(dds, model=None):
    if model is None:
        model = get_model()
    model = DebertaWrapper(model)
    args = TrainingArguments(
        output_dir="outputs",
        learning_rate=lr,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        fp16=_fp16,
        eval_strategy="no",
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs * 2,
        num_train_epochs=epochs,
        weight_decay=wd,
        report_to="none",
        dataloader_num_workers=min(16, os.cpu_count() or 2),
        dataloader_pin_memory=True,
        remove_unused_columns=False,
    )
    return Trainer(
        model=model,
        args=args,
        train_dataset=dds["train"],
        eval_dataset=dds["test"],
        tokenizer=tokz,
        compute_metrics=corr_d,
    )




## === cell 7
sep = " [s] "
df["section"] = df.context.str[0]
df["sectok"] = "[" + df["section"] + "]"

sectoks = list(df["sectok"].unique())
tokz.add_special_tokens({"additional_special_tokens": sectoks})

df["inputs"] = (
    df["sectok"] + sep + df.context + sep + df.anchor.str.lower() + sep + df.target
)

dds = get_dds(df)



## === cell 8
model = get_model()
model.resize_token_embeddings(len(tokz))


trainer = get_trainer(dds, model=model)
trainer.train()



## === cell 9
sep = " [s] "
eval_df["section"] = eval_df.context.str[0]
eval_df["sectok"] = "[" + eval_df["section"] + "]"
eval_df["inputs"] = (
    eval_df["sectok"]
    + sep
    + eval_df.context
    + sep
    + eval_df.anchor.str.lower()
    + sep
    + eval_df.target
)

eval_ds = Dataset.from_pandas(eval_df, preserve_index=False).map(
    tok_func,
    batched=True,
    batch_size=2000,
    num_proc=2,
    remove_columns=("id", "anchor", "target", "context", "section", "sectok", "inputs"),
)

preds = trainer.predict(eval_ds).predictions
preds = np.clip(preds.astype(float).flatten(), 0, 1)

submission_df = pd.DataFrame({"id": eval_df["id"], "score": preds})
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)

print("=== sample submission ===")
print(pd.read_csv(path / "sample_submission.csv").head(3))

print("\n=== our submission ===")
print(submission_df.head(3))



## === cell 10
print("Current directory:", os.getcwd())
print("Files:")
for f in os.listdir("."):
    if f.endswith(".csv"):
        print(" -", f)
