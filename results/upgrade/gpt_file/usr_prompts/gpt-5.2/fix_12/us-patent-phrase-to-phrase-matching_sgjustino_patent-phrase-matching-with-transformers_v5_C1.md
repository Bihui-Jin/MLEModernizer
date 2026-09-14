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

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")



## === cell 1
import pandas as pd
import numpy as np
import warnings, logging
import random

import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import Dataset

warnings.simplefilter("ignore")
logging.disable(logging.WARNING)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass

if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device



## === cell 2
train_df = pd.read_csv("/kaggle/input/us-patent-phrase-to-phrase-matching/train.csv")
test_df = pd.read_csv("/kaggle/input/us-patent-phrase-to-phrase-matching/test.csv")

print(train_df.head())
print(train_df.shape, test_df.shape)



## === cell 3
pass



## === cell 4
pass



## === cell 5
pass



## === cell 6
pass



## === cell 7
pass



## === cell 8
model_path = "microsoft/deberta-v3-small"
tokenizer_deberta = AutoTokenizer.from_pretrained(model_path, use_fast=True)



## === cell 9
sep = tokenizer_deberta.sep_token

train_ctx = train_df["context"].astype("string")
train_anc = train_df["anchor"].astype("string")
train_tgt = train_df["target"].astype("string")
train_df["inputs"] = train_ctx + sep + train_anc + sep + train_tgt

test_ctx = test_df["context"].astype("string")
test_anc = test_df["anchor"].astype("string")
test_tgt = test_df["target"].astype("string")
test_df["inputs"] = test_ctx + sep + test_anc + sep + test_tgt

train_df[["inputs"]].head()



## === cell 10
train_ds = Dataset.from_pandas(
    train_df[["inputs", "score"]]
    .rename(columns={"score": "label"})
    .assign(label=lambda d: d["label"].astype("float32")),
    preserve_index=False,
)
test_ids = test_df["id"].copy()
test_ds = Dataset.from_pandas(
    test_df[["inputs"]],
    preserve_index=False,
)

train_ds, test_ds




## === cell 11
def token_func(examples):
    return tokenizer_deberta(
        examples["inputs"], padding=False, truncation=True, max_length=48
    )




## === cell 12
cpu = os.cpu_count() or 2
num_proc = min(4, max(1, cpu // 2))

tokenized_train_ds = train_ds.map(
    token_func,
    batched=True,
    batch_size=4096,
    num_proc=num_proc,
    desc="Tokenizing train",
    load_from_cache_file=True,
)
tokenized_test_ds = test_ds.map(
    token_func,
    batched=True,
    batch_size=4096,
    num_proc=num_proc,
    desc="Tokenizing test",
    load_from_cache_file=True,
)

tokenized_train_ds[0]



## === cell 13
if "inputs" in tokenized_train_ds.column_names:
    tokenized_train_ds = tokenized_train_ds.remove_columns(["inputs"])
if "inputs" in tokenized_test_ds.column_names:
    tokenized_test_ds = tokenized_test_ds.remove_columns(["inputs"])

train_cols = ["input_ids", "attention_mask", "label"]
test_cols = ["input_ids", "attention_mask"]
if "token_type_ids" in tokenized_train_ds.column_names:
    train_cols.append("token_type_ids")
if "token_type_ids" in tokenized_test_ds.column_names:
    test_cols.append("token_type_ids")

tokenized_train_ds = tokenized_train_ds.with_format("torch", columns=train_cols)
tokenized_test_ds = tokenized_test_ds.with_format("torch", columns=test_cols)

tokenized_train_ds[0]



## === cell 14
dataset_split_hf = tokenized_train_ds.train_test_split(
    test_size=0.2, seed=SEED, shuffle=True
)
dataset_split = {"train": dataset_split_hf["train"], "test": dataset_split_hf["test"]}
dataset_split




## === cell 15
def corr(eval_pred):
    preds, labels = eval_pred
    preds = np.asarray(preds).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    if preds.std() == 0 or labels.std() == 0:
        return {"pearson": 0.0}
    return {"pearson": float(np.corrcoef(preds, labels)[0][1])}




## === cell 16
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer_deberta,
    padding="max_length",
    max_length=48,
)

_num_workers = min(4, max(1, (os.cpu_count() or 2) // 2))

ta_kwargs = dict(
    output_dir="outputs",
    learning_rate=8e-5,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    fp16=False,
    per_device_train_batch_size=256,
    per_device_eval_batch_size=256,
    num_train_epochs=5,
    weight_decay=0.01,
    report_to="none",
    logging_steps=200,
    seed=SEED,
    dataloader_num_workers=_num_workers,
    dataloader_pin_memory=torch.cuda.is_available(),
    dataloader_persistent_workers=(_num_workers > 0),
    dataloader_prefetch_factor=4 if _num_workers > 0 else None,
    remove_unused_columns=False,
    label_names=["label"],
    save_strategy="no",
    disable_tqdm=True,
)

try:
    args = TrainingArguments(
        **ta_kwargs,
        evaluation_strategy="epoch",
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        torch_compile=True if torch.cuda.is_available() else False,
    )
except TypeError:
    args = TrainingArguments(
        **ta_kwargs,
        eval_strategy="epoch",
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        torch_compile=True if torch.cuda.is_available() else False,
    )

deberta_model = AutoModelForSequenceClassification.from_pretrained(
    model_path, num_labels=1
)
try:
    deberta_model.config.problem_type = "regression"
except Exception:
    pass

try:
    if hasattr(deberta_model, "gradient_checkpointing_disable"):
        deberta_model.gradient_checkpointing_disable()
except Exception:
    pass

try:
    if hasattr(deberta_model.config, "use_cache"):
        deberta_model.config.use_cache = False
except Exception:
    pass

deberta_model.to(device)

deberta_trainer = Trainer(
    model=deberta_model,
    args=args,
    train_dataset=dataset_split["train"],
    eval_dataset=dataset_split["test"],
    tokenizer=tokenizer_deberta,
    data_collator=data_collator,
    compute_metrics=corr,
)



## === cell 17
training_outcome = deberta_trainer.train()
training_outcome



## === cell 18
pred_out = deberta_trainer.predict(tokenized_test_ds)
test_prediction = pred_out.predictions.reshape(-1).astype(np.float32)

test_prediction = np.clip(test_prediction, 0.0, 1.0)
test_prediction[:10], test_prediction.min(), test_prediction.max()



## === cell 19
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



## === cell 20
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
