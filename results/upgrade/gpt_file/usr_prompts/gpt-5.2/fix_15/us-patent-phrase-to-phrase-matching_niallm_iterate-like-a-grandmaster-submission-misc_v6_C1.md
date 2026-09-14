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

No external packages required in the script and installed.

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

0.8084755399348172

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
from pathlib import Path



## === cell 1
iskaggle = (
    bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "")) or Path("/kaggle").exists()
)
iskaggle




## === cell 2
def resolve_comp_path() -> Path:
    candidates = [
        Path("/kaggle/input/us-patent-phrase-to-phrase-matching"),
        Path("/kaggle/data/us-patent-phrase-to-phrase-matching"),
        Path("../input/us-patent-phrase-to-phrase-matching"),
        Path.home() / "data" / "us-patent-phrase-to-phrase-matching",
    ]
    for p in candidates:
        if (p / "train.csv").exists() and (p / "test.csv").exists():
            return p
    base = Path("/kaggle/input")
    if base.exists():
        for p in base.rglob("us-patent-phrase-to-phrase-matching"):
            if (p / "train.csv").exists() and (p / "test.csv").exists():
                return p
    raise FileNotFoundError(
        "Could not locate competition dataset folder containing train.csv/test.csv"
    )


path = resolve_comp_path()
path



## === cell 3
df = pd.read_csv(path / "train.csv")
eval_df = pd.read_csv(path / "test.csv")
df.shape, eval_df.shape



## === cell 4
import warnings, logging, torch
from torch.utils.data import Dataset

warnings.simplefilter("ignore")
logging.disable(logging.WARNING)

import transformers
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import DataCollatorWithPadding

transformers.__version__



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 5
model_nm = "microsoft/deberta-v3-small"
model_nm



## === cell 6
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass
os.environ["PYTHONHASHSEED"] = str(SEED)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

torch.backends.cudnn.benchmark = False

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass



## === cell 7
tokz = AutoTokenizer.from_pretrained(model_nm, use_fast=True)
sep = tokz.sep_token
sep



## === cell 8
df["section"] = df.context.str[0]
df["inputs"] = df.context + sep + df.anchor + sep + df.target

eval_df["section"] = eval_df.context.str[0]
eval_df["inputs"] = eval_df.context + sep + eval_df.anchor + sep + eval_df.target

df[["context", "anchor", "target", "inputs"]].head()



## === cell 9
MAX_LEN = 512
MAX_LEN




## === cell 10
def tokenize_texts(texts, tokenizer, max_len=512):
    texts = list(texts)
    return tokenizer(
        texts,
        truncation=True,
        max_length=int(max_len),
        padding=False,  # keep dynamic padding via DataCollatorWithPadding
    )


class TokenizedEncDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.input_ids = torch.as_tensor(encodings["input_ids"], dtype=torch.long)
        self.attention_mask = torch.as_tensor(
            encodings["attention_mask"], dtype=torch.long
        )
        self.token_type_ids = None
        if "token_type_ids" in encodings:
            self.token_type_ids = torch.as_tensor(
                encodings["token_type_ids"], dtype=torch.long
            )

        self.labels = None
        if labels is not None:
            self.labels = torch.as_tensor(
                np.asarray(labels, dtype=np.float32), dtype=torch.float32
            )

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, idx):
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }
        if self.token_type_ids is not None:
            item["token_type_ids"] = self.token_type_ids[idx]
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item




## === cell 11
anchors = df.anchor.unique()
np.random.seed(42)
np.random.shuffle(anchors)

val_prop = 0.25
val_sz = int(len(anchors) * val_prop)
val_anchors = anchors[:val_sz]

is_val = np.isin(df.anchor, val_anchors)
idxs = np.arange(len(df))
val_idxs = idxs[is_val]
trn_idxs = idxs[~is_val]
len(val_idxs), len(trn_idxs)




## === cell 12
def _cache_key():
    return f"enc_{model_nm.replace('/','_')}_maxlen{MAX_LEN}_ntr{len(trn_idxs)}_nva{len(val_idxs)}_nte{len(eval_df)}.pt"


cache_file = Path("tokenized_cache") / _cache_key()
cache_file.parent.mkdir(parents=True, exist_ok=True)

if cache_file.exists():
    cache = torch.load(cache_file, map_location="cpu")
    trn_enc = cache["trn_enc"]
    val_enc = cache["val_enc"]
    tst_enc = cache["tst_enc"]
else:
    trn_texts = df.iloc[trn_idxs]["inputs"].values
    val_texts = df.iloc[val_idxs]["inputs"].values
    tst_texts = eval_df["inputs"].values

    trn_enc = tokenize_texts(trn_texts, tokz, MAX_LEN)
    val_enc = tokenize_texts(val_texts, tokz, MAX_LEN)
    tst_enc = tokenize_texts(tst_texts, tokz, MAX_LEN)

    torch.save({"trn_enc": trn_enc, "val_enc": val_enc, "tst_enc": tst_enc}, cache_file)

train_ds = TokenizedEncDataset(
    encodings=trn_enc,
    labels=df.iloc[trn_idxs]["score"].values,
)
valid_ds = TokenizedEncDataset(
    encodings=val_enc,
    labels=df.iloc[val_idxs]["score"].values,
)
test_ds = TokenizedEncDataset(
    encodings=tst_enc,
    labels=None,
)

df.iloc[trn_idxs].score.mean(), df.iloc[val_idxs].score.mean()




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2385208118.py in <cell line: 0>()
     25     torch.save({"trn_enc": trn_enc, "val_enc": val_enc, "tst_enc": tst_enc}, cache_file)
     26 
---> 27 train_ds = TokenizedEncDataset(
     28     encodings=trn_enc,
     29     labels=df.iloc[trn_idxs]["score"].values,

/tmp/ipykernel_11/507421372.py in __init__(self, encodings, labels)
     15 class TokenizedEncDataset(Dataset):
     16     def __init__(self, encodings, labels=None):
---> 17         self.input_ids = torch.as_tensor(encodings["input_ids"], dtype=torch.long)
     18         self.attention_mask = torch.as_tensor(
     19             encodings["attention_mask"], dtype=torch.long

ValueError: expected sequence of length 11 at dim 1 (got 9)

## === cell 13
def corr(eval_pred):
    preds, labels = eval_pred
    preds = np.asarray(preds, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    return {"pearson": np.corrcoef(preds, labels)[0, 1]}




## === cell 14
lr, bs = 8e-5, 128
wd, epochs = 0.01, 4




## === cell 15
def make_training_args():
    n_cpu = os.cpu_count() or 2

    dl_workers = 2 if iskaggle else min(8, max(2, n_cpu // 2))

    common = dict(
        output_dir="outputs",
        learning_rate=lr,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        fp16=bool(torch.cuda.is_available()),
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs * 2,
        num_train_epochs=epochs,
        weight_decay=wd,
        report_to="none",
        save_strategy="no",  # avoid checkpoint I/O
        logging_strategy="steps",
        logging_steps=200,
        disable_tqdm=True,
        dataloader_num_workers=dl_workers,
        dataloader_pin_memory=bool(torch.cuda.is_available()),
        remove_unused_columns=True,
        seed=SEED,
        data_seed=SEED,
        group_by_length=True,
        eval_accumulation_steps=32,
        dataloader_prefetch_factor=2 if dl_workers > 0 else None,
        dataloader_persistent_workers=bool(dl_workers > 0),
    )

    if common["dataloader_prefetch_factor"] is None:
        common.pop("dataloader_prefetch_factor", None)

    try:
        return TrainingArguments(**common, evaluation_strategy="epoch")
    except TypeError:
        return TrainingArguments(**common, eval_strategy="epoch")


args = make_training_args()
args



## === cell 16
data_collator = DataCollatorWithPadding(tokenizer=tokz, pad_to_multiple_of=8)

model = AutoModelForSequenceClassification.from_pretrained(model_nm, num_labels=1)

if hasattr(torch, "compile"):
    try:
        model = torch.compile(model)  # PyTorch 2.x
    except Exception:
        pass

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    tokenizer=tokz,
    data_collator=data_collator,
    compute_metrics=corr,
)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3403406412.py in <cell line: 0>()
     14     model=model,
     15     args=args,
---> 16     train_dataset=train_ds,
     17     eval_dataset=valid_ds,
     18     tokenizer=tokz,

NameError: name 'train_ds' is not defined

## === cell 17
trainer.train()



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3352579090.py in <cell line: 0>()
----> 1 trainer.train()
      2 

NameError: name 'trainer' is not defined

## === cell 18
prediction_results = trainer.predict(test_ds)
preds = np.asarray(prediction_results.predictions).reshape(-1)

submission_df = pd.DataFrame({"id": eval_df["id"].values, "score": preds})
submission_df.to_csv("submission.csv", index=False)

submission_df.head(), submission_df.shape



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1414503665.py in <cell line: 0>()
----> 1 prediction_results = trainer.predict(test_ds)
      2 preds = np.asarray(prediction_results.predictions).reshape(-1)
      3 
      4 submission_df = pd.DataFrame({"id": eval_df["id"].values, "score": preds})
      5 submission_df.to_csv("submission.csv", index=False)

NameError: name 'trainer' is not defined

## === cell 19
sample_sub = pd.read_csv(path / "sample_submission.csv")
assert list(submission_df.columns) == ["id", "score"]
assert len(submission_df) == len(sample_sub)
print("Wrote submission.csv with shape:", submission_df.shape)

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3893209674.py in <cell line: 0>()
      1 sample_sub = pd.read_csv(path / "sample_submission.csv")
----> 2 assert list(submission_df.columns) == ["id", "score"]
      3 assert len(submission_df) == len(sample_sub)
      4 print("Wrote submission.csv with shape:", submission_df.shape)

NameError: name 'submission_df' is not defined
