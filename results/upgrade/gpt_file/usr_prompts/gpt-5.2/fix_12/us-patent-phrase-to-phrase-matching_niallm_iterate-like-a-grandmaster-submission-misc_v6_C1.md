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
from torch.utils.data import DataLoader

warnings.simplefilter("ignore")
logging.disable(logging.WARNING)

import transformers
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import DataCollatorWithPadding

from datasets import Dataset, DatasetDict, load_from_disk

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
def tok_func(x):
    return tokz(
        x["inputs"],
        truncation=True,
        max_length=MAX_LEN,
        padding=False,  # dynamic padding in collator
    )


num_proc = 1  # keep deterministic and avoid fork overhead in constrained env
writer_batch_size = 50_000  # fewer disk writes than 10k; same data

cache_root = Path("hf_cache_tok")
tok_id = model_nm.replace("/", "_")
cache_train = cache_root / f"train_tok_{tok_id}_len{MAX_LEN}_fast{int(tokz.is_fast)}"
cache_test = cache_root / f"test_tok_{tok_id}_len{MAX_LEN}_fast{int(tokz.is_fast)}"

if cache_train.exists() and cache_test.exists():
    tok_ds = load_from_disk(str(cache_train))
    tok_eval_ds = load_from_disk(str(cache_test))
else:
    ds = Dataset.from_pandas(df, preserve_index=False).rename_column("score", "label")
    eval_ds = Dataset.from_pandas(eval_df, preserve_index=False)

    tok_ds = ds.map(
        tok_func,
        batched=True,
        batch_size=4096,  # larger batches reduce Python overhead; identical results
        num_proc=num_proc,
        load_from_cache_file=True,
        remove_columns=("anchor", "target", "context", "inputs", "id", "section"),
        writer_batch_size=writer_batch_size,
        desc="Tokenizing train",
    )
    tok_eval_ds = eval_ds.map(
        tok_func,
        batched=True,
        batch_size=4096,
        num_proc=num_proc,
        load_from_cache_file=True,
        remove_columns=("anchor", "target", "context", "inputs", "id", "section"),
        writer_batch_size=writer_batch_size,
        desc="Tokenizing test",
    )

    cache_root.mkdir(parents=True, exist_ok=True)
    tok_ds.save_to_disk(str(cache_train))
    tok_eval_ds.save_to_disk(str(cache_test))

tok_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
tok_eval_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

tok_ds[0]



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
val_indices = np.flatnonzero(is_val)
trn_indices = np.flatnonzero(~is_val)

train_ds = tok_ds.select(trn_indices)
valid_ds = tok_ds.select(val_indices)

df.iloc[trn_idxs].score.mean(), df.iloc[val_idxs].score.mean()




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
    dl_workers = 4 if iskaggle else min(8, max(2, n_cpu // 2))

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
        remove_unused_columns=False,
        seed=SEED,
        data_seed=SEED,
        group_by_length=True,  # buckets by length to reduce padding -> faster, same data/epochs
        length_column_name="length",  # will be added below if missing; harmless otherwise
        eval_accumulation_steps=32,
        dataloader_prefetch_factor=4 if dl_workers > 0 else None,
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
if "length" not in train_ds.column_names:
    train_ds = train_ds.map(
        lambda x: {"length": len(x["input_ids"])},
        desc="Adding length (train)",
    )
if "length" not in valid_ds.column_names:
    valid_ds = valid_ds.map(
        lambda x: {"length": len(x["input_ids"])},
        desc="Adding length (valid)",
    )
if "length" not in tok_eval_ds.column_names:
    tok_eval_ds = tok_eval_ds.map(
        lambda x: {"length": len(x["input_ids"])},
        desc="Adding length (test)",
    )

data_collator = DataCollatorWithPadding(tokenizer=tokz, pad_to_multiple_of=8)

model = AutoModelForSequenceClassification.from_pretrained(model_nm, num_labels=1)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    tokenizer=tokz,
    data_collator=data_collator,
    compute_metrics=corr,
)



## === cell 17
trainer.train()



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3352579090.py in <cell line: 0>()
----> 1 trainer.train()
      2 

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in train(self, resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)
   2204                 hf_hub_utils.enable_progress_bars()
   2205         else:
-> 2206             return inner_training_loop(
   2207                 args=args,
   2208                 resume_from_checkpoint=resume_from_checkpoint,

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in _inner_training_loop(self, batch_size, args, resume_from_checkpoint, trial, ignore_keys_for_eval)
   2546                     )
   2547                     with context():
-> 2548                         tr_loss_step = self.training_step(model, inputs, num_items_in_batch)
   2549 
   2550                     if (

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in training_step(self, model, inputs, num_items_in_batch)
   3747 
   3748         with self.compute_loss_context_manager():
-> 3749             loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
   3750 
   3751         del inputs

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in compute_loss(self, model, inputs, return_outputs, num_items_in_batch)
   3834                 loss_kwargs["num_items_in_batch"] = num_items_in_batch
   3835             inputs = {**inputs, **loss_kwargs}
-> 3836         outputs = model(**inputs)
   3837         # Save past state if it exists
   3838         # TODO: this needs to be fixed and made cleaner later.

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

TypeError: DebertaV2ForSequenceClassification.forward() got an unexpected keyword argument 'length'

## === cell 18
prediction_results = trainer.predict(tok_eval_ds)
preds = np.asarray(prediction_results.predictions).reshape(-1)

submission_df = pd.DataFrame({"id": eval_df["id"].values, "score": preds})
submission_df.to_csv("submission.csv", index=False)

submission_df.head(), submission_df.shape



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1321061306.py in <cell line: 0>()
----> 1 prediction_results = trainer.predict(tok_eval_ds)
      2 preds = np.asarray(prediction_results.predictions).reshape(-1)
      3 
      4 submission_df = pd.DataFrame({"id": eval_df["id"].values, "score": preds})
      5 submission_df.to_csv("submission.csv", index=False)

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in predict(self, test_dataset, ignore_keys, metric_key_prefix)
   4275 
   4276         eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
-> 4277         output = eval_loop(
   4278             test_dataloader, description="Prediction", ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
   4279         )

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in evaluation_loop(self, dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)
   4392 
   4393             # Prediction step
-> 4394             losses, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
   4395             main_input_name = getattr(self.model, "main_input_name", "input_ids")
   4396             inputs_decode = (

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in prediction_step(self, model, inputs, prediction_loss_only, ignore_keys)
   4618                     loss = None
   4619                     with self.compute_loss_context_manager():
-> 4620                         outputs = model(**inputs)
   4621                     if isinstance(outputs, dict):
   4622                         logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

TypeError: DebertaV2ForSequenceClassification.forward() got an unexpected keyword argument 'length'

## === cell 19
sample_sub = pd.read_csv(path / "sample_submission.csv")
assert list(submission_df.columns) == ["id", "score"]
assert len(submission_df) == len(sample_sub)
print("Wrote submission.csv with shape:", submission_df.shape)



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1549849452.py in <cell line: 0>()
      1 sample_sub = pd.read_csv(path / "sample_submission.csv")
----> 2 assert list(submission_df.columns) == ["id", "score"]
      3 assert len(submission_df) == len(sample_sub)
      4 print("Wrote submission.csv with shape:", submission_df.shape)
      5 

NameError: name 'submission_df' is not defined

## === cell 20
n_folds = 4
n_folds
