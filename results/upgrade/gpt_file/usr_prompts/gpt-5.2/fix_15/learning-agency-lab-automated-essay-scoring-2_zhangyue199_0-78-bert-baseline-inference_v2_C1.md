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
Predict the score of student essays.

## Metric
Quadratic weighted kappa.

## Submission Format
For each `essay_id` in the test set, you must predict the corresponding `score` (between 1-6, see [rubric](https://storage.googleapis.com/kaggle-forum-message-attachments/2733927/20538/Rubric_%20Holistic%20Essay%20Scoring.pdf) for more details). The file should contain a header and have the following format:

```
essay_id,score
000d118,3
000fe60,3
001ab80,4
...
```

## Dataset
- **train.csv** - Essays and scores to be used as training data.
    - `essay_id` - The unique ID of the essay
    - `full_text` - The full essay response
    - `score` - Holistic score of the essay on a 1-6 scale
- **test.csv** - The essays to be used as test data. Contains the same fields as `train.csv`, aside from exclusion of `score`.
- **sample_submission.csv** - A submission file in the correct format.
    - `essay_id` - The unique ID of the essay
    - `score` - The predicted holistic score of the essay on a 1-6 scale

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
            description.md (153 lines)
            sample_submission.csv (1732 lines)
            sample_submission.csv.zip (9.4 kB)
            test.csv (15336 lines)
            test.csv.zip (1.2 MB)
            train.csv (139231 lines)
            train.csv.zip (11.0 MB)
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
        input/
            description.md (153 lines)
            sample_submission.csv (1732 lines)
            sample_submission.csv.zip (9.4 kB)
            test.csv (15336 lines)
            test.csv.zip (1.2 MB)
            train.csv (139231 lines)
            train.csv.zip (11.0 MB)
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
        working/
            learning-agency-lab-automated-essay-scoring-2/
                description.md (153 lines)
                sample_submission.csv (1732 lines)
                ... and 5 other files
                learning-agency-lab-automated-essay-scoring-2/
```

-> data/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv has 1731 rows and 2 columns.
The columns are: essay_id, score

-> data/learning-agency-lab-automated-essay-scoring-2/test.csv has 15335 rows and 2 columns.
The columns are: essay_id, full_text

-> data/learning-agency-lab-automated-essay-scoring-2/train.csv has 139230 rows and 3 columns.
The columns are: essay_id, full_text, score

-> data/sample_submission.csv has 1731 rows and 2 columns.
The columns are: essay_id, score

-> data/test.csv has 15335 rows and 2 columns.
The columns are: essay_id, full_text

-> data/train.csv has 139230 rows and 3 columns.
The columns are: essay_id, full_text, score

-> (stopped after 10 files for performance)

# 5. Target score

0.7788391867553548

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import re
import random
import numpy as np
import pandas as pd

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
    DataCollatorWithPadding,
)

from datasets import Dataset as HFDataset

SEED = 42
set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

train_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
test_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"

train_df = pd.read_csv(train_path, usecols=["essay_id", "full_text", "score"])
test_df = pd.read_csv(test_path, usecols=["essay_id", "full_text"])

MAX_LEN = 512
MODEL_ID = "bert-base-uncased"

print("train_df:", train_df.shape, "test_df:", test_df.shape)
print("train columns:", train_df.columns.tolist())
print("test columns:", test_df.columns.tolist())




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
_CLEAN_RE = re.compile(r"[^a-zA-Z0-9]+")
_WS_RE = re.compile(r"\s+")


def clean_text_series(s: pd.Series) -> pd.Series:
    s = s.astype(str)
    s = s.str.replace(_CLEAN_RE, " ", regex=True)
    s = s.str.replace(_WS_RE, " ", regex=True)
    return s.str.strip()


train_df["full_text"] = clean_text_series(train_df["full_text"])
test_df["full_text"] = clean_text_series(test_df["full_text"])

train_df["label"] = (train_df["score"].astype(int) - 1).clip(0, 5)

print(train_df[["essay_id", "score", "label"]].head())




## === cell 2
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=6)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

if hasattr(torch, "compile"):
    try:
        model = torch.compile(model)  # preserves outputs; reduces overhead
        print("Enabled torch.compile")
    except Exception as e:
        print("torch.compile unavailable/failed, continuing without it:", repr(e))




## === cell 3
val_frac = 0.05
val_size = max(1, int(len(train_df) * val_frac))
train_split = train_df.iloc[:-val_size].reset_index(drop=True)
val_split = train_df.iloc[-val_size:].reset_index(drop=True)

_cpu = os.cpu_count() or 1

train_dataset = HFDataset.from_pandas(
    train_split[["full_text", "label"]], preserve_index=False
).rename_column("label", "labels")
val_dataset = HFDataset.from_pandas(
    val_split[["full_text", "label"]], preserve_index=False
).rename_column("label", "labels")
test_dataset = HFDataset.from_pandas(test_df[["full_text"]], preserve_index=False)


def tokenize_and_add_length(batch):
    enc = tokenizer(
        batch["full_text"],
        truncation=True,
        max_length=MAX_LEN,
        padding=False,  # dynamic padding done by DataCollatorWithPadding
        return_attention_mask=True,
    )
    enc["length"] = np.fromiter(
        (len(x) for x in enc["input_ids"]), dtype=np.int32
    ).tolist()
    if "labels" in batch:
        enc["labels"] = batch["labels"]
    return enc


cache_dir = "./hf_cache_tokenized"
os.makedirs(cache_dir, exist_ok=True)

tok_proc = 1

train_dataset = train_dataset.map(
    tokenize_and_add_length,
    batched=True,
    num_proc=tok_proc,
    desc="Tokenizing train",
    remove_columns=["full_text"],
    load_from_cache_file=True,
    cache_file_name=os.path.join(
        cache_dir, f"train_{MODEL_ID.replace('/','_')}_{MAX_LEN}.arrow"
    ),
)
val_dataset = val_dataset.map(
    tokenize_and_add_length,
    batched=True,
    num_proc=tok_proc,
    desc="Tokenizing val",
    remove_columns=["full_text"],
    load_from_cache_file=True,
    cache_file_name=os.path.join(
        cache_dir, f"val_{MODEL_ID.replace('/','_')}_{MAX_LEN}.arrow"
    ),
)
test_dataset = test_dataset.map(
    tokenize_and_add_length,
    batched=True,
    num_proc=tok_proc,
    desc="Tokenizing test",
    remove_columns=["full_text"],
    load_from_cache_file=True,
    cache_file_name=os.path.join(
        cache_dir, f"test_{MODEL_ID.replace('/','_')}_{MAX_LEN}.arrow"
    ),
)

train_dataset = train_dataset.with_format(
    "torch", columns=["input_ids", "attention_mask", "labels", "length"]
)
val_dataset = val_dataset.with_format(
    "torch", columns=["input_ids", "attention_mask", "labels", "length"]
)
test_dataset = test_dataset.with_format(
    "torch", columns=["input_ids", "attention_mask", "length"]
)

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer, pad_to_multiple_of=8 if torch.cuda.is_available() else None
)

print("train/val/test sizes:", len(train_dataset), len(val_dataset), len(test_dataset))
print("tokenize num_proc:", tok_proc)




## === cell 4
num_workers = 0

per_device_bs = 16 if torch.cuda.is_available() else 8
grad_accum = 2 if torch.cuda.is_available() else 1

if hasattr(model, "gradient_checkpointing_enable"):
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass

args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    report_to="none",
    seed=SEED,
    dataloader_drop_last=False,
    do_train=True,
    do_eval=True,
    eval_strategy="epoch",
    save_strategy="no",
    per_device_train_batch_size=per_device_bs,
    per_device_eval_batch_size=per_device_bs,
    gradient_accumulation_steps=grad_accum,
    num_train_epochs=1,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.06,
    logging_steps=50,
    fp16=False,
    dataloader_num_workers=num_workers,
    dataloader_pin_memory=True if torch.cuda.is_available() else False,
    dataloader_prefetch_factor=None,
    dataloader_persistent_workers=False,
    group_by_length=True,
    length_column_name="length",
)

trainer = Trainer(
    model=model,
    args=args,
    tokenizer=tokenizer,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

train_result = trainer.train()
eval_result = trainer.evaluate()
print("train_result:", {k: float(v) for k, v in train_result.metrics.items()})
print("eval_result:", {k: float(v) for k, v in eval_result.items()})




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/313294029.py in <cell line: 0>()
     47 )
     48 
---> 49 train_result = trainer.train()
     50 eval_result = trainer.evaluate()
     51 print("train_result:", {k: float(v) for k, v in train_result.metrics.items()})

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in train(self, resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)
   2204                 hf_hub_utils.enable_progress_bars()
   2205         else:
-> 2206             return inner_training_loop(
   2207                 args=args,
   2208                 resume_from_checkpoint=resume_from_checkpoint,

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in _inner_training_loop(self, batch_size, args, resume_from_checkpoint, trial, ignore_keys_for_eval)
   2254         logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
   2255         # Data loader and number of training steps
-> 2256         train_dataloader = self.get_train_dataloader()
   2257         if self.is_fsdp_xla_v2_enabled:
   2258             train_dataloader = tpu_spmd_dataloader(train_dataloader)

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in get_train_dataloader(self)
   1051             raise ValueError("Trainer: training requires a train_dataset.")
   1052 
-> 1053         return self._get_dataloader(
   1054             dataset=self.train_dataset,
   1055             description="Training",

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in _get_dataloader(self, dataset, description, batch_size, sampler_fn, is_training, dataloader_key)
   1005         data_collator = self.data_collator
   1006         if is_datasets_available() and isinstance(dataset, datasets.Dataset):
-> 1007             dataset = self._remove_unused_columns(dataset, description=description)
   1008         else:
   1009             data_collator = self._get_collator_with_removed_columns(self.data_collator, description=description)

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in _remove_unused_columns(self, dataset, description)
    931         columns = [k for k in signature_columns if k in dataset.column_names]
    932         if len(columns) == 0:
--> 933             raise ValueError(
    934                 f"No columns in the dataset match the model's forward method signature: ({', '.join(signature_columns)}). "
    935                 f"The following columns have been ignored: [{', '.join(ignored_columns)}]. "

ValueError: No columns in the dataset match the model's forward method signature: (args, kwargs, label, label_ids). The following columns have been ignored: [attention_mask, length, token_type_ids, input_ids, labels]. Please check the dataset and model. You may need to set `remove_unused_columns=False` in `TrainingArguments`.

## === cell 5
pred_out = trainer.predict(test_dataset)
preds = pred_out.predictions

if isinstance(preds, (tuple, list)):
    preds = preds[0]

pred_labels = np.argmax(preds, axis=1).astype(int)
scores = (pred_labels + 1).clip(1, 6).astype(int)

sub = test_df[["essay_id"]].copy()
sub["score"] = scores
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
assert list(sub.columns) == ["essay_id", "score"]
assert sub.shape[0] == test_df.shape[0]
assert os.path.exists("submission.csv")

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/4163067646.py in <cell line: 0>()
----> 1 pred_out = trainer.predict(test_dataset)
      2 preds = pred_out.predictions
      3 
      4 if isinstance(preds, (tuple, list)):
      5     preds = preds[0]

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in predict(self, test_dataset, ignore_keys, metric_key_prefix)
   4271         self._memory_tracker.start()
   4272 
-> 4273         test_dataloader = self.get_test_dataloader(test_dataset)
   4274         start_time = time.time()
   4275 

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in get_test_dataloader(self, test_dataset)
   1154                 `model.forward()` method are automatically removed. It must implement `__len__`.
   1155         """
-> 1156         return self._get_dataloader(
   1157             dataset=test_dataset,
   1158             description="test",

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in _get_dataloader(self, dataset, description, batch_size, sampler_fn, is_training, dataloader_key)
   1005         data_collator = self.data_collator
   1006         if is_datasets_available() and isinstance(dataset, datasets.Dataset):
-> 1007             dataset = self._remove_unused_columns(dataset, description=description)
   1008         else:
   1009             data_collator = self._get_collator_with_removed_columns(self.data_collator, description=description)

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in _remove_unused_columns(self, dataset, description)
    931         columns = [k for k in signature_columns if k in dataset.column_names]
    932         if len(columns) == 0:
--> 933             raise ValueError(
    934                 f"No columns in the dataset match the model's forward method signature: ({', '.join(signature_columns)}). "
    935                 f"The following columns have been ignored: [{', '.join(ignored_columns)}]. "

ValueError: No columns in the dataset match the model's forward method signature: (args, kwargs, label, label_ids). The following columns have been ignored: [token_type_ids, input_ids, attention_mask, length]. Please check the dataset and model. You may need to set `remove_unused_columns=False` in `TrainingArguments`.
