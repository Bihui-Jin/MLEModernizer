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

3.11

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

0.8066012823938029

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
from pathlib import Path
import os

path = Path("/kaggle/input/us-patent-phrase-to-phrase-matching")
assert (path / "train.csv").exists(), f"Missing train.csv at {path}"



## === cell 1
print("Using data path:", path)



## === cell 2
import pandas as pd

df = pd.read_csv(
    path / "train.csv", usecols=["id", "anchor", "target", "context", "score"]
)
print("train shape:", df.shape)



## === cell 3
pass



## === cell 4
pass



## === cell 5
df["input"] = (
    "TEXT1: " + df["context"] + "; TEXT2: " + df["target"] + "; ANC: " + df["anchor"]
)



## === cell 6
from datasets import Dataset, DatasetDict
import datasets as _datasets
import os as _os

_cache_dir = _os.environ.get("HF_DATASETS_CACHE", "/kaggle/working/hf_datasets_cache")
_os.environ["HF_DATASETS_CACHE"] = _cache_dir
_datasets.config.HF_DATASETS_CACHE = _cache_dir

ds = Dataset.from_pandas(df, preserve_index=False)



## === cell 7
_ = ds



## === cell 8
_candidate_model_dirs = [
    Path("/kaggle/input/deberta-v3-small"),
    Path("/kaggle/input/microsoft-deberta-v3-small"),
    Path("/kaggle/input/deberta-v3-small-hf"),
    Path("/kaggle/input/deberta-v3-base"),
]
model_name = None
for p in _candidate_model_dirs:
    if p.exists():
        model_name = str(p)
        break

if model_name is None:
    model_name = "microsoft/deberta-v3-small"

print("Using model:", model_name)



## === cell 9
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokz = AutoTokenizer.from_pretrained(model_name, use_fast=True)



## === cell 10
pass



## === cell 11
_MAX_LEN = min(getattr(tokz, "model_max_length", 512) or 512, 512)


def tok_func(x):
    return tokz(x["input"], truncation=True, max_length=_MAX_LEN)




## === cell 12
_num_proc = 1

tok_ds = ds.map(
    tok_func,
    batched=True,
    num_proc=_num_proc,
    load_from_cache_file=False,
    desc="Tokenizing train",
)



## === cell 13
pass



## === cell 14
pass



## === cell 15
tok_ds = tok_ds.rename_columns({"score": "labels"})



## === cell 16
_ = tok_ds[0]



## === cell 17
eval_df = pd.read_csv(path / "test.csv", usecols=["id", "anchor", "target", "context"])



## === cell 18
pass



## === cell 19
dds = tok_ds.train_test_split(0.25, seed=42)



## === cell 20
eval_df["input"] = (
    "TEXT1: "
    + eval_df["context"]
    + "; TEXT2: "
    + eval_df["target"]
    + "; ANC: "
    + eval_df["anchor"]
)

from datasets import Dataset as _Dataset

eval_ds = _Dataset.from_pandas(eval_df, preserve_index=False).map(
    tok_func,
    batched=True,
    num_proc=_num_proc,
    load_from_cache_file=False,
    desc="Tokenizing test",
)



## === cell 21
_keep_cols = {"input_ids", "attention_mask", "token_type_ids", "labels", "id"}


def _prune_columns(dset):
    cols = set(dset.column_names)
    remove = [c for c in cols if c not in _keep_cols]
    if remove:
        dset = dset.remove_columns(remove)
    return dset


dds = DatasetDict({k: _prune_columns(v) for k, v in dds.items()})
eval_ds = _prune_columns(eval_ds)

for k in dds:
    dds[k].set_format(
        type="torch",
        columns=[
            c
            for c in ["input_ids", "attention_mask", "token_type_ids", "labels"]
            if c in dds[k].column_names
        ],
    )
eval_ds.set_format(
    type="torch",
    columns=[
        c
        for c in ["input_ids", "attention_mask", "token_type_ids"]
        if c in eval_ds.column_names
    ],
)



## === cell 22
import numpy as np


def corr(x, y):
    return np.corrcoef(x, y)[0][1]


def corr_d(eval_pred):
    preds, labels = eval_pred
    preds = np.asarray(preds).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    return {"pearson": corr(preds, labels)}




## === cell 23
from transformers import TrainingArguments, Trainer
import transformers

print("transformers version:", transformers.__version__)


def make_training_args(**kwargs):
    try:
        return TrainingArguments(**kwargs)
    except TypeError as e:
        if (
            "evaluation_strategy" in kwargs
            and "unexpected keyword argument 'evaluation_strategy'" in str(e)
        ):
            kwargs2 = dict(kwargs)
            kwargs2["eval_strategy"] = kwargs2.pop("evaluation_strategy")
            return TrainingArguments(**kwargs2)
        raise




## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 24
bs = 128
epochs = 4
lr = 8e-5



## === cell 25
import torch
import random

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

use_fp16 = torch.cuda.is_available()

_num_workers = min(4, (_os.cpu_count() or 2))
args = make_training_args(
    output_dir="outputs",
    learning_rate=lr,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    fp16=use_fp16,
    evaluation_strategy="no",
    per_device_train_batch_size=bs,
    per_device_eval_batch_size=bs * 2,
    num_train_epochs=epochs,
    weight_decay=0.01,
    report_to="none",
    logging_steps=200,
    remove_unused_columns=True,
    disable_tqdm=True,
    dataloader_num_workers=_num_workers,
    dataloader_pin_memory=torch.cuda.is_available(),
    dataloader_persistent_workers=(_num_workers > 0),
    group_by_length=True,
    length_column_name="attention_mask",  # Trainer will derive lengths; keeping explicit for compatibility
    seed=seed,
    data_seed=seed,
)



## === cell 26
from transformers import DataCollatorWithPadding, AutoConfig

data_collator = DataCollatorWithPadding(tokenizer=tokz)

cfg = AutoConfig.from_pretrained(model_name)
cfg.num_labels = 1
cfg.problem_type = "regression"

model = AutoModelForSequenceClassification.from_pretrained(model_name, config=cfg)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dds["train"],
    eval_dataset=dds[
        "test"
    ],  # kept to preserve semantics/structure even if eval is disabled
    tokenizer=tokz,
    data_collator=data_collator,
    compute_metrics=corr_d,
)



## === cell 27
trainer.train()



## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
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
   2500                 update_step += 1
   2501                 num_batches = args.gradient_accumulation_steps if update_step != (total_updates - 1) else remainder
-> 2502                 batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, num_batches, args.device)
   2503                 for i, inputs in enumerate(batch_samples):
   2504                     step += 1

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in get_batch_samples(self, epoch_iterator, num_batches, device)
   5298         for _ in range(num_batches):
   5299             try:
-> 5300                 batch_samples.append(next(epoch_iterator))
   5301             except StopIteration:
   5302                 break

/usr/local/lib/python3.11/dist-packages/accelerate/data_loader.py in __iter__(self)
    562 
    563         self.set_epoch(self.iteration)
--> 564         dataloader_iter = self.base_dataloader.__iter__()
    565         # We iterate one batch ahead to check when we are at the end
    566         try:

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __iter__(self)
    484         if self.persistent_workers and self.num_workers > 0:
    485             if self._iterator is None:
--> 486                 self._iterator = self._get_iterator()
    487             else:
    488                 self._iterator._reset(self)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _get_iterator(self)
    420         else:
    421             self.check_worker_number_rationality()
--> 422             return _MultiProcessingDataLoaderIter(self)
    423 
    424     @property

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __init__(self, loader)
   1197         _utils.signal_handling._set_SIGCHLD_handler()
   1198         self._worker_pids_set = True
-> 1199         self._reset(loader, first_iter=True)
   1200 
   1201     def _reset(self, loader, first_iter=False):

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _reset(self, loader, first_iter)
   1234         # prime the prefetch loop
   1235         for _ in range(self._prefetch_factor * self._num_workers):
-> 1236             self._try_put_index()
   1237 
   1238     def _try_get_data(self, timeout=_utils.MP_STATUS_CHECK_INTERVAL):

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _try_put_index(self)
   1484 
   1485         try:
-> 1486             index = self._next_index()
   1487         except StopIteration:
   1488             return

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_index(self)
    696 
    697     def _next_index(self):
--> 698         return next(self._sampler_iter)  # may raise StopIteration
    699 
    700     def _next_data(self):

/usr/local/lib/python3.11/dist-packages/torch/utils/data/sampler.py in __iter__(self)
    335     def __iter__(self) -> Iterator[List[int]]:
    336         # Implemented based on the benchmarking in https://github.com/pytorch/pytorch/pull/76951
--> 337         sampler_iter = iter(self.sampler)
    338         if self.drop_last:
    339             # Create multiple references to the same iterator

/usr/local/lib/python3.11/dist-packages/transformers/trainer_pt_utils.py in __iter__(self)
    657 
    658     def __iter__(self):
--> 659         indices = get_length_grouped_indices(self.lengths, self.batch_size, generator=self.generator)
    660         return iter(indices)
    661 

/usr/local/lib/python3.11/dist-packages/transformers/trainer_pt_utils.py in get_length_grouped_indices(lengths, batch_size, mega_batch_mult, generator)
    603     megabatch_size = mega_batch_mult * batch_size
    604     megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
--> 605     megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    606 
    607     # The rest is to get the biggest batch first.

/usr/local/lib/python3.11/dist-packages/transformers/trainer_pt_utils.py in <listcomp>(.0)
    603     megabatch_size = mega_batch_mult * batch_size
    604     megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
--> 605     megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    606 
    607     # The rest is to get the biggest batch first.

RuntimeError: The size of tensor a (20) must match the size of tensor b (19) at non-singleton dimension 0

## === cell 28
pred_out = trainer.predict(eval_ds)
preds = np.asarray(pred_out.predictions).astype(float).reshape(-1)

preds = np.clip(preds, 0, 1)

submission = pd.DataFrame({"id": eval_df["id"].values, "score": preds})
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Saved at:", os.path.abspath("submission.csv"))

## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/500875141.py in <cell line: 0>()
----> 1 pred_out = trainer.predict(eval_ds)
      2 preds = np.asarray(pred_out.predictions).astype(float).reshape(-1)
      3 
      4 preds = np.clip(preds, 0, 1)
      5 

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in predict(self, test_dataset, ignore_keys, metric_key_prefix)
   4275 
   4276         eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
-> 4277         output = eval_loop(
   4278             test_dataloader, description="Prediction", ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
   4279         )

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in evaluation_loop(self, dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)
   4382 
   4383         # Main evaluation loop
-> 4384         for step, inputs in enumerate(dataloader):
   4385             # Update the observed num examples
   4386             observed_batch_size = find_batch_size(inputs)

/usr/local/lib/python3.11/dist-packages/accelerate/data_loader.py in __iter__(self)
    562 
    563         self.set_epoch(self.iteration)
--> 564         dataloader_iter = self.base_dataloader.__iter__()
    565         # We iterate one batch ahead to check when we are at the end
    566         try:

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __iter__(self)
    484         if self.persistent_workers and self.num_workers > 0:
    485             if self._iterator is None:
--> 486                 self._iterator = self._get_iterator()
    487             else:
    488                 self._iterator._reset(self)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _get_iterator(self)
    420         else:
    421             self.check_worker_number_rationality()
--> 422             return _MultiProcessingDataLoaderIter(self)
    423 
    424     @property

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __init__(self, loader)
   1197         _utils.signal_handling._set_SIGCHLD_handler()
   1198         self._worker_pids_set = True
-> 1199         self._reset(loader, first_iter=True)
   1200 
   1201     def _reset(self, loader, first_iter=False):

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _reset(self, loader, first_iter)
   1234         # prime the prefetch loop
   1235         for _ in range(self._prefetch_factor * self._num_workers):
-> 1236             self._try_put_index()
   1237 
   1238     def _try_get_data(self, timeout=_utils.MP_STATUS_CHECK_INTERVAL):

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _try_put_index(self)
   1484 
   1485         try:
-> 1486             index = self._next_index()
   1487         except StopIteration:
   1488             return

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_index(self)
    696 
    697     def _next_index(self):
--> 698         return next(self._sampler_iter)  # may raise StopIteration
    699 
    700     def _next_data(self):

/usr/local/lib/python3.11/dist-packages/torch/utils/data/sampler.py in __iter__(self)
    335     def __iter__(self) -> Iterator[List[int]]:
    336         # Implemented based on the benchmarking in https://github.com/pytorch/pytorch/pull/76951
--> 337         sampler_iter = iter(self.sampler)
    338         if self.drop_last:
    339             # Create multiple references to the same iterator

/usr/local/lib/python3.11/dist-packages/transformers/trainer_pt_utils.py in __iter__(self)
    657 
    658     def __iter__(self):
--> 659         indices = get_length_grouped_indices(self.lengths, self.batch_size, generator=self.generator)
    660         return iter(indices)
    661 

/usr/local/lib/python3.11/dist-packages/transformers/trainer_pt_utils.py in get_length_grouped_indices(lengths, batch_size, mega_batch_mult, generator)
    603     megabatch_size = mega_batch_mult * batch_size
    604     megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
--> 605     megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    606 
    607     # The rest is to get the biggest batch first.

/usr/local/lib/python3.11/dist-packages/transformers/trainer_pt_utils.py in <listcomp>(.0)
    603     megabatch_size = mega_batch_mult * batch_size
    604     megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
--> 605     megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    606 
    607     # The rest is to get the biggest batch first.

RuntimeError: The size of tensor a (20) must match the size of tensor b (21) at non-singleton dimension 0
