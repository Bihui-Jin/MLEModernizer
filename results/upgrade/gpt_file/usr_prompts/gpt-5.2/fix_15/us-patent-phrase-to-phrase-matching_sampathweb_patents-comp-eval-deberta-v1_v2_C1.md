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

0.7954764062840406

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved nan) has done: 'I fix the import-time protobuf/TF conflict causing the `MessageFactory.GetPrototype` crash by setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` before importing `transformers/datasets`. Then I fix the dataset formatting/collation issues by using the expected `labels` field (not `label`) and by disabling `remove_unused_columns` so the Trainer keeps `input_ids`/`attention_mask`. Finally, I keep the same model and training loop but ensure prediction runs and writes a valid `submission.csv` with `id,score` clipped to `[0,1]`.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import numpy as np
import pandas as pd

print("Kaggle input root exists:", os.path.exists("/kaggle/input"))
print("Kaggle data root exists:", os.path.exists("/kaggle/data"))



## === cell 1
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed,
)
from datasets import Dataset as HFDataset

set_seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

try:
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
except Exception:
    pass


class CompatTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if "num_items_in_batch" in inputs:
            inputs = dict(inputs)
            inputs.pop("num_items_in_batch", None)
        return super().compute_loss(
            model, inputs, return_outputs=return_outputs, **kwargs
        )


class DataCollatorWithPaddingDropKeys(DataCollatorWithPadding):
    def __call__(self, features):
        for f in features:
            if "length" in f:
                f.pop("length")
        return super().__call__(features)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
def prepare_df(df, tokenizer):
    if "score" in df.columns:
        df = df.rename(columns={"score": "label"})
    df = df.copy()

    sep = " " + (tokenizer.sep_token or "[SEP]") + " "

    ctx = df["context"].astype(str)
    anc = df["anchor"].astype(str).str.lower()
    tgt = df["target"].astype(str).str.lower()

    section = ctx.str.strip().str[0]
    sec_tok = "[" + section + "]"

    df["inputs"] = sec_tok + sep + ctx + sep + anc + sep + tgt
    return df


def tokenize_dataset_fast(df, tokenizer, with_labels: bool):
    keep_cols = ["inputs"] + (["label"] if with_labels else [])
    hfd = HFDataset.from_pandas(df[keep_cols], preserve_index=False)

    def _tok(batch):
        enc = tokenizer(
            batch["inputs"],
            padding=False,  # dynamic padding via data collator
            truncation=True,
            max_length=128,
            return_attention_mask=True,
            return_length=True,  # enables fast/accurate length bucketing
        )
        if with_labels:
            enc["labels"] = np.asarray(batch["label"], dtype=np.float32)
        return enc

    hfd = hfd.map(
        _tok,
        batched=True,
        batch_size=2048,
        num_proc=1,
        remove_columns=hfd.column_names,
        desc="Tokenizing",
    )

    cols = ["input_ids", "attention_mask"] + (["labels"] if with_labels else [])
    hfd.set_format(type="torch", columns=cols)
    return hfd




## === cell 3
MODEL_NAME = "microsoft/deberta-v3-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=1,
    problem_type="regression",
)

if torch.cuda.is_available():
    pass

try:
    model.gradient_checkpointing_enable()
except Exception:
    pass

data_collator = DataCollatorWithPaddingDropKeys(tokenizer=tokenizer)



## === cell 4
train_path = "/kaggle/input/us-patent-phrase-to-phrase-matching/train.csv"
test_path = "/kaggle/input/us-patent-phrase-to-phrase-matching/test.csv"
if not os.path.exists(train_path):
    train_path = "/kaggle/data/us-patent-phrase-to-phrase-matching/train.csv"
if not os.path.exists(test_path):
    test_path = "/kaggle/data/us-patent-phrase-to-phrase-matching/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

train_df = prepare_df(train_df, tokenizer)
test_df = prepare_df(test_df, tokenizer)

train_ds = tokenize_dataset_fast(train_df, tokenizer, with_labels=True)
test_ds = tokenize_dataset_fast(test_df, tokenizer, with_labels=False)

cpu_cnt = os.cpu_count() or 1
dl_workers = min(4, cpu_cnt)

training_args = TrainingArguments(
    output_dir="out",
    overwrite_output_dir=True,
    do_train=True,
    do_eval=False,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=128,
    learning_rate=2e-5,
    num_train_epochs=1.0,
    weight_decay=0.01,
    logging_steps=200,
    save_strategy="no",
    report_to=[],
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=dl_workers,
    dataloader_pin_memory=torch.cuda.is_available(),
    dataloader_persistent_workers=(dl_workers > 0),
    dataloader_prefetch_factor=2 if dl_workers > 0 else None,
    remove_unused_columns=False,
    optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
    group_by_length=True,
    length_column_name="length",  # make grouping deterministic + fast using the computed token lengths
    prediction_loss_only=True,
)

trainer = CompatTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_ds,
)

trainer.train()

pred_out = trainer.predict(test_ds)
logits = pred_out.predictions

if isinstance(logits, (tuple, list)):
    logits = logits[0]
logits = np.asarray(logits)

if logits.ndim == 2 and logits.shape[1] == 1:
    preds = logits[:, 0]
elif logits.ndim == 1:
    preds = logits
else:
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    idx = np.arange(probs.shape[1], dtype=np.float32)
    if probs.shape[1] > 1:
        idx = idx / (probs.shape[1] - 1)
    preds = (probs * idx[None, :]).sum(axis=1)

preds = preds.astype(float)
preds = np.clip(preds, 0.0, 1.0).reshape(-1)

sub_df = pd.DataFrame({"id": test_df["id"].values, "score": preds})
sub_df.to_csv("submission.csv", index=False)

print(sub_df.head())
print("Wrote submission.csv with shape:", sub_df.shape)
print("submission.csv exists:", os.path.exists("submission.csv"))

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1612841921.py in <cell line: 0>()
     52 )
     53 
---> 54 trainer.train()
     55 
     56 pred_out = trainer.predict(test_ds)

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

/usr/local/lib/python3.11/dist-packages/transformers/trainer_pt_utils.py in <lambda>(i)
    603     megabatch_size = mega_batch_mult * batch_size
    604     megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
--> 605     megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    606 
    607     # The rest is to get the biggest batch first.

/usr/local/lib/python3.11/dist-packages/datasets/arrow_dataset.py in __getitem__(self, key)
    680             else:
    681                 source = self.source._fast_select_column(self.column_name)
--> 682             return source[key][self.column_name]
    683         elif isinstance(key, int):
    684             return self.source[key][self.column_name]

KeyError: 'length'
