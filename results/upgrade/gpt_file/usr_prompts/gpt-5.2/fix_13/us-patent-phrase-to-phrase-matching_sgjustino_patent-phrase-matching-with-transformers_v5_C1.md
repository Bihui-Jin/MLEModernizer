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

# 5. Target score

0.7805410791891036

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.01981) has done: 'The timeout is dominated by training compute (5 epochs) and by avoidable overhead in data handling. I keep the exact same model, loss, epochs, batch sizes, and evaluation, but reduce overhead by (1) switching to dynamic padding during tokenization (so we don’t pad every example to length 48 twice), (2) using Hugging Face `Dataset`’s native `train_test_split` instead of a Python-level subset wrapper (faster indexing/data loading), and (3) enabling safe runtime settings (`torch.set_float32_matmul_precision("high")`, `gradient_checkpointing`, and disabling `use_cache`) that preserve numerical results while reducing memory pressure and speeding throughput on GPU. I also ensure the `Trainer` uses the correct label field and avoids unnecessary column processing.'
- What this solution (achieved 0.01981) has done: 'The timeout is dominated by fine-tuning DeBERTa for 5 epochs on ~26k training rows with evaluation each epoch; the current setup also pays extra overhead in collation and uses `torch.compile`, which is often slow to compile relative to this workload. I keep the exact same model, loss, and training loop semantics, but reduce overhead by (1) switching to a fixed-length collator (avoids per-batch dynamic padding cost), (2) enabling gradient checkpointing (trades compute for large memory/batch stability and often improves throughput on Kaggle GPUs by avoiding memory pressure/stalls), (3) turning off `torch.compile` to avoid compilation time, and (4) ensuring dataloaders are maximally efficient and stable. Tokenization stays identical (same max_length/truncation), and the train/val split and metric remain unchanged.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")



## === cell 1
import pandas as pd
import numpy as np
import warnings, logging
import random
import math

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



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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

per_device_train_bs = 256
grad_accum = 1
num_epochs = 5
train_len = len(dataset_split["train"])
steps_per_epoch = math.ceil(train_len / (per_device_train_bs * grad_accum))
max_steps = steps_per_epoch * num_epochs

ta_kwargs = dict(
    output_dir="outputs",
    learning_rate=8e-5,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    fp16=False,
    per_device_train_batch_size=per_device_train_bs,
    per_device_eval_batch_size=256,
    gradient_accumulation_steps=grad_accum,
    num_train_epochs=num_epochs,
    max_steps=max_steps,  # deterministic total training work; matches epochs above
    weight_decay=0.01,
    report_to="none",
    logging_steps=200,
    seed=SEED,
    data_seed=SEED,
    dataloader_num_workers=_num_workers,
    dataloader_pin_memory=torch.cuda.is_available(),
    dataloader_persistent_workers=(_num_workers > 0),
    dataloader_prefetch_factor=4 if _num_workers > 0 else None,
    remove_unused_columns=False,
    label_names=["label"],
    save_strategy="no",
    disable_tqdm=True,
    group_by_length=True,
    length_column_name="attention_mask",
)

try:
    args = TrainingArguments(
        **ta_kwargs,
        evaluation_strategy="no",
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        torch_compile=True if torch.cuda.is_available() else False,
    )
except TypeError:
    args = TrainingArguments(
        **ta_kwargs,
        eval_strategy="no",
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



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/3122752704.py in <cell line: 0>()
----> 1 training_outcome = deberta_trainer.train()
      2 training_outcome
      3 

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

RuntimeError: The size of tensor a (12) must match the size of tensor b (9) at non-singleton dimension 0

## === cell 18
eval_metrics = deberta_trainer.evaluate(dataset_split["test"])
eval_metrics



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/1619609441.py in <cell line: 0>()
      1 # --- SPEED: run evaluation once after training (instead of every epoch) to eliminate repeated eval overhead.
----> 2 eval_metrics = deberta_trainer.evaluate(dataset_split["test"])
      3 eval_metrics
      4 

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in evaluate(self, eval_dataset, ignore_keys, metric_key_prefix)
   4197 
   4198         eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
-> 4199         output = eval_loop(
   4200             eval_dataloader,
   4201             description="Evaluation",

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

RuntimeError: Boolean value of Tensor with more than one value is ambiguous

## === cell 19
pred_out = deberta_trainer.predict(tokenized_test_ds)
test_prediction = pred_out.predictions.reshape(-1).astype(np.float32)

test_prediction = np.clip(test_prediction, 0.0, 1.0)
test_prediction[:10], test_prediction.min(), test_prediction.max()



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/1506665986.py in <cell line: 0>()
----> 1 pred_out = deberta_trainer.predict(tokenized_test_ds)
      2 test_prediction = pred_out.predictions.reshape(-1).astype(np.float32)
      3 
      4 test_prediction = np.clip(test_prediction, 0.0, 1.0)
      5 test_prediction[:10], test_prediction.min(), test_prediction.max()

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

RuntimeError: The size of tensor a (8) must match the size of tensor b (9) at non-singleton dimension 0

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



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1039108322.py in <cell line: 0>()
      2     {
      3         "id": test_ids.values,
----> 4         "score": test_prediction,
      5     }
      6 )

NameError: name 'test_prediction' is not defined

## === cell 21
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())

## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3000660677.py in <cell line: 0>()
----> 1 submission.to_csv("submission.csv", index=False)
      2 print("Wrote submission.csv with shape:", submission.shape)
      3 print(submission.head())

NameError: name 'submission' is not defined
