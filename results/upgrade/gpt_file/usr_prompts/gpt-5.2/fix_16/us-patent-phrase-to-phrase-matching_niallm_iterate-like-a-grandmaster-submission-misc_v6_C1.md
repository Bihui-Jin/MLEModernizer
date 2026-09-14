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

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

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
import warnings, logging, random
import torch
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
        self.encodings = encodings
        self.labels = labels  # keep as numpy/float; cast in __getitem__

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
        }
        if "token_type_ids" in self.encodings:
            item["token_type_ids"] = self.encodings["token_type_ids"][idx]
        if self.labels is not None:
            item["labels"] = float(self.labels[idx])
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



## === cell 17
trainer.train()



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
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
    565         # We iterate one batch ahead to check when we are at the end
    566         try:
--> 567             current_batch = next(dataloader_iter)
    568         except StopIteration:
    569             self.end()

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
   1478                 del self._task_info[idx]
   1479                 self._rcvd_idx += 1
-> 1480                 return self._process_data(data)
   1481 
   1482     def _try_put_index(self):

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _process_data(self, data)
   1503         self._try_put_index()
   1504         if isinstance(data, ExceptionWrapper):
-> 1505             data.reraise()
   1506         return data
   1507 

/usr/local/lib/python3.11/dist-packages/torch/_utils.py in reraise(self)
    731             # instantiate since we don't know how to
    732             raise RuntimeError(msg) from None
--> 733         raise exception
    734 
    735 

ValueError: Caught ValueError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 55, in fetch
    return self.collate_fn(data)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/trainer_utils.py", line 872, in __call__
    return self.data_collator(features)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/data/data_collator.py", line 272, in __call__
    batch = pad_without_fast_tokenizer_warning(
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/data/data_collator.py", line 67, in pad_without_fast_tokenizer_warning
    padded = tokenizer.pad(*pad_args, **pad_kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py", line 3291, in pad
    raise ValueError(
ValueError: You should supply an encoding or a list of encodings to this method that includes input_ids, but you provided []


## === cell 18
prediction_results = trainer.predict(test_ds)
preds = np.asarray(prediction_results.predictions).reshape(-1)

submission_df = pd.DataFrame({"id": eval_df["id"].values, "score": preds})
submission_df.to_csv("submission.csv", index=False)

submission_df.head(), submission_df.shape



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1414503665.py in <cell line: 0>()
----> 1 prediction_results = trainer.predict(test_ds)
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
   4382 
   4383         # Main evaluation loop
-> 4384         for step, inputs in enumerate(dataloader):
   4385             # Update the observed num examples
   4386             observed_batch_size = find_batch_size(inputs)

/usr/local/lib/python3.11/dist-packages/accelerate/data_loader.py in __iter__(self)
    565         # We iterate one batch ahead to check when we are at the end
    566         try:
--> 567             current_batch = next(dataloader_iter)
    568         except StopIteration:
    569             self.end()

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
   1478                 del self._task_info[idx]
   1479                 self._rcvd_idx += 1
-> 1480                 return self._process_data(data)
   1481 
   1482     def _try_put_index(self):

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _process_data(self, data)
   1503         self._try_put_index()
   1504         if isinstance(data, ExceptionWrapper):
-> 1505             data.reraise()
   1506         return data
   1507 

/usr/local/lib/python3.11/dist-packages/torch/_utils.py in reraise(self)
    731             # instantiate since we don't know how to
    732             raise RuntimeError(msg) from None
--> 733         raise exception
    734 
    735 

ValueError: Caught ValueError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 55, in fetch
    return self.collate_fn(data)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/trainer_utils.py", line 872, in __call__
    return self.data_collator(features)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/data/data_collator.py", line 272, in __call__
    batch = pad_without_fast_tokenizer_warning(
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/data/data_collator.py", line 67, in pad_without_fast_tokenizer_warning
    padded = tokenizer.pad(*pad_args, **pad_kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py", line 3291, in pad
    raise ValueError(
ValueError: You should supply an encoding or a list of encodings to this method that includes input_ids, but you provided []


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
