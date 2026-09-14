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

0.8038069821115165

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

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




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
    ds = Dataset.from_pandas(df, preserve_index=False).rename_column("score", "label")
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


def get_trainer(dds, model=None):
    if model is None:
        model = get_model()
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

if hasattr(torch, "compile"):
    model = torch.compile(model)

trainer = get_trainer(dds, model=model)
trainer.train()




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/385718519.py in <cell line: 0>()
      7 
      8 trainer = get_trainer(dds, model=model)
----> 9 trainer.train()
     10 
     11 

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
  File "/usr/local/lib/python3.11/dist-packages/transformers/data/data_collator.py", line 272, in __call__
    batch = pad_without_fast_tokenizer_warning(
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/data/data_collator.py", line 67, in pad_without_fast_tokenizer_warning
    padded = tokenizer.pad(*pad_args, **pad_kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py", line 3291, in pad
    raise ValueError(
ValueError: You should supply an encoding or a list of encodings to this method that includes input_ids, but you provided ['label']


## === cell 9
eval_df["inputs"] = eval_df.context + sep + eval_df.anchor + sep + eval_df.target
eval_ds = Dataset.from_pandas(eval_df, preserve_index=False).map(
    tok_func,
    batched=True,
    batch_size=2000,
    num_proc=2,  # keep low to avoid overhead
)

preds = trainer.predict(eval_ds).predictions.astype(float)
preds = np.clip(preds, 0, 1).flatten()

submission_df = pd.DataFrame({"id": eval_df["id"], "score": preds})
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)

print("=== sample submission ===")
print(pd.read_csv(path / "sample_submission.csv").head(3))

print("\n=== our submission ===")
print(submission_df.head(3))




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1879507538.py in <cell line: 0>()
      7 )
      8 
----> 9 preds = trainer.predict(eval_ds).predictions.astype(float)
     10 preds = np.clip(preds, 0, 1).flatten()
     11 

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

ValueError: No columns in the dataset match the model's forward method signature: (args, kwargs, label_ids, label). The following columns have been ignored: [attention_mask, token_type_ids, id, context, inputs, input_ids, anchor, target]. Please check the dataset and model. You may need to set `remove_unused_columns=False` in `TrainingArguments`.

## === cell 10
print("Current directory:", os.getcwd())
print("Files:")
for f in os.listdir("."):
    if f.endswith(".csv"):
        print(" -", f)
