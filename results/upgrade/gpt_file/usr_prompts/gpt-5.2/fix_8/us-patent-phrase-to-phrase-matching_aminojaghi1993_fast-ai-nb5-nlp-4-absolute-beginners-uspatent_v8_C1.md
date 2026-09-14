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

0.8039326304000577

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
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault(
    "CUBLAS_WORKSPACE_CONFIG", ":4096:8"
)  # deterministic cublas where applicable

from pathlib import Path
import random
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset as TorchDataset

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

torch.use_deterministic_algorithms(False)

try:
    torch.set_num_threads(min(4, os.cpu_count() or 1))
except Exception:
    pass

print("torch:", torch.__version__)
import transformers as _tf

print("transformers:", _tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
path = Path("../input/us-patent-phrase-to-phrase-matching")
if not path.exists():
    alt = Path("/kaggle/input/us-patent-phrase-to-phrase-matching")
    if alt.exists():
        path = alt
if not path.exists():
    alt2 = Path("/kaggle/data/us-patent-phrase-to-phrase-matching")
    if alt2.exists():
        path = alt2

assert (path / "train.csv").exists(), f"train.csv not found under: {path}"
assert (path / "test.csv").exists(), f"test.csv not found under: {path}"
assert (
    path / "sample_submission.csv"
).exists(), f"sample_submission.csv not found under: {path}"

print("Using data path:", path)



## === cell 2
df = pd.read_csv(path / "train.csv")

df["context"] = df["context"].astype(str)
df["target"] = df["target"].astype(str)
df["anchor"] = df["anchor"].astype(str)
df["input"] = (
    "TEXT1: " + df["context"] + "; TEXT2: " + df["target"] + "; ANC1: " + df["anchor"]
)

print(df[["id", "input", "score"]].head())



## === cell 3
candidate_paths = [
    Path("../input/ms-deberta-v3-small/pytorch/small/1/ms-deberta-v3-small-local"),
    Path("/kaggle/input/ms-deberta-v3-small/pytorch/small/1/ms-deberta-v3-small-local"),
]
model_path = None
for p in candidate_paths:
    if p.exists():
        model_path = str(p)
        break

if model_path is None:
    model_path = "microsoft/deberta-v3-small"

tokz = AutoTokenizer.from_pretrained(
    model_path, local_files_only=(model_path != "microsoft/deberta-v3-small")
)

MAX_LEN = int(getattr(tokz, "model_max_length", 512))
if MAX_LEN > 512:
    MAX_LEN = 512
print("Model:", model_path)
print("MAX_LEN:", MAX_LEN)




## === cell 4
class RawTextRegressionDataset(TorchDataset):
    def __init__(self, texts, labels=None):
        self.texts = list(texts)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.float32)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {"text": self.texts[idx]}
        if self.labels is not None:
            item["labels"] = float(self.labels[idx])
        return item


rng = np.random.RandomState(42)
idx = np.arange(len(df))
rng.shuffle(idx)
n_valid = int(0.2 * len(df))
valid_idx = idx[:n_valid]
train_idx = idx[n_valid:]

train_texts = df.loc[train_idx, "input"].tolist()
train_labels = df.loc[train_idx, "score"].to_numpy()
valid_texts = df.loc[valid_idx, "input"].tolist()
valid_labels = df.loc[valid_idx, "score"].to_numpy()

train_ds = RawTextRegressionDataset(train_texts, train_labels)
valid_ds = RawTextRegressionDataset(valid_texts, valid_labels)

print("Train size:", len(train_ds), "Valid size:", len(valid_ds))



## === cell 5
eval_df = pd.read_csv(path / "test.csv")
eval_df["context"] = eval_df["context"].astype(str)
eval_df["target"] = eval_df["target"].astype(str)
eval_df["anchor"] = eval_df["anchor"].astype(str)
eval_df["input"] = (
    "TEXT1: "
    + eval_df["context"]
    + "; TEXT2: "
    + eval_df["target"]
    + "; ANC1: "
    + eval_df["anchor"]
)

eval_ds = RawTextRegressionDataset(eval_df["input"].tolist(), labels=None)
print("Test size:", len(eval_ds))




## === cell 6
def corr_d(eval_pred):
    preds, labels = eval_pred
    preds = np.asarray(preds, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    preds -= preds.mean()
    labels -= labels.mean()
    denom = np.sqrt((preds * preds).sum()) * np.sqrt((labels * labels).sum())
    pearson = float((preds * labels).sum() / denom) if denom != 0 else 0.0
    return {"pearson": pearson}




## === cell 7
bs = 128
epochs = 2
lr = 8e-5

_cpu = os.cpu_count() or 1
_num_workers = 2 if _cpu >= 4 else 1

args = TrainingArguments(
    output_dir="outputs",
    learning_rate=lr,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    fp16=True,
    eval_strategy="epoch",
    per_device_train_batch_size=bs,
    per_device_eval_batch_size=bs * 2,
    num_train_epochs=epochs,
    weight_decay=0.01,
    report_to="none",
    logging_strategy="steps",
    logging_steps=50,
    save_strategy="no",
    remove_unused_columns=False,
    dataloader_num_workers=_num_workers,
    dataloader_pin_memory=True,
    dataloader_prefetch_factor=2 if _num_workers > 0 else None,
    seed=42,
    data_seed=42,
)

print("TrainingArguments created OK. num_workers:", _num_workers)



## === cell 8
model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    num_labels=1,
    local_files_only=(model_path != "microsoft/deberta-v3-small"),
)

try:
    model.config.problem_type = "regression"
except Exception:
    pass

data_collator = DataCollatorWithPadding(tokenizer=tokz, pad_to_multiple_of=8)


def collate_and_tokenize(features):
    texts = [f["text"] for f in features]
    batch = tokz(
        texts,
        truncation=True,
        max_length=MAX_LEN,
        padding=True,  # actual padding handled here (still dynamic per batch)
        return_attention_mask=True,
    )
    if "labels" in features[0]:
        batch["labels"] = torch.tensor(
            [f["labels"] for f in features], dtype=torch.float32
        )
    return batch


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    tokenizer=tokz,
    data_collator=collate_and_tokenize,
    compute_metrics=corr_d,
)

print("Trainer created OK.")



## === cell 9
trainer.train()



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
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

/usr/local/lib/python3.11/dist-packages/transformers/models/deberta_v2/modeling_deberta_v2.py in forward(self, input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds, labels, output_attentions, output_hidden_states, return_dict)
   1077         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
   1078 
-> 1079         outputs = self.deberta(
   1080             input_ids,
   1081             token_type_ids=token_type_ids,

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

/usr/local/lib/python3.11/dist-packages/transformers/models/deberta_v2/modeling_deberta_v2.py in forward(self, input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds, output_attentions, output_hidden_states, return_dict)
    763         elif input_ids is not None:
    764             self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
--> 765             input_shape = input_ids.size()
    766         elif inputs_embeds is not None:
    767             input_shape = inputs_embeds.size()[:-1]

AttributeError: 'list' object has no attribute 'size'

## === cell 10
pred_out = trainer.predict(eval_ds)
preds = np.asarray(pred_out.predictions, dtype=np.float64).reshape(-1)
preds = np.clip(preds, 0.0, 1.0)

submission = pd.DataFrame({"id": eval_df["id"].values, "score": preds})
submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2128262932.py in <cell line: 0>()
----> 1 pred_out = trainer.predict(eval_ds)
      2 preds = np.asarray(pred_out.predictions, dtype=np.float64).reshape(-1)
      3 preds = np.clip(preds, 0.0, 1.0)
      4 
      5 submission = pd.DataFrame({"id": eval_df["id"].values, "score": preds})

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

/usr/local/lib/python3.11/dist-packages/transformers/models/deberta_v2/modeling_deberta_v2.py in forward(self, input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds, labels, output_attentions, output_hidden_states, return_dict)
   1077         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
   1078 
-> 1079         outputs = self.deberta(
   1080             input_ids,
   1081             token_type_ids=token_type_ids,

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

/usr/local/lib/python3.11/dist-packages/transformers/models/deberta_v2/modeling_deberta_v2.py in forward(self, input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds, output_attentions, output_hidden_states, return_dict)
    763         elif input_ids is not None:
    764             self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
--> 765             input_shape = input_ids.size()
    766         elif inputs_embeds is not None:
    767             input_shape = inputs_embeds.size()[:-1]

AttributeError: 'list' object has no attribute 'size'

## === cell 11
print("samples submission----------------")
sam_sub = pd.read_csv(path / "sample_submission.csv")
print(sam_sub.head(3))
print(sam_sub.dtypes)

print("our submission----------------")
our_sub = pd.read_csv("submission.csv")
print(our_sub.head(3))
print(our_sub.dtypes)

assert list(our_sub.columns) == ["id", "score"]
assert len(our_sub) == len(sam_sub)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1299182897.py in <cell line: 0>()
      5 
      6 print("our submission----------------")
----> 7 our_sub = pd.read_csv("submission.csv")
      8 print(our_sub.head(3))
      9 print(our_sub.dtypes)

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: 'submission.csv'

## === cell 12
import os as _os
import shutil as _shutil

print(_os.getcwd())
print("----")
print("Files in CWD:")
print("\n".join(sorted(_os.listdir("."))))

_shutil.rmtree("outputs", ignore_errors=True)
print("----")
print("Files in CWD after removing outputs (if existed):")
print("\n".join(sorted(_os.listdir("."))))
