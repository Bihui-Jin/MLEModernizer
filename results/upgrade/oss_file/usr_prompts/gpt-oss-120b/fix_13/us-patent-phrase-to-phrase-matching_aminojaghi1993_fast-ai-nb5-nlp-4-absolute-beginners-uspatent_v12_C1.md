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

try:
    from google.protobuf import message_factory

    if not hasattr(message_factory.MessageFactory, "GetPrototype"):
        def _get_prototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        message_factory.MessageFactory.GetPrototype = _get_prototype
except Exception:
    pass

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
    ds = Dataset.from_pandas(df, preserve_index=False).rename_column("score", "labels")
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
        remove_unused_columns=False,
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
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2594271584.py in <cell line: 0>()
      6 
      7 trainer = get_trainer(dds, model=model)
----> 8 trainer.train()
      9 

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

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/eval_frame.py in _fn(*args, **kwargs)
    572 
    573             try:
--> 574                 return fn(*args, **kwargs)
    575             finally:
    576                 # Restore the dynamic layer stack depth if necessary.

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

TypeError: DebertaV2ForSequenceClassification.forward() got an unexpected keyword argument 'num_items_in_batch'

## === cell 9
eval_df["inputs"] = eval_df.context + sep + eval_df.anchor + sep + eval_df.target
eval_ds = Dataset.from_pandas(eval_df, preserve_index=False).map(
    tok_func,
    batched=True,
    batch_size=2000,
    num_proc=2,  # keep low to avoid overhead
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

preds = trainer.predict(eval_ds).predictions
preds = np.clip(preds.astype(float).flatten(), 0, 1)

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
/tmp/ipykernel_11/1520881223.py in <cell line: 0>()
      1 eval_df["inputs"] = eval_df.context + sep + eval_df.anchor + sep + eval_df.target
----> 2 eval_ds = Dataset.from_pandas(eval_df, preserve_index=False).map(
      3     tok_func,
      4     batched=True,
      5     batch_size=2000,

/usr/local/lib/python3.11/dist-packages/datasets/arrow_dataset.py in wrapper(*args, **kwargs)
    560         }
    561         # apply actual function
--> 562         out: Union["Dataset", "DatasetDict"] = func(self, *args, **kwargs)
    563         datasets: list["Dataset"] = list(out.values()) if isinstance(out, dict) else [out]
    564         # re-apply format to the output

/usr/local/lib/python3.11/dist-packages/datasets/arrow_dataset.py in map(self, function, with_indices, with_rank, input_columns, batched, batch_size, drop_last_batch, remove_columns, keep_in_memory, load_from_cache_file, cache_file_name, writer_batch_size, features, disable_nullable, fn_kwargs, num_proc, suffix_template, new_fingerprint, desc, try_original_type)
   3107             missing_columns = set(remove_columns) - set(self._data.column_names)
   3108             if missing_columns:
-> 3109                 raise ValueError(
   3110                     f"Column to remove {list(missing_columns)} not in the dataset. Current columns in the dataset: {self._data.column_names}"
   3111                 )

ValueError: Column to remove ['section', 'sectok'] not in the dataset. Current columns in the dataset: ['id', 'anchor', 'target', 'context', 'inputs']

## === cell 10
print("Current directory:", os.getcwd())
print("Files:")
for f in os.listdir("."):
    if f.endswith(".csv"):
        print(" -", f)
