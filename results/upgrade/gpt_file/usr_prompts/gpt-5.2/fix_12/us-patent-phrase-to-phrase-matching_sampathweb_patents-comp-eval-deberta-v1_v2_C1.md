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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

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
from datasets import disable_caching

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




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
def prepare_df(df, tokenizer):
    df = df.rename(columns={"score": "label"}).copy()
    sep = " " + (tokenizer.sep_token or "[SEP]") + " "

    ctx = df["context"].astype(str)
    anc = df["anchor"].astype(str).str.lower()
    tgt = df["target"].astype(str).str.lower()

    section = ctx.str.strip().str[0]
    sec_tok = "[" + section + "]"

    df["inputs"] = sec_tok + sep + ctx + sep + anc + sep + tgt
    return df


def tokenize_dataset_fast(df, tokenizer, with_labels: bool, num_proc: int):
    keep_cols = ["inputs"] + (["label"] if with_labels else [])
    hfd = HFDataset.from_pandas(df[keep_cols], preserve_index=False)

    def _tok(batch):
        enc = tokenizer(
            batch["inputs"],
            padding=False,  # dynamic padding via data collator
            truncation=True,
            max_length=128,
            return_attention_mask=True,
        )
        enc["length"] = [len(x) for x in enc["input_ids"]]
        if with_labels:
            enc["labels"] = np.asarray(batch["label"], dtype=np.float32)
        return enc

    hfd = hfd.map(
        _tok,
        batched=True,
        batch_size=2048,
        num_proc=max(1, num_proc),
        remove_columns=hfd.column_names,
        desc="Tokenizing",
    )
    cols = ["input_ids", "attention_mask", "length"] + (
        ["labels"] if with_labels else []
    )
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

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)



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

cpu_cnt = os.cpu_count() or 1

tok_proc = min(4, cpu_cnt)

train_ds = tokenize_dataset_fast(
    train_df, tokenizer, with_labels=True, num_proc=tok_proc
)
test_ds = tokenize_dataset_fast(
    test_df, tokenizer, with_labels=False, num_proc=tok_proc
)

dl_workers = min(4, cpu_cnt)

training_args = TrainingArguments(
    output_dir="out",
    overwrite_output_dir=True,
    do_train=True,
    do_eval=False,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=128,  # Speed: larger eval batch (no gradients) reduces overhead; predictions identical.
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
    length_column_name="length",
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
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/583056800.py in <cell line: 0>()
     61 )
     62 
---> 63 trainer.train()
     64 
     65 # Speed: predict uses eval batch size; keep compute pipeline minimal.

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

/tmp/ipykernel_11/1279606859.py in compute_loss(self, model, inputs, return_outputs, **kwargs)
     39             inputs = dict(inputs)
     40             inputs.pop("num_items_in_batch", None)
---> 41         return super().compute_loss(
     42             model, inputs, return_outputs=return_outputs, **kwargs
     43         )

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
