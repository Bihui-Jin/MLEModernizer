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

0.7819492614928613

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved nan) has done: 'The timeout is dominated by training DeBERTa for 4 epochs with max_length=256 on ~24k samples, and by the overhead of the HuggingFace `Trainer` evaluation/prediction loops. To finish under 600s without changing the model or training semantics, I keep the same architecture/epochs/loss/features, but reduce runtime overhead by (1) using length-based dynamic padding (via `pad_to_multiple_of=8`) instead of always padding to 256, (2) enabling compilation on PyTorch 2.x when available, (3) using a more efficient Pearson computation (no `np.corrcoef`), and (4) removing redundant Python overhead in collation by using a padding collator on already-tokenized inputs. These are provably equivalent in meaning (same tokens, same truncation, same batches/epochs) and only affect speed/overhead, with at most negligible floating-point differences.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import random

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ["TOKENIZERS_PARALLELISM"] = "true"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


set_seed(42)



## === cell 1
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from transformers.trainer_utils import EvalPrediction
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

from transformers import __version__ as transformers_version

print("transformers:", transformers_version)

import torch

print("torch:", torch.__version__, "cuda_available:", torch.cuda.is_available())

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass
torch.backends.cudnn.benchmark = False

try:
    torch.set_num_threads(min(4, os.cpu_count() or 2))
except Exception:
    pass

try:
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
except Exception:
    pass




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
def corr(x, y):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = x.size
    if n == 0 or y.size == 0:
        return 0.0
    xm = x.mean()
    ym = y.mean()
    xv = x - xm
    yv = y - ym
    denom = np.sqrt(np.dot(xv, xv) * np.dot(yv, yv))
    if denom == 0.0:
        return 0.0
    c = float(np.dot(xv, yv) / denom)
    if np.isnan(c):
        return 0.0
    return c




## === cell 3
def corr_d(eval_pred: EvalPrediction):
    preds = eval_pred.predictions
    labels = eval_pred.label_ids
    preds = np.asarray(preds).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    return {"pearson": corr(preds, labels)}




## === cell 4
path = "/kaggle/input/us-patent-phrase-to-phrase-matching/"
train_data = pd.read_csv(path + "train.csv")
test_data = pd.read_csv(path + "test.csv")

print(train_data.head())
print(test_data.head())
print("train shape:", train_data.shape, "test shape:", test_data.shape)



## === cell 5
train_data["section"] = train_data.context.str[0]
test_data["section"] = test_data.context.str[0]
print(train_data.section.value_counts().head())



## === cell 6
model_nm = "microsoft/deberta-v3-small"


def load_tokenizer_and_model(model_id: str):
    candidate_paths = [model_id]
    for base in ["/kaggle/input", "/kaggle/working"]:
        cand = os.path.join(base, model_id.replace("/", "_"))
        candidate_paths.append(cand)

    last_err = None
    for p in candidate_paths:
        try:
            tok = AutoTokenizer.from_pretrained(p, use_fast=True)
            mdl = AutoModelForSequenceClassification.from_pretrained(p, num_labels=1)
            return tok, mdl
        except Exception as e:
            last_err = e
    raise RuntimeError(
        f"Could not load tokenizer/model for '{model_id}'. Last error: {last_err}"
    )


tokenizer, model = load_tokenizer_and_model(model_nm)

sep = tokenizer.sep_token if tokenizer.sep_token is not None else "[SEP]"
print("sep token:", sep)

try:
    if hasattr(model, "config") and model.config is not None:
        model.config.output_hidden_states = False
        model.config.output_attentions = False
        model.config.return_dict = True
except Exception:
    pass

try:
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "config") and model.config is not None:
            model.config.use_cache = False
except Exception:
    pass

ENABLE_TORCH_COMPILE = False
if ENABLE_TORCH_COMPILE:
    try:
        if hasattr(torch, "compile"):
            model = torch.compile(model)
            print("torch.compile enabled")
    except Exception as e:
        print("torch.compile not enabled:", repr(e))




## === cell 7
def prepare_data(df: pd.DataFrame):
    df["_ctx"] = df["context"].astype(str).to_numpy()
    df["_anc"] = df["anchor"].astype(str).to_numpy()
    df["_tgt"] = df["target"].astype(str).to_numpy()


prepare_data(train_data)
prepare_data(test_data)

print(train_data[["anchor", "target", "context", "score"]].head())



## === cell 8
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


def tokenize_texts(ctx, anc, tgt, tokenizer, max_length: int):
    first = [c + sep + a for c, a in zip(ctx, anc)]
    enc = tokenizer(
        first,
        list(tgt),
        truncation=True,
        max_length=max_length,
        padding=False,  # dynamic padding via collator
        return_attention_mask=True,
    )
    return enc["input_ids"], enc["attention_mask"]


class TextRegressionTokenizedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        input_ids,
        attention_mask,
        labels=None,
        with_labels: bool = True,
        lengths=None,
    ):
        self.with_labels = with_labels

        self.input_ids = input_ids
        self.attention_mask = attention_mask

        self.labels = None
        if with_labels:
            self.labels = torch.as_tensor(np.asarray(labels, dtype=np.float32))

        self.lengths = None
        if lengths is not None:
            self.lengths = torch.as_tensor(np.asarray(lengths, dtype=np.int32))

    def __len__(self):
        return int(len(self.input_ids))

    def __getitem__(self, idx: int):
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }
        if self.with_labels:
            item["labels"] = self.labels[idx]
        if self.lengths is not None:
            item["length"] = int(self.lengths[idx])
        return item


@dataclass
class FastRegressionCollator:
    tokenizer: Any
    pad_to_multiple_of: Optional[int] = 8

    def __post_init__(self):
        self.pad_id = int(self.tokenizer.pad_token_id or 0)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        has_labels = "labels" in features[0]

        if has_labels:
            labels = (
                torch.stack([f["labels"] for f in features]).view(-1).to(torch.float32)
            )
        else:
            labels = None

        input_ids_list = [f["input_ids"] for f in features]
        attn_list = [f["attention_mask"] for f in features]

        max_len = 0
        for ids in input_ids_list:
            l = len(ids)
            if l > max_len:
                max_len = l

        if (
            self.pad_to_multiple_of is not None
            and max_len % self.pad_to_multiple_of != 0
        ):
            max_len = (
                (max_len // self.pad_to_multiple_of) + 1
            ) * self.pad_to_multiple_of

        bs = len(features)
        input_ids = torch.full((bs, max_len), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((bs, max_len), dtype=torch.long)

        for i, (ids, am) in enumerate(zip(input_ids_list, attn_list)):
            l = len(ids)
            input_ids[i, :l] = torch.as_tensor(ids, dtype=torch.long)
            attention_mask[i, :l] = torch.as_tensor(am, dtype=torch.long)

        batch = {"input_ids": input_ids, "attention_mask": attention_mask}
        if labels is not None:
            batch["labels"] = labels
        return batch


rng = np.random.RandomState(42)
idx = np.arange(len(train_data))
rng.shuffle(idx)
cut = int(len(idx) * (1 - 0.25))
train_idx, valid_idx = idx[:cut], idx[cut:]

train_df = train_data.iloc[train_idx].copy()
valid_df = train_data.iloc[valid_idx].copy()

max_length = 256


def _cache_path(split: str) -> str:
    safe_model = model_nm.replace("/", "_")
    return os.path.join(
        "/kaggle/working", f"tokcache_{safe_model}_ml{max_length}_{split}.npz"
    )


def _load_or_tokenize(split: str, ctx, anc, tgt) -> Tuple[list, list]:
    cpath = _cache_path(split)
    if os.path.exists(cpath):
        data = np.load(cpath, allow_pickle=True)
        return data["input_ids"].tolist(), data["attention_mask"].tolist()
    input_ids, attn = tokenize_texts(ctx, anc, tgt, tokenizer, max_length)
    np.savez_compressed(
        cpath,
        input_ids=np.array(input_ids, dtype=object),
        attention_mask=np.array(attn, dtype=object),
    )
    return input_ids, attn


train_input_ids, train_attn = _load_or_tokenize(
    "train",
    train_df["_ctx"].values,
    train_df["_anc"].values,
    train_df["_tgt"].values,
)
valid_input_ids, valid_attn = _load_or_tokenize(
    "valid",
    valid_df["_ctx"].values,
    valid_df["_anc"].values,
    valid_df["_tgt"].values,
)
test_input_ids, test_attn = _load_or_tokenize(
    "test",
    test_data["_ctx"].values,
    test_data["_anc"].values,
    test_data["_tgt"].values,
)

train_lengths = np.fromiter(
    (len(x) for x in train_input_ids), count=len(train_input_ids), dtype=np.int32
)
valid_lengths = np.fromiter(
    (len(x) for x in valid_input_ids), count=len(valid_input_ids), dtype=np.int32
)
test_lengths = np.fromiter(
    (len(x) for x in test_input_ids), count=len(test_input_ids), dtype=np.int32
)

train_labels = train_df["score"].to_numpy(dtype=np.float32, copy=True)
valid_labels = valid_df["score"].to_numpy(dtype=np.float32, copy=True)

train_ds = TextRegressionTokenizedDataset(
    train_input_ids,
    train_attn,
    labels=train_labels,
    with_labels=True,
    lengths=train_lengths,
)
valid_ds = TextRegressionTokenizedDataset(
    valid_input_ids,
    valid_attn,
    labels=valid_labels,
    with_labels=True,
    lengths=valid_lengths,
)
eval_ds = TextRegressionTokenizedDataset(
    test_input_ids, test_attn, labels=None, with_labels=False, lengths=test_lengths
)

data_collator = FastRegressionCollator(tokenizer=tokenizer, pad_to_multiple_of=8)

print("train/valid sizes:", len(train_ds), len(valid_ds), "test:", len(eval_ds))




## === cell 9
class CompatTrainer(Trainer):
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        inputs.pop("num_items_in_batch", None)
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        return (loss, outputs) if return_outputs else loss

    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ):
        if self.compute_metrics is None:
            return super().evaluation_loop(
                dataloader,
                description,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        model.eval()

        n = len(dataloader.dataset)
        preds_out = np.empty((n,), dtype=np.float32)
        labels_out = np.empty((n,), dtype=np.float32)
        have_labels = None

        losses_sum = 0.0
        losses_n = 0

        offset = 0
        with torch.inference_mode():
            for inputs in dataloader:
                inputs = self._prepare_inputs(inputs)
                outputs = model(**inputs)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
                logits = (
                    outputs["logits"] if isinstance(outputs, dict) else outputs.logits
                )

                bs = logits.shape[0]
                preds_out[offset : offset + bs] = (
                    logits.detach().float().view(-1).cpu().numpy()
                )

                if have_labels is None:
                    have_labels = "labels" in inputs
                if have_labels:
                    labels_out[offset : offset + bs] = (
                        inputs["labels"].detach().float().view(-1).cpu().numpy()
                    )

                if loss is not None:
                    losses_sum += float(loss.detach().float().cpu().item())
                    losses_n += 1

                offset += bs

        preds = preds_out[:offset]
        labels = labels_out[:offset] if have_labels else np.array([], dtype=np.float32)

        metrics = (
            self.compute_metrics(EvalPrediction(predictions=preds, label_ids=labels))
            if labels.size
            else {}
        )

        from transformers.trainer_utils import EvalLoopOutput

        metrics = {f"{metric_key_prefix}_{k}": v for k, v in metrics.items()}
        if losses_n:
            metrics[f"{metric_key_prefix}_loss"] = float(losses_sum / losses_n)

        return EvalLoopOutput(
            predictions=preds,
            label_ids=labels,
            metrics=metrics,
            num_samples=len(dataloader.dataset),
        )


bs = 32
epochs = 4
lr = 8e-5

use_fp16 = torch.cuda.is_available()

if torch.cuda.is_available():
    num_workers = 0
    persistent = False
    prefetch = None
else:
    num_workers = 0
    persistent = False
    prefetch = None

args = TrainingArguments(
    output_dir="outputs",
    learning_rate=lr,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    fp16=use_fp16,
    eval_strategy="epoch",
    per_device_train_batch_size=bs,
    per_device_eval_batch_size=bs * 2,
    num_train_epochs=epochs,
    weight_decay=0.01,
    report_to="none",
    save_strategy="no",
    logging_steps=50,
    remove_unused_columns=False,
    dataloader_num_workers=num_workers,
    dataloader_pin_memory=torch.cuda.is_available(),
    dataloader_persistent_workers=persistent,
    dataloader_prefetch_factor=prefetch,
    group_by_length=True,
    length_column_name="length",
    optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
    eval_accumulation_steps=32,
    prediction_loss_only=False,
)

trainer = CompatTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=corr_d,
)



## === cell 10
trainer.train()



## --- ERROR in cell 10, traceback:
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
   2546                     )
   2547                     with context():
-> 2548                         tr_loss_step = self.training_step(model, inputs, num_items_in_batch)
   2549 
   2550                     if (

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in training_step(***failed resolving arguments***)
   3795                 kwargs["scale_wrt_gas"] = False
   3796 
-> 3797             self.accelerator.backward(loss, **kwargs)
   3798 
   3799             return loss.detach()

/usr/local/lib/python3.11/dist-packages/accelerate/accelerator.py in backward(self, loss, **kwargs)
   2576             self.lomo_backward(loss, learning_rate)
   2577         else:
-> 2578             loss.backward(**kwargs)
   2579 
   2580     def set_trigger(self):

/usr/local/lib/python3.11/dist-packages/torch/_tensor.py in backward(self, gradient, retain_graph, create_graph, inputs)
    624                 inputs=inputs,
    625             )
--> 626         torch.autograd.backward(
    627             self, gradient, retain_graph, create_graph, inputs=inputs
    628         )

/usr/local/lib/python3.11/dist-packages/torch/autograd/__init__.py in backward(tensors, grad_tensors, retain_graph, create_graph, grad_variables, inputs)
    345     # some Python versions print out the first line of a multi-line function
    346     # calls in the traceback and some print out the last line
--> 347     _engine_run_backward(
    348         tensors,
    349         grad_tensors_,

/usr/local/lib/python3.11/dist-packages/torch/autograd/graph.py in _engine_run_backward(t_outputs, *args, **kwargs)
    821         unregister_hooks = _register_logging_hooks_on_whole_graph(t_outputs)
    822     try:
--> 823         return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
    824             t_outputs, *args, **kwargs
    825         )  # Calls into the C++ engine to run the backward pass

/usr/local/lib/python3.11/dist-packages/torch/autograd/function.py in apply(self, *args)
    305             )
    306         user_fn = vjp_fn if vjp_fn is not Function.vjp else backward_fn
--> 307         return user_fn(self, *args)
    308 
    309     def apply_jvp(self, *args):

/usr/local/lib/python3.11/dist-packages/torch/utils/checkpoint.py in backward(ctx, *args)
    319                 " this checkpoint() is not necessary"
    320             )
--> 321         torch.autograd.backward(outputs_with_grad, args_with_grad)
    322         grads = tuple(
    323             inp.grad if isinstance(inp, torch.Tensor) else None

/usr/local/lib/python3.11/dist-packages/torch/autograd/__init__.py in backward(tensors, grad_tensors, retain_graph, create_graph, grad_variables, inputs)
    345     # some Python versions print out the first line of a multi-line function
    346     # calls in the traceback and some print out the last line
--> 347     _engine_run_backward(
    348         tensors,
    349         grad_tensors_,

/usr/local/lib/python3.11/dist-packages/torch/autograd/graph.py in _engine_run_backward(t_outputs, *args, **kwargs)
    821         unregister_hooks = _register_logging_hooks_on_whole_graph(t_outputs)
    822     try:
--> 823         return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
    824             t_outputs, *args, **kwargs
    825         )  # Calls into the C++ engine to run the backward pass

RuntimeError: Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed). Saved intermediate values of the graph are freed when you call .backward() or autograd.grad(). Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved tensors after calling backward.

## === cell 11
with torch.inference_mode():
    pred_out = trainer.predict(eval_ds)
preds = np.asarray(pred_out.predictions, dtype=float).reshape(-1)
preds = np.clip(preds, 0.0, 1.0)

score = np.select(
    [
        preds >= 0.875,
        preds >= 0.625,
        preds >= 0.375,
        preds >= 0.125,
    ],
    [1.0, 0.75, 0.5, 0.25],
    default=0.0,
).astype(float)

print("preds range:", float(np.min(preds)), float(np.max(preds)))
print("unique snapped scores:", sorted(set(score.tolist())))



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/4097543787.py in <cell line: 0>()
      1 with torch.inference_mode():
----> 2     pred_out = trainer.predict(eval_ds)
      3 preds = np.asarray(pred_out.predictions, dtype=float).reshape(-1)
      4 preds = np.clip(preds, 0.0, 1.0)
      5 

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in predict(self, test_dataset, ignore_keys, metric_key_prefix)
   4275 
   4276         eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
-> 4277         output = eval_loop(
   4278             test_dataloader, description="Prediction", ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
   4279         )

/tmp/ipykernel_11/1064354451.py in evaluation_loop(self, dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)
     41                 inputs = self._prepare_inputs(inputs)
     42                 outputs = model(**inputs)
---> 43                 loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
     44                 logits = (
     45                     outputs["logits"] if isinstance(outputs, dict) else outputs.logits

/usr/local/lib/python3.11/dist-packages/transformers/utils/generic.py in __getitem__(self, k)
    447         if isinstance(k, str):
    448             inner_dict = dict(self.items())
--> 449             return inner_dict[k]
    450         else:
    451             return self.to_tuple()[k]

KeyError: 'loss'

## === cell 12
submission = pd.DataFrame({"id": test_data["id"].values, "score": score})
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Saved to:", os.path.abspath("submission.csv"))

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/87345754.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({"id": test_data["id"].values, "score": score})
      2 submission.to_csv("submission.csv", index=False)
      3 print(submission.head())
      4 print("Wrote submission.csv with shape:", submission.shape)
      5 print("Saved to:", os.path.abspath("submission.csv"))

NameError: name 'score' is not defined
