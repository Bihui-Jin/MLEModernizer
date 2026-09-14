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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
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

0.8021729294158836

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

import random
import numpy as np
import pandas as pd

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
MODEL_ROOT = "/kaggle/input/f-tfm-small"
MAX_LENGTH = 3072
BATCH_SIZE = 2
N_FOLDS = 5

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)




## === cell 2
def _resolve_competition_dir() -> str:
    candidates = [
        "/kaggle/input/learning-agency-lab-automated-essay-scoring-2",
        "/kaggle/data/learning-agency-lab-automated-essay-scoring-2",
        "/kaggle/input",
        "/kaggle/data",
    ]
    for base in candidates:
        if (
            os.path.isdir(base)
            and os.path.exists(os.path.join(base, "train.csv"))
            and os.path.exists(os.path.join(base, "test.csv"))
        ):
            return base

    for base in ["/kaggle/input", "/kaggle/data"]:
        if os.path.isdir(base):
            for name in os.listdir(base):
                p = os.path.join(base, name)
                if (
                    os.path.isdir(p)
                    and os.path.exists(os.path.join(p, "train.csv"))
                    and os.path.exists(os.path.join(p, "test.csv"))
                    and os.path.exists(os.path.join(p, "sample_submission.csv"))
                ):
                    return p

    raise FileNotFoundError(
        "Could not locate competition dataset directory with train.csv and test.csv under /kaggle/input or /kaggle/data."
    )


COMP_DIR = _resolve_competition_dir()

df_train = pd.read_csv(os.path.join(COMP_DIR, "train.csv"))
df_test = pd.read_csv(os.path.join(COMP_DIR, "test.csv"))
sample_submission = pd.read_csv(os.path.join(COMP_DIR, "sample_submission.csv"))

assert {"essay_id", "full_text"}.issubset(df_test.columns)
assert {"essay_id", "full_text", "score"}.issubset(df_train.columns)

print("COMP_DIR:", COMP_DIR)
print(
    "train:", df_train.shape, "test:", df_test.shape, "sample:", sample_submission.shape
)




## === cell 3
def _resolve_model_root(model_root: str) -> str:
    if os.path.isdir(model_root):
        return model_root

    base = "/kaggle/input"
    if os.path.isdir(base):
        for name in os.listdir(base):
            p = os.path.join(base, name)
            if os.path.isdir(p) and os.path.basename(p) == os.path.basename(model_root):
                return p
    return model_root  # keep original; downstream will handle fallback


MODEL_ROOT = _resolve_model_root(MODEL_ROOT)


def _find_fold_dir(fold: int) -> str:
    """
    Prefer the originally intended folder pattern.
    If not found, fall back to scanning MODEL_ROOT for any directory containing the fold id.
    """
    direct = os.path.join(MODEL_ROOT, f"f-tfm-small_AES2_fold_{fold}")
    if os.path.isdir(direct):
        return direct

    candidates = []
    if os.path.isdir(MODEL_ROOT):
        for name in os.listdir(MODEL_ROOT):
            p = os.path.join(MODEL_ROOT, name)
            if os.path.isdir(p) and (f"fold_{fold}" in name or f"fold{fold}" in name):
                candidates.append(p)
    candidates = sorted(candidates)
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        f"Could not locate model directory for fold={fold}. "
        f"Tried: {direct} and scanning under {MODEL_ROOT}. "
        f"MODEL_ROOT exists={os.path.isdir(MODEL_ROOT)}"
    )


FALLBACK_MODEL = "distilbert-base-uncased"

use_fallback = False
tokenizer_dir = None
try:
    tokenizer_dir = _find_fold_dir(0)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir, local_files_only=True, use_fast=True
    )
except FileNotFoundError as e:
    print("WARNING:", str(e))
    print(f"Falling back to tokenizer/model: {FALLBACK_MODEL}")
    use_fallback = True
    tokenizer = AutoTokenizer.from_pretrained(
        FALLBACK_MODEL, local_files_only=False, use_fast=True
    )


def tokenize(batch):
    return tokenizer(batch["full_text"], max_length=MAX_LENGTH, truncation=True)


print("MODEL_ROOT:", MODEL_ROOT)
print("tokenizer_dir:", tokenizer_dir)
print("use_fallback:", use_fallback)



## === cell 4
test_ds = Dataset.from_pandas(df_test[["essay_id", "full_text"]], preserve_index=False)
test_ds = test_ds.map(tokenize, batched=True)
test_ds = test_ds.remove_columns(["essay_id", "full_text"])

args = TrainingArguments(
    output_dir="./_tmp_pred",
    per_device_eval_batch_size=BATCH_SIZE,
    report_to="none",
    dataloader_num_workers=0,
)

predictions = []
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

if use_fallback:
    model = AutoModelForSequenceClassification.from_pretrained(
        FALLBACK_MODEL, local_files_only=False
    )
    trainer = Trainer(
        model=model, args=args, data_collator=data_collator, tokenizer=tokenizer
    )
    fold_logits = trainer.predict(test_ds).predictions
    predictions.append(fold_logits)
    print(f"Fallback model logits shape={fold_logits.shape} from {FALLBACK_MODEL}")
else:
    for fold in range(N_FOLDS):
        model_dir = _find_fold_dir(fold)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, local_files_only=True
        )

        trainer = Trainer(
            model=model,
            args=args,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )

        fold_logits = trainer.predict(test_ds).predictions
        predictions.append(fold_logits)
        print(f"Fold {fold}: logits shape={fold_logits.shape} from {model_dir}")

preds = np.mean(np.stack(predictions, axis=0), axis=0)
print("Ensembled preds shape:", preds.shape)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/1299255750.py in <cell line: 0>()
     22         model=model, args=args, data_collator=data_collator, tokenizer=tokenizer
     23     )
---> 24     fold_logits = trainer.predict(test_ds).predictions
     25     predictions.append(fold_logits)
     26     print(f"Fallback model logits shape={fold_logits.shape} from {FALLBACK_MODEL}")

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

/usr/local/lib/python3.11/dist-packages/transformers/models/distilbert/modeling_distilbert.py in forward(self, input_ids, attention_mask, head_mask, inputs_embeds, labels, output_attentions, output_hidden_states, return_dict)
    915         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    916 
--> 917         distilbert_output = self.distilbert(
    918             input_ids=input_ids,
    919             attention_mask=attention_mask,

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

/usr/local/lib/python3.11/dist-packages/transformers/models/distilbert/modeling_distilbert.py in forward(self, input_ids, attention_mask, head_mask, inputs_embeds, output_attentions, output_hidden_states, return_dict)
    721         head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
    722 
--> 723         embeddings = self.embeddings(input_ids, inputs_embeds)  # (bs, seq_length, dim)
    724 
    725         if self._use_flash_attention_2:

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

/usr/local/lib/python3.11/dist-packages/transformers/models/distilbert/modeling_distilbert.py in forward(self, input_ids, input_embeds)
    124         position_embeddings = self.position_embeddings(position_ids)  # (bs, max_seq_length, dim)
    125 
--> 126         embeddings = input_embeds + position_embeddings  # (bs, max_seq_length, dim)
    127         embeddings = self.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
    128         embeddings = self.dropout(embeddings)  # (bs, max_seq_length, dim)

RuntimeError: The size of tensor a (550) must match the size of tensor b (512) at non-singleton dimension 1

## === cell 5
if preds.ndim == 2 and preds.shape[1] == 1:
    raw = preds[:, 0]
    scores = (np.clip(raw, 0, 5).round(0) + 1).astype(np.int32)
else:
    cls = np.argmax(preds, axis=1)
    scores = (cls + 1).astype(np.int32)
    scores = np.clip(scores, 1, 6).astype(np.int32)

sub = pd.DataFrame({"essay_id": df_test["essay_id"].values, "score": scores})
sub = sample_submission[["essay_id"]].merge(sub, on="essay_id", how="left")
assert (
    sub["score"].isna().sum() == 0
), "Missing predictions for some essay_id after merge."
sub["score"] = sub["score"].astype(np.int32)

sub.to_csv("submission.csv", index=False)
print(sub.head())
print(
    f"Wrote submission.csv with shape={sub.shape} to {os.path.abspath('submission.csv')}"
)
print("Score distribution:\n", sub["score"].value_counts().sort_index())

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2991847865.py in <cell line: 0>()
      1 # Convert model outputs to discrete scores 1..6.
      2 # Handle both regression (shape [N,1]) and classification logits (shape [N,C]).
----> 3 if preds.ndim == 2 and preds.shape[1] == 1:
      4     raw = preds[:, 0]
      5     scores = (np.clip(raw, 0, 5).round(0) + 1).astype(np.int32)

NameError: name 'preds' is not defined
