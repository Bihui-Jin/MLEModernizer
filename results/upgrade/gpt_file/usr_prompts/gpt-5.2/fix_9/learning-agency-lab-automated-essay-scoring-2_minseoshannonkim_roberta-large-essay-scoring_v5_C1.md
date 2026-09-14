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

No external packages required in the script and installed.

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

0.8077619023103844

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the `datasets` dependency that’s crashing (protobuf/MessageFactory) and replace it with a minimal pandas-based pipeline that preserves the same core idea: a RoBERTa sequence classifier over essay text with labels shifted to 0–5. I also fix the invalid `from_pretrained('/kaggle/input/roberta-large')` path usage by loading a real pretrained checkpoint (`roberta-large`) directly (Kaggle has it cached), ensuring `model`/`tokenizer` are defined in submission mode. Finally, I replace the `pipeline()` inference (which is slow and returns string labels) with a batched `model(**tokenized)` forward pass and argmax to produce integer scores 1–6, then write `submission.csv` with the required columns.'
- What this solution (achieved 0.0) has done: 'I fix the crash coming from importing `Trainer/TrainingArguments` (it triggers a protobuf `MessageFactory` error in this Kaggle Python 3.12 environment) by removing those imports and leaving the rest of the RoBERTa inference pipeline intact. Since `is_submission=True`, none of the training code paths are needed; this keeps the core logic (RoBERTa large + argmax over 6 labels, score = pred+1) unchanged while making the notebook run end-to-end. I also ensure the data path is robust (fallback to the non-nested `/kaggle/input/...` files if needed) and that a valid `submission.csv` is always written with the required columns. This should move the score from 0.0 (crash/invalid) to a non-zero valid score without changing modeling semantics.'
- What this solution (achieved 0.0) has done: 'I fix the protobuf/transformers import crash that prevents the notebook from running by pinning protobuf to the compatible pure-Python implementation before importing `transformers`, which resolves the `MessageFactory.GetPrototype` error in many Kaggle Py3.12 images. I also make the data-path selection more robust (still using the same files) and ensure we always write a correctly-formatted `submission.csv` aligned to `test_df` order. These changes are execution/stability fixes and should move the score from 0.0 (failed/invalid run) to a valid non-zero score without changing the model’s core inference logic (RoBERTa-large argmax over 6 labels, mapped to 1–6).'
- What this solution (achieved 0.0) has done: 'I fix the runtime crash in the `transformers` import caused by an incompatible protobuf version in this Kaggle Python 3.12 image by forcing a compatible protobuf configuration and (if needed) safely downgrading protobuf at runtime before importing `transformers`. I also add a small fallback to load the model/tokenizer from a local Kaggle input directory if online/cached resolution fails, without changing the core inference logic (RoBERTa-large, argmax over 6 labels, map to 1–6). Finally, I ensure the submission is always written as a valid `submission.csv` with the correct columns and row alignment to `test_df`.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is almost certainly coming from using a base RoBERTa checkpoint with a randomly-initialized 6-class classification head, which produces essentially random labels. To move the score toward your target with minimal change and identical core logic (RoBERTa sequence classifier + argmax + map to 1–6), I keep your inference pipeline but load a checkpoint that is actually fine-tuned for this competition if it exists locally under `/kaggle/input` (common in Kaggle notebooks), otherwise fall back to your current behavior. I also force `local_files_only=True` first to avoid failing silently due to internet restrictions and to prefer Kaggle-mounted models, then allow the default resolution as a fallback. This should turn predictions from random to meaningful (non-zero QWK) while keeping the same architecture/inference semantics and still writing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd



## === cell 1
data_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2"
train_file_path = f"{data_path}/train.csv"
test_file_path = f"{data_path}/test.csv"

if not os.path.exists(train_file_path):
    train_file_path = "/kaggle/input/train.csv"
    test_file_path = "/kaggle/input/test.csv"

train_df = pd.read_csv(train_file_path, usecols=["essay_id", "full_text", "score"])
test_df = pd.read_csv(test_file_path, usecols=["essay_id", "full_text"])

train_df = train_df.rename(columns={"full_text": "text", "score": "labels"})
test_df = test_df.rename(columns={"full_text": "text"})

train_df["labels"] = train_df["labels"].astype(int) - 1  # 0..5

print(train_df[["essay_id", "text", "labels"]].head())
print(test_df[["essay_id", "text"]].head())
print("train shape:", train_df.shape, "test shape:", test_df.shape)



## === cell 2
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
class EssayDataset(Dataset):
    def __init__(
        self,
        df,
        tokenizer=None,
        with_labels=True,
        max_length=512,
        encodings=None,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.with_labels = with_labels
        self.max_length = max_length
        self.encodings = encodings  # if provided, must align with df rows

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.encodings is not None:
            item = {k: v[idx] for k, v in self.encodings.items()}
        else:
            row = self.df.iloc[idx]
            item = self.tokenizer(
                row["text"],
                truncation=True,
                max_length=self.max_length,
            )
        if self.with_labels:
            item["labels"] = int(self.df.iloc[idx]["labels"])
        return item




## === cell 4
id2label = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}
label2id = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}

is_submission = True

model_name = "roberta-large"



## === cell 5
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True




## === cell 6
def _try_load_tokenizer_and_model(primary_name: str):
    last_err = None

    candidates = [
        primary_name,
        "/kaggle/input/roberta-large",
        "/kaggle/input/roberta-large-squad2",
        "/kaggle/input/roberta-base",
    ]

    for local_only in (True, False):
        for cand in candidates:
            try:
                tok = AutoTokenizer.from_pretrained(
                    cand, local_files_only=local_only, use_fast=True
                )
                mdl = AutoModelForSequenceClassification.from_pretrained(
                    cand,
                    num_labels=6,
                    id2label=id2label,
                    label2id=label2id,
                    local_files_only=local_only,
                )
                print(
                    f"Loaded model/tokenizer from: {cand} (local_files_only={local_only})"
                )
                return tok, mdl
            except Exception as e:
                last_err = e
                continue

    raise RuntimeError(f"Failed to load model/tokenizer. Last error: {last_err}")


tokenizer, model = _try_load_tokenizer_and_model(model_name)
model.to(device)




## === cell 7
def quadratic_weighted_kappa(y_true, y_pred, min_rating=0, max_rating=5):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    n_ratings = max_rating - min_rating + 1
    O = np.zeros((n_ratings, n_ratings), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        O[a - min_rating, b - min_rating] += 1.0

    act_hist = np.bincount(y_true - min_rating, minlength=n_ratings).astype(np.float64)
    pred_hist = np.bincount(y_pred - min_rating, minlength=n_ratings).astype(np.float64)

    E = np.outer(act_hist, pred_hist)
    E = E / E.sum() * O.sum()

    W = np.zeros((n_ratings, n_ratings), dtype=np.float64)
    for i in range(n_ratings):
        for j in range(n_ratings):
            W[i, j] = ((i - j) ** 2) / ((n_ratings - 1) ** 2)

    num = (W * O).sum()
    den = (W * E).sum()
    return 1.0 - num / den if den != 0 else 0.0


train_split_df, valid_split_df = train_test_split(
    train_df,
    train_size=0.98,  # keep most data for training; small valid for sanity QWK
    random_state=seed,
    stratify=train_df["labels"],
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

MAX_LEN = 512

tok_workers = min(4, os.cpu_count() or 1)
train_encodings = tokenizer(
    train_split_df["text"].tolist(),
    truncation=True,
    max_length=MAX_LEN,
    padding=False,
    return_attention_mask=True,
    num_workers=tok_workers,
    batch_size=1024,
)
valid_encodings = tokenizer(
    valid_split_df["text"].tolist(),
    truncation=True,
    max_length=MAX_LEN,
    padding=False,
    return_attention_mask=True,
    num_workers=tok_workers,
    batch_size=1024,
)

train_data = EssayDataset(
    train_split_df,
    tokenizer=None,
    with_labels=True,
    max_length=MAX_LEN,
    encodings=train_encodings,
)
valid_data = EssayDataset(
    valid_split_df,
    tokenizer=None,
    with_labels=True,
    max_length=MAX_LEN,
    encodings=valid_encodings,
)

num_workers = min(4, os.cpu_count() or 1)
pin_memory = torch.cuda.is_available()

train_loader = DataLoader(
    train_data,
    batch_size=8,
    shuffle=True,
    collate_fn=data_collator,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=(num_workers > 0),
    prefetch_factor=2 if num_workers > 0 else None,
)
valid_loader = DataLoader(
    valid_data,
    batch_size=16,
    shuffle=False,
    collate_fn=data_collator,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=(num_workers > 0),
    prefetch_factor=2 if num_workers > 0 else None,
)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
model.train()

epochs = 1
for ep in range(epochs):
    total_loss = 0.0
    n_steps = 0
    for batch in train_loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu().item())
        n_steps += 1
    print(f"epoch {ep+1}/{epochs} - train loss: {total_loss/max(n_steps,1):.4f}")

model.eval()
v_true, v_pred = [], []
with torch.no_grad():
    for batch in valid_loader:
        labels = batch["labels"].numpy()
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        logits = model(**batch).logits
        preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        v_true.append(labels)
        v_pred.append(preds)
v_true = np.concatenate(v_true)
v_pred = np.concatenate(v_pred)
print("valid QWK (labels 0..5):", quadratic_weighted_kappa(v_true, v_pred))



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1362530371.py in <cell line: 0>()
     38 # only executes faster. For fast tokenizers, `num_workers` parallelizes the pre-tokenization.
     39 tok_workers = min(4, os.cpu_count() or 1)
---> 40 train_encodings = tokenizer(
     41     train_split_df["text"].tolist(),
     42     truncation=True,

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in __call__(self, text, text_pair, text_target, text_pair_target, add_special_tokens, padding, truncation, max_length, stride, is_split_into_words, pad_to_multiple_of, padding_side, return_tensors, return_token_type_ids, return_attention_mask, return_overflowing_tokens, return_special_tokens_mask, return_offsets_mapping, return_length, verbose, **kwargs)
   2853             if not self._in_target_context_manager:
   2854                 self._switch_to_input_mode()
-> 2855             encodings = self._call_one(text=text, text_pair=text_pair, **all_kwargs)
   2856         if text_target is not None:
   2857             self._switch_to_target_mode()

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in _call_one(self, text, text_pair, add_special_tokens, padding, truncation, max_length, stride, is_split_into_words, pad_to_multiple_of, padding_side, return_tensors, return_token_type_ids, return_attention_mask, return_overflowing_tokens, return_special_tokens_mask, return_offsets_mapping, return_length, verbose, split_special_tokens, **kwargs)
   2941                 )
   2942             batch_text_or_text_pairs = list(zip(text, text_pair)) if text_pair is not None else text
-> 2943             return self.batch_encode_plus(
   2944                 batch_text_or_text_pairs=batch_text_or_text_pairs,
   2945                 add_special_tokens=add_special_tokens,

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in batch_encode_plus(self, batch_text_or_text_pairs, add_special_tokens, padding, truncation, max_length, stride, is_split_into_words, pad_to_multiple_of, padding_side, return_tensors, return_token_type_ids, return_attention_mask, return_overflowing_tokens, return_special_tokens_mask, return_offsets_mapping, return_length, verbose, split_special_tokens, **kwargs)
   3142         )
   3143 
-> 3144         return self._batch_encode_plus(
   3145             batch_text_or_text_pairs=batch_text_or_text_pairs,
   3146             add_special_tokens=add_special_tokens,

/usr/local/lib/python3.11/dist-packages/transformers/models/roberta/tokenization_roberta_fast.py in _batch_encode_plus(self, *args, **kwargs)
    215         )
    216 
--> 217         return super()._batch_encode_plus(*args, **kwargs)
    218 
    219     def _encode_plus(self, *args, **kwargs) -> BatchEncoding:

TypeError: PreTrainedTokenizerFast._batch_encode_plus() got an unexpected keyword argument 'num_workers'

## === cell 8
print(model.__class__.__name__)



## === cell 9
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

test_encodings = tokenizer(
    test_df["text"].tolist(),
    truncation=True,
    max_length=MAX_LEN,
    padding=False,
    return_attention_mask=True,
    num_workers=tok_workers,
    batch_size=1024,
)
test_dataset = EssayDataset(
    test_df,
    tokenizer=None,
    with_labels=False,
    max_length=MAX_LEN,
    encodings=test_encodings,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    collate_fn=data_collator,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=(num_workers > 0),
    prefetch_factor=2 if num_workers > 0 else None,
)

model.eval()
all_preds = []

with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        outputs = model(**batch)
        preds = torch.argmax(outputs.logits, dim=-1).detach().cpu().numpy()
        all_preds.append(preds)

all_preds = np.concatenate(all_preds, axis=0)
scores = (all_preds + 1).astype(int)

print("Pred score range:", int(scores.min()), int(scores.max()), "n=", len(scores))



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1049175233.py in <cell line: 0>()
      2 
      3 # Speed fix: same fast multiprocess tokenization for test; identical semantics.
----> 4 test_encodings = tokenizer(
      5     test_df["text"].tolist(),
      6     truncation=True,

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in __call__(self, text, text_pair, text_target, text_pair_target, add_special_tokens, padding, truncation, max_length, stride, is_split_into_words, pad_to_multiple_of, padding_side, return_tensors, return_token_type_ids, return_attention_mask, return_overflowing_tokens, return_special_tokens_mask, return_offsets_mapping, return_length, verbose, **kwargs)
   2853             if not self._in_target_context_manager:
   2854                 self._switch_to_input_mode()
-> 2855             encodings = self._call_one(text=text, text_pair=text_pair, **all_kwargs)
   2856         if text_target is not None:
   2857             self._switch_to_target_mode()

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in _call_one(self, text, text_pair, add_special_tokens, padding, truncation, max_length, stride, is_split_into_words, pad_to_multiple_of, padding_side, return_tensors, return_token_type_ids, return_attention_mask, return_overflowing_tokens, return_special_tokens_mask, return_offsets_mapping, return_length, verbose, split_special_tokens, **kwargs)
   2941                 )
   2942             batch_text_or_text_pairs = list(zip(text, text_pair)) if text_pair is not None else text
-> 2943             return self.batch_encode_plus(
   2944                 batch_text_or_text_pairs=batch_text_or_text_pairs,
   2945                 add_special_tokens=add_special_tokens,

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in batch_encode_plus(self, batch_text_or_text_pairs, add_special_tokens, padding, truncation, max_length, stride, is_split_into_words, pad_to_multiple_of, padding_side, return_tensors, return_token_type_ids, return_attention_mask, return_overflowing_tokens, return_special_tokens_mask, return_offsets_mapping, return_length, verbose, split_special_tokens, **kwargs)
   3142         )
   3143 
-> 3144         return self._batch_encode_plus(
   3145             batch_text_or_text_pairs=batch_text_or_text_pairs,
   3146             add_special_tokens=add_special_tokens,

/usr/local/lib/python3.11/dist-packages/transformers/models/roberta/tokenization_roberta_fast.py in _batch_encode_plus(self, *args, **kwargs)
    215         )
    216 
--> 217         return super()._batch_encode_plus(*args, **kwargs)
    218 
    219     def _encode_plus(self, *args, **kwargs) -> BatchEncoding:

TypeError: PreTrainedTokenizerFast._batch_encode_plus() got an unexpected keyword argument 'num_workers'

## === cell 10
submission_df = pd.DataFrame(
    {
        "essay_id": test_df["essay_id"].astype(str).values,
        "score": scores,
    }
)

submission_df["score"] = submission_df["score"].astype(int).clip(1, 6)

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print("Wrote:", submission_path)
print(submission_df.head())
print("submission shape:", submission_df.shape)

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1320729523.py in <cell line: 0>()
      2     {
      3         "essay_id": test_df["essay_id"].astype(str).values,
----> 4         "score": scores,
      5     }
      6 )

NameError: name 'scores' is not defined
