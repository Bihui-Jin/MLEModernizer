# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
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

# 5. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader

torch.manual_seed(42)
np.random.seed(42)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

try:
    import google.protobuf.message_factory as _mf

    if hasattr(_mf, "MessageFactory") and not hasattr(
        _mf.MessageFactory, "GetPrototype"
    ):

        def _GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        _mf.MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass



## === cell 1
test_data = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
)
test_data.head()



## === cell 2
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PRIMARY_MODEL_NAME = "microsoft/deberta-v3-xsmall"
FALLBACK_MODEL_NAME = "microsoft/deberta-v3-base"  # broadly available
SECOND_FALLBACK_MODEL_NAME = "microsoft/deberta-v3-small"  # extra robustness

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_tokenizer(name: str):
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(name, num_labels=6)
    return tok, mdl


_last_err = None
for _name in (PRIMARY_MODEL_NAME, FALLBACK_MODEL_NAME, SECOND_FALLBACK_MODEL_NAME):
    try:
        tokenizer, model = load_model_and_tokenizer(_name)
        _last_err = None
        break
    except Exception as e:
        _last_err = e

if _last_err is not None:
    raise _last_err

model = model.to(device)
model.eval()

if torch.cuda.is_available():
    try:
        model = model.to(memory_format=torch.channels_last)
    except Exception:
        pass

model



## === cell 3
model_max_len = getattr(tokenizer, "model_max_length", 512)
if model_max_len is None or model_max_len > 1024:
    model_max_len = 512

max_length = min(512, int(model_max_len))

len(test_data), max_length



## === cell 4

train_df = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
)
train_texts = train_df["full_text"].astype(str).tolist()
train_labels = (train_df["score"].astype(int).values - 1).clip(0, 5).astype(np.int64)

train_enc = tokenizer(
    train_texts,
    truncation=True,
    padding=True,
    max_length=max_length,
    return_tensors="pt",
)


class EncodedWithLabelsDataset(TorchDataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.keys = tuple(encodings.keys())

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        item = {k: self.encodings[k][idx] for k in self.keys}
        item["labels"] = self.labels[idx]
        return item


def collate_encoded_with_labels(batch):
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}


train_dataset = EncodedWithLabelsDataset(train_enc, train_labels)

train_batch_size = 16 if torch.cuda.is_available() else 8
train_loader = DataLoader(
    train_dataset,
    batch_size=train_batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=False,
    collate_fn=collate_encoded_with_labels,
)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

epochs = 1  # minimal change to move away from 0.0 while keeping runtime under 600s
model.train()
for ep in range(epochs):
    for batch in train_loader:
        optimizer.zero_grad(set_to_none=True)
        batch = {
            k: v.to(device, non_blocking=torch.cuda.is_available())
            for k, v in batch.items()
            if k in ("input_ids", "attention_mask", "labels")
        }
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

model.eval()
float(loss.detach().cpu())



## === cell 5
texts = test_data["full_text"].astype(str).tolist()
test_enc = tokenizer(
    texts,
    truncation=True,
    padding=True,
    max_length=max_length,
    return_tensors="pt",
)


class EncodedDataset(TorchDataset):
    def __init__(self, encodings):
        self.encodings = encodings
        self.keys = tuple(encodings.keys())

    def __len__(self):
        return self.encodings[self.keys[0]].shape[0]

    def __getitem__(self, idx):
        return {k: self.encodings[k][idx] for k in self.keys}


def collate_encoded(batch):
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}


encoded_test_dataset = EncodedDataset(test_enc)

batch_size = 192 if torch.cuda.is_available() else 48
num_workers = 0

test_loader = DataLoader(
    encoded_test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=False,
    collate_fn=collate_encoded,
)

n_test = len(encoded_test_dataset)
num_labels = 6
logits = np.empty((n_test, num_labels), dtype=np.float32)

model.eval()
row = 0
with torch.inference_mode():
    for batch in test_loader:
        batch = {
            k: v.to(device, non_blocking=torch.cuda.is_available())
            for k, v in batch.items()
            if k in ("input_ids", "attention_mask")
        }

        outputs = model(**batch)
        out = outputs.logits  # [bs, 6]
        out = out.detach().to("cpu").numpy()
        bsz = out.shape[0]

        if out.shape[-1] != 6:
            if out.shape[-1] > 6:
                out = out[:, :6]
            else:
                pad = np.zeros((bsz, 6 - out.shape[-1]), dtype=out.dtype)
                out = np.concatenate([out, pad], axis=1)

        logits[row : row + bsz] = out
        row += bsz

logits = logits[:row]
logits.shape



## === cell 6
pred_class = np.argmax(logits, axis=-1)

num_labels = logits.shape[-1]
if num_labels == 6:
    predicted_scores = (pred_class + 1).astype(np.int32)
else:
    predicted_scores = np.rint(1 + (pred_class / max(1, num_labels - 1)) * 5).astype(
        np.int32
    )

predicted_scores = np.clip(predicted_scores, 1, 6)

predicted_scores[:10], predicted_scores.min(), predicted_scores.max(), num_labels



## === cell 7
submission = pd.DataFrame(
    {
        "essay_id": test_data["essay_id"].values,
        "score": predicted_scores,
    }
)

submission = submission[["essay_id", "score"]]
submission.to_csv("submission.csv", index=False)
submission.head()



## === cell 8
submission
