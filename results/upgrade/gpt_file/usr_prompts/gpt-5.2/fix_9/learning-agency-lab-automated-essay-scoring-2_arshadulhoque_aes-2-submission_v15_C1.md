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
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

torch.set_grad_enabled(False)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True



## === cell 1
test_data = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
)
test_data



## === cell 2
MODEL_NAME = "microsoft/deberta-v3-xsmall"

try:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, local_files_only=True, use_fast=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, local_files_only=True
    )
except Exception:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

if getattr(model.config, "num_labels", None) != 6:
    model = (
        AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=6,
            ignore_mismatched_sizes=True,
            local_files_only=True,
        )
        if os.path.isdir(os.path.expanduser("~/.cache/huggingface"))
        else AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=6, ignore_mismatched_sizes=True
        )
    )

model.eval()



## === cell 3
MAX_LEN = 1536
texts = test_data["full_text"].tolist()

tok_kwargs = dict(
    truncation=True,
    max_length=MAX_LEN,
    padding=False,
    return_attention_mask=True,
    return_token_type_ids=False,
)

encodings = tokenizer(texts, **tok_kwargs)


class PreTokenizedDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
        self.length = len(encodings["input_ids"])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}


test_dataset = PreTokenizedDataset(encodings)



## === cell 4
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model.to(device)

if hasattr(model, "config"):
    try:
        model.config.use_cache = False
    except Exception:
        pass

if use_cuda:
    try:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
    except Exception:
        pass

pad_to_multiple_of = 8 if use_cuda else None
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding="longest",
    max_length=MAX_LEN,
    pad_to_multiple_of=pad_to_multiple_of,
    return_tensors="pt",
)

num_workers = 0
prefetch_factor = None

dl_kwargs = dict(
    dataset=test_dataset,
    batch_size=(64 if use_cuda else 8),
    shuffle=False,
    num_workers=num_workers,
    pin_memory=use_cuda,
    collate_fn=data_collator,
    persistent_workers=False,
)
test_loader = DataLoader(**dl_kwargs)

if use_cuda:
    try:
        model = model.to(memory_format=torch.channels_last)
    except Exception:
        pass



## === cell 5
pred_classes = np.empty(len(test_dataset), dtype=np.int64)

model.eval()
offset = 0

with torch.inference_mode():
    for batch in test_loader:
        if use_cuda:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        else:
            batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        batch_pred = torch.argmax(outputs.logits, dim=-1).cpu().numpy().astype(np.int64)

        bs = batch_pred.shape[0]
        pred_classes[offset : offset + bs] = batch_pred
        offset += bs

assert offset == len(test_dataset)



## === cell 6
predicted_scores = (pred_classes + 1).clip(1, 6).astype("int32")



## === cell 7
submission = pd.DataFrame(
    {"essay_id": test_data["essay_id"].values, "score": predicted_scores}
)

submission.to_csv("submission.csv", index=False)
submission



## === cell 8
assert submission.shape[0] == test_data.shape[0]
assert list(submission.columns) == ["essay_id", "score"]
assert submission["score"].between(1, 6).all()
print("Wrote submission.csv with shape:", submission.shape)
