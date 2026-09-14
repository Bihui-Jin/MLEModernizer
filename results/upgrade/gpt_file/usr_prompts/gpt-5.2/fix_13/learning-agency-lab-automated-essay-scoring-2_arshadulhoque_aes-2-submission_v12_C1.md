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
from transformers import AutoTokenizer, AutoModelForSequenceClassification

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

try:
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
except Exception:
    pass

test_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
test_data = pd.read_csv(test_path)
test_data.head()



## === cell 1
model_name = "microsoft/deberta-v3-xsmall"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6)

if tokenizer.pad_token is None and tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

if torch.cuda.is_available():
    try:
        model = torch.compile(model, mode="reduce-overhead")
    except Exception:
        pass



## === cell 2
MAX_LEN = 1536


class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


test_dataset = TextDataset(test_data["full_text"].tolist())
len(test_dataset), test_dataset[0][:200]



## === cell 3
per_device_bs = 64 if torch.cuda.is_available() else 16


def collate_texts(batch_texts):
    enc = tokenizer(
        batch_texts,
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt",
    )
    return enc["input_ids"].contiguous(), enc["attention_mask"].contiguous()


num_workers = min(4, (os.cpu_count() or 1)) if torch.cuda.is_available() else 0

loader = DataLoader(
    test_dataset,
    batch_size=per_device_bs,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=2 if num_workers > 0 else None,
    collate_fn=collate_texts,
)

n = len(test_dataset)
predicted_class = np.empty(n, dtype=np.int64)

use_amp = torch.cuda.is_available()
amp_ctx = (
    torch.autocast(device_type="cuda", dtype=torch.float16)
    if use_amp
    else torch.cpu.amp.autocast(enabled=False)
)

offset = 0
with torch.inference_mode():
    with amp_ctx:
        for batch_input_ids, batch_attention_mask in loader:
            batch_input_ids = batch_input_ids.to(device, non_blocking=True)
            batch_attention_mask = batch_attention_mask.to(device, non_blocking=True)
            out = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            bs = out.logits.shape[0]
            predicted_class[offset : offset + bs] = (
                out.logits.argmax(dim=-1).detach().cpu().numpy()
            )
            offset += bs

predicted_scores = (predicted_class + 1).astype(np.int32)
predicted_scores = np.clip(predicted_scores, 1, 6).astype(np.int32)

predicted_scores[:10], predicted_scores.min(), predicted_scores.max(), len(
    predicted_scores
)



## === cell 4
submission = pd.DataFrame(
    {
        "essay_id": test_data["essay_id"].values,
        "score": predicted_scores,
    }
)

assert submission.shape[0] == test_data.shape[0]
assert list(submission.columns) == ["essay_id", "score"]

submission.to_csv("submission.csv", index=False)
submission.head()



## === cell 5
print("Wrote:", os.path.abspath("submission.csv"))
print("Submission shape:", submission.shape)
print(submission["score"].value_counts().sort_index())
