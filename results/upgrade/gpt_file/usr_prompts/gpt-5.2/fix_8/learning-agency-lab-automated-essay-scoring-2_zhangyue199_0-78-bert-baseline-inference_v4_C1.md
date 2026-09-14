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
import re
import numpy as np
import pandas as pd

import torch
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True




## === cell 1
model_path = "roberta-base"
MAX_LEN = 512

test_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
test_df = pd.read_csv(test_path)




## === cell 2
_non_alnum = re.compile(r"[^a-zA-Z0-9]")
_whitespace = re.compile(r"\s+")

s = test_df["full_text"].astype(str)
s = s.str.replace(_non_alnum, " ", regex=True)
s = s.str.replace(_whitespace, " ", regex=True)
test_df["full_text"] = s.str.strip()




## === cell 3
tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

if device.type == "cuda":
    model = model.to(memory_format=torch.channels_last)

if hasattr(torch, "compile"):
    try:
        model = torch.compile(model, mode="reduce-overhead")
    except Exception:
        pass

texts = test_df["full_text"].tolist()
n = len(texts)
num_labels = int(model.config.num_labels)

batch_size = 128 if device.type == "cuda" else 32
tokenize_chunk = (
    2048 if device.type == "cuda" else 512
)  # controls peak CPU/RAM; does not change outputs

preds = np.empty((n, num_labels), dtype=np.float32)

offset = 0
with torch.inference_mode():
    for start in range(0, n, tokenize_chunk):
        end = min(start + tokenize_chunk, n)
        enc = tokenizer(
            texts[start:end],
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )

        input_ids_all = enc["input_ids"]
        attention_mask_all = enc["attention_mask"]

        for b0 in range(0, input_ids_all.size(0), batch_size):
            b1 = min(b0 + batch_size, input_ids_all.size(0))
            input_ids = input_ids_all[b0:b1]
            attention_mask = attention_mask_all[b0:b1]

            if device.type == "cuda":
                input_ids = input_ids.to(device, non_blocking=True)
                attention_mask = attention_mask.to(device, non_blocking=True)
            else:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)

            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits.detach().cpu().numpy()

            bs = logits.shape[0]
            preds[offset : offset + bs] = logits
            offset += bs

scores = (np.argmax(preds, axis=1) + 1).astype(int)
scores = np.clip(scores, 1, 6)

sub = test_df[["essay_id"]].copy()
sub["score"] = scores

sub_path = "submission.csv"
sub.to_csv(sub_path, index=False)

print("Wrote:", sub_path)
print(sub.head())
print("Rows:", len(sub), "Columns:", list(sub.columns))
