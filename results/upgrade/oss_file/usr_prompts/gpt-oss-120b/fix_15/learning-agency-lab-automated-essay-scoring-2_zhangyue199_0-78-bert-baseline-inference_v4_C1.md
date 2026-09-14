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

# 5. Target score

0.7753155262158988

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import re
import numpy as np
import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

torch.manual_seed(42)
np.random.seed(42)

torch.set_float32_matmul_precision("high")
torch.set_num_threads(os.cpu_count() or 1)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
model_name = "roberta-base"
MAX_LEN = 512

test_df = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
)



## === cell 2
test_df["full_text"] = (
    test_df["full_text"]
    .str.replace(r"\s+", " ", regex=True)
    .str.replace(r"[^a-zA-Z0-9]", " ", regex=True)
    .str.strip()
)



## === cell 3
tokenizer = RobertaTokenizer.from_pretrained(model_name, use_fast=True)
encoded = tokenizer(
    test_df["full_text"].tolist(),
    padding="max_length",
    truncation=True,
    max_length=MAX_LEN,
    return_tensors="pt",
)

test_input_ids = encoded["input_ids"].pin_memory()
test_attention_mask = encoded["attention_mask"].pin_memory()



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/1105354651.py in <cell line: 0>()
      8 )
      9 
---> 10 test_input_ids = encoded["input_ids"].pin_memory()
     11 test_attention_mask = encoded["attention_mask"].pin_memory()
     12 

RuntimeError: Cannot access accelerator device when none is available.

## === cell 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=6)
model.to(device)

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
    model.half()  # half‑precision on GPU
    eval_batch_size = 1024  # larger GPU batch reduces loop overhead
else:
    eval_batch_size = 2048  # larger CPU batch reduces loop overhead



## === cell 5
model.eval()
total_samples = test_input_ids.size(0)
scores = np.empty(total_samples, dtype=np.int32)

with torch.inference_mode():
    for start in range(0, total_samples, eval_batch_size):
        end = min(start + eval_batch_size, total_samples)
        batch_input_ids = test_input_ids[start:end].to(device, non_blocking=True)
        batch_attention_mask = test_attention_mask[start:end].to(
            device, non_blocking=True
        )

        outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
        logits = outputs.logits  # (batch, 6)

        batch_scores = torch.argmax(logits, dim=1).cpu().numpy() + 1
        scores[start:end] = batch_scores



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1064360315.py in <cell line: 0>()
      1 model.eval()
----> 2 total_samples = test_input_ids.size(0)
      3 scores = np.empty(total_samples, dtype=np.int32)
      4 
      5 with torch.inference_mode():

NameError: name 'test_input_ids' is not defined

## === cell 6
sub = test_df[["essay_id"]].copy()
sub["score"] = scores
sub.to_csv("submission.csv", index=False)
display(sub.head())

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/803957998.py in <cell line: 0>()
      1 sub = test_df[["essay_id"]].copy()
----> 2 sub["score"] = scores
      3 sub.to_csv("submission.csv", index=False)
      4 display(sub.head())

NameError: name 'scores' is not defined
