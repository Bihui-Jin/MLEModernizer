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

3.13

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

0.7846490997226996

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import torch

print(f"PyTorch version: {torch.__version__}")

import transformers

print(f"Hugging Face Transformers version: {transformers.__version__}")

import datasets

print(f"Hugging Face Datasets version: {datasets.__version__}")



## === cell 1
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

MODEL_NAME = "bert-base-cased"
INPUT_DIR = "/kaggle/input/"
MODEL_DIR = "/kaggle/input/aes2-persuade-bertbase-5ep-results/"
CHECKPOINT_SUBDIR = "checkpoint-2500"

MAX_LENGTH = 1024
RANDOM_SEED = 42
EVAL_USE_PRETRAIN = 1
SUBMISSION = 1

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("device:", device)
if torch.cuda.is_available():
    print("cuda current_device:", torch.cuda.current_device())
else:
    print("cuda not available; running on CPU")

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)



## === cell 2
ckpt_path = Path(MODEL_DIR) / CHECKPOINT_SUBDIR
if not ckpt_path.exists():
    raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_path}")

tokenizer = transformers.AutoTokenizer.from_pretrained(
    str(ckpt_path), local_files_only=True
)
print("Load tokenizer successfully:", ckpt_path)

model = transformers.AutoModelForSequenceClassification.from_pretrained(
    str(ckpt_path), local_files_only=True
)
print("Load model successfully:", ckpt_path)

use_fp16 = bool(torch.cuda.is_available())

training_args = transformers.TrainingArguments(
    output_dir=".",
    per_device_eval_batch_size=32,
    report_to="none",
    fp16=use_fp16,
)

trainer = transformers.Trainer(
    model=model,
    args=training_args,
)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/1573498675.py in <cell line: 0>()
      3 ckpt_path = Path(MODEL_DIR) / CHECKPOINT_SUBDIR
      4 if not ckpt_path.exists():
----> 5     raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_path}")
      6 
      7 tokenizer = transformers.AutoTokenizer.from_pretrained(

FileNotFoundError: Checkpoint directory not found: /kaggle/input/aes2-persuade-bertbase-5ep-results/checkpoint-2500

## === cell 3
df = pd.read_csv(INPUT_DIR + "learning-agency-lab-automated-essay-scoring-2/test.csv")
texts = df["full_text"].astype(str).values.tolist()
print("Read csv successfully:", df.shape)

encodings = tokenizer(
    texts,
    truncation=True,
    padding="max_length",
    max_length=MAX_LENGTH,
)

dataset = datasets.Dataset.from_dict(encodings)
dataset = dataset.with_format("torch", columns=["input_ids", "attention_mask"])
print("Dataset built:", dataset)

pred_out = trainer.predict(dataset)
submission_logits = pred_out.predictions
submission_preds = (
    torch.argmax(torch.tensor(submission_logits), dim=1).cpu().numpy() + 1
)
print(
    "Predictions generated:",
    submission_preds.shape,
    "min/max:",
    submission_preds.min(),
    submission_preds.max(),
)

submission_csv = pd.read_csv(
    INPUT_DIR + "learning-agency-lab-automated-essay-scoring-2/sample_submission.csv"
)
submission_csv["score"] = submission_preds.astype(int)

submission_csv = submission_csv[["essay_id", "score"]]

submission_path = "submission.csv"
submission_csv.to_csv(submission_path, index=False)
print("Wrote:", submission_path)
print(submission_csv.head())

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3529094337.py in <cell line: 0>()
      5 # Fix: provide max_length so the model input length matches training setup and avoid dynamic padding issues.
      6 # Also return attention_mask; Dataset will be set to torch for Trainer.
----> 7 encodings = tokenizer(
      8     texts,
      9     truncation=True,

NameError: name 'tokenizer' is not defined
