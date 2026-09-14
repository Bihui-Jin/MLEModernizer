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

0.781539931926555

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path  # (Python3.4+)

MODEL_NAME = "bert-base-cased"
INPUT_DIR = "/kaggle/input/"
MODEL_DIR = "/kaggle/input/aes2-bertbase-5ep-results/aes2_bertbase_5ep_results/"
MAX_LENGTH = 1024
RANDOM_SEED = 42
EVAL_USE_PRETRAIN = 1
SUBMISSION = 1

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)
if torch.cuda.is_available():
    try:
        print(torch.cuda.current_device())
    except Exception as e:
        print(f"CUDA device query failed: {e}")



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1059635055.py in <cell line: 0>()
     13 SUBMISSION = 1
     14 
---> 15 device = "cuda:0" if torch.cuda.is_available() else "cpu"
     16 print(device)
     17 # Avoid calling torch.cuda.current_device() on systems without a GPU

NameError: name 'torch' is not defined

## === cell 1
if os.path.isdir(MODEL_DIR):
    checkpoint_path = os.path.join(MODEL_DIR, "checkpoint-1000")
    if os.path.isdir(checkpoint_path):
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            checkpoint_path, local_files_only=True
        )
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            checkpoint_path, local_files_only=True
        )
        print("Loaded tokenizer and model from local checkpoint.")
    else:
        print(
            "Local checkpoint not found, loading pretrained model from HuggingFace hub."
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=6
        )
else:
    print("Model directory not present, loading pretrained model from HuggingFace hub.")
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=6
    )

training_args = transformers.TrainingArguments(
    ".",
    per_device_eval_batch_size=32,
    report_to="none",
    fp16=False,
)

trainer = transformers.Trainer(
    model=model,
    args=training_args,
)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/521605500.py in <cell line: 0>()
     20 else:
     21     print("Model directory not present, loading pretrained model from HuggingFace hub.")
---> 22     tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
     23     model = transformers.AutoModelForSequenceClassification.from_pretrained(
     24         MODEL_NAME, num_labels=6

NameError: name 'transformers' is not defined

## === cell 2
df = pd.read_csv(INPUT_DIR + "learning-agency-lab-automated-essay-scoring-2/test.csv")
texts = df["full_text"].values.tolist()
print("Read csv successfully")

encodings = tokenizer(texts, truncation=True, padding=True)
print(encodings[0])
print("Encode texts successfully")

dataset = datasets.Dataset.from_dict(encodings)
print("Establish Dataset successfully")

prediction_output = trainer.predict(dataset)
logits = prediction_output.predictions
pred_ids = torch.argmax(torch.tensor(logits), dim=1) + 1  # scores are 1‑6
submission_preds = pred_ids.cpu().numpy()
print("Predict successfully")

submission_csv = pd.read_csv(
    INPUT_DIR + "learning-agency-lab-automated-essay-scoring-2/sample_submission.csv"
)
submission_csv["score"] = submission_preds.astype(int)
submission_csv.to_csv("submission.csv", index=False)
print("Submission file created:")
print(submission_csv.head())

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/492938282.py in <cell line: 0>()
      3 print("Read csv successfully")
      4 
----> 5 encodings = tokenizer(texts, truncation=True, padding=True)
      6 print(encodings[0])
      7 print("Encode texts successfully")

NameError: name 'tokenizer' is not defined
