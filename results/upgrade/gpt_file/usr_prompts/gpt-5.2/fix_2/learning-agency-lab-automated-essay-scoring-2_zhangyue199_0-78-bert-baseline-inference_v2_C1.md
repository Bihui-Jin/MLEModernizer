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

0.7788391867553548

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
from torch.utils.data import Dataset

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
model_path = "/kaggle/input/bert-baseline-train/output/bert-base"
FALLBACK_MODEL_PATH = "/kaggle/input/bert-base-uncased"

MAX_LEN = 512

test_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
test_df = pd.read_csv(test_path)




## === cell 2
def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text))
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    return text.strip()


test_df["full_text"] = test_df["full_text"].apply(clean_text)



## === cell 3
effective_model_path = model_path if os.path.isdir(model_path) else FALLBACK_MODEL_PATH
if not os.path.isdir(effective_model_path):
    raise FileNotFoundError(
        f"Neither model_path nor fallback exists.\n"
        f"model_path={model_path}\n"
        f"fallback={FALLBACK_MODEL_PATH}\n"
        f"Directory listing of /kaggle/input: {os.listdir('/kaggle/input')}"
    )

tokenizer = BertTokenizer.from_pretrained(effective_model_path, local_files_only=True)
model = BertForSequenceClassification.from_pretrained(
    effective_model_path, local_files_only=True
)

if getattr(model.config, "num_labels", None) != 6:
    model.config.num_labels = 6
    model.classifier = torch.nn.Linear(model.classifier.in_features, 6)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/4264091586.py in <cell line: 0>()
      3 effective_model_path = model_path if os.path.isdir(model_path) else FALLBACK_MODEL_PATH
      4 if not os.path.isdir(effective_model_path):
----> 5     raise FileNotFoundError(
      6         f"Neither model_path nor fallback exists.\n"
      7         f"model_path={model_path}\n"

FileNotFoundError: Neither model_path nor fallback exists.
model_path=/kaggle/input/bert-baseline-train/output/bert-base
fallback=/kaggle/input/bert-base-uncased
Directory listing of /kaggle/input: ['description.md', 'sample_submission.csv', 'sample_submission.csv.zip', 'learning-agency-lab-automated-essay-scoring-2', 'test.csv', 'train.csv.zip', 'train.csv', 'test.csv.zip']

## === cell 4
class EssayTestDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=512):
        self.texts = list(texts)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in enc.items()}


test_dataset = EssayTestDataset(test_df["full_text"].values, tokenizer, MAX_LEN)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/521003601.py in <cell line: 0>()
     21 
     22 
---> 23 test_dataset = EssayTestDataset(test_df["full_text"].values, tokenizer, MAX_LEN)
     24 

NameError: name 'tokenizer' is not defined

## === cell 5
args = TrainingArguments(
    output_dir=".",
    per_device_eval_batch_size=8,
    report_to="none",
    do_train=False,
    do_eval=False,
    do_predict=True,
    dataloader_drop_last=False,
)

trainer = Trainer(
    model=model,
    args=args,
    tokenizer=tokenizer,
)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2405656292.py in <cell line: 0>()
     11 
     12 trainer = Trainer(
---> 13     model=model,
     14     args=args,
     15     tokenizer=tokenizer,

NameError: name 'model' is not defined

## === cell 6
pred_out = trainer.predict(test_dataset)
preds = pred_out.predictions

if isinstance(preds, (tuple, list)):
    preds = preds[0]

scores = np.argmax(preds, axis=1) + 1
scores = np.clip(scores, 1, 6).astype(int)

sub = test_df[["essay_id"]].copy()
sub["score"] = scores
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
assert list(sub.columns) == ["essay_id", "score"]
assert sub.shape[0] == test_df.shape[0]
assert os.path.exists("submission.csv")

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3597955492.py in <cell line: 0>()
----> 1 pred_out = trainer.predict(test_dataset)
      2 preds = pred_out.predictions
      3 
      4 # Handle possible tuple output
      5 if isinstance(preds, (tuple, list)):

NameError: name 'trainer' is not defined
