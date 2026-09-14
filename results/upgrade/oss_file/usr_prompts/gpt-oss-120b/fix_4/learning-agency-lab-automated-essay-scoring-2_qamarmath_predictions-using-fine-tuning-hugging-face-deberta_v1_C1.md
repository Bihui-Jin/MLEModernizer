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

0.7964847912419482

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.37982) has done: 'I fixed the protobuf import issue, switched to a reliable tokenizer/model loading strategy (using a local checkpoint if it exists, otherwise falling back to the public DeBERTa‑large model), corrected the dataset tokenisation pipeline, and ensured the prediction array is generated and saved as a proper `submission.csv` file.'
- What this solution (achieved 0.06138) has done: 'We avoid the Trainer (which triggers the protobuf error) and run inference manually with a DataLoader, then compute scores as the expected value of the class probabilities instead of a simple arg‑max. This fixes the runtime crash and usually yields a higher quadratic weighted‑kappa, moving the score toward the target while keeping the original model and tokenisation logic unchanged.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import gc
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset

TEST_DATA_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
BASE_MODEL_NAME = "microsoft/deberta-large"
MAX_LENGTH = 1024
EVAL_BATCH_SIZE = 1



## === cell 1
if os.path.isdir(MODEL_PATH) and os.path.isfile(
    os.path.join(MODEL_PATH, "tokenizer_config.json")
):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
else:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)


def tokenize(sample):
    return tokenizer(sample["full_text"], max_length=MAX_LENGTH, truncation=True)


df_test = pd.read_csv(TEST_DATA_PATH)
ds = Dataset.from_pandas(df_test)
ds = ds.map(tokenize, batched=False, remove_columns=["essay_id", "full_text"])




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3923976435.py in <cell line: 0>()
----> 1 if os.path.isdir(MODEL_PATH) and os.path.isfile(
      2     os.path.join(MODEL_PATH, "tokenizer_config.json")
      3 ):
      4     tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
      5 else:

NameError: name 'MODEL_PATH' is not defined

## === cell 2
class DataCollator:
    def __call__(self, features):
        model_inputs = [
            {"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]}
            for f in features
        ]
        batch = tokenizer.pad(
            model_inputs,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
            pad_to_multiple_of=16,
        )
        return batch


collator = DataCollator()



## === cell 3
model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL_NAME, num_labels=6
)
model.eval()
model.to("cuda" if torch.cuda.is_available() else "cpu")

loader = DataLoader(ds, batch_size=EVAL_BATCH_SIZE, collate_fn=collator)

all_logits = []
device = next(model.parameters()).device
with torch.no_grad():
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits.cpu().numpy()
        all_logits.append(logits)

predictions = np.concatenate(all_logits, axis=0)  # shape (num_examples, 6)

del model, loader
torch.cuda.empty_cache()
gc.collect()



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 4
probs = F.softmax(torch.from_numpy(predictions), dim=1).numpy()
class_vals = np.arange(1, 7)  # [1,2,3,4,5,6]
expected_scores = (probs * class_vals).sum(axis=1)
preds = np.clip(np.rint(expected_scores), 1, 6).astype(int)

df_test["score"] = preds
submission = df_test[["essay_id", "score"]]
submission.to_csv("submission.csv", index=False)

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/728789558.py in <cell line: 0>()
----> 1 probs = F.softmax(torch.from_numpy(predictions), dim=1).numpy()
      2 class_vals = np.arange(1, 7)  # [1,2,3,4,5,6]
      3 expected_scores = (probs * class_vals).sum(axis=1)
      4 preds = np.clip(np.rint(expected_scores), 1, 6).astype(int)
      5 

NameError: name 'predictions' is not defined
