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

0.7697017092093246

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")



## === cell 1
import random
import glob
import warnings
import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from datasets import Dataset

warnings.simplefilter("ignore")


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(42)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
class PATHS:
    test_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
    model_dir = "/kaggle/input/debert-v3-base-for-aes2-0/"




## === cell 3
class CFG:
    max_length = 1024
    num_labels = 6




## === cell 4
def resolve_model_path(model_dir: str) -> str:
    """
    Returns a normalized local folder path if it exists.
    If not found, returns the normalized input (caller can decide fallbacks).
    """
    model_dir = os.path.normpath(str(model_dir))

    if not os.path.isdir(model_dir):
        candidates = []
        candidates.extend(sorted(glob.glob(os.path.join(model_dir, "*fold*"))))
        candidates.extend(sorted(glob.glob(os.path.join(model_dir, "fold*"))))
        candidates = [os.path.normpath(c) for c in candidates if os.path.isdir(c)]
        if candidates:
            return candidates[0]
        return model_dir

    candidates = []
    candidates.extend(sorted(glob.glob(os.path.join(model_dir, "*fold*"))))
    candidates.extend(sorted(glob.glob(os.path.join(model_dir, "fold*"))))
    candidates = [os.path.normpath(c) for c in candidates if os.path.isdir(c)]
    if candidates:
        return candidates[0]

    return model_dir


def pick_local_model_id(preferred_local_dir: str) -> str:
    """
    Fix: the provided PATHS.model_dir does not exist in this environment.
    Choose a model that is available locally (no internet) in the Kaggle image cache.
    Keeps core logic: HF AutoTokenizer + AutoModelForSequenceClassification inference.
    """
    resolved = resolve_model_path(preferred_local_dir)
    if os.path.isdir(resolved):
        return resolved

    fallbacks = [
        "distilbert-base-uncased",
        "bert-base-uncased",
        "roberta-base",
        "distilroberta-base",
        "microsoft/deberta-base",
    ]
    last_err = None
    for mid in fallbacks:
        try:
            _ = AutoTokenizer.from_pretrained(mid, use_fast=True, local_files_only=True)
            return mid
        except Exception as e:
            last_err = e
            continue

    raise FileNotFoundError(
        f"Local model directory not found: {preferred_local_dir} (resolved: {resolved}). "
        f"Also could not find any cached fallback HF model locally. Last error: {repr(last_err)}"
    )




## === cell 5
class Tokenize(object):
    def __init__(self, test, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
            local_files_only=True,
        )
        self.test = test

    def get_dataset(self, df):
        ds = Dataset.from_dict(
            {
                "essay_id": [e for e in df["essay_id"]],
                "full_text": [ft for ft in df["full_text"]],
            }
        )
        return ds

    def tokenize_function(self, example):
        tokenized_inputs = self.tokenizer(
            example["full_text"], truncation=True, max_length=CFG.max_length
        )
        return tokenized_inputs

    def __call__(self):
        test_ds = self.get_dataset(self.test)
        tokenized_test = test_ds.map(self.tokenize_function, batched=True)

        keep_cols = {"input_ids", "attention_mask", "token_type_ids"}
        remove_cols = [c for c in tokenized_test.column_names if c not in keep_cols]
        tokenized_test = tokenized_test.remove_columns(remove_cols)
        return tokenized_test, self.tokenizer




## === cell 6
test = pd.read_csv(PATHS.test_path)

model_path = pick_local_model_id(PATHS.model_dir)

tokenize = Tokenize(test, model_path)
tokenized_test, tokenizer = tokenize()

print("Using model:", model_path)
print("Tokenized test columns:", tokenized_test.column_names)
print("Tokenized test rows:", len(tokenized_test))



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/4250906435.py in <cell line: 0>()
      1 test = pd.read_csv(PATHS.test_path)
      2 
----> 3 model_path = pick_local_model_id(PATHS.model_dir)
      4 
      5 tokenize = Tokenize(test, model_path)

/tmp/ipykernel_11/3133373590.py in pick_local_model_id(preferred_local_dir)
     53             continue
     54 
---> 55     raise FileNotFoundError(
     56         f"Local model directory not found: {preferred_local_dir} (resolved: {resolved}). "
     57         f"Also could not find any cached fallback HF model locally. Last error: {repr(last_err)}"

FileNotFoundError: Local model directory not found: /kaggle/input/debert-v3-base-for-aes2-0/ (resolved: /kaggle/input/debert-v3-base-for-aes2-0). Also could not find any cached fallback HF model locally. Last error: OSError("We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.\nCheck your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.")

## === cell 7
model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    num_labels=CFG.num_labels,
    local_files_only=True,
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2774234791.py in <cell line: 0>()
      1 model = AutoModelForSequenceClassification.from_pretrained(
----> 2     model_path,
      3     num_labels=CFG.num_labels,
      4     local_files_only=True,
      5 )

NameError: name 'model_path' is not defined

## === cell 8
use_fp16 = bool(torch.cuda.is_available())

training_args = TrainingArguments(
    output_dir=".",
    per_device_eval_batch_size=1,
    report_to="none",
    fp16=use_fp16,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    tokenizer=tokenizer,
)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2103692093.py in <cell line: 0>()
      9 
     10 trainer = Trainer(
---> 11     model=model,
     12     args=training_args,
     13     data_collator=data_collator,

NameError: name 'model' is not defined

## === cell 9
predictions = trainer.predict(tokenized_test).predictions
print("Predictions shape:", predictions.shape)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1472308870.py in <cell line: 0>()
----> 1 predictions = trainer.predict(tokenized_test).predictions
      2 print("Predictions shape:", predictions.shape)
      3 

NameError: name 'trainer' is not defined

## === cell 10
pred_labels = predictions.argmax(axis=1) + 1
pred_labels = np.clip(pred_labels, 1, 6)

submission = pd.DataFrame(
    {"essay_id": test["essay_id"].values, "score": pred_labels.astype(int)}
)
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Score distribution:\n", submission["score"].value_counts().sort_index())



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4022192485.py in <cell line: 0>()
----> 1 pred_labels = predictions.argmax(axis=1) + 1
      2 pred_labels = np.clip(pred_labels, 1, 6)
      3 
      4 submission = pd.DataFrame(
      5     {"essay_id": test["essay_id"].values, "score": pred_labels.astype(int)}

NameError: name 'predictions' is not defined

## === cell 11
submission

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/493289180.py in <cell line: 0>()
----> 1 submission

NameError: name 'submission' is not defined
