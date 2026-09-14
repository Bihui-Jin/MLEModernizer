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

0.794443268230026

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
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
    max_length = 512
    num_labels = 6




## === cell 4
def resolve_model_path(model_dir: str) -> str:
    """
    Bugfix (runtime): ensure we pass a valid local directory containing config.json.
    If the configured directory doesn't exist in this Kaggle environment, fall back
    to searching /kaggle/input for a compatible local HF model directory.
    """

    def _resolve_under(root_dir: str) -> str:
        root_dir = os.path.normpath(root_dir)

        if os.path.isdir(root_dir) and os.path.exists(
            os.path.join(root_dir, "config.json")
        ):
            return root_dir

        fold_candidates = sorted(
            [p for p in glob.glob(os.path.join(root_dir, "*fold*")) if os.path.isdir(p)]
        )
        for p in fold_candidates:
            if os.path.exists(os.path.join(p, "config.json")):
                return os.path.normpath(p)

        config_hits = glob.glob(
            os.path.join(root_dir, "**", "config.json"), recursive=True
        )
        if len(config_hits) > 0:
            config_hits.sort(key=lambda x: (x.count(os.sep), x))
            return os.path.normpath(os.path.dirname(config_hits[0]))

        raise FileNotFoundError

    try:
        return _resolve_under(model_dir)
    except FileNotFoundError:
        pass

    try:
        return _resolve_under("/kaggle/input")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Could not find a local HuggingFace model (config.json) under: {model_dir} "
            f"or under /kaggle/input. Please ensure the model dataset is attached."
        )




## === cell 5
class Tokenize(object):
    def __init__(self, test, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True
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
        return tokenized_test, self.tokenizer




## === cell 6
test = pd.read_csv(PATHS.test_path)

model_path = resolve_model_path(PATHS.model_dir)

tokenize = Tokenize(test, model_path)
tokenized_test, tokenizer = tokenize()

keep_cols = {"input_ids", "attention_mask", "token_type_ids"}
present_keep_cols = keep_cols.intersection(set(tokenized_test.column_names))
remove_cols = [c for c in tokenized_test.column_names if c not in present_keep_cols]
tokenized_test = tokenized_test.remove_columns(remove_cols)

tokenized_test = tokenized_test.with_format("torch")



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/594187717.py in resolve_model_path(model_dir)
     41     try:
---> 42         return _resolve_under("/kaggle/input")
     43     except FileNotFoundError:

/tmp/ipykernel_11/594187717.py in _resolve_under(root_dir)
     30 
---> 31         raise FileNotFoundError
     32 

FileNotFoundError: 

During handling of the above exception, another exception occurred:

FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2006211901.py in <cell line: 0>()
      1 test = pd.read_csv(PATHS.test_path)
      2 
----> 3 model_path = resolve_model_path(PATHS.model_dir)
      4 
      5 tokenize = Tokenize(test, model_path)

/tmp/ipykernel_11/594187717.py in resolve_model_path(model_dir)
     42         return _resolve_under("/kaggle/input")
     43     except FileNotFoundError:
---> 44         raise FileNotFoundError(
     45             f"Could not find a local HuggingFace model (config.json) under: {model_dir} "
     46             f"or under /kaggle/input. Please ensure the model dataset is attached."

FileNotFoundError: Could not find a local HuggingFace model (config.json) under: /kaggle/input/debert-v3-base-for-aes2-0/ or under /kaggle/input. Please ensure the model dataset is attached.

## === cell 7
model = AutoModelForSequenceClassification.from_pretrained(
    model_path, num_labels=CFG.num_labels, local_files_only=True
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/54786963.py in <cell line: 0>()
      1 model = AutoModelForSequenceClassification.from_pretrained(
----> 2     model_path, num_labels=CFG.num_labels, local_files_only=True
      3 )
      4 
      5 data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

NameError: name 'model_path' is not defined

## === cell 8
use_fp16 = torch.cuda.is_available()

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
/tmp/ipykernel_11/3222491426.py in <cell line: 0>()
      9 
     10 trainer = Trainer(
---> 11     model=model,
     12     args=training_args,
     13     data_collator=data_collator,

NameError: name 'model' is not defined

## === cell 9
predictions = trainer.predict(tokenized_test).predictions

pred_scores = predictions.argmax(axis=1).astype(np.int64) + 1
pred_scores = np.clip(pred_scores, 1, 6)

if len(pred_scores) != len(test):
    raise RuntimeError(
        f"Prediction length mismatch: {len(pred_scores)} vs test {len(test)}"
    )

submission = pd.DataFrame({"essay_id": test["essay_id"].values, "score": pred_scores})
submission.to_csv("submission.csv", index=False)

submission

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/472546532.py in <cell line: 0>()
----> 1 predictions = trainer.predict(tokenized_test).predictions
      2 
      3 pred_scores = predictions.argmax(axis=1).astype(np.int64) + 1
      4 pred_scores = np.clip(pred_scores, 1, 6)
      5 

NameError: name 'trainer' is not defined
