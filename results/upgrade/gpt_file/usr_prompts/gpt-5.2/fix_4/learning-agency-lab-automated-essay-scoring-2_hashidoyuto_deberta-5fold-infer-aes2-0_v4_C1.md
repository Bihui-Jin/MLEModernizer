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

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")



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




## === cell 2
class PATHS:
    test_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
    test_path_alt = (
        "/kaggle/data/learning-agency-lab-automated-essay-scoring-2/test.csv"
    )
    model_dir = "/kaggle/input/groupkfold-deberta-aes2-0/"




## === cell 3
class CFG:
    max_length = 512
    num_labels = 6




## === cell 4
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(42)




## === cell 5
class Tokenize(object):
    def __init__(self, test, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.test = test

    def get_dataset(self, df):
        ds = Dataset.from_dict(
            {
                "essay_id": df["essay_id"].to_numpy(),
                "full_text": df["full_text"].to_numpy(),
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
if os.path.exists(PATHS.test_path):
    test = pd.read_csv(PATHS.test_path)
elif os.path.exists(PATHS.test_path_alt):
    test = pd.read_csv(PATHS.test_path_alt)
else:
    raise FileNotFoundError(
        f"Could not find test.csv at {PATHS.test_path} or {PATHS.test_path_alt}"
    )

model_paths = []
if os.path.exists(PATHS.model_dir):
    model_paths = glob.glob(os.path.join(PATHS.model_dir, "*fold*"))
    model_paths.sort()

if len(model_paths) == 0:
    fallback_model = "microsoft/deberta-v3-base"
    model_paths = [fallback_model]

model_paths



## === cell 7
use_fp16 = torch.cuda.is_available()



## === cell 8
tokenize = Tokenize(test, model_paths[0])
tokenized_test, tokenizer = tokenize()

keep_cols = {"essay_id", "input_ids", "attention_mask", "token_type_ids"}
remove_cols = [c for c in tokenized_test.column_names if c not in keep_cols]
if len(remove_cols) > 0:
    tokenized_test = tokenized_test.remove_columns(remove_cols)

tensor_cols = [
    c
    for c in ["input_ids", "attention_mask", "token_type_ids"]
    if c in tokenized_test.column_names
]
tokenized_test.set_format(type="torch", columns=tensor_cols)

per_device_eval_bs = 16 if torch.cuda.is_available() else 8

predictions = []
for i, model_path in enumerate(model_paths):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=CFG.num_labels
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=".",
        per_device_eval_batch_size=per_device_eval_bs,
        report_to="none",
        fp16=use_fp16,
        dataloader_num_workers=2,
        dataloader_pin_memory=torch.cuda.is_available(),
        dataloader_persistent_workers=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    pre_preds = trainer.predict(tokenized_test).predictions  # (n_test, num_labels)
    predictions.append(pre_preds)

if len(predictions) == 0:
    raise RuntimeError(
        "No predictions were produced; check model_paths and inference pipeline."
    )

len(predictions), predictions[0].shape



## === cell 9
n_test = len(test)
for j, p in enumerate(predictions):
    if p.ndim != 2 or p.shape[0] != n_test or p.shape[1] != CFG.num_labels:
        raise ValueError(f"Unexpected prediction shape at model {j}: {p.shape}")



## === cell 10
final_pred = np.zeros_like(predictions[0], dtype=np.float64)
for p in predictions:
    final_pred += p.astype(np.float64)

final_pred /= len(predictions)
final_pred = final_pred.argmax(axis=1) + 1  # class 0-5 -> score 1-6

submission = pd.DataFrame(
    {"essay_id": test["essay_id"].values, "score": final_pred.astype(int)}
)
submission.to_csv("submission.csv", index=False)

submission.head()



## === cell 11
submission.shape, submission.dtypes, submission.head()
