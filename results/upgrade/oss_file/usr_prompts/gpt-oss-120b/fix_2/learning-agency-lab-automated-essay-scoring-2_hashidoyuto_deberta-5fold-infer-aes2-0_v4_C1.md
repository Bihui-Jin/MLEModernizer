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

0.8053301059475746

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

import warnings

warnings.simplefilter("ignore")
import numpy as np
import pandas as pd
import torch
import glob
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import Dataset




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
class PATHS:
    test_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
    model_dir = "/kaggle/input/groupkfold-deberta-aes2-0/"


class CFG:
    max_length = 512
    num_labels = 6




## === cell 2
class Tokenize:
    def __init__(self, df, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.df = df

    def get_dataset(self):
        return Dataset.from_dict(
            {
                "essay_id": self.df["essay_id"].tolist(),
                "full_text": self.df["full_text"].tolist(),
            }
        )

    def tokenize_function(self, example):
        return self.tokenizer(
            example["full_text"], truncation=True, max_length=CFG.max_length
        )

    def __call__(self):
        ds = self.get_dataset()
        tokenized = ds.map(self.tokenize_function, batched=True)
        return tokenized, self.tokenizer




## === cell 3
test = pd.read_csv(PATHS.test_path)

model_paths = sorted(glob.glob(os.path.join(PATHS.model_dir, "*fold*")))
if not model_paths:
    raise FileNotFoundError(f"No model checkpoints found in {PATHS.model_dir}")



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2171460344.py in <cell line: 0>()
      5 model_paths = sorted(glob.glob(os.path.join(PATHS.model_dir, "*fold*")))
      6 if not model_paths:
----> 7     raise FileNotFoundError(f"No model checkpoints found in {PATHS.model_dir}")
      8 

FileNotFoundError: No model checkpoints found in /kaggle/input/groupkfold-deberta-aes2-0/

## === cell 4
predictions = []

for model_path in model_paths:
    tokenize = Tokenize(test, model_path)
    tokenized_test, tokenizer = tokenize()

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=CFG.num_labels
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir="./tmp",
        per_device_eval_batch_size=1,
        report_to="none",
        fp16=False,  # use fp16 only when GPU is available
        no_cuda=not torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    preds = trainer.predict(tokenized_test).predictions
    predictions.append(preds)



## === cell 5
num_folds = len(predictions)
if num_folds == 0:
    raise RuntimeError("No predictions were generated.")

final_pred = np.zeros_like(predictions[0])

for p in predictions:
    final_pred += p
final_pred = final_pred / num_folds

final_labels = final_pred.argmax(axis=1) + 1



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/710685876.py in <cell line: 0>()
      2 num_folds = len(predictions)
      3 if num_folds == 0:
----> 4     raise RuntimeError("No predictions were generated.")
      5 
      6 # Initialise accumulator with correct shape

RuntimeError: No predictions were generated.

## === cell 6
submission = pd.DataFrame({"essay_id": test["essay_id"].values, "score": final_labels})
submission.to_csv("submission.csv", index=False)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/826703433.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({"essay_id": test["essay_id"].values, "score": final_labels})
      2 submission.to_csv("submission.csv", index=False)
      3 

NameError: name 'final_labels' is not defined

## === cell 7
submission.head()

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3365464162.py in <cell line: 0>()
----> 1 submission.head()

NameError: name 'submission' is not defined
