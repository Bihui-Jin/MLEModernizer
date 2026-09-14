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

0.807150201953024

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import re
import numpy as np
import pandas as pd

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
COMP_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2"
test_path = os.path.join(COMP_PATH, "test.csv")

candidate_model_paths = [
    "/kaggle/input/bert-baseline-train/output/bert-base",
    "/kaggle/input/bert-baseline-train/bert-base",
    "/kaggle/input/bert-baseline-train",
]
model_path = next((p for p in candidate_model_paths if os.path.isdir(p)), None)
if model_path is None:
    model_path = "bert-base-uncased"

MAX_LEN = 1024

test_df = pd.read_csv(test_path)
assert {"essay_id", "full_text"}.issubset(test_df.columns)



## === cell 2
s = test_df["full_text"].astype(str)
s = s.str.replace(r"\s+", " ", regex=True)
s = s.str.replace(r"[^a-zA-Z0-9]", " ", regex=True)
s = s.str.strip()
test_df["full_text"] = s



## === cell 3
import torch

from datasets import Dataset

is_local_dir = os.path.isdir(model_path)

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=is_local_dir,
    use_fast=True,
)

tokenizer_max = getattr(tokenizer, "model_max_length", 512)
if tokenizer_max is None or tokenizer_max > 100000:
    tokenizer_max = 512
MAX_LEN = int(min(MAX_LEN, tokenizer_max))

model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    local_files_only=is_local_dir,
)

torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

test_ds = Dataset.from_pandas(test_df[["essay_id", "full_text"]], preserve_index=False)


def _tokenize_batch(batch):
    return tokenizer(
        batch["full_text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
    )


test_ds = test_ds.with_format("torch", columns=["essay_id", "full_text"])
test_ds.set_transform(_tokenize_batch)


def _collate(features):
    batch = {}
    for k in ("input_ids", "attention_mask", "token_type_ids"):
        if k in features[0]:
            batch[k] = torch.stack([f[k] for f in features], dim=0)
    return batch


num_workers = min(4, (os.cpu_count() or 2))
args = TrainingArguments(
    output_dir=".",
    per_device_eval_batch_size=32,  # safe speedup for inference; does not change core logic/semantics
    report_to="none",
    dataloader_drop_last=False,
    dataloader_num_workers=num_workers,
    dataloader_pin_memory=True,
)

trainer = Trainer(
    model=model,
    args=args,
    tokenizer=tokenizer,
    data_collator=_collate,
)

pred_out = trainer.predict(test_ds)
preds = pred_out.predictions

preds = np.asarray(preds)

if preds.ndim == 2 and preds.shape[1] > 1:
    scores = np.argmax(preds, axis=1) + 1
else:
    scores = np.ravel(preds)
    scores = np.rint(scores).astype(int)

scores = np.clip(scores, 1, 6).astype(int)

sub = test_df[["essay_id"]].copy()
sub["score"] = scores
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
assert os.path.exists("submission.csv") and sub.shape[1] == 2
assert list(sub.columns) == ["essay_id", "score"]
assert len(sub) == len(test_df)

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3883344064.py in <cell line: 0>()
     77 )
     78 
---> 79 pred_out = trainer.predict(test_ds)
     80 preds = pred_out.predictions
     81 

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in predict(self, test_dataset, ignore_keys, metric_key_prefix)
   4271         self._memory_tracker.start()
   4272 
-> 4273         test_dataloader = self.get_test_dataloader(test_dataset)
   4274         start_time = time.time()
   4275 

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in get_test_dataloader(self, test_dataset)
   1154                 `model.forward()` method are automatically removed. It must implement `__len__`.
   1155         """
-> 1156         return self._get_dataloader(
   1157             dataset=test_dataset,
   1158             description="test",

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in _get_dataloader(self, dataset, description, batch_size, sampler_fn, is_training, dataloader_key)
   1005         data_collator = self.data_collator
   1006         if is_datasets_available() and isinstance(dataset, datasets.Dataset):
-> 1007             dataset = self._remove_unused_columns(dataset, description=description)
   1008         else:
   1009             data_collator = self._get_collator_with_removed_columns(self.data_collator, description=description)

/usr/local/lib/python3.11/dist-packages/transformers/trainer.py in _remove_unused_columns(self, dataset, description)
    931         columns = [k for k in signature_columns if k in dataset.column_names]
    932         if len(columns) == 0:
--> 933             raise ValueError(
    934                 f"No columns in the dataset match the model's forward method signature: ({', '.join(signature_columns)}). "
    935                 f"The following columns have been ignored: [{', '.join(ignored_columns)}]. "

ValueError: No columns in the dataset match the model's forward method signature: (input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, labels, output_attentions, output_hidden_states, return_dict, label_ids, label, labels). The following columns have been ignored: [full_text, essay_id]. Please check the dataset and model. You may need to set `remove_unused_columns=False` in `TrainingArguments`.
