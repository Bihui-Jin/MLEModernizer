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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

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

torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

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

test_ds = Dataset.from_pandas(test_df[["essay_id", "full_text"]], preserve_index=False)


def _tokenize_batch(batch):
    return tokenizer(
        batch["full_text"],
        padding=False,
        truncation=True,
        max_length=MAX_LEN,
    )


num_proc = min(4, (os.cpu_count() or 2))
test_ds = test_ds.map(
    _tokenize_batch,
    batched=True,
    num_proc=num_proc,
    remove_columns=["full_text"],
    desc="Tokenizing test",
)

test_ds.set_format(
    type="torch",
    columns=[
        c
        for c in ["input_ids", "attention_mask", "token_type_ids"]
        if c in test_ds.column_names
    ],
    output_all_columns=True,
)

from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer, padding="longest", return_tensors="pt"
)

batch_size = 32
num_workers = min(4, (os.cpu_count() or 2))

pin_memory = torch.cuda.is_available()

loader = DataLoader(
    test_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
    collate_fn=data_collator,
)

pred_batches = []
with torch.inference_mode():
    for batch in loader:
        batch = {k: v.to(device, non_blocking=pin_memory) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        pred_batches.append(logits.detach().cpu())

preds = torch.cat(pred_batches, dim=0).numpy()

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
/tmp/ipykernel_11/59287853.py in <cell line: 0>()
     96 pred_batches = []
     97 with torch.inference_mode():
---> 98     for batch in loader:
     99         # batch from collator contains only model inputs; essay_id is excluded by collator.
    100         batch = {k: v.to(device, non_blocking=pin_memory) for k, v in batch.items()}

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
   1478                 del self._task_info[idx]
   1479                 self._rcvd_idx += 1
-> 1480                 return self._process_data(data)
   1481 
   1482     def _try_put_index(self):

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _process_data(self, data)
   1503         self._try_put_index()
   1504         if isinstance(data, ExceptionWrapper):
-> 1505             data.reraise()
   1506         return data
   1507 

/usr/local/lib/python3.11/dist-packages/torch/_utils.py in reraise(self)
    731             # instantiate since we don't know how to
    732             raise RuntimeError(msg) from None
--> 733         raise exception
    734 
    735 

ValueError: Caught ValueError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py", line 767, in convert_to_tensors
    tensor = as_tensor(value)
             ^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py", line 729, in as_tensor
    return torch.tensor(value)
           ^^^^^^^^^^^^^^^^^^^
ValueError: too many dimensions 'str'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 55, in fetch
    return self.collate_fn(data)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/data/data_collator.py", line 272, in __call__
    batch = pad_without_fast_tokenizer_warning(
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/data/data_collator.py", line 67, in pad_without_fast_tokenizer_warning
    padded = tokenizer.pad(*pad_args, **pad_kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py", line 3374, in pad
    return BatchEncoding(batch_outputs, tensor_type=return_tensors)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py", line 240, in __init__
    self.convert_to_tensors(tensor_type=tensor_type, prepend_batch_axis=prepend_batch_axis)
  File "/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py", line 783, in convert_to_tensors
    raise ValueError(
ValueError: Unable to create tensor, you should probably activate truncation and/or padding with 'padding=True' 'truncation=True' to have batched tensors with the same length. Perhaps your features (`essay_id` in this case) have excessive nesting (inputs type `list` where type `int` is expected).
