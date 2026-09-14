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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
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

0.8030895862887903

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.0) has done: 'I fix the environment/import crash in the first cell by forcing the pure-Python protobuf implementation (this avoids the `MessageFactory.GetPrototype` error that happens with some package combinations). Then I fix model/tokenizer loading so a local Kaggle dataset path like `/kaggle/input/...` is treated as a filesystem directory (not a Hub repo id), which resolves the `HFValidationError`. Finally, I make the inference-to-submission step correct for a sequence-classification model by converting logits to class predictions (1–6) rather than clipping raw logits; this preserves the intended evaluation semantics and produces a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import random
import pandas as pd
import numpy as np

from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset
from scipy.special import softmax
import torch



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
MODEL_NAME = "/kaggle/input/f-tfm-small/funnel-small-ft_ver4"
MAX_LENGTH = 3072
BATCH_SIZE = 2



## === cell 2
df_train = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
)
df_test = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
)
sample_submission = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv"
)



## === cell 3
if not os.path.isdir(MODEL_NAME):
    raise FileNotFoundError(
        f"MODEL_NAME path does not exist or is not a directory: {MODEL_NAME}"
    )

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, local_files_only=True
)

model.eval()
if torch.cuda.is_available():
    model.cuda()




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/620466539.py in <cell line: 0>()
      2 # Ensure Transformers treats MODEL_NAME as a local directory, not a hub repo_id.
      3 if not os.path.isdir(MODEL_NAME):
----> 4     raise FileNotFoundError(
      5         f"MODEL_NAME path does not exist or is not a directory: {MODEL_NAME}"
      6     )

FileNotFoundError: MODEL_NAME path does not exist or is not a directory: /kaggle/input/f-tfm-small/funnel-small-ft_ver4

## === cell 4
def tokenize(sample):
    return tokenizer(sample["full_text"], max_length=MAX_LENGTH, truncation=True)




## === cell 5
train = Dataset.from_pandas(df_train)

test = (
    Dataset.from_pandas(df_test)
    .map(tokenize, batched=True)
    .remove_columns(["essay_id", "full_text"])
)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3160323468.py in <cell line: 0>()
      4 test = (
      5     Dataset.from_pandas(df_test)
----> 6     .map(tokenize, batched=True)
      7     .remove_columns(["essay_id", "full_text"])
      8 )

/usr/local/lib/python3.11/dist-packages/datasets/arrow_dataset.py in wrapper(*args, **kwargs)
    560         }
    561         # apply actual function
--> 562         out: Union["Dataset", "DatasetDict"] = func(self, *args, **kwargs)
    563         datasets: list["Dataset"] = list(out.values()) if isinstance(out, dict) else [out]
    564         # re-apply format to the output

/usr/local/lib/python3.11/dist-packages/datasets/arrow_dataset.py in map(self, function, with_indices, with_rank, input_columns, batched, batch_size, drop_last_batch, remove_columns, keep_in_memory, load_from_cache_file, cache_file_name, writer_batch_size, features, disable_nullable, fn_kwargs, num_proc, suffix_template, new_fingerprint, desc, try_original_type)
   3339                 else:
   3340                     for unprocessed_kwargs in unprocessed_kwargs_per_job:
-> 3341                         for rank, done, content in Dataset._map_single(**unprocessed_kwargs):
   3342                             check_if_shard_done(rank, done, content)
   3343 

/usr/local/lib/python3.11/dist-packages/datasets/arrow_dataset.py in _map_single(shard, function, with_indices, with_rank, input_columns, batched, batch_size, drop_last_batch, remove_columns, keep_in_memory, cache_file_name, writer_batch_size, features, disable_nullable, fn_kwargs, new_fingerprint, rank, offset, try_original_type)
   3695                 else:
   3696                     _time = time.time()
-> 3697                     for i, batch in iter_outputs(shard_iterable):
   3698                         num_examples_in_batch = len(i)
   3699                         if update_data:

/usr/local/lib/python3.11/dist-packages/datasets/arrow_dataset.py in iter_outputs(shard_iterable)
   3645             else:
   3646                 for i, example in shard_iterable:
-> 3647                     yield i, apply_function(example, i, offset=offset)
   3648 
   3649         num_examples_progress_update = 0

/usr/local/lib/python3.11/dist-packages/datasets/arrow_dataset.py in apply_function(pa_inputs, indices, offset)
   3568             """Utility to apply the function on a selection of columns."""
   3569             inputs, fn_args, additional_args, fn_kwargs = prepare_inputs(pa_inputs, indices, offset=offset)
-> 3570             processed_inputs = function(*fn_args, *additional_args, **fn_kwargs)
   3571             return prepare_outputs(pa_inputs, inputs, processed_inputs)
   3572 

/tmp/ipykernel_11/2095828209.py in tokenize(sample)
      1 def tokenize(sample):
----> 2     return tokenizer(sample["full_text"], max_length=MAX_LENGTH, truncation=True)
      3 
      4 

NameError: name 'tokenizer' is not defined

## === cell 6
predictions = []

args = TrainingArguments(
    output_dir=".",
    per_device_eval_batch_size=BATCH_SIZE,
    report_to="none",
)

trainer = Trainer(
    args=args,
    model=model,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
)

preds = trainer.predict(test).predictions  # logits: (n_examples, n_labels)
predictions.append(preds)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/928648025.py in <cell line: 0>()
      9 trainer = Trainer(
     10     args=args,
---> 11     model=model,
     12     tokenizer=tokenizer,
     13     data_collator=DataCollatorWithPadding(tokenizer),

NameError: name 'model' is not defined

## === cell 7
preds = np.mean(predictions, axis=0)



## === cell 8
if preds.ndim == 2:
    pred_labels = np.argmax(preds, axis=1)  # 0..num_labels-1
    df_test["score"] = (pred_labels + 1).astype(int)
else:
    df_test["score"] = (np.clip(preds, 0, 5).round(0) + 1).astype(int)

submission = df_test[["essay_id", "score"]]

submission = sample_submission[["essay_id"]].merge(
    submission, on="essay_id", how="left"
)
if submission["score"].isna().any():
    raise ValueError(
        "Some test essay_ids are missing predictions after merge; check alignment."
    )
submission["score"] = submission["score"].astype(int)

submission.to_csv("submission.csv", index=False)
print(submission.head())
print(f"Wrote submission.csv with shape: {submission.shape}")
