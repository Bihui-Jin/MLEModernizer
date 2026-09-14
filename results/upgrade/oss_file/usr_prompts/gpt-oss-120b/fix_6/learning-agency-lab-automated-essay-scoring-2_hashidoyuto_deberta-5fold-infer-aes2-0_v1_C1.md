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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import warnings

warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import random

from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.benchmark = True




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
class PATHS:
    base = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2"
    train_path = os.path.join(base, "train.csv")
    test_path = os.path.join(base, "test.csv")
    sample_submission_path = os.path.join(base, "sample_submission.csv")




## === cell 2
class CFG:
    model_name = "distilbert-base-uncased"
    max_length = 512
    num_labels = 6
    epochs = 2
    per_device_train_batch_size = 8
    per_device_eval_batch_size = 8
    learning_rate = 2e-5
    seed = 42


torch.manual_seed(CFG.seed)
np.random.seed(CFG.seed)
random.seed(CFG.seed)



## === cell 3
train_df = pd.read_csv(PATHS.train_path)
test_df = pd.read_csv(PATHS.test_path)

train_df["label"] = train_df["score"] - 1

train_df, val_df = train_test_split(
    train_df,
    test_size=0.1,
    stratify=train_df["label"],
    random_state=CFG.seed,
)



## === cell 4
tokenizer = AutoTokenizer.from_pretrained(CFG.model_name, use_fast=True)


class EssayDataset(TorchDataset):
    """Dataset that stores pre‑tokenised tensors."""

    def __init__(self, encodings, labels=None):
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.labels = labels

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


def prepare_dataset(df, is_test=False):
    encodings = tokenizer(
        df["full_text"].tolist(),
        truncation=True,
        max_length=CFG.max_length,
        padding="max_length",
        return_tensors="pt",
        num_workers=4,  # <-- added for speed
    )
    if is_test:
        return EssayDataset(encodings)
    else:
        labels = torch.tensor(df["label"].values, dtype=torch.long)
        return EssayDataset(encodings, labels)


train_dataset = prepare_dataset(train_df, is_test=False)
val_dataset = prepare_dataset(val_df, is_test=False)
test_dataset = prepare_dataset(test_df, is_test=True)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1697844405.py in <cell line: 0>()
     42 
     43 
---> 44 train_dataset = prepare_dataset(train_df, is_test=False)
     45 val_dataset = prepare_dataset(val_df, is_test=False)
     46 test_dataset = prepare_dataset(test_df, is_test=True)

/tmp/ipykernel_11/1697844405.py in prepare_dataset(df, is_test)
     27 def prepare_dataset(df, is_test=False):
     28     # Parallel tokenization via num_workers (uses all CPU cores efficiently)
---> 29     encodings = tokenizer(
     30         df["full_text"].tolist(),
     31         truncation=True,

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in __call__(self, text, text_pair, text_target, text_pair_target, add_special_tokens, padding, truncation, max_length, stride, is_split_into_words, pad_to_multiple_of, padding_side, return_tensors, return_token_type_ids, return_attention_mask, return_overflowing_tokens, return_special_tokens_mask, return_offsets_mapping, return_length, verbose, **kwargs)
   2853             if not self._in_target_context_manager:
   2854                 self._switch_to_input_mode()
-> 2855             encodings = self._call_one(text=text, text_pair=text_pair, **all_kwargs)
   2856         if text_target is not None:
   2857             self._switch_to_target_mode()

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in _call_one(self, text, text_pair, add_special_tokens, padding, truncation, max_length, stride, is_split_into_words, pad_to_multiple_of, padding_side, return_tensors, return_token_type_ids, return_attention_mask, return_overflowing_tokens, return_special_tokens_mask, return_offsets_mapping, return_length, verbose, split_special_tokens, **kwargs)
   2941                 )
   2942             batch_text_or_text_pairs = list(zip(text, text_pair)) if text_pair is not None else text
-> 2943             return self.batch_encode_plus(
   2944                 batch_text_or_text_pairs=batch_text_or_text_pairs,
   2945                 add_special_tokens=add_special_tokens,

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in batch_encode_plus(self, batch_text_or_text_pairs, add_special_tokens, padding, truncation, max_length, stride, is_split_into_words, pad_to_multiple_of, padding_side, return_tensors, return_token_type_ids, return_attention_mask, return_overflowing_tokens, return_special_tokens_mask, return_offsets_mapping, return_length, verbose, split_special_tokens, **kwargs)
   3142         )
   3143 
-> 3144         return self._batch_encode_plus(
   3145             batch_text_or_text_pairs=batch_text_or_text_pairs,
   3146             add_special_tokens=add_special_tokens,

TypeError: PreTrainedTokenizerFast._batch_encode_plus() got an unexpected keyword argument 'num_workers'

## === cell 5
model = AutoModelForSequenceClassification.from_pretrained(
    CFG.model_name,
    num_labels=CFG.num_labels,
)

training_args = TrainingArguments(
    output_dir="./model_output",
    num_train_epochs=CFG.epochs,
    per_device_train_batch_size=CFG.per_device_train_batch_size,
    per_device_eval_batch_size=CFG.per_device_eval_batch_size,
    learning_rate=CFG.learning_rate,
    eval_strategy="no",
    save_strategy="no",
    logging_strategy="no",
    load_best_model_at_end=False,
    fp16=torch.cuda.is_available(),
    report_to="none",
    seed=CFG.seed,
    dataloader_num_workers=4,  # increased to match tokenization workers
    dataloader_pin_memory=True,
    disable_tqdm=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/834153159.py in <cell line: 0>()
     25     model=model,
     26     args=training_args,
---> 27     train_dataset=train_dataset,
     28     tokenizer=tokenizer,
     29 )

NameError: name 'train_dataset' is not defined

## === cell 6
trainer.train()



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3352579090.py in <cell line: 0>()
----> 1 trainer.train()
      2 

NameError: name 'trainer' is not defined

## === cell 7
preds = trainer.predict(test_dataset).predictions
pred_labels = np.argmax(preds, axis=1) + 1  # back to 1‑6 scale

submission = pd.DataFrame(
    {
        "essay_id": test_df["essay_id"],
        "score": pred_labels,
    }
)
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3175501451.py in <cell line: 0>()
----> 1 preds = trainer.predict(test_dataset).predictions
      2 pred_labels = np.argmax(preds, axis=1) + 1  # back to 1‑6 scale
      3 
      4 submission = pd.DataFrame(
      5     {

NameError: name 'trainer' is not defined

## === cell 8
submission.head()

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3365464162.py in <cell line: 0>()
----> 1 submission.head()

NameError: name 'submission' is not defined
