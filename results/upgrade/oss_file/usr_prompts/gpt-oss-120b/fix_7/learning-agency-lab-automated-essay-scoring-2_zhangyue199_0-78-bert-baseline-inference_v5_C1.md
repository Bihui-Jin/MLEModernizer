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
from datasets import Dataset
import pandas as pd
import numpy as np
import re
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
import torch




## === cell 1
model_path = "/kaggle/input/bert-baseline-train/output/bert-base"
MAX_LEN = 512  # BERT's max token length
test_df = pd.read_csv(
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
)




## === cell 2
def clean_text_series(series: pd.Series) -> pd.Series:
    series = series.str.replace(r"\s+", " ", regex=True)
    series = series.str.replace(r"[^a-zA-Z0-9]", " ", regex=True)
    return series.str.strip()


test_df["full_text"] = clean_text_series(test_df["full_text"])




## === cell 3
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, local_files_only=True
    )
except Exception:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoTokenizer.from_pretrained(
        "bert-base-uncased", num_labels=6, problem_type="single_label_classification"
    )  # original logic retained

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

import os

torch.set_num_threads(os.cpu_count() or 8)

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True  # enable cuDNN auto‑tuning




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1486199114.py in <cell line: 0>()
     11 
     12 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
---> 13 model.to(device)
     14 model.eval()
     15 

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in __getattr__(self, key)
   1098 
   1099         if key not in self.__dict__:
-> 1100             raise AttributeError(f"{self.__class__.__name__} has no attribute {key}")
   1101         else:
   1102             return super().__getattr__(key)

AttributeError: BertTokenizerFast has no attribute to

## === cell 4
encodings = tokenizer(
    test_df["full_text"].tolist(),
    padding="max_length",
    truncation=True,
    max_length=MAX_LEN,
    return_tensors="pt",
)

input_ids = encodings["input_ids"].to(device)
attention_mask = encodings["attention_mask"].to(device)

batch_size = 128  # fits typical GPU/CPU memory; adjust if needed
total_samples = input_ids.size(0)

scores_np = np.empty(total_samples, dtype=np.int32)

with torch.inference_mode():  # lightweight context for inference
    for start_idx in range(0, total_samples, batch_size):
        end_idx = min(start_idx + batch_size, total_samples)
        batch_input_ids = input_ids[start_idx:end_idx]
        batch_attention_mask = attention_mask[start_idx:end_idx]

        outputs = model(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
        )
        batch_scores = torch.argmax(outputs.logits, dim=1).cpu().numpy() + 1
        scores_np[start_idx:end_idx] = batch_scores

sub = test_df[["essay_id"]].copy()
sub["score"] = scores_np
sub.to_csv("submission.csv", index=False)
display(sub.head())

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/272404761.py in <cell line: 0>()
     22         batch_attention_mask = attention_mask[start_idx:end_idx]
     23 
---> 24         outputs = model(
     25             input_ids=batch_input_ids,
     26             attention_mask=batch_attention_mask,

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in __call__(self, text, text_pair, text_target, text_pair_target, add_special_tokens, padding, truncation, max_length, stride, is_split_into_words, pad_to_multiple_of, padding_side, return_tensors, return_token_type_ids, return_attention_mask, return_overflowing_tokens, return_special_tokens_mask, return_offsets_mapping, return_length, verbose, **kwargs)
   2847         all_kwargs.update(kwargs)
   2848         if text is None and text_target is None:
-> 2849             raise ValueError("You need to specify either `text` or `text_target`.")
   2850         if text is not None:
   2851             # The context manager will send the inputs as normal texts and not text_target, but we shouldn't change the

ValueError: You need to specify either `text` or `text_target`.
