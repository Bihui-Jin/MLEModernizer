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

0.7527003751565937

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

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch

torch.manual_seed(42)
np.random.seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
device



## === cell 1
TEST_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
if not os.path.exists(TEST_PATH):
    TEST_PATH = "/kaggle/input/test.csv"

TRAIN_PATH = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
if not os.path.exists(TRAIN_PATH):
    TRAIN_PATH = "/kaggle/input/train.csv"

SAMPLE_SUB_PATH = (
    "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv"
)
if not os.path.exists(SAMPLE_SUB_PATH):
    SAMPLE_SUB_PATH = "/kaggle/input/sample_submission.csv"

test_data = pd.read_csv(TEST_PATH)
train_data = pd.read_csv(TRAIN_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

test_data.head(), test_data.shape, train_data.shape, sample_sub.shape



## === cell 2
from pathlib import Path


def find_local_hf_model_dir(preferred_names):
    """
    Try common Kaggle/HF cache locations for a directory containing config.json.
    Returns a local path string or None.
    """
    candidates = []
    roots = [
        Path("/kaggle/input"),
        Path("/kaggle/working"),
        Path.home() / ".cache" / "huggingface" / "hub",
        Path("/root/.cache/huggingface/hub"),
    ]
    for root in roots:
        if not root.exists():
            continue
        for name in preferred_names:
            candidates.append(root / name)
        for name in preferred_names:
            if "/" in name:
                org, repo = name.split("/", 1)
                candidates.append(root / f"models--{org}--{repo}" / "snapshots")
    for c in candidates:
        try:
            if c.is_dir():
                if c.name == "snapshots":
                    for snap in sorted(c.iterdir()):
                        if (snap / "config.json").exists():
                            return str(snap)
                else:
                    if (c / "config.json").exists():
                        return str(c)
        except Exception:
            pass
    return None


MODEL_NAME = "microsoft/deberta-v3-base"
local_model_dir = find_local_hf_model_dir([MODEL_NAME])

local_model_dir



## === cell 3
use_transformers = False
tokenizer = None
model = None

if local_model_dir is not None:
    try:
        from datasets import Dataset
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            Trainer,
            TrainingArguments,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            local_model_dir, local_files_only=True
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            local_model_dir,
            num_labels=6,
            local_files_only=True,
        ).to(device)
        model.eval()
        use_transformers = True
    except Exception as e:
        use_transformers = False
        tokenizer = None
        model = None
        print(
            "Falling back to sklearn baseline due to Transformers load error:", repr(e)
        )

use_transformers



## === cell 4
predicted_scores = None

if use_transformers:
    from datasets import Dataset
    from transformers import Trainer, TrainingArguments

    max_len = min(1024, int(getattr(tokenizer, "model_max_length", 1024) or 1024))

    test_encodings = tokenizer(
        test_data["full_text"].astype(str).tolist(),
        truncation=True,
        padding=True,
        max_length=max_len,
    )

    test_dataset = Dataset.from_dict(test_encodings)
    test_dataset = test_dataset.add_column(
        "essay_id", test_data["essay_id"].astype(str).tolist()
    )

    predict_args = TrainingArguments(
        output_dir=".",
        per_device_eval_batch_size=4,
        dataloader_drop_last=False,
        report_to="none",
    )

    trainer = Trainer(model=model, args=predict_args, tokenizer=tokenizer)

    predictions = trainer.predict(test_dataset)
    logits = predictions.predictions
    pred_class = np.argmax(logits, axis=-1).astype(np.int32)
    predicted_scores = (pred_class + 1).astype(np.int32)
else:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge

    X_train_text = train_data["full_text"].astype(str).values
    y_train = train_data["score"].astype(np.float32).values
    X_test_text = test_data["full_text"].astype(str).values

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=200_000,
        min_df=2,
        strip_accents="unicode",
        lowercase=True,
    )
    Xtr = vectorizer.fit_transform(X_train_text)
    Xte = vectorizer.transform(X_test_text)

    reg = Ridge(alpha=1.0, random_state=42)
    reg.fit(Xtr, y_train)
    y_pred = reg.predict(Xte)

    predicted_scores = np.rint(y_pred).astype(np.int32)

predicted_scores[:10], int(predicted_scores.min()), int(predicted_scores.max())



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py in _solve_sparse_cg(X, y, alpha, max_iter, tol, verbose, X_offset, X_scale, sample_weight_sqrt)
    114             try:
--> 115                 coef, info = sp_linalg.cg(C, y_column, tol=tol, atol="legacy")
    116             except TypeError:

TypeError: cg() got an unexpected keyword argument 'tol'

During handling of the above exception, another exception occurred:

TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3972900223.py in <cell line: 0>()
     54 
     55     reg = Ridge(alpha=1.0, random_state=42)
---> 56     reg.fit(Xtr, y_train)
     57     y_pred = reg.predict(Xte)
     58 

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py in fit(self, X, y, sample_weight)
   1132             y_numeric=True,
   1133         )
-> 1134         return super().fit(X, y, sample_weight=sample_weight)
   1135 
   1136 

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py in fit(self, X, y, sample_weight)
    898                 params = {}
    899 
--> 900             self.coef_, self.n_iter_ = _ridge_regression(
    901                 X,
    902                 y,

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py in _ridge_regression(X, y, alpha, sample_weight, solver, max_iter, tol, verbose, positive, random_state, return_n_iter, return_intercept, X_scale, X_offset, check_input, fit_intercept)
    669     n_iter = None
    670     if solver == "sparse_cg":
--> 671         coef = _solve_sparse_cg(
    672             X,
    673             y,

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py in _solve_sparse_cg(X, y, alpha, max_iter, tol, verbose, X_offset, X_scale, sample_weight_sqrt)
    116             except TypeError:
    117                 # old scipy
--> 118                 coef, info = sp_linalg.cg(C, y_column, tol=tol)
    119             coefs[i] = X1.rmatvec(coef)
    120         else:

TypeError: cg() got an unexpected keyword argument 'tol'

## === cell 5
submission = pd.DataFrame(
    {
        "essay_id": test_data["essay_id"].astype(str).values,
        "score": predicted_scores,
    }
)

submission["score"] = submission["score"].clip(1, 6).astype(np.int32)

if "essay_id" in sample_sub.columns and sample_sub.shape[0] == submission.shape[0]:
    submission = sample_sub[["essay_id"]].merge(submission, on="essay_id", how="left")
    submission["score"] = submission["score"].fillna(3).clip(1, 6).astype(np.int32)

submission.to_csv("submission.csv", index=False)
submission.head()



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1855051797.py in <cell line: 0>()
      7 
      8 # Ensure valid range and integer dtype
----> 9 submission["score"] = submission["score"].clip(1, 6).astype(np.int32)
     10 
     11 # Ensure ordering matches sample_submission if needed (safety against any accidental reordering)

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in astype(self, dtype, copy, errors)
   6641         else:
   6642             # else, only a single dtype is given
-> 6643             new_data = self._mgr.astype(dtype=dtype, copy=copy, errors=errors)
   6644             res = self._constructor_from_mgr(new_data, axes=new_data.axes)
   6645             return res.__finalize__(self, method="astype")

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in astype(self, dtype, copy, errors)
    428             copy = False
    429 
--> 430         return self.apply(
    431             "astype",
    432             dtype=dtype,

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in apply(self, f, align_keys, **kwargs)
    361                 applied = b.apply(f, **kwargs)
    362             else:
--> 363                 applied = getattr(b, f)(**kwargs)
    364             result_blocks = extend_blocks(applied, result_blocks)
    365 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/blocks.py in astype(self, dtype, copy, errors, using_cow, squeeze)
    756             values = values[0, :]  # type: ignore[call-overload]
    757 
--> 758         new_values = astype_array_safe(values, dtype, copy=copy, errors=errors)
    759 
    760         new_values = maybe_coerce_values(new_values)

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in astype_array_safe(values, dtype, copy, errors)
    235 
    236     try:
--> 237         new_values = astype_array(values, dtype, copy=copy)
    238     except (ValueError, TypeError):
    239         # e.g. _astype_nansafe can fail on object-dtype of strings

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in astype_array(values, dtype, copy)
    180 
    181     else:
--> 182         values = _astype_nansafe(values, dtype, copy=copy)
    183 
    184     # in pandas we don't store numpy str dtypes, so convert to object

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in _astype_nansafe(arr, dtype, copy, skipna)
    131     if copy or arr.dtype == object or dtype == object:
    132         # Explicit copy, or required since NumPy can't view from / to object.
--> 133         return arr.astype(dtype, copy=True)
    134 
    135     return arr.astype(dtype, copy=copy)

TypeError: int() argument must be a string, a bytes-like object or a real number, not 'NoneType'

## === cell 6
assert (
    submission.shape[0] == test_data.shape[0]
), "Submission row count must match test set."
assert list(submission.columns) == [
    "essay_id",
    "score",
], "Submission must have columns: essay_id, score"
assert submission["score"].between(1, 6).all(), "Scores must be in [1, 6]"
assert submission["essay_id"].astype(str).isna().sum() == 0, "essay_id must be non-null"
submission.describe(include="all")

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_11/2152251145.py in <cell line: 0>()
      6     "score",
      7 ], "Submission must have columns: essay_id, score"
----> 8 assert submission["score"].between(1, 6).all(), "Scores must be in [1, 6]"
      9 assert submission["essay_id"].astype(str).isna().sum() == 0, "essay_id must be non-null"
     10 submission.describe(include="all")

AssertionError: Scores must be in [1, 6]
