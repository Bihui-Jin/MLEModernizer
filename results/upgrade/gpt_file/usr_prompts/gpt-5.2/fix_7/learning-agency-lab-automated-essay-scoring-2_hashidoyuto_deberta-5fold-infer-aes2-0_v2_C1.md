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
os.environ.setdefault("HF_DATASETS_DISABLE_PROTOBUF", "1")



## === cell 1
import random
import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.simplefilter("ignore")


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(42)

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from datasets import Dataset




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
class PATHS:
    test_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/test.csv"
    train_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/train.csv"
    model_dir = "/kaggle/input/debert-v3-base-for-aes2-0/"

    sample_sub_path = "/kaggle/input/learning-agency-lab-automated-essay-scoring-2/sample_submission.csv"




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


def find_any_local_hf_model_dir(search_root="/kaggle/input"):
    """
    In offline Kaggle images, we cannot download HF models.
    Try to find any directory under /kaggle/input that looks like an HF model
    (contains config.json and some model weight file).
    """
    root = Path(search_root)
    if not root.exists():
        return None

    weight_names = {
        "pytorch_model.bin",
        "model.safetensors",
        "pytorch_model.safetensors",
    }
    for config_path in root.rglob("config.json"):
        try:
            parent = config_path.parent
            if any((parent / w).exists() for w in weight_names):
                return str(parent)
        except Exception:
            continue
    return None


def pick_local_model_id(preferred_local_dir: str) -> str | None:
    """
    Choose a locally available HF model dir. If none exists, return None to trigger
    a guaranteed offline baseline.
    """
    resolved = resolve_model_path(preferred_local_dir)
    if os.path.isdir(resolved) and os.path.isfile(
        os.path.join(resolved, "config.json")
    ):
        return resolved

    discovered = find_any_local_hf_model_dir("/kaggle/input")
    if discovered is not None:
        return discovered

    return None




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

print("Resolved model_path:", model_path)
print("Test shape:", test.shape)
print("Test columns:", list(test.columns))



## === cell 7
use_transformer = model_path is not None
pred_labels = None  # Fix: ensure always defined for submission writing.

if use_transformer:
    tokenize = Tokenize(test, model_path)
    tokenized_test, tokenizer = tokenize()

    print("Using local HF model:", model_path)
    print("Tokenized test columns:", tokenized_test.column_names)
    print("Tokenized test rows:", len(tokenized_test))

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=CFG.num_labels,
        local_files_only=True,
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

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

    predictions = trainer.predict(tokenized_test).predictions
    print("Predictions shape:", predictions.shape)

    pred_labels = predictions.argmax(axis=1) + 1
    pred_labels = np.clip(pred_labels, 1, 6).astype(int)

else:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge
    from scipy.sparse import hstack

    train = pd.read_csv(PATHS.train_path)
    X_train_text = train["full_text"].astype(str).values
    y_train = train["score"].values.astype(float)
    X_test_text = test["full_text"].astype(str).values

    tfidf_word = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=150_000,
        strip_accents="unicode",
        lowercase=True,
    )
    tfidf_char = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_features=200_000,
        lowercase=False,
    )

    Xw_tr = tfidf_word.fit_transform(X_train_text)
    Xc_tr = tfidf_char.fit_transform(X_train_text)
    X_tr = hstack([Xw_tr, Xc_tr]).tocsr()

    Xw_te = tfidf_word.transform(X_test_text)
    Xc_te = tfidf_char.transform(X_test_text)
    X_te = hstack([Xw_te, Xc_te]).tocsr()

    reg = Ridge(alpha=1.0, random_state=42, solver="sag")
    reg.fit(X_tr, y_train)
    pred = reg.predict(X_te)

    pred_labels = np.rint(pred).astype(int)
    pred_labels = np.clip(pred_labels, 1, 6).astype(int)

    print("Used offline TF-IDF + Ridge baseline. Pred label distribution (counts):")
    print(pd.Series(pred_labels).value_counts().sort_index())



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py in _solve_sparse_cg(X, y, alpha, max_iter, tol, verbose, X_offset, X_scale, sample_weight_sqrt)
    114             try:
--> 115                 coef, info = sp_linalg.cg(C, y_column, tol=tol, atol="legacy")
    116             except TypeError:

TypeError: cg() got an unexpected keyword argument 'tol'

During handling of the above exception, another exception occurred:

TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3055388843.py in <cell line: 0>()
     75     # Use a dense-compatible solver that avoids cg entirely.
     76     reg = Ridge(alpha=1.0, random_state=42, solver="sag")
---> 77     reg.fit(X_tr, y_train)
     78     pred = reg.predict(X_te)
     79 

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

## === cell 8
sample_sub = pd.read_csv(PATHS.sample_sub_path)

pred_df = pd.DataFrame(
    {"essay_id": test["essay_id"].values, "score": pred_labels.astype(int)}
)
pred_df = pred_df.drop_duplicates(subset=["essay_id"])

submission = sample_sub[["essay_id"]].merge(pred_df, on="essay_id", how="left")

if submission["score"].isna().any():
    submission["score"] = submission["score"].fillna(3).astype(int)

submission["score"] = submission["score"].clip(1, 6).astype(int)

submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Score distribution:\n", submission["score"].value_counts().sort_index())
print("Submission columns:", list(submission.columns))
print("Any NaNs:", submission.isna().any().to_dict())

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3003467598.py in <cell line: 0>()
      3 
      4 pred_df = pd.DataFrame(
----> 5     {"essay_id": test["essay_id"].values, "score": pred_labels.astype(int)}
      6 )
      7 pred_df = pred_df.drop_duplicates(subset=["essay_id"])

AttributeError: 'NoneType' object has no attribute 'astype'
