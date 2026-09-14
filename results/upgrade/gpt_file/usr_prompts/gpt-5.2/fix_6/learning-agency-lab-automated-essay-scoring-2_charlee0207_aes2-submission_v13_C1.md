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

3.13

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

0.7600545627569122

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import torch

print(f"PyTorch version: {torch.__version__}")

import transformers

print(f"Hugging Face Transformers version: {transformers.__version__}")

import datasets

print(f"Hugging Face Datasets version: {datasets.__version__}")



## === cell 1
import os
import numpy as np
import pandas as pd
from pathlib import Path

MODEL_NAME = "bert-base-cased"  # kept for fallback only
INPUT_DIR = "/kaggle/input/"
MAX_LENGTH = 1024
RANDOM_SEED = 42
SUBMISSION = 1

has_cuda = torch.cuda.is_available()
device = "cuda:0" if has_cuda else "cpu"
print("device:", device)
if has_cuda:
    print("cuda current_device:", torch.cuda.current_device())
else:
    print("CUDA not available; running on CPU.")

if not Path(INPUT_DIR).exists():
    INPUT_DIR = "/kaggle/data/"
print("Using INPUT_DIR:", INPUT_DIR)

COMP_DIR = Path(INPUT_DIR) / "learning-agency-lab-automated-essay-scoring-2"
print("Competition dir exists:", COMP_DIR.exists())

CANDIDATE_MODEL_DIRS = [
    Path(INPUT_DIR) / "aes2-persuade-bertbase-2ep-results" / "checkpoint-7000",
    Path(INPUT_DIR) / "aes2-persuade-bertbase-2ep-results",
    COMP_DIR,
    Path(INPUT_DIR),
]


def _looks_like_hf_dir(p: Path) -> bool:
    if not p.exists() or not p.is_dir():
        return False
    expected_any = [
        "config.json",
        "model.safetensors",
        "pytorch_model.bin",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
        "merges.txt",
        "special_tokens_map.json",
    ]
    return any((p / f).exists() for f in expected_any)


LOCAL_MODEL_DIR = None
for p in CANDIDATE_MODEL_DIRS:
    if _looks_like_hf_dir(p):
        LOCAL_MODEL_DIR = p
        break

print("Detected LOCAL_MODEL_DIR:", str(LOCAL_MODEL_DIR) if LOCAL_MODEL_DIR else None)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)



## === cell 2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

train_path = str(COMP_DIR / "train.csv")
test_path = str(COMP_DIR / "test.csv")
sample_path = str(COMP_DIR / "sample_submission.csv")

if not Path(train_path).exists():
    train_path = os.path.join(INPUT_DIR, "train.csv")
if not Path(test_path).exists():
    test_path = os.path.join(INPUT_DIR, "test.csv")
if not Path(sample_path).exists():
    sample_path = os.path.join(INPUT_DIR, "sample_submission.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
print("Read train.csv successfully:", train_df.shape)
print("Read test.csv successfully:", test_df.shape)

X_train_text = train_df["full_text"].astype(str).fillna("").tolist()
y_train = train_df["score"].astype(float).values
X_test_text = test_df["full_text"].astype(str).fillna("").tolist()

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    strip_accents="unicode",
    lowercase=True,
    sublinear_tf=True,
)
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)
print("TF-IDF shapes:", X_train.shape, X_test.shape)

model = Ridge(alpha=1.0, random_state=RANDOM_SEED)
model.fit(X_train, y_train)

preds_cont = model.predict(X_test)
submission_preds = np.rint(preds_cont).astype(int)
submission_preds = np.clip(submission_preds, 1, 6)

print("Predict successfully. Pred shape:", submission_preds.shape)
print(
    "Pred value counts:",
    pd.Series(submission_preds).value_counts().sort_index().to_dict(),
)

submission_csv = pd.DataFrame(
    {"essay_id": test_df["essay_id"].values, "score": submission_preds}
)
submission_csv = submission_csv[["essay_id", "score"]]
submission_csv.to_csv("submission.csv", index=False)
print("Wrote submission.csv:", submission_csv.shape)
print(submission_csv.head())

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_ridge.py in _solve_sparse_cg(X, y, alpha, max_iter, tol, verbose, X_offset, X_scale, sample_weight_sqrt)
    114             try:
--> 115                 coef, info = sp_linalg.cg(C, y_column, tol=tol, atol="legacy")
    116             except TypeError:

TypeError: cg() got an unexpected keyword argument 'tol'

During handling of the above exception, another exception occurred:

TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_56/3481302423.py in <cell line: 0>()
     40 
     41 model = Ridge(alpha=1.0, random_state=RANDOM_SEED)
---> 42 model.fit(X_train, y_train)
     43 
     44 preds_cont = model.predict(X_test)

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
