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
Given a dataset of comments from Wikipedia's talk page edits, predict the probability of each comment being toxic.

## Metric
Mean column-wise ROC AUC; the average of the individual AUCs of each predicted column.

## Submission Format
For each `id` in the test set, you must predict a probability for each of the six possible types of comment toxicity (toxic, severe_toxic, obscene, threat, insult, identity_hate). The columns must be in the same order as shown below. The file should contain a header and have the following format:

```
id,toxic,severe_toxic,obscene,threat,insult,identity_hate
00001cee341fdb12,0.5,0.5,0.5,0.5,0.5,0.5
0000247867823ef7,0.5,0.5,0.5,0.5,0.5,0.5
etc.
```

## Dataset 
- **train.csv** - the training set, contains comments with their binary labels
- **test.csv** - the test set, you must predict the toxicity probabilities for these comments.
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (69 lines)
            sample_submission.csv (153165 lines)
            sample_submission.csv.zip (1.5 MB)
            test.csv (552889 lines)
            test.csv.zip (24.6 MB)
            train.csv (561809 lines)
            train.csv.zip (27.7 MB)
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
        input/
            description.md (69 lines)
            sample_submission.csv (153165 lines)
            sample_submission.csv.zip (1.5 MB)
            test.csv (552889 lines)
            test.csv.zip (24.6 MB)
            train.csv (561809 lines)
            train.csv.zip (27.7 MB)
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
        working/
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
```

-> data/jigsaw-toxic-comment-classification-challenge/sample_submission.csv has 153164 rows and 7 columns.
The columns are: id, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/jigsaw-toxic-comment-classification-challenge/test.csv has 552888 rows and 2 columns.
The columns are: id, comment_text

-> data/jigsaw-toxic-comment-classification-challenge/train.csv has 561808 rows and 8 columns.
The columns are: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/sample_submission.csv has 153164 rows and 7 columns.
The columns are: id, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/test.csv has 552888 rows and 2 columns.
The columns are: id, comment_text

-> data/train.csv has 561808 rows and 8 columns.
The columns are: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> (stopped after 10 files for performance)

# 5. Target score

0.9863068319863196

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.5) has done: 'The failure is because the notebook tries to ensemble submissions from other Kaggle datasets that are not present in your `/kaggle/input` tree, so the first `read_csv` raises `FileNotFoundError` and all later variables are undefined. I keep the ensemble core logic (weighted averaging) but make it robust: dynamically discover which candidate submission files actually exist, load only those, and compute a properly normalized weighted mean. To ensure a valid end-to-end run and a correct `.csv` output, I also align/merge predictions by `id` against `sample_submission.csv`, enforce the required column order, clip probabilities to `[0,1]`, and always write `submission.csv`. If none of the external submissions exist, the code fall back to using `sample_submission.csv` (0.5s) so it still produces a valid submission file.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from pathlib import Path

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

print("Listing /kaggle/input (top-level):")
try:
    print(os.listdir("/kaggle/input")[:50])
except FileNotFoundError:
    print("No /kaggle/input found; will rely on /kaggle/data paths.")

label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

DATA_ROOTS = [
    Path("/kaggle/data/jigsaw-toxic-comment-classification-challenge"),
    Path("/kaggle/data"),
    Path("/kaggle/input/jigsaw-toxic-comment-classification-challenge"),
    Path("/kaggle/input"),
]


def first_existing(*candidates: Path) -> Path:
    for p in candidates:
        if p is not None and p.exists():
            return p
    return None


sample_path = first_existing(*[r / "sample_submission.csv" for r in DATA_ROOTS])
train_path = first_existing(*[r / "train.csv" for r in DATA_ROOTS])
test_path = first_existing(*[r / "test.csv" for r in DATA_ROOTS])

if sample_path is None:
    raise FileNotFoundError(
        "Could not locate sample_submission.csv under known Kaggle roots."
    )
if train_path is None:
    raise FileNotFoundError("Could not locate train.csv under known Kaggle roots.")
if test_path is None:
    raise FileNotFoundError("Could not locate test.csv under known Kaggle roots.")

sample_sub = pd.read_csv(sample_path)
missing = [c for c in (["id"] + label_cols) if c not in sample_sub.columns]
if missing:
    raise ValueError(f"sample_submission.csv missing required columns: {missing}")

print("Using sample_submission:", str(sample_path))
print("Using train:", str(train_path))
print("Using test:", str(test_path))
print("sample_submission shape:", sample_sub.shape)



## === cell 1
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import scipy.sparse as sp
import multiprocessing as mp

train_usecols = ["id", "comment_text"] + label_cols
test_usecols = ["id", "comment_text"]
train_dtypes = {
    "id": "string",
    "comment_text": "string",
    "toxic": "int8",
    "severe_toxic": "int8",
    "obscene": "int8",
    "threat": "int8",
    "insult": "int8",
    "identity_hate": "int8",
}
test_dtypes = {"id": "string", "comment_text": "string"}

train_df = pd.read_csv(train_path, usecols=train_usecols, dtype=train_dtypes)
test_df = pd.read_csv(test_path, usecols=test_usecols, dtype=test_dtypes)

req_train = ["id", "comment_text"] + label_cols
missing_train = [c for c in req_train if c not in train_df.columns]
if missing_train:
    raise ValueError(f"train.csv missing required columns: {missing_train}")
req_test = ["id", "comment_text"]
missing_test = [c for c in req_test if c not in test_df.columns]
if missing_test:
    raise ValueError(f"test.csv missing required columns: {missing_test}")

X_train_text = train_df["comment_text"].fillna("").astype(str).to_numpy()
X_test_text = test_df["comment_text"].fillna("").astype(str).to_numpy()
y_train = train_df[label_cols].to_numpy(dtype=np.int8, copy=False)

n_jobs_vec = max(1, (mp.cpu_count() or 2) - 1)

word_tfidf = TfidfVectorizer(
    strip_accents="unicode",
    analyzer="word",
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.9,
    sublinear_tf=True,
    n_jobs=n_jobs_vec,
)
char_tfidf = TfidfVectorizer(
    strip_accents="unicode",
    analyzer="char",
    ngram_range=(3, 5),
    min_df=3,
    max_df=0.9,
    sublinear_tf=True,
    n_jobs=n_jobs_vec,
)

vectorizer = FeatureUnion([("word", word_tfidf), ("char", char_tfidf)])

print("Vectorizing text...")
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

X_train = X_train.tocsr()
X_test = X_test.tocsr()

print("X_train shape:", X_train.shape, "X_test shape:", X_test.shape)
print("X_train nnz:", X_train.nnz, "X_test nnz:", X_test.nnz)

n_jobs_ovr = max(1, (mp.cpu_count() or 2) - 1)
base_lr = LogisticRegression(
    C=4.0,
    solver="saga",
    max_iter=200,
    random_state=0,
    n_jobs=1,  # parallelism is handled at OVR level to avoid nested oversubscription
)

clf = OneVsRestClassifier(base_lr, n_jobs=n_jobs_ovr)

print("Fitting classifier...")
clf.fit(X_train, y_train)

print("Predicting probabilities...")
test_pred = clf.predict_proba(X_test)  # shape (n_test, 6)

base_ids = sample_sub["id"].astype(str).to_numpy()
test_ids = test_df["id"].astype(str).to_numpy()

id_to_idx = pd.Series(np.arange(test_ids.shape[0], dtype=np.int32), index=test_ids)
mapped_idx = id_to_idx.reindex(base_ids).to_numpy()

p_res = sample_sub[["id"] + label_cols].copy()
vals = np.empty((base_ids.shape[0], len(label_cols)), dtype=np.float64)

mask = ~pd.isna(mapped_idx)
mapped_idx_int = mapped_idx[mask].astype(np.int64, copy=False)
vals[mask] = test_pred[mapped_idx_int].astype(np.float64, copy=False)

if (~mask).any():
    n_miss = int((~mask).sum())
    print(
        f"Warning: model predictions missing {n_miss} ids after alignment; filling with 0.5"
    )
    vals[~mask] = 0.5

p_res[label_cols] = np.clip(vals, 0.0, 1.0)

print("Result head:")
print(p_res.head())
print("Result shape:", p_res.shape)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4060322650.py in <cell line: 0>()
     43 n_jobs_vec = max(1, (mp.cpu_count() or 2) - 1)
     44 
---> 45 word_tfidf = TfidfVectorizer(
     46     strip_accents="unicode",
     47     analyzer="word",

TypeError: TfidfVectorizer.__init__() got an unexpected keyword argument 'n_jobs'

## === cell 2
expected_cols = list(sample_sub.columns)
if list(p_res.columns) != expected_cols:
    for c in expected_cols:
        if c not in p_res.columns:
            p_res[c] = 0.5
    p_res = p_res[expected_cols]

assert p_res["id"].notna().all()
assert p_res.shape[0] == sample_sub.shape[0]
for c in label_cols:
    assert ((p_res[c] >= 0.0) & (p_res[c] <= 1.0)).all()



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1885721509.py in <cell line: 0>()
      1 expected_cols = list(sample_sub.columns)
----> 2 if list(p_res.columns) != expected_cols:
      3     for c in expected_cols:
      4         if c not in p_res.columns:
      5             p_res[c] = 0.5

NameError: name 'p_res' is not defined

## === cell 3
out_path = "submission.csv"
p_res.to_csv(out_path, index=False)
print(f"Wrote {out_path} with shape {p_res.shape} and columns {list(p_res.columns)}")
print(p_res.head())

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2783051757.py in <cell line: 0>()
      1 out_path = "submission.csv"
----> 2 p_res.to_csv(out_path, index=False)
      3 print(f"Wrote {out_path} with shape {p_res.shape} and columns {list(p_res.columns)}")
      4 print(p_res.head())

NameError: name 'p_res' is not defined
