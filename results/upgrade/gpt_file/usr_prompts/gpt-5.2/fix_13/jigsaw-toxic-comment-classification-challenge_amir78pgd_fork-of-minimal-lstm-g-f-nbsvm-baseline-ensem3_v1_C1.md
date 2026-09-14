# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

3.7

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

# 5. Code solution

## === cell 0
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

np.random.seed(42)

COMP_DIR_CANDIDATES = [
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge",
    "/kaggle/data/jigsaw-toxic-comment-classification-challenge",
    "/kaggle/input",
    "/kaggle/data",
]


def _first_existing_file(relpath_options):
    for base in COMP_DIR_CANDIDATES:
        for rel in relpath_options:
            p = os.path.join(base, rel)
            if os.path.exists(p):
                return p
    return None


train_path = _first_existing_file(
    ["train.csv", "jigsaw-toxic-comment-classification-challenge/train.csv"]
)
test_path = _first_existing_file(
    ["test.csv", "jigsaw-toxic-comment-classification-challenge/test.csv"]
)
sample_path = _first_existing_file(
    [
        "sample_submission.csv",
        "jigsaw-toxic-comment-classification-challenge/sample_submission.csv",
    ]
)

if train_path is None or test_path is None or sample_path is None:
    raise FileNotFoundError(
        f"Could not find required files. "
        f"Resolved paths: train={train_path}, test={test_path}, sample={sample_path}. "
        f"Checked bases: {COMP_DIR_CANDIDATES}"
    )

label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
train_usecols = ["id", "comment_text"] + label_cols
test_usecols = ["id", "comment_text"]
sample_usecols = ["id"] + label_cols

train_dtypes = {
    "id": "string",
    "comment_text": "string",
    **{c: "int8" for c in label_cols},
}
test_dtypes = {"id": "string", "comment_text": "string"}
sample_dtypes = {"id": "string", **{c: "float32" for c in label_cols}}

train = pd.read_csv(train_path, usecols=train_usecols, dtype=train_dtypes)
test = pd.read_csv(test_path, usecols=test_usecols, dtype=test_dtypes)
sample = pd.read_csv(sample_path, usecols=sample_usecols, dtype=sample_dtypes)

assert "id" in train.columns and "comment_text" in train.columns
assert "id" in test.columns and "comment_text" in test.columns
for c in label_cols:
    assert c in train.columns, f"Missing label col {c} in train.csv"
for c in ["id"] + label_cols:
    assert c in sample.columns, f"Missing col {c} in sample_submission.csv"

train["comment_text"] = train["comment_text"].fillna("")
test["comment_text"] = test["comment_text"].fillna("")

X_train_text = train["comment_text"].to_numpy(dtype=object)
X_test_text = test["comment_text"].to_numpy(dtype=object)
Y = train[label_cols].to_numpy(dtype=np.int8, copy=False)

print("Paths:", train_path, test_path, sample_path)
print(train.shape, test.shape, sample.shape)



## === cell 1
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from scipy import sparse

_VECT_CACHE = {}


def _as_csr_float32(X):
    if not sparse.isspmatrix_csr(X):
        X = X.tocsr(copy=False)
    if X.dtype != np.float32:
        X = X.astype(np.float32, copy=False)
    return X


def _vect_cache_key(vectorizer, X_tr, X_te):
    return (id(vectorizer), id(X_tr), id(X_te))


def get_vectorized_mats(vectorizer, X_tr, X_te):
    cache_key = _vect_cache_key(vectorizer, X_tr, X_te)
    cached = _VECT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    vectorizer.fit(X_tr)
    X_all = np.concatenate([X_tr, X_te], axis=0)
    X_all_mat = _as_csr_float32(vectorizer.transform(X_all))
    n_tr = X_tr.shape[0]
    Xtr = X_all_mat[:n_tr]
    Xte = X_all_mat[n_tr:]
    _VECT_CACHE[cache_key] = (Xtr, Xte)
    return Xtr, Xte


def fit_predict_proba_from_mats(Xtr, Y_tr, Xte, C=4.0, max_iter=300):
    n_labels = Y_tr.shape[1]
    preds = np.zeros((Xte.shape[0], n_labels), dtype=np.float32)

    y_min = Y_tr.min(axis=0)
    y_max = Y_tr.max(axis=0)
    is_const = y_min == y_max

    n_jobs = min(os.cpu_count() or 2, 4)

    nonconst_idx = np.flatnonzero(~is_const)
    if nonconst_idx.size:
        base = LogisticRegression(
            C=C,
            solver="liblinear",
            max_iter=max_iter,
            random_state=42,
        )
        ovr = OneVsRestClassifier(base, n_jobs=n_jobs)
        ovr.fit(Xtr, Y_tr[:, nonconst_idx])

        proba = ovr.predict_proba(Xte)
        preds[:, nonconst_idx] = np.asarray(proba, dtype=np.float32)

    if is_const.any():
        const_idx = np.flatnonzero(is_const)
        preds[:, const_idx] = Y_tr[0, const_idx].astype(np.float32, copy=False)

    return preds


def fit_predict_proba(vectorizer, X_tr, Y_tr, X_te, C=4.0, max_iter=300):
    Xtr, Xte = get_vectorized_mats(vectorizer, X_tr, X_te)
    return fit_predict_proba_from_mats(Xtr, Y_tr, Xte, C=C, max_iter=max_iter)


vec1 = TfidfVectorizer(
    strip_accents="unicode",
    lowercase=True,
    analyzer="word",
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.9,
    sublinear_tf=True,
    token_pattern=r"(?u)\b\w\w+\b",
    dtype=np.float32,
)
vec2 = TfidfVectorizer(
    strip_accents="unicode",
    lowercase=True,
    analyzer="word",
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    token_pattern=r"(?u)\b\w\w+\b",
    dtype=np.float32,
)
vec3 = TfidfVectorizer(
    strip_accents="unicode",
    lowercase=True,
    analyzer="char",
    ngram_range=(3, 5),
    min_df=3,
    max_df=0.9,
    sublinear_tf=True,
    dtype=np.float32,
)
vec4 = CountVectorizer(
    strip_accents="unicode",
    lowercase=True,
    analyzer="word",
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.9,
    token_pattern=r"(?u)\b\w\w+\b",
    dtype=np.float32,
)



## === cell 2
Xtr1, Xte1 = get_vectorized_mats(vec1, X_train_text, X_test_text)
pred1 = fit_predict_proba_from_mats(Xtr1, Y, Xte1, C=4.0, max_iter=300)

Xtr2, Xte2 = get_vectorized_mats(vec2, X_train_text, X_test_text)
pred2 = fit_predict_proba_from_mats(Xtr2, Y, Xte2, C=3.0, max_iter=300)

Xtr3, Xte3 = get_vectorized_mats(vec3, X_train_text, X_test_text)
pred3 = fit_predict_proba_from_mats(Xtr3, Y, Xte3, C=2.0, max_iter=300)

Xtr4, Xte4 = get_vectorized_mats(vec4, X_train_text, X_test_text)
pred4 = fit_predict_proba_from_mats(Xtr4, Y, Xte4, C=4.0, max_iter=300)

pred = (pred1 + pred2 + pred3 + pred4) / 4.0
pred = np.clip(pred, 1e-6, 1 - 1e-6)

print("Pred shape:", pred.shape)



## === cell 3
submission = pd.DataFrame(pred, columns=label_cols)
submission.insert(0, "id", test["id"].to_numpy())
submission = submission[["id"] + label_cols]

assert submission.shape[0] == test.shape[0]
assert list(submission.columns) == ["id"] + label_cols
print(submission.head())



## === cell 4
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
