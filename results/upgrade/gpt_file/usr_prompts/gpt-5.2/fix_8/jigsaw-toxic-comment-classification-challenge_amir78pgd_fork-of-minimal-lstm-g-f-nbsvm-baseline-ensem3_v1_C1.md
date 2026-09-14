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

# 5. Target score

0.98390496049508

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

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
Y = train[label_cols].to_numpy(dtype=np.float32, copy=False)

print(train.shape, test.shape, sample.shape)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1106794363.py in <cell line: 0>()
     23 sample_dtypes = {"id": "string", **{c: "float32" for c in label_cols}}
     24 
---> 25 train = pd.read_csv(train_path, usecols=train_usecols, dtype=train_dtypes)
     26 test = pd.read_csv(test_path, usecols=test_usecols, dtype=test_dtypes)
     27 sample = pd.read_csv(sample_path, usecols=sample_usecols, dtype=sample_dtypes)

NameError: name 'train_path' is not defined

## === cell 1
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from scipy import sparse

_VECT_CACHE = {}


def _ensure_fast_csr_float32(X):
    if not sparse.isspmatrix_csr(X):
        X = X.tocsr(copy=False)
    if X.dtype != np.float32:
        X = X.astype(np.float32, copy=False)
    return X


def _vect_cache_key(vectorizer, X_tr, X_te):
    params = tuple(sorted(vectorizer.get_params(deep=True).items()))
    return (params, id(X_tr), id(X_te))


def get_vectorized_mats(vectorizer, X_tr, X_te):
    cache_key = _vect_cache_key(vectorizer, X_tr, X_te)
    cached = _VECT_CACHE.get(cache_key)
    if cached is None:
        vectorizer.fit(X_tr)
        Xtr = _ensure_fast_csr_float32(vectorizer.transform(X_tr))
        Xte = _ensure_fast_csr_float32(vectorizer.transform(X_te))
        _VECT_CACHE[cache_key] = (Xtr, Xte)
    else:
        Xtr, Xte = cached
    return Xtr, Xte


def fit_predict_proba_from_mats(Xtr, Y_tr, Xte, C=4.0, max_iter=300):
    """
    Core logic preserved: one-vs-rest LogisticRegression('liblinear') per label with same hyperparams.
    Accepts precomputed sparse matrices to avoid repeated vectorization work.
    """
    n_labels = Y_tr.shape[1]
    preds = np.zeros((Xte.shape[0], n_labels), dtype=np.float32)

    y0 = Y_tr[0, :]
    y_min = Y_tr.min(axis=0)
    y_max = Y_tr.max(axis=0)
    is_const = y_min == y_max

    for j in range(n_labels):
        if is_const[j]:
            preds[:, j] = float(y0[j])
            continue

        y = Y_tr[:, j]
        clf = LogisticRegression(
            C=C,
            solver="liblinear",
            max_iter=max_iter,
            random_state=42,
            n_jobs=1,
        )
        clf.fit(Xtr, y)
        preds[:, j] = clf.predict_proba(Xte)[:, 1].astype(np.float32, copy=False)

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
)
vec3 = TfidfVectorizer(
    strip_accents="unicode",
    lowercase=True,
    analyzer="char",
    ngram_range=(3, 5),
    min_df=3,
    max_df=0.9,
    sublinear_tf=True,
)
vec4 = CountVectorizer(
    strip_accents="unicode",
    lowercase=True,
    analyzer="word",
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.9,
    token_pattern=r"(?u)\b\w\w+\b",
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




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2216402821.py in <cell line: 0>()
      1 # Speed: explicitly compute and reuse vectorized matrices (fit/transform once per vectorizer).
----> 2 Xtr1, Xte1 = get_vectorized_mats(vec1, X_train_text, X_test_text)
      3 pred1 = fit_predict_proba_from_mats(Xtr1, Y, Xte1, C=4.0, max_iter=300)
      4 
      5 Xtr2, Xte2 = get_vectorized_mats(vec2, X_train_text, X_test_text)

NameError: name 'X_train_text' is not defined

## === cell 3
submission = pd.DataFrame(pred, columns=label_cols)
submission.insert(0, "id", test["id"].to_numpy())
submission = submission[["id"] + label_cols]

assert submission.shape[0] == test.shape[0]
assert list(submission.columns) == ["id"] + label_cols
print(submission.head())




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3686530695.py in <cell line: 0>()
----> 1 submission = pd.DataFrame(pred, columns=label_cols)
      2 submission.insert(0, "id", test["id"].to_numpy())
      3 submission = submission[["id"] + label_cols]
      4 
      5 assert submission.shape[0] == test.shape[0]

NameError: name 'pred' is not defined

## === cell 4
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2160108896.py in <cell line: 0>()
----> 1 submission.to_csv("submission.csv", index=False)
      2 print("Wrote submission.csv with shape:", submission.shape)

NameError: name 'submission' is not defined
