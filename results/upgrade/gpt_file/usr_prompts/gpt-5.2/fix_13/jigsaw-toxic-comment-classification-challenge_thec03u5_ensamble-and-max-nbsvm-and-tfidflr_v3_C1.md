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

3.6

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

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

BASE_CANDIDATES = [
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge",
    "/kaggle/data/jigsaw-toxic-comment-classification-challenge",
    "/kaggle/input",
    "/kaggle/data",
]
DATA_DIR = None
for c in BASE_CANDIDATES:
    if os.path.exists(os.path.join(c, "train.csv")) and os.path.exists(
        os.path.join(c, "test.csv")
    ):
        DATA_DIR = c
        break

if DATA_DIR is None:
    raise FileNotFoundError(
        "Could not locate train.csv/test.csv in expected Kaggle paths. "
        "Checked: " + ", ".join(BASE_CANDIDATES)
    )

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sub_path = os.path.join(DATA_DIR, "sample_submission.csv")

usecols_train = ["id", "comment_text"] + label_cols
dtype_train = {c: np.int8 for c in label_cols}

train = pd.read_csv(
    train_path, usecols=usecols_train, dtype=dtype_train, low_memory=False
)
test = pd.read_csv(test_path, usecols=["id", "comment_text"], low_memory=False)
sample_sub = pd.read_csv(sub_path, low_memory=False)

train["comment_text"] = train["comment_text"].fillna("")
test["comment_text"] = test["comment_text"].fillna("")

missing = [c for c in label_cols if c not in train.columns]
if missing:
    raise KeyError(f"Missing label columns in train.csv: {missing}")

print("Loaded:", train.shape, test.shape, sample_sub.shape)
print("Using DATA_DIR:", DATA_DIR)




## === cell 1
from scipy import sparse
import joblib

CACHE_DIR = "/kaggle/working/tfidf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

train_text = train["comment_text"].values
test_text = test["comment_text"].values
n_tr = train_text.shape[0]

word_vectorizer = TfidfVectorizer(
    strip_accents="unicode",
    analyzer="word",
    token_pattern=r"\w{1,}",
    ngram_range=(1, 2),
    max_features=200000,
    min_df=3,
    sublinear_tf=True,
    dtype=np.float32,
)

char_vectorizer = TfidfVectorizer(
    strip_accents="unicode",
    analyzer="char",
    ngram_range=(3, 5),
    max_features=200000,
    min_df=3,
    sublinear_tf=True,
    dtype=np.float32,
)

wv_path = os.path.join(CACHE_DIR, "word_vectorizer_trainfit.joblib")
cv_path = os.path.join(CACHE_DIR, "char_vectorizer_trainfit.joblib")
Xtr_path = os.path.join(CACHE_DIR, "X_tr_trainfit.npz")
Xte_path = os.path.join(CACHE_DIR, "X_te_trainfit.npz")

if (
    os.path.exists(wv_path)
    and os.path.exists(cv_path)
    and os.path.exists(Xtr_path)
    and os.path.exists(Xte_path)
):
    word_vectorizer = joblib.load(wv_path)
    char_vectorizer = joblib.load(cv_path)
    X_tr = sparse.load_npz(Xtr_path)
    X_te = sparse.load_npz(Xte_path)
    print("Loaded cached TF-IDF matrices:", X_tr.shape, X_te.shape)
else:
    Xw_tr = word_vectorizer.fit_transform(train_text)
    Xc_tr = char_vectorizer.fit_transform(train_text)

    Xw_te = word_vectorizer.transform(test_text)
    Xc_te = char_vectorizer.transform(test_text)

    print(
        "Word TFIDF:", Xw_tr.shape, Xw_te.shape, "Char TFIDF:", Xc_tr.shape, Xc_te.shape
    )

    X_tr = sparse.hstack((Xw_tr, Xc_tr), format="csr", dtype=np.float32)
    X_te = sparse.hstack((Xw_te, Xc_te), format="csr", dtype=np.float32)
    del Xw_tr, Xw_te, Xc_tr, Xc_te

    X_tr.sort_indices()
    X_te.sort_indices()

    if X_tr.indices.dtype != np.int32:
        X_tr.indices = X_tr.indices.astype(np.int32, copy=False)
    if X_tr.indptr.dtype != np.int32:
        X_tr.indptr = X_tr.indptr.astype(np.int32, copy=False)
    if X_te.indices.dtype != np.int32:
        X_te.indices = X_te.indices.astype(np.int32, copy=False)
    if X_te.indptr.dtype != np.int32:
        X_te.indptr = X_te.indptr.astype(np.int32, copy=False)

    joblib.dump(word_vectorizer, wv_path, compress=3)
    joblib.dump(char_vectorizer, cv_path, compress=3)
    sparse.save_npz(Xtr_path, X_tr)
    sparse.save_npz(Xte_path, X_te)

print("Final sparse matrices:", X_tr.shape, X_te.shape)
print("nnz:", X_tr.nnz, X_te.nnz)
print("TF-IDF dtype:", X_tr.dtype, X_te.dtype)




## === cell 2
from sklearn.base import clone

Y = train[label_cols].to_numpy(dtype=np.int8, copy=False)

cpu_cnt = os.cpu_count() or 1
LR_NJOBS = max(1, min(cpu_cnt, 4))

base_clf = LogisticRegression(
    solver="saga",
    penalty="l2",
    C=4.0,
    max_iter=200,
    n_jobs=LR_NJOBS,
    random_state=RANDOM_STATE,
)

preds = np.zeros((X_te.shape[0], len(label_cols)), dtype=np.float64)

for i, col in enumerate(label_cols):
    y = Y[:, i]
    clf = clone(base_clf)
    clf.fit(X_tr, y)
    preds[:, i] = clf.predict_proba(X_te)[:, 1]
    print(f"Trained {col}: positive_rate={float(y.mean()):.6f}")

print("Preds shape:", preds.shape)




## === cell 3
submission = sample_sub.copy()

sub_ids = submission["id"].values
test_ids = test["id"].values

if (
    len(submission) == len(test)
    and sub_ids.shape == test_ids.shape
    and np.array_equal(sub_ids, test_ids)
):
    for j, col in enumerate(label_cols):
        submission[col] = preds[:, j]
else:
    test_id_to_row = pd.Series(np.arange(test.shape[0]), index=test_ids)
    row_idx = test_id_to_row.reindex(sub_ids)
    if row_idx.isna().any():
        tmp = pd.DataFrame({"id": test_ids})
        for j, col in enumerate(label_cols):
            tmp[col] = preds[:, j]
        submission = submission[["id"]].merge(tmp, on="id", how="left")
    else:
        row_idx = row_idx.astype(np.int64).to_numpy()
        for j, col in enumerate(label_cols):
            submission[col] = preds[row_idx, j]

for col in label_cols:
    submission[col] = submission[col].clip(0.0, 1.0)

submission = submission[["id"] + label_cols]

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)

print("Wrote:", submission_path)
print(submission.head())




## === cell 4
print(
    "Skipped redundant writes: submissionAvg.csv, submissionMax.csv, submissionCondMax.csv"
)
