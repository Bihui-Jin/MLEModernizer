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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
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
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

CANDIDATE_BASES = [
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge",
    "/kaggle/data/jigsaw-toxic-comment-classification-challenge",
    "/kaggle/input",
    "/kaggle/data",
    "../input/jigsaw-toxic-comment-classification-challenge",
    "../input",
]


def _find_file(filename: str):
    for base in CANDIDATE_BASES:
        path = os.path.join(base, filename)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Could not find {filename} under any of: {CANDIDATE_BASES}"
    )


train_path = _find_file("train.csv")
test_path = _find_file("test.csv")
sample_path = _find_file("sample_submission.csv")

print("Using files:")
print("train:", train_path)
print("test :", test_path)
print("sample:", sample_path)



## === cell 1
try:
    from sklearnex import patch_sklearn

    patch_sklearn()
    print("sklearnex patch enabled")
except Exception as e:
    print("sklearnex patch not enabled:", repr(e))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import roc_auc_score

try:
    from sklearnex.linear_model import LogisticRegression  # type: ignore

    _USING_SKLEARNEX_LR = True
except Exception:
    from sklearn.linear_model import LogisticRegression  # type: ignore

    _USING_SKLEARNEX_LR = False

class_names = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

train_df = pd.read_csv(
    train_path,
    usecols=["id", "comment_text"] + class_names,
    dtype={c: np.int8 for c in class_names},
)
test_df = pd.read_csv(test_path, usecols=["id", "comment_text"])
sample_submission = pd.read_csv(sample_path)

train_text = train_df["comment_text"].fillna("").astype(str).to_numpy()
test_text = test_df["comment_text"].fillna("").astype(str).to_numpy()

target = train_df[class_names]  # already int8 via dtype above

print("Train shape:", train_df.shape, "Test shape:", test_df.shape)
print("Target cols:", list(target.columns))
print("Using sklearnex LogisticRegression:", _USING_SKLEARNEX_LR)



## === cell 2
import os
from joblib import parallel_backend

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

try:
    from threadpoolctl import threadpool_limits
except Exception:
    threadpool_limits = None

_CPU = os.cpu_count() or 1
N_JOBS = max(1, min(_CPU, 8))
LR_N_JOBS = 1

tfidf_word = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents="unicode",
    analyzer="word",
    token_pattern=r"\w{1,}",
    ngram_range=(1, 2),
    max_features=200000,
    min_df=2,
    dtype=np.float32,  # negligible FP diffs; large speed/memory win on big sparse matrices
)
tfidf_char = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents="unicode",
    analyzer="char",
    ngram_range=(3, 5),
    max_features=200000,
    min_df=2,
    dtype=np.float32,  # negligible FP diffs; large speed/memory win on big sparse matrices
)

vectorizer = FeatureUnion([("word", tfidf_word), ("char", tfidf_char)], n_jobs=N_JOBS)


class _CombinedText:
    __slots__ = ("_a", "_b", "_len_a", "_len")

    def __init__(self, a, b):
        self._a = a
        self._b = b
        self._len_a = len(a)
        self._len = self._len_a + len(b)

    def __len__(self):
        return self._len

    def __iter__(self):
        yield from self._a
        yield from self._b

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self._len)
            if step != 1:
                return [self[i] for i in range(start, stop, step)]
            if stop <= self._len_a:
                return self._a[start:stop]
            if start >= self._len_a:
                return self._b[start - self._len_a : stop - self._len_a]
            return list(self._a[start:]) + list(self._b[: stop - self._len_a])
        if idx < 0:
            idx += self._len
        if idx < self._len_a:
            return self._a[idx]
        return self._b[idx - self._len_a]


all_text = _CombinedText(train_text, test_text)

if threadpool_limits is not None:
    _tpl_ctx = threadpool_limits(limits=1)
else:

    class _NoopCtx:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    _tpl_ctx = _NoopCtx()

with _tpl_ctx:
    with parallel_backend("threading", n_jobs=N_JOBS):
        all_features = vectorizer.fit_transform(all_text)

all_features = all_features.tocsr()
if all_features.indices.dtype != np.int32:
    all_features.indices = all_features.indices.astype(np.int32, copy=False)
if all_features.indptr.dtype != np.int32:
    all_features.indptr = all_features.indptr.astype(np.int32, copy=False)

n_train = len(train_df)
train_features = all_features[:n_train]
test_features = all_features[n_train:]

del all_features

print("CPU count:", _CPU, "Using N_JOBS (vectorizer):", N_JOBS, "LR_N_JOBS:", LR_N_JOBS)
print("Train features:", train_features.shape, "Test features:", test_features.shape)
print(
    "Sparse dtype:",
    train_features.dtype,
    "nnz(train):",
    train_features.nnz,
    "nnz(test):",
    test_features.nnz,
)



## === cell 3
from scipy.special import expit

scores = []
submission = sample_submission.copy()

X_train = train_features
X_test = test_features
n_test = X_test.shape[0]

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

_dummy_X = np.zeros(len(train_df), dtype=np.uint8)

test_pred_sum = np.empty(n_test, dtype=np.float64)

for class_name in class_names:
    y = target[class_name].to_numpy(copy=False)  # int8, no copy
    splits = list(skf.split(_dummy_X, y))

    fold_aucs = []
    test_pred_sum.fill(0.0)

    for tr_idx, va_idx in splits:
        X_tr = X_train[tr_idx]
        y_tr = y[tr_idx]
        X_va = X_train[va_idx]
        y_va = y[va_idx]

        clf = LogisticRegression(
            C=0.1,
            solver="sag",
            max_iter=1000,
            n_jobs=LR_N_JOBS,
            random_state=42,
        )
        clf.fit(X_tr, y_tr)

        va_pred = expit(clf.decision_function(X_va))
        fold_aucs.append(roc_auc_score(y_va, va_pred))

        test_pred_sum += expit(clf.decision_function(X_test))

    cv_score = float(np.mean(fold_aucs))
    scores.append(cv_score)
    print(f"CV score for class {class_name} is {cv_score}")

    submission[class_name] = (test_pred_sum / len(splits)).astype(np.float64)



## === cell 4
print("Total CV score is {}".format(float(np.mean(scores))))

submission = submission[["id"] + class_names]

out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)

print("Wrote submission.csv to:", out_path)
print("Submission shape:", submission.shape)
print(submission.head())
