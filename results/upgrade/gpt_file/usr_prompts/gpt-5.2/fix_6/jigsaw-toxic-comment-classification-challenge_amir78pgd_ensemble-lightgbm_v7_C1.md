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

# 5. Code solution

## === cell 0
import os
import glob

CANDIDATE_INPUT_DIRS = [
    "../input/jigsaw-toxic-comment-classification-challenge",
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge",
    "/kaggle/data/jigsaw-toxic-comment-classification-challenge",
    "/kaggle/data",
    "../input",
    "../data",
    "/kaggle/input",
    "/kaggle/data",
]


def find_file(filename):
    for d in CANDIDATE_INPUT_DIRS:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    for d in CANDIDATE_INPUT_DIRS:
        if os.path.exists(d):
            hits = glob.glob(os.path.join(d, "**", filename), recursive=True)
            if hits:
                return hits[0]
    raise FileNotFoundError(
        f"Could not find {filename} in candidate dirs: {CANDIDATE_INPUT_DIRS}"
    )


train_path = find_file("train.csv")
test_path = find_file("test.csv")
sub_path = find_file("sample_submission.csv")

(train_path, test_path, sub_path)



## === cell 1
import numpy as np
import pandas as pd

label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
train = pd.read_csv(train_path, usecols=["id", "comment_text"] + label_cols)
test = pd.read_csv(test_path, usecols=["id", "comment_text"])
sample_sub = pd.read_csv(sub_path)

assert all(c in train.columns for c in ["id", "comment_text"] + label_cols)
assert all(c in test.columns for c in ["id", "comment_text"])
assert (
    list(sample_sub.columns) == ["id"] + label_cols
), "Sample submission columns mismatch."

train["comment_text"] = train["comment_text"].fillna("")
test["comment_text"] = test["comment_text"].fillna("")

X_train = train["comment_text"].values
X_test = test["comment_text"].values
Y = train[label_cols].astype(np.float32).values

test_ids = test["id"].values

(train.shape, test.shape, sample_sub.shape)



## === cell 2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.multiclass import OneVsRestClassifier

from scipy import sparse

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

_MODEL_CACHE = {}
_PROBA_CACHE = {}


def _estimator_cache_key(estimator):
    p = estimator.get_params(deep=True)
    items = []
    for k in sorted(p.keys()):
        v = p[k]
        if isinstance(v, (str, int, float, bool, type(None))):
            items.append((k, v))
        elif isinstance(v, tuple):
            items.append((k, v))
        else:
            items.append((k, v.__class__.__name__))
    return (estimator.__class__.__name__, tuple(items))


def _ensure_csr_float32(X):
    if not sparse.isspmatrix_csr(X):
        X = X.tocsr()
    if X.dtype != np.float32:
        X = X.astype(np.float32, copy=False)
    return X


def fit_vectorizer_cached(vectorizer, X_train_text, X_test_text):
    Xt = vectorizer.fit_transform(X_train_text)
    Xv = vectorizer.transform(X_test_text)
    Xt = _ensure_csr_float32(Xt)
    Xv = _ensure_csr_float32(Xv)
    return Xt, Xv


def _ovr_n_jobs_for_estimator(estimator):
    if isinstance(estimator, LogisticRegression) and estimator.solver == "liblinear":
        return 1
    return -1


def fit_predict_proba_from_matrices(estimator, Xtr, Ytr, Xte):
    est_key = _estimator_cache_key(estimator)
    proba_key = (est_key, id(Xtr), id(Xte))
    cached = _PROBA_CACHE.get(proba_key)
    if cached is not None:
        return cached, _MODEL_CACHE[(est_key, id(Xtr))]

    model_key = (est_key, id(Xtr))
    clf = _MODEL_CACHE.get(model_key)
    if clf is None:
        clf = OneVsRestClassifier(
            estimator, n_jobs=_ovr_n_jobs_for_estimator(estimator)
        )
        clf.fit(Xtr, Ytr)
        _MODEL_CACHE[model_key] = clf

    proba = clf.predict_proba(Xte)
    proba = np.clip(proba, 1e-6, 1 - 1e-6)
    _PROBA_CACHE[proba_key] = proba
    return proba, clf


vec_word = TfidfVectorizer(
    strip_accents="unicode",
    lowercase=True,
    analyzer="word",
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.9,
    sublinear_tf=True,
    norm="l2",
)

vec_char = TfidfVectorizer(
    strip_accents="unicode",
    lowercase=True,
    analyzer="char",
    ngram_range=(3, 5),
    min_df=3,
    max_df=0.9,
    sublinear_tf=True,
    norm="l2",
)

Xtr_word, Xte_word = fit_vectorizer_cached(vec_word, X_train, X_test)
Xtr_char, Xte_char = fit_vectorizer_cached(vec_char, X_train, X_test)

p_slgbm_arr, _ = fit_predict_proba_from_matrices(
    LogisticRegression(solver="liblinear", C=4.0, max_iter=1000),
    Xtr_word,
    Y,
    Xte_word,
)

p_nbsvm_arr, _ = fit_predict_proba_from_matrices(
    ComplementNB(alpha=0.5),
    Xtr_word,
    Y,
    Xte_word,
)

svc = LinearSVC(C=2.0)
cal_svc = CalibratedClassifierCV(svc, method="sigmoid", cv=3)
p_caps_gru_arr, _ = fit_predict_proba_from_matrices(
    cal_svc,
    Xtr_char,
    Y,
    Xte_char,
)

p_dual_embed_pl_arr, _ = fit_predict_proba_from_matrices(
    LogisticRegression(solver="liblinear", C=2.0, max_iter=1000),
    Xtr_char,
    Y,
    Xte_char,
)

p_dual_embed_mish_arr, _ = fit_predict_proba_from_matrices(
    LogisticRegression(solver="liblinear", C=1.0, max_iter=1000),
    Xtr_word,
    Y,
    Xte_word,
)

p_lstm_glove_tta_arr, _ = fit_predict_proba_from_matrices(
    LogisticRegression(solver="liblinear", C=8.0, max_iter=1000),
    Xtr_word,
    Y,
    Xte_word,
)

p_dual_embed_dehyp_arr, _ = fit_predict_proba_from_matrices(
    ComplementNB(alpha=1.0),
    Xtr_word,
    Y,
    Xte_word,
)

p_lstm_fast_arr, _ = fit_predict_proba_from_matrices(
    ComplementNB(alpha=0.2),
    Xtr_word,
    Y,
    Xte_word,
)

p_bi_post_arr, _ = fit_predict_proba_from_matrices(
    CalibratedClassifierCV(LinearSVC(C=1.0), method="sigmoid", cv=3),
    Xtr_char,
    Y,
    Xte_char,
)

p_dpcnn_arr, _ = fit_predict_proba_from_matrices(
    CalibratedClassifierCV(LinearSVC(C=3.0), method="sigmoid", cv=3),
    Xtr_char,
    Y,
    Xte_char,
)

p_dmcnn_arr, _ = fit_predict_proba_from_matrices(
    LogisticRegression(solver="liblinear", C=4.0, max_iter=1000),
    Xtr_char,
    Y,
    Xte_char,
)

p_rcn_arr, _ = fit_predict_proba_from_matrices(
    LogisticRegression(solver="liblinear", C=1.0, max_iter=1000),
    Xtr_char,
    Y,
    Xte_char,
)

p_attn_300d_arr, _ = fit_predict_proba_from_matrices(
    LogisticRegression(solver="liblinear", C=8.0, max_iter=1000),
    Xtr_char,
    Y,
    Xte_char,
)


def proba_to_df(proba_arr):
    df = pd.DataFrame(proba_arr, columns=label_cols)
    df.insert(0, "id", test_ids)
    return df


p_slgbm = proba_to_df(p_slgbm_arr)
p_nbsvm = proba_to_df(p_nbsvm_arr)
p_caps_gru = proba_to_df(p_caps_gru_arr)
p_dual_embed_pl = proba_to_df(p_dual_embed_pl_arr)

for df in [p_slgbm, p_caps_gru, p_dual_embed_pl, p_nbsvm]:
    assert df.shape[0] == test.shape[0] and list(df.columns) == ["id"] + label_cols
    assert (df["id"].values == test_ids).all()

(p_caps_gru.head(), p_slgbm.head())




## === cell 3
def assert_aligned_ids(arr_ids, other_ids, name):
    if not np.array_equal(arr_ids, other_ids):
        raise ValueError(f"ID alignment failed vs {name}")


base_ids = test_ids
for name, arr in [
    ("p_slgbm", p_slgbm_arr),
    ("p_caps_gru", p_caps_gru_arr),
    ("p_dual_embed_pl", p_dual_embed_pl_arr),
    ("p_dual_embed_mish", p_dual_embed_mish_arr),
    ("p_lstm_glove_tta", p_lstm_glove_tta_arr),
    ("p_dual_embed_dehyp", p_dual_embed_dehyp_arr),
    ("p_lstm_fast", p_lstm_fast_arr),
    ("p_nbsvm", p_nbsvm_arr),
    ("p_bi_post", p_bi_post_arr),
    ("p_dpcnn", p_dpcnn_arr),
    ("p_dmcnn", p_dmcnn_arr),
    ("p_rcn", p_rcn_arr),
    ("p_attn_300d", p_attn_300d_arr),
]:
    assert_aligned_ids(base_ids, test_ids, name)

ens = p_slgbm_arr.copy()
ens += p_caps_gru_arr
ens += p_dual_embed_pl_arr
ens += p_dual_embed_mish_arr
ens += p_lstm_glove_tta_arr
ens += p_dual_embed_dehyp_arr
ens += p_lstm_fast_arr
ens += p_nbsvm_arr
ens += p_bi_post_arr * 3
ens += p_dpcnn_arr * 3
ens += p_dmcnn_arr * 3
ens += p_rcn_arr * 3
ens += p_attn_300d_arr * 3
ens /= 23.0

ens = np.clip(ens, 1e-6, 1 - 1e-6)

p_res = pd.DataFrame(ens, columns=label_cols)
p_res.insert(0, "id", test_ids)

p_res.head()



## === cell 4
assert np.array_equal(
    sample_sub["id"].values, test_ids
), "sample_submission ids differ from test ids/order"

submission = pd.DataFrame({"id": test_ids})
for i, c in enumerate(label_cols):
    submission[c] = ens[:, i]

submission[label_cols] = submission[label_cols].fillna(0.5).clip(1e-6, 1 - 1e-6)
submission = submission[["id"] + label_cols]

assert submission.shape[0] == sample_sub.shape[0]
assert list(submission.columns) == list(sample_sub.columns)

submission.head()



## === cell 5
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv", submission.shape)
print(submission.describe().T)
