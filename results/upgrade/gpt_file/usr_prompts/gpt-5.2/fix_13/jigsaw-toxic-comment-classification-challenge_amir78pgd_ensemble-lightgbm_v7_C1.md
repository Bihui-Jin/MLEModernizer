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

0.986308862412295

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

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
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

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

X_train = train["comment_text"].to_numpy()
X_test = test["comment_text"].to_numpy()
Y = train[label_cols].to_numpy(dtype=np.float32, copy=False)

test_ids = test["id"].to_numpy()

(train.shape, test.shape, sample_sub.shape)



## === cell 2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.multiclass import OneVsRestClassifier
from scipy import sparse

import hashlib
import time


def fit_vectorizer_fast(vectorizer, X_train_text, X_test_text, vocab_fit_max=200_000):
    t0 = time.time()

    X_all = np.concatenate([X_train_text, X_test_text], axis=0)
    if vocab_fit_max is not None and len(X_all) > vocab_fit_max:
        X_fit = X_all[:vocab_fit_max]
    else:
        X_fit = X_all

    vectorizer.fit(X_fit)
    X_all_vec = vectorizer.transform(X_all)

    if not sparse.isspmatrix_csr(X_all_vec):
        X_all_vec = X_all_vec.tocsr()

    if X_all_vec.dtype != np.float32:
        X_all_vec = X_all_vec.astype(np.float32, copy=False)

    X_all_vec.sort_indices()

    n_tr = len(X_train_text)
    Xt = X_all_vec[:n_tr]
    Xv = X_all_vec[n_tr:]

    print(
        f"Vectorizer {vectorizer.analyzer}-{vectorizer.ngram_range} "
        f"vocab={len(vectorizer.vocabulary_):,} "
        f"Xtr nnz={Xt.nnz:,} Xte nnz={Xv.nnz:,} "
        f"in {time.time()-t0:.1f}s"
    )
    return Xt, Xv


def _ovr_n_jobs_for_estimator(estimator):
    if isinstance(estimator, LogisticRegression) and estimator.solver == "liblinear":
        return 1
    return 1  # avoid nested parallelism


_svm_calib_cache = {}


def _calibrated_linsvc_ovr_predict_proba(Xtr, Ytr, Xte, C, cv=3, method="sigmoid"):
    key = (C, cv, method, Xtr.shape, Xtr.nnz, Xte.shape, Xte.nnz)
    if key in _svm_calib_cache:
        return _svm_calib_cache[key]

    t0 = time.time()
    base = LinearSVC(C=C)
    est = CalibratedClassifierCV(base, method=method, cv=cv, n_jobs=1)
    clf = OneVsRestClassifier(est, n_jobs=1)
    clf.fit(Xtr, Ytr)
    proba = clf.predict_proba(Xte)
    proba = np.clip(proba, 1e-6, 1 - 1e-6)
    _svm_calib_cache[key] = proba
    print(f"Calibrated LinearSVC(C={C}) OVR done in {time.time()-t0:.1f}s")
    return proba


def fit_predict_proba_from_matrices(estimator, Xtr, Ytr, Xte):
    if isinstance(estimator, CalibratedClassifierCV) and isinstance(
        estimator.estimator, LinearSVC
    ):
        return _calibrated_linsvc_ovr_predict_proba(
            Xtr,
            Ytr,
            Xte,
            C=estimator.estimator.C,
            cv=estimator.cv,
            method=estimator.method,
        )

    if isinstance(estimator, (LogisticRegression, ComplementNB)):
        estimator.fit(Xtr, Ytr)
        proba = estimator.predict_proba(Xte)
        if isinstance(proba, list):
            proba = np.column_stack([p[:, 1] for p in proba])
    else:
        clf = OneVsRestClassifier(
            estimator, n_jobs=_ovr_n_jobs_for_estimator(estimator)
        )
        clf.fit(Xtr, Ytr)
        proba = clf.predict_proba(Xte)

    proba = np.clip(proba, 1e-6, 1 - 1e-6)
    return proba


vec_word = TfidfVectorizer(
    strip_accents="unicode",
    lowercase=True,
    analyzer="word",
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.9,
    sublinear_tf=True,
    norm="l2",
    dtype=np.float32,
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
    dtype=np.float32,
)

Xtr_word, Xte_word = fit_vectorizer_fast(
    vec_word, X_train, X_test, vocab_fit_max=200_000
)
Xtr_char, Xte_char = fit_vectorizer_fast(
    vec_char, X_train, X_test, vocab_fit_max=200_000
)

svm_n_jobs = 1

tasks = [
    (
        "p_slgbm_arr",
        LogisticRegression(solver="liblinear", C=4.0, max_iter=1000),
        Xtr_word,
        Xte_word,
    ),
    ("p_nbsvm_arr", ComplementNB(alpha=0.5), Xtr_word, Xte_word),
    (
        "p_caps_gru_arr",
        CalibratedClassifierCV(
            LinearSVC(C=2.0), method="sigmoid", cv=3, n_jobs=svm_n_jobs
        ),
        Xtr_char,
        Xte_char,
    ),
    (
        "p_dual_embed_pl_arr",
        LogisticRegression(solver="liblinear", C=2.0, max_iter=1000),
        Xtr_char,
        Xte_char,
    ),
    (
        "p_dual_embed_mish_arr",
        LogisticRegression(solver="liblinear", C=1.0, max_iter=1000),
        Xtr_word,
        Xte_word,
    ),
    (
        "p_lstm_glove_tta_arr",
        LogisticRegression(solver="liblinear", C=8.0, max_iter=1000),
        Xtr_word,
        Xte_word,
    ),
    ("p_dual_embed_dehyp_arr", ComplementNB(alpha=1.0), Xtr_word, Xte_word),
    ("p_lstm_fast_arr", ComplementNB(alpha=0.2), Xtr_word, Xte_word),
    (
        "p_bi_post_arr",
        CalibratedClassifierCV(
            LinearSVC(C=1.0), method="sigmoid", cv=3, n_jobs=svm_n_jobs
        ),
        Xtr_char,
        Xte_char,
    ),
    (
        "p_dpcnn_arr",
        CalibratedClassifierCV(
            LinearSVC(C=3.0), method="sigmoid", cv=3, n_jobs=svm_n_jobs
        ),
        Xtr_char,
        Xte_char,
    ),
    (
        "p_dmcnn_arr",
        LogisticRegression(solver="liblinear", C=4.0, max_iter=1000),
        Xtr_char,
        Xte_char,
    ),
    (
        "p_rcn_arr",
        LogisticRegression(solver="liblinear", C=1.0, max_iter=1000),
        Xtr_char,
        Xte_char,
    ),
    (
        "p_attn_300d_arr",
        LogisticRegression(solver="liblinear", C=8.0, max_iter=1000),
        Xtr_char,
        Xte_char,
    ),
]

_pred_cache = {}


def _cache_key(name, est, Xtr, Xte):
    est_params = (
        repr(est.get_params(deep=True)) if hasattr(est, "get_params") else repr(est)
    )
    h = hashlib.md5(est_params.encode("utf-8")).hexdigest()
    return (h, Xtr.shape, Xtr.nnz, Xte.shape, Xte.nnz)


def _run_task(name, est, Xtr, Xte):
    t0 = time.time()
    key = _cache_key(name, est, Xtr, Xte)
    if key in _pred_cache:
        out = _pred_cache[key]
        print(f"{name}: cache hit in {time.time()-t0:.1f}s")
        return name, out
    out = fit_predict_proba_from_matrices(est, Xtr, Y, Xte)
    _pred_cache[key] = out
    print(f"{name}: done in {time.time()-t0:.1f}s")
    return name, out


results = [_run_task(name, est, Xtr, Xte) for (name, est, Xtr, Xte) in tasks]
proba_map = dict(results)

p_slgbm_arr = proba_map["p_slgbm_arr"]
p_nbsvm_arr = proba_map["p_nbsvm_arr"]
p_caps_gru_arr = proba_map["p_caps_gru_arr"]
p_dual_embed_pl_arr = proba_map["p_dual_embed_pl_arr"]
p_dual_embed_mish_arr = proba_map["p_dual_embed_mish_arr"]
p_lstm_glove_tta_arr = proba_map["p_lstm_glove_tta_arr"]
p_dual_embed_dehyp_arr = proba_map["p_dual_embed_dehyp_arr"]
p_lstm_fast_arr = proba_map["p_lstm_fast_arr"]
p_bi_post_arr = proba_map["p_bi_post_arr"]
p_dpcnn_arr = proba_map["p_dpcnn_arr"]
p_dmcnn_arr = proba_map["p_dmcnn_arr"]
p_rcn_arr = proba_map["p_rcn_arr"]
p_attn_300d_arr = proba_map["p_attn_300d_arr"]

assert p_slgbm_arr.shape == (test.shape[0], len(label_cols))
assert p_caps_gru_arr.shape == (test.shape[0], len(label_cols))

(p_caps_gru_arr[:2], p_slgbm_arr[:2])




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1298423650.py in <cell line: 0>()
    241 # CHANGE (timeout fix): Run tasks sequentially to avoid joblib process-spawning and copying huge sparse matrices
    242 # into multiple processes (major wall-time + memory blow-up). This is equivalent work, just without overhead.
--> 243 results = [_run_task(name, est, Xtr, Xte) for (name, est, Xtr, Xte) in tasks]
    244 proba_map = dict(results)
    245 

/tmp/ipykernel_11/1298423650.py in <listcomp>(.0)
    241 # CHANGE (timeout fix): Run tasks sequentially to avoid joblib process-spawning and copying huge sparse matrices
    242 # into multiple processes (major wall-time + memory blow-up). This is equivalent work, just without overhead.
--> 243 results = [_run_task(name, est, Xtr, Xte) for (name, est, Xtr, Xte) in tasks]
    244 proba_map = dict(results)
    245 

/tmp/ipykernel_11/1298423650.py in _run_task(name, est, Xtr, Xte)
    233         print(f"{name}: cache hit in {time.time()-t0:.1f}s")
    234         return name, out
--> 235     out = fit_predict_proba_from_matrices(est, Xtr, Y, Xte)
    236     _pred_cache[key] = out
    237     print(f"{name}: done in {time.time()-t0:.1f}s")

/tmp/ipykernel_11/1298423650.py in fit_predict_proba_from_matrices(estimator, Xtr, Ytr, Xte)
     95 
     96     if isinstance(estimator, (LogisticRegression, ComplementNB)):
---> 97         estimator.fit(Xtr, Ytr)
     98         proba = estimator.predict_proba(Xte)
     99         if isinstance(proba, list):

/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_logistic.py in fit(self, X, y, sample_weight)
   1194             _dtype = [np.float64, np.float32]
   1195 
-> 1196         X, y = self._validate_data(
   1197             X,
   1198             y,

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
    582                 y = check_array(y, input_name="y", **check_y_params)
    583             else:
--> 584                 X, y = check_X_y(X, y, **check_params)
    585             out = X, y
    586 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_X_y(X, y, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, multi_output, ensure_min_samples, ensure_min_features, y_numeric, estimator)
   1120     )
   1121 
-> 1122     y = _check_y(y, multi_output=multi_output, y_numeric=y_numeric, estimator=estimator)
   1123 
   1124     check_consistent_length(X, y)

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in _check_y(y, multi_output, y_numeric, estimator)
   1141     else:
   1142         estimator_name = _check_estimator_name(estimator)
-> 1143         y = column_or_1d(y, warn=True)
   1144         _assert_all_finite(y, input_name="y", estimator_name=estimator_name)
   1145         _ensure_no_complex_data(y)

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in column_or_1d(y, dtype, warn)
   1200         return _asarray_with_order(xp.reshape(y, -1), order="C", xp=xp)
   1201 
-> 1202     raise ValueError(
   1203         "y should be a 1d array, got an array of shape {} instead.".format(shape)
   1204     )

ValueError: y should be a 1d array, got an array of shape (159571, 6) instead.

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



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/418244742.py in <cell line: 0>()
      6 base_ids = test_ids
      7 for name, arr in [
----> 8     ("p_slgbm", p_slgbm_arr),
      9     ("p_caps_gru", p_caps_gru_arr),
     10     ("p_dual_embed_pl", p_dual_embed_pl_arr),

NameError: name 'p_slgbm_arr' is not defined

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



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/962703275.py in <cell line: 0>()
      5 submission = pd.DataFrame({"id": test_ids})
      6 for i, c in enumerate(label_cols):
----> 7     submission[c] = ens[:, i]
      8 
      9 submission[label_cols] = submission[label_cols].fillna(0.5).clip(1e-6, 1 - 1e-6)

NameError: name 'ens' is not defined

## === cell 5
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv", submission.shape)
print(submission.describe().T)

## --- ERROR in outputing the csv:
Invalid submission: Submission is missing the following columns: {'severe_toxic', 'insult', 'threat', 'toxic', 'identity_hate', 'obscene'}
