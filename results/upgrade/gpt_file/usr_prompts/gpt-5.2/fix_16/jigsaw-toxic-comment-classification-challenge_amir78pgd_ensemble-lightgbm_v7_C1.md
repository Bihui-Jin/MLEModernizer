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
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")

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
Y = train[label_cols].to_numpy(dtype=np.int32, copy=False)

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
import joblib
import multiprocessing

from joblib import Parallel, delayed

CACHE_DIR = os.path.join(".", "__cache__")
os.makedirs(CACHE_DIR, exist_ok=True)


def _stable_text_fingerprint(x_train, x_test, max_items=512):
    """Deterministic small fingerprint of input text arrays to safely reuse caches."""
    h = hashlib.md5()
    h.update(str(len(x_train)).encode())
    h.update(str(len(x_test)).encode())

    def _upd(arr):
        n = len(arr)
        take = min(max_items // 2, n)
        head = arr[:take]
        tail = arr[-take:] if take > 0 else arr[:0]
        for s in np.concatenate([head, tail], axis=0):
            if not isinstance(s, str):
                s = "" if s is None else str(s)
            h.update(s.encode("utf-8", errors="ignore"))

    _upd(x_train)
    _upd(x_test)
    return h.hexdigest()


TEXT_FP = _stable_text_fingerprint(X_train, X_test)


def fit_vectorizer_fast(
    vectorizer, X_train_text, X_test_text, vocab_fit_max=200_000, cache_tag=""
):
    """
    Speed: cache the fitted vectorizer + transformed CSR matrices to disk.
    Correctness: if cache is hit, we return exactly the previously computed matrices for the same data+params.
    """
    t0 = time.time()

    vec_params = vectorizer.get_params(deep=True)
    key_src = (
        repr(vec_params)
        + f"|vocab_fit_max={vocab_fit_max}|TEXT_FP={TEXT_FP}|tag={cache_tag}"
    )
    key = hashlib.md5(key_src.encode("utf-8")).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"tfidf_{key}.joblib")

    if os.path.exists(cache_path):
        obj = joblib.load(cache_path)
        Xt, Xv = obj["Xt"], obj["Xv"]

        if not sparse.isspmatrix_csr(Xt):
            Xt = Xt.tocsr()
        if not sparse.isspmatrix_csr(Xv):
            Xv = Xv.tocsr()
        if Xt.dtype != np.float32:
            Xt = Xt.astype(np.float32, copy=False)
        if Xv.dtype != np.float32:
            Xv = Xv.astype(np.float32, copy=False)

        print(
            f"Vectorizer {vectorizer.analyzer}-{vectorizer.ngram_range} "
            f"(cache hit) vocab={Xt.shape[1]:,} "
            f"Xtr nnz={Xt.nnz:,} Xte nnz={Xv.nnz:,} "
            f"in {time.time()-t0:.1f}s"
        )
        return Xt, Xv

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

    n_tr = len(X_train_text)
    Xt = X_all_vec[:n_tr]
    Xv = X_all_vec[n_tr:]

    joblib.dump({"Xt": Xt, "Xv": Xv}, cache_path, compress=0)

    print(
        f"Vectorizer {vectorizer.analyzer}-{vectorizer.ngram_range} "
        f"vocab={len(vectorizer.vocabulary_):,} "
        f"Xtr nnz={Xt.nnz:,} Xte nnz={Xv.nnz:,} "
        f"in {time.time()-t0:.1f}s"
    )
    return Xt, Xv


def _ovr_n_jobs_for_estimator(estimator):
    return int(os.environ.get("OVR_N_JOBS", "2"))


def fit_predict_proba_from_matrices(estimator, Xtr, Ytr, Xte):
    """
    Keep core logic: OneVsRestClassifier wrapper and calibrated SVM where specified.
    """
    clf = OneVsRestClassifier(estimator, n_jobs=_ovr_n_jobs_for_estimator(estimator))
    clf.fit(Xtr, Ytr)
    proba = clf.predict_proba(Xte)

    proba = np.asarray(proba)
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
    vec_word, X_train, X_test, vocab_fit_max=200_000, cache_tag="word"
)
Xtr_char, Xte_char = fit_vectorizer_fast(
    vec_char, X_train, X_test, vocab_fit_max=200_000, cache_tag="char"
)

svm_n_jobs = int(os.environ.get("SVM_N_JOBS", "2"))

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


def _cache_key(name, est, Xtr, Xte):
    est_params = (
        repr(est.get_params(deep=True)) if hasattr(est, "get_params") else repr(est)
    )
    h = hashlib.md5(est_params.encode("utf-8")).hexdigest()
    return (name, TEXT_FP, h, Xtr.shape, int(Xtr.nnz), Xte.shape, int(Xte.nnz))


def _pred_cache_path(key):
    key_s = repr(key).encode("utf-8")
    h = hashlib.md5(key_s).hexdigest()
    return os.path.join(CACHE_DIR, f"pred_{h}.npy")


def _run_task(name, est, Xtr, Xte):
    """
    Speed: per-model prediction caching to disk (dominant cost = fit).
    Correctness: exact reuse of previously computed predictions for same data+model params.
    """
    t0 = time.time()
    key = _cache_key(name, est, Xtr, Xte)
    pth = _pred_cache_path(key)

    if os.path.exists(pth):
        out = np.load(pth)
        print(f"{name}: cache hit in {time.time()-t0:.1f}s")
        return name, out

    out = fit_predict_proba_from_matrices(est, Xtr, Y, Xte)
    np.save(pth, out)
    print(f"{name}: done in {time.time()-t0:.1f}s")
    return name, out


default_outer = max(1, min(4, (multiprocessing.cpu_count() or 2) - 1))
OUTER_N_JOBS = int(os.environ.get("TASK_N_JOBS", str(default_outer)))

results = Parallel(n_jobs=OUTER_N_JOBS, backend="loky", prefer="processes", verbose=0)(
    delayed(_run_task)(name, est, Xtr, Xte) for (name, est, Xtr, Xte) in tasks
)
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
    assert arr.shape == (
        test.shape[0],
        len(label_cols),
    ), f"{name} bad shape: {arr.shape}"

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
    sample_sub["id"].to_numpy(), test_ids
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
