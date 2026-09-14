# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Build a model that recognizes toxicity and minimizes unintended bias with respect to mentions of identities.

## Metric
We combine several submetrics: An overall ROC-AUC for the full evaluation set, along with the ROC-AUCs on three specific subsets of the test set capturing different aspects of bias.

The final model score looks like:

$$
\text { score }=w_0 A U C_{\text {overall }}+\sum_{a=1}^A w_a M_p\left(m_{s, a}\right)
$$
where:
$A=$ number of submetrics $(3)$
$m_{s, a}=$ bias metric for identity subgroup $s$ using submetric $a$
$w_a=$ a weighting for the relative importance of each submetric; all four $w$ values set to 0.25

Overall AUC: This is the ROC-AUC for the full evaluation set.

### Bias AUCs
To measure unintended bias, we again calculate the ROC-AUC, this time on three specific subsets of the test set for each identity, each capturing a different aspect of unintended bias. 

**Subgroup AUC**: Here, we restrict the data set to only the examples that mention the specific identity subgroup. *A low value in this metric means the model does a poor job of distinguishing between toxic and non-toxic comments that mention the identity*.

**BPSN (Background Positive, Subgroup Negative) AUC**: Here, we restrict the test set to the non-toxic examples that mention the identity and the toxic examples that do not. *A low value in this metric means that the model confuses non-toxic examples that mention the identity with toxic examples that do not*, likely meaning that the model predicts higher toxicity scores than it should for non-toxic examples mentioning the identity.

**BNSP (Background Negative, Subgroup Positive) AUC**: Here, we restrict the test set to the toxic examples that mention the identity and the non-toxic examples that do not. *A low value here means that the model confuses toxic examples that mention the identity with non-toxic examples that do not*, likely meaning that the model predicts lower toxicity scores than it should for toxic examples mentioning the identity.

#### Generalized Mean of Bias AUCs
To combine the per-identity Bias AUCs into one overall measure, we calculate their generalized mean as defined below:

$$
M_p\left(m_s\right)=\left(\frac{1}{N} \sum_{s=1}^N m_s^p\right)^{\frac{1}{p}}
$$

where:
$M_p=$ the $p$ th power-mean function
$m_s=$ the bias metric $m$ calulated for subgroup $S$
$N=$ number of identity subgroups

For this competition, we use a $p$ value of -5 to encourage competitors to improve the model for the identity subgroups with the lowest model performance.

## Submission Format
```
id,prediction
7000000,0.0
7000001,0.0
etc.

```

## Dataset
The text of the individual comment is found in the `comment_text` column. Each comment in Train has a toxicity label (`target`), and models should predict the `target` toxicity for the Test data. This attribute (and all others) are fractional values which represent the fraction of human raters who believed the attribute applied to the given comment. For evaluation, test set examples with `target >= 0.5` will be considered to be in the positive class (toxic).

The data also has several additional toxicity subtype attributes. Models do not need to predict these attributes for the competition, they are included as an additional avenue for research. Subtype attributes are:

- severe_toxicity
- obscene
- threat
- insult
- identity_attack
- sexual_explicit

Additionally, a subset of comments have been labelled with a variety of identity attributes, representing the identities that are *mentioned* in the comment. The columns corresponding to identity attributes are listed below. Only identities shown below will be included in the evaluation calculation.

- **male**
- **female**
- **homosexual_gay_or_lesbian**
- **christian**
- **jewish**
- **muslim**
- **black**
- **white**
- **psychiatric_or_mental_illness**

### Files
- **train.csv** - the training set, which includes toxicity labels and subgroups
- **test.csv** - the test set, which does **not** include toxicity labels or subgroups
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
            description.md (286 lines)
            sample_submission.csv (97321 lines)
            sample_submission.csv.zip (230.8 kB)
            test.csv (205781 lines)
            test.csv.zip (12.5 MB)
            train.csv (3820210 lines)
            train.csv.zip (285.9 MB)
            jigsaw-unintended-bias-in-toxicity-classification/
                description.md (286 lines)
                sample_submission.csv (97321 lines)
                ... and 5 other files
                jigsaw-unintended-bias-in-toxicity-classification/
        input/
            description.md (286 lines)
            sample_submission.csv (97321 lines)
            sample_submission.csv.zip (230.8 kB)
            test.csv (205781 lines)
            test.csv.zip (12.5 MB)
            train.csv (3820210 lines)
            train.csv.zip (285.9 MB)
            jigsaw-unintended-bias-in-toxicity-classification/
                description.md (286 lines)
                sample_submission.csv (97321 lines)
                ... and 5 other files
                jigsaw-unintended-bias-in-toxicity-classification/
        working/
            jigsaw-unintended-bias-in-toxicity-classification/
                description.md (286 lines)
                sample_submission.csv (97321 lines)
                ... and 5 other files
                jigsaw-unintended-bias-in-toxicity-classification/
```

-> data/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv has 97320 rows and 2 columns.
The columns are: id, prediction

-> data/jigsaw-unintended-bias-in-toxicity-classification/test.csv has 205780 rows and 2 columns.
The columns are: id, comment_text

-> data/jigsaw-unintended-bias-in-toxicity-classification/train.csv has 3820209 rows and 45 columns.
The columns are: id, target, comment_text, severe_toxicity, obscene, identity_attack, insult, threat, asian, atheist, bisexual, black, buddhist, christian, female... and 30 more columns

-> data/sample_submission.csv has 97320 rows and 2 columns.
The columns are: id, prediction

-> data/test.csv has 205780 rows and 2 columns.
The columns are: id, comment_text

-> data/train.csv has 3820209 rows and 45 columns.
The columns are: id, target, comment_text, severe_toxicity, obscene, identity_attack, insult, threat, asian, atheist, bisexual, black, buddhist, christian, female... and 30 more columns

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

INPUT_DIR = "/kaggle/input/jigsaw-unintended-bias-in-toxicity-classification"
if not os.path.exists(INPUT_DIR):
    INPUT_DIR = "/kaggle/input"

print("Listing /kaggle/input:")
print(os.listdir("/kaggle/input")[:50])




## === cell 1
train_path = os.path.join(INPUT_DIR, "train.csv")
test_path = os.path.join(INPUT_DIR, "test.csv")
sample_path = os.path.join(INPUT_DIR, "sample_submission.csv")

train_zip = train_path + ".zip"
test_zip = test_path + ".zip"
sample_zip = sample_path + ".zip"

if not (
    os.path.exists(train_path)
    and os.path.exists(test_path)
    and os.path.exists(sample_path)
):
    nested = os.path.join(
        INPUT_DIR, "jigsaw-unintended-bias-in-toxicity-classification"
    )
    train_path = os.path.join(nested, "train.csv")
    test_path = os.path.join(nested, "test.csv")
    sample_path = os.path.join(nested, "sample_submission.csv")
    train_zip = train_path + ".zip"
    test_zip = test_path + ".zip"
    sample_zip = sample_path + ".zip"

train_read_path = train_zip if os.path.exists(train_zip) else train_path
test_read_path = test_zip if os.path.exists(test_zip) else test_path
sample_read_path = sample_zip if os.path.exists(sample_zip) else sample_path

assert os.path.exists(train_read_path), f"train not found at {train_read_path}"
assert os.path.exists(test_read_path), f"test not found at {test_read_path}"
assert os.path.exists(
    sample_read_path
), f"sample_submission not found at {sample_read_path}"

print("Using paths:")
print("train:", train_read_path)
print("test :", test_read_path)
print("sample:", sample_read_path)




## === cell 2
identity_cols = [
    "male",
    "female",
    "homosexual_gay_or_lesbian",
    "christian",
    "jewish",
    "muslim",
    "black",
    "white",
    "psychiatric_or_mental_illness",
]
usecols_train = ["comment_text", "target"] + identity_cols
usecols_test = ["id", "comment_text"]

dtype_train = {c: "float32" for c in (["target"] + identity_cols)}
dtype_train["comment_text"] = "object"
dtype_test = {"id": "int64", "comment_text": "object"}

train = pd.read_csv(
    train_read_path, usecols=usecols_train, dtype=dtype_train, engine="c"
)
test = pd.read_csv(test_read_path, usecols=usecols_test, dtype=dtype_test, engine="c")
sample_sub = pd.read_csv(sample_read_path, engine="c")

print(train.shape, test.shape, sample_sub.shape)
print("Train target NA:", train["target"].isna().mean())
print("Train text NA:", train["comment_text"].isna().mean())
print("Test text NA:", test["comment_text"].isna().mean())




## === cell 3
train_text_series = train["comment_text"].fillna("")
X_text = train_text_series.to_numpy(copy=False)

test_text = test["comment_text"].fillna("").to_numpy(copy=False)

y_cont_all = train["target"].to_numpy(dtype=np.float32, copy=False)
y = (y_cont_all >= 0.5).astype(np.int8, copy=False)

idn_mat = train[identity_cols].fillna(0.0).to_numpy(dtype=np.float32, copy=False)

subgroup_mentioned = (idn_mat >= 0.5).any(axis=1).astype(np.float32, copy=False)
sample_weight = np.where(subgroup_mentioned > 0, 1.5, 1.0).astype(
    np.float32, copy=False
)

print("y (binary) stats:", int(np.min(y)), float(np.mean(y)), int(np.max(y)))
print(
    "subgroup_mentioned rate (in used train):", float(np.mean(subgroup_mentioned > 0))
)




## === cell 4
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import scipy.sparse as sp
import numpy as np
import os
import hashlib
import json

import joblib

sp.csr_matrix.sort_indices  # ensure scipy imported

_CPU = os.cpu_count() or 2
N_JOBS = max(1, min(_CPU - 1, 4))
VEC_JOBS = max(1, _CPU)  # vectorization is the main bottleneck; use all cores

word_tfidf = TfidfVectorizer(
    strip_accents="unicode",
    lowercase=True,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.9,
    max_features=200000,
    sublinear_tf=True,
    dtype=np.float32,
)

char_tfidf = TfidfVectorizer(
    strip_accents="unicode",
    lowercase=True,
    analyzer="char",
    ngram_range=(3, 5),
    min_df=3,
    max_df=0.9,
    max_features=150000,
    sublinear_tf=True,
    dtype=np.float32,
)

clf = LogisticRegression(
    solver="saga",
    penalty="l2",
    max_iter=300,
    C=4.0,
    n_jobs=N_JOBS,
    random_state=42,
    class_weight="balanced",
    warm_start=True,
)

print("Using n_jobs (LR) =", N_JOBS)
print("Using n_jobs (TFIDF/joblib) =", VEC_JOBS)
print("Vectorizers created.")
print("Classifier:", clf)




## === cell 5
import numpy as np
import scipy.sparse as sp

CACHE_DIR = "/kaggle/working/tfidf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def _jsonify(obj):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_jsonify(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, type):
        return f"{obj.__module__}.{obj.__qualname__}"
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    return str(obj)


def _stable_params(obj):
    return _jsonify(obj.get_params(deep=True))


cache_key_payload = {
    "train_rows": int(len(X_text)),
    "test_rows": int(len(test_text)),
    "word_params": _stable_params(word_tfidf),
    "char_params": _stable_params(char_tfidf),
}
cache_key = hashlib.md5(
    json.dumps(cache_key_payload, sort_keys=True).encode("utf-8")
).hexdigest()

xtrain_path = os.path.join(CACHE_DIR, f"X_train_{cache_key}.npz")
xtest_path = os.path.join(CACHE_DIR, f"X_test_{cache_key}.npz")


def save_sparse_fast(path, mat):
    mat = mat.tocsr(copy=False)
    if mat.dtype != np.float32:
        mat = mat.astype(np.float32, copy=False)
    mat.sort_indices()
    sp.save_npz(path, mat, compressed=False)


def load_sparse_fast(path):
    mat = sp.load_npz(path)
    mat = mat.tocsr(copy=False)
    if mat.dtype != np.float32:
        mat = mat.astype(np.float32, copy=False)
    mat.sort_indices()
    return mat


use_cache = os.path.exists(xtrain_path) and os.path.exists(xtest_path)

if use_cache:
    X_train = load_sparse_fast(xtrain_path)
    X_test = load_sparse_fast(xtest_path)
    print("Loaded cached TF-IDF features:", X_train.shape, X_test.shape)
else:
    all_text = np.concatenate([X_text, test_text], axis=0)

    with joblib.parallel_backend("threading", n_jobs=VEC_JOBS):
        Xw_all = word_tfidf.fit_transform(all_text)
        Xc_all = char_tfidf.fit_transform(all_text)

    n_tr = len(X_text)
    X_train = sp.hstack([Xw_all[:n_tr], Xc_all[:n_tr]], format="csr", dtype=np.float32)
    X_test = sp.hstack([Xw_all[n_tr:], Xc_all[n_tr:]], format="csr", dtype=np.float32)
    X_train.sort_indices()
    X_test.sort_indices()

    save_sparse_fast(xtrain_path, X_train)
    save_sparse_fast(xtest_path, X_test)
    print("Built + cached TF-IDF features:", X_train.shape, X_test.shape)

X_train = X_train.tocsr(copy=False)
if X_train.dtype != np.float32:
    X_train = X_train.astype(np.float32, copy=False)
X_train.sort_indices()

clf.fit(X_train, y, sample_weight=sample_weight)




## === cell 6
X_test = X_test.tocsr(copy=False)
if X_test.dtype != np.float32:
    X_test = X_test.astype(np.float32, copy=False)
X_test.sort_indices()

pred = clf.predict_proba(X_test)[:, 1]
pred = np.clip(pred, 0.0, 1.0)




## === cell 7
sub = pd.DataFrame({"id": test["id"].to_numpy(copy=False), "prediction": pred})

assert list(sub.columns) == list(
    sample_sub.columns
), f"Submission columns {sub.columns.tolist()} != {sample_sub.columns.tolist()}"
assert len(sub) == len(
    sample_sub
), f"Submission rows {len(sub)} != sample rows {len(sample_sub)}"
assert sub["id"].isna().sum() == 0
assert sub["prediction"].isna().sum() == 0

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv:", sub.shape)
print(sub.head())




## === cell 8
print(
    "submission.csv written. Prediction range:",
    float(np.min(pred)),
    float(np.max(pred)),
)
