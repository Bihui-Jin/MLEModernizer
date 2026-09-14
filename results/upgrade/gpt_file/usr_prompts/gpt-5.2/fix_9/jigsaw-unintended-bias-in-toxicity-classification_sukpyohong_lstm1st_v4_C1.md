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

# 5. Target score

0.9018441105821952

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.68777) has done: 'The current notebook fails because it tries to copy a submission from a non-existent input dataset (`../input/bidirection-lstm11/submission.csv`), so `sub` is never created and no `submission.csv` is written. I replace that with a minimal, fully self-contained baseline that reads the provided `train.csv` and `test.csv`, trains a lightweight text model, and writes a valid `submission.csv` with the required `id,prediction` columns. To keep runtime under control with the very large training set, the code trains on a capped sample size and uses a scikit-learn pipeline that is available in your environment. This reliably yield a valid submission and a reasonable score (and can be tuned later toward the exact target if you share the achieved score).'
- What this solution (achieved 0.70266) has done: 'I fix the training failure by converting the continuous `target` labels into a binary classification target (`target >= 0.5`), which is what the competition metric uses and what `LogisticRegression` requires. I also keep the existing sample-weighting scheme but apply it to the downsampled rows consistently by subsetting the identity matrix before computing `subgroup_mentioned`. Finally, I make inference robust by ensuring the model is fit before calling `predict_proba`, and I always write a valid `submission.csv` with the required `id,prediction` columns.'

# 9. Code solution

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

assert os.path.exists(train_path), f"train.csv not found at {train_path}"
assert os.path.exists(test_path), f"test.csv not found at {test_path}"
assert os.path.exists(sample_path), f"sample_submission.csv not found at {sample_path}"

print("Using paths:")
print(train_path)
print(test_path)
print(sample_path)




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

train = pd.read_csv(train_path, usecols=usecols_train, dtype=dtype_train)
test = pd.read_csv(test_path, usecols=usecols_test, dtype=dtype_test)
sample_sub = pd.read_csv(sample_path)

print(train.shape, test.shape, sample_sub.shape)
print("Train target NA:", train["target"].isna().mean())
print("Train text NA:", train["comment_text"].isna().mean())
print("Test text NA:", test["comment_text"].isna().mean())




## === cell 3
train_text_series = train["comment_text"].fillna("")
X_text = train_text_series.to_numpy()

test_text = test["comment_text"].fillna("").to_numpy()

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
import joblib
import hashlib
import json

sp.csr_matrix.sort_indices  # touch to ensure scipy is imported

N_JOBS = max(1, (os.cpu_count() or 2) - 1)

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
    n_jobs=N_JOBS,  # char analyzer benefits substantially; preserves exact semantics
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

print("Using n_jobs =", N_JOBS)
print("Vectorizers:", word_tfidf, char_tfidf)
print("Classifier:", clf)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4013625512.py in <cell line: 0>()
     24 )
     25 
---> 26 char_tfidf = TfidfVectorizer(
     27     strip_accents="unicode",
     28     lowercase=True,

TypeError: TfidfVectorizer.__init__() got an unexpected keyword argument 'n_jobs'

## === cell 5
CACHE_DIR = "/kaggle/working/tfidf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def _stable_params(obj):
    return obj.get_params(deep=True)


cache_key_payload = {
    "train_rows": int(len(X_text)),
    "test_rows": int(len(test_text)),
    "word_params": _stable_params(word_tfidf),
    "char_params": _stable_params(char_tfidf),
}
cache_key = hashlib.md5(
    json.dumps(cache_key_payload, sort_keys=True).encode("utf-8")
).hexdigest()

word_path = os.path.join(CACHE_DIR, f"word_{cache_key}.joblib")
char_path = os.path.join(CACHE_DIR, f"char_{cache_key}.joblib")
xtrain_path = os.path.join(CACHE_DIR, f"X_train_{cache_key}.npz")
xtest_path = os.path.join(CACHE_DIR, f"X_test_{cache_key}.npz")


def save_sparse_csr(path, mat):
    mat = mat.tocsr()
    mat.sort_indices()
    np.savez_compressed(
        path,
        data=mat.data,
        indices=mat.indices,
        indptr=mat.indptr,
        shape=np.array(mat.shape, dtype=np.int64),
    )


def load_sparse_csr(path):
    loader = np.load(path, allow_pickle=False)
    shape = tuple(loader["shape"].tolist())
    mat = sp.csr_matrix(
        (loader["data"], loader["indices"], loader["indptr"]), shape=shape
    )
    mat.sort_indices()
    return mat


use_cache = (
    os.path.exists(word_path)
    and os.path.exists(char_path)
    and os.path.exists(xtrain_path)
    and os.path.exists(xtest_path)
)

if use_cache:
    word_tfidf = joblib.load(word_path)
    char_tfidf = joblib.load(char_path)
    X_train = load_sparse_csr(xtrain_path)
    X_test = load_sparse_csr(xtest_path)
    print("Loaded cached TF-IDF features:", X_train.shape, X_test.shape)
else:
    all_text = np.concatenate([X_text, test_text], axis=0)

    Xw_all = word_tfidf.fit_transform(all_text)
    Xc_all = char_tfidf.fit_transform(all_text)

    n_tr = len(X_text)
    Xw_train, Xw_test = Xw_all[:n_tr], Xw_all[n_tr:]
    Xc_train, Xc_test = Xc_all[:n_tr], Xc_all[n_tr:]

    X_train = sp.hstack([Xw_train, Xc_train], format="csr", dtype=np.float32)
    X_test = sp.hstack([Xw_test, Xc_test], format="csr", dtype=np.float32)

    X_train.sort_indices()
    X_test.sort_indices()

    joblib.dump(word_tfidf, word_path, compress=3)
    joblib.dump(char_tfidf, char_path, compress=3)
    save_sparse_csr(xtrain_path, X_train)
    save_sparse_csr(xtest_path, X_test)
    print("Built + cached TF-IDF features:", X_train.shape, X_test.shape)

clf.fit(X_train, y, sample_weight=sample_weight)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2162740217.py in <cell line: 0>()
     14     "test_rows": int(len(test_text)),
     15     "word_params": _stable_params(word_tfidf),
---> 16     "char_params": _stable_params(char_tfidf),
     17 }
     18 cache_key = hashlib.md5(

NameError: name 'char_tfidf' is not defined

## === cell 6
pred = clf.predict_proba(X_test)[:, 1]
pred = np.clip(pred, 0.0, 1.0)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/302129740.py in <cell line: 0>()
----> 1 pred = clf.predict_proba(X_test)[:, 1]
      2 pred = np.clip(pred, 0.0, 1.0)
      3 
      4 

NameError: name 'clf' is not defined

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




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4001107800.py in <cell line: 0>()
----> 1 sub = pd.DataFrame({"id": test["id"].to_numpy(copy=False), "prediction": pred})
      2 
      3 assert list(sub.columns) == list(
      4     sample_sub.columns
      5 ), f"Submission columns {sub.columns.tolist()} != {sample_sub.columns.tolist()}"

NameError: name 'pred' is not defined

## === cell 8
print(
    "submission.csv written. Prediction range:",
    float(np.min(pred)),
    float(np.max(pred)),
)

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1537048734.py in <cell line: 0>()
      1 print(
      2     "submission.csv written. Prediction range:",
----> 3     float(np.min(pred)),
      4     float(np.max(pred)),
      5 )

NameError: name 'pred' is not defined
