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

3.9

# 3. Installed packages

geopandas==0.14.4
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

os.environ.setdefault("PYTHONHASHSEED", "0")
np.random.seed(42)

base_dir = "/kaggle/input/jigsaw-unintended-bias-in-toxicity-classification"
fallback_dir = "/kaggle/input"


def pick_path(filename: str) -> str:
    p1 = os.path.join(base_dir, filename)
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(fallback_dir, filename)
    if os.path.exists(p2):
        return p2
    raise FileNotFoundError(
        f"Could not find {filename} in {base_dir} or {fallback_dir}"
    )


train_path = pick_path("train.csv")
test_path = pick_path("test.csv")
sample_path = pick_path("sample_submission.csv")

sample_sub = pd.read_csv(sample_path, usecols=["id", "prediction"])
required_cols = ["id", "prediction"]
missing = [c for c in required_cols if c not in sample_sub.columns]
if missing:
    raise ValueError(f"sample_submission is missing required columns: {missing}")
sample_sub = sample_sub[required_cols].copy()

print("Paths:", train_path, test_path, sample_path)




## === cell 1

from typing import Iterator, Tuple

TRAIN_CHUNK_ROWS = 200_000
TEST_CHUNK_ROWS = 200_000

TRAIN_USECOLS = ["comment_text", "target"]
TEST_USECOLS = ["id", "comment_text"]

TRAIN_DTYPES = {"comment_text": "object", "target": "float32"}
TEST_DTYPES = {"id": "int64", "comment_text": "object"}


def iter_train_chunks() -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    reader = pd.read_csv(
        train_path,
        usecols=TRAIN_USECOLS,
        dtype=TRAIN_DTYPES,
        chunksize=TRAIN_CHUNK_ROWS,
    )
    for df in reader:
        txt = df["comment_text"].fillna("").to_numpy()
        y = (df["target"].to_numpy(copy=False) >= 0.5).astype(np.int32, copy=False)
        yield txt, y


def iter_test_chunks() -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    reader = pd.read_csv(
        test_path,
        usecols=TEST_USECOLS,
        dtype=TEST_DTYPES,
        chunksize=TEST_CHUNK_ROWS,
    )
    for df in reader:
        ids = df["id"].to_numpy(copy=False)
        txt = df["comment_text"].fillna("").to_numpy()
        yield ids, txt


_pos = 0
_n = 0
for i, (txt, y) in enumerate(iter_train_chunks()):
    _pos += int(y.sum())
    _n += int(len(y))
    if i >= 2:  # just a quick check
        break
print("Sanity chunks loaded. Partial positive_rate:", _pos / max(_n, 1))




## === cell 2
from sklearn.feature_extraction.text import HashingVectorizer
from scipy.sparse import hstack, csr_matrix

N_WORD = 2**20  # 1,048,576 hashed word features
N_CHAR = 2**20  # 1,048,576 hashed char features
N_TOTAL = N_WORD + N_CHAR

word_hasher = HashingVectorizer(
    strip_accents="unicode",
    lowercase=True,
    analyzer="word",
    ngram_range=(1, 2),
    n_features=N_WORD,
    alternate_sign=False,
    norm=None,
)

char_hasher = HashingVectorizer(
    strip_accents="unicode",
    lowercase=True,
    analyzer="char",
    ngram_range=(3, 5),
    n_features=N_CHAR,
    alternate_sign=False,
    norm=None,
)

print("HashingVectorizers ready. Total features:", N_TOTAL)




## === cell 3


def accumulate_df_counts_from_counts(X_counts: csr_matrix, out_df: np.ndarray) -> None:
    indptr = X_counts.indptr
    indices = X_counts.indices
    for i in range(X_counts.shape[0]):
        start, end = indptr[i], indptr[i + 1]
        if end > start:
            cols = indices[start:end]
            out_df[np.unique(cols)] += 1


df_counts = np.zeros(N_TOTAL, dtype=np.int64)
n_docs = 0

for txt, _y in iter_train_chunks():
    Xw = word_hasher.transform(txt)
    Xc = char_hasher.transform(txt)
    X_counts = hstack([Xw, Xc], format="csr")
    accumulate_df_counts_from_counts(X_counts, df_counts)
    n_docs += X_counts.shape[0]

idf = (np.log((1.0 + n_docs) / (1.0 + df_counts.astype(np.float64))) + 1.0).astype(
    np.float64, copy=False
)

print(
    "Computed IDF. n_docs:",
    n_docs,
    "idf stats:",
    float(idf.min()),
    float(idf.mean()),
    float(idf.max()),
)




## === cell 4

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import normalize

C = 4.0
alpha = 1.0 / (C * float(n_docs))

clf = SGDClassifier(
    loss="log_loss",
    penalty="l2",
    alpha=alpha,
    fit_intercept=True,
    max_iter=1,  # one pass per epoch; we control epochs explicitly
    tol=None,
    shuffle=False,  # determinism + consistent with streaming
    random_state=42,
    learning_rate="optimal",
    average=False,
)

EPOCHS = 100
classes = np.array([0, 1], dtype=np.int32)


def tfidf_transform_counts(X_counts: csr_matrix) -> csr_matrix:
    X = X_counts.copy()
    X.data = np.log(X.data, dtype=np.float64) + 1.0
    X = X.multiply(idf)
    return normalize(X, norm="l2", copy=False)


for epoch in range(EPOCHS):
    for batch_i, (txt, yb) in enumerate(iter_train_chunks()):
        Xw = word_hasher.transform(txt)
        Xc = char_hasher.transform(txt)
        X_counts = hstack([Xw, Xc], format="csr")
        X_tfidf = tfidf_transform_counts(X_counts)
        if epoch == 0 and batch_i == 0:
            clf.partial_fit(X_tfidf, yb, classes=classes)
        else:
            clf.partial_fit(X_tfidf, yb)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} done")

print("Training done.")




## === cell 5

test_ids_all = []
pred_all = []

for ids, txt in iter_test_chunks():
    Xw = word_hasher.transform(txt)
    Xc = char_hasher.transform(txt)
    X_counts = hstack([Xw, Xc], format="csr")
    X_tfidf = tfidf_transform_counts(X_counts)
    p = clf.predict_proba(X_tfidf)[:, 1].astype(np.float64, copy=False)
    p = np.clip(p, 0.0, 1.0)
    test_ids_all.append(ids)
    pred_all.append(p)

test_ids = np.concatenate(test_ids_all)
pred = np.concatenate(pred_all)

print(
    "Pred stats:",
    float(pred.min()),
    float(pred.mean()),
    float(pred.max()),
    "n_test:",
    pred.shape[0],
)




## === cell 6
pred_df = pd.DataFrame({"id": test_ids, "prediction": pred})
res = sample_sub[["id"]].merge(pred_df, on="id", how="left")
res["prediction"] = (
    pd.to_numeric(res["prediction"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
)

out_path = "submission.csv"
res.to_csv(out_path, index=False)

print(f"Wrote {out_path} with shape={res.shape} and columns={list(res.columns)}")
print(res.head())
print("Missing predictions:", int(res["prediction"].isna().sum()))
