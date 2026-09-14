# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.654157714273941

# 6. Current score

0.98006

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.98006) has done: 'The timeout is dominated by fitting TF‑IDF on the concatenated train+test (over 1.1M rows) and by building/merging two large vectorizers via `FeatureUnion`, which creates heavy intermediate objects and overhead. We keep identical features and model but avoid fitting on test by using `vocabulary_` from train for the test transform (semantically equivalent at inference time and standard for this competition), and we replace `FeatureUnion` with explicit vectorizers plus a single `hstack`, eliminating parallel union overhead and extra copies. We also minimize memory churn by using `copy=False`, avoiding unnecessary dtype conversions, and ensuring CSR/int32 indices once. LogisticRegression/OneVsRest and all hyperparameters remain unchanged.'
- What this solution (achieved 0.98006) has done: 'Your current score (0.98006) is far above the target (0.65416), so to move *toward* the target we should intentionally reduce model performance while keeping the same pipeline and producing a valid submission. The smallest, lowest-risk way is to keep the exact same TF‑IDF features and LogisticRegression/OVR training, but apply stronger post-processing shrinkage that pulls predictions toward 0.5 (which reduces AUC without changing model core logic). This preserves evaluation semantics (still valid probabilities per class) and keeps runtime essentially unchanged. I add a single calibration/shrink step after `predict_proba` controlled by a fixed constant chosen to substantially lower separability.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

BASE = "/kaggle/input/jigsaw-toxic-comment-classification-challenge"
train_path = os.path.join(BASE, "train.csv")
test_path = os.path.join(BASE, "test.csv")
sample_path = os.path.join(BASE, "sample_submission.csv")

print("Found files in BASE:", os.listdir(BASE)[:10])
print("train_path exists:", os.path.exists(train_path))
print("test_path exists:", os.path.exists(test_path))
print("sample_path exists:", os.path.exists(sample_path))



## === cell 1
target_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

train_df = pd.read_csv(
    train_path,
    usecols=["id", "comment_text"] + target_cols,
    dtype={**{c: np.int8 for c in target_cols}, "id": "string"},
)
test_df = pd.read_csv(
    test_path,
    usecols=["id", "comment_text"],
    dtype={"id": "string"},
)
sample_sub = pd.read_csv(sample_path, dtype={"id": "string"})

train_text = train_df["comment_text"].fillna("").to_numpy()
test_text = test_df["comment_text"].fillna("").to_numpy()

y = train_df[target_cols].to_numpy(dtype=np.int8, copy=False)

print(train_df.shape, test_df.shape, sample_sub.shape)
print(
    "Targets prevalence:\n",
    pd.Series(y.mean(axis=0), index=target_cols).sort_values(ascending=False),
)



## === cell 2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from scipy import sparse

word_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.9,
    max_features=200_000,
    strip_accents="unicode",
    lowercase=True,
    sublinear_tf=True,
    dtype=np.float32,
)

char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=3,
    max_df=0.9,
    max_features=300_000,
    strip_accents="unicode",
    lowercase=True,
    sublinear_tf=True,
    dtype=np.float32,
)

Xw_train = word_vectorizer.fit_transform(train_text)
Xc_train = char_vectorizer.fit_transform(train_text)
X_train = sparse.hstack([Xw_train, Xc_train], format="csr")

Xw_test = word_vectorizer.transform(test_text)
Xc_test = char_vectorizer.transform(test_text)
X_test = sparse.hstack([Xw_test, Xc_test], format="csr")

X_train.sort_indices()
X_test.sort_indices()

X_train = X_train.astype(np.float32, copy=False)
X_test = X_test.astype(np.float32, copy=False)

if X_train.indices.dtype != np.int32:
    X_train.indices = X_train.indices.astype(np.int32, copy=False)
if X_train.indptr.dtype != np.int32:
    X_train.indptr = X_train.indptr.astype(np.int32, copy=False)
if X_test.indices.dtype != np.int32:
    X_test.indices = X_test.indices.astype(np.int32, copy=False)
if X_test.indptr.dtype != np.int32:
    X_test.indptr = X_test.indptr.astype(np.int32, copy=False)

base_clf = LogisticRegression(
    solver="saga",
    C=4.0,
    max_iter=1000,
    tol=1e-3,  # keep as provided
    n_jobs=-1,
    random_state=0,
)

clf = OneVsRestClassifier(base_clf, n_jobs=-1)

print("Vectorized shapes:", X_train.shape, X_test.shape)
print("nnz train/test:", X_train.nnz, X_test.nnz)



## === cell 3
clf.fit(X_train, y)

test_pred = clf.predict_proba(X_test)

SHRINK_ALPHA = 0.15  # 1.0 = original preds; smaller -> closer to 0.5 -> lower AUC
test_pred = 0.5 + SHRINK_ALPHA * (test_pred - 0.5)

test_pred = np.clip(test_pred, 0.0, 1.0)

print("Pred shape:", test_pred.shape)
print("Pred min/max:", float(test_pred.min()), float(test_pred.max()))



## === cell 4
submission = pd.DataFrame(test_pred, columns=target_cols)
submission.insert(0, "id", test_df["id"].to_numpy())
submission = submission[["id"] + target_cols]

assert list(submission.columns) == list(sample_sub.columns), (
    submission.columns,
    sample_sub.columns,
)
assert submission.shape[0] == sample_sub.shape[0], (submission.shape, sample_sub.shape)

submission.head()



## === cell 5
out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path, "with shape:", submission.shape)
print(pd.read_csv(out_path, nrows=3))
