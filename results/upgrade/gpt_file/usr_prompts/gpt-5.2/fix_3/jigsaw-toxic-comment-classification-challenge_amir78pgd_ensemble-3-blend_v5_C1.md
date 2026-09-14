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

print("Listing /kaggle/input (top-level):")
try:
    print(os.listdir("/kaggle/input")[:50])
except FileNotFoundError:
    print("No /kaggle/input found; will rely on /kaggle/data paths.")



## === cell 1
import numpy as np
import pandas as pd
from pathlib import Path


label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

DATA_ROOTS = [
    Path("/kaggle/data/jigsaw-toxic-comment-classification-challenge"),
    Path("/kaggle/data"),
    Path("/kaggle/input/jigsaw-toxic-comment-classification-challenge"),
    Path("/kaggle/input"),
]


def first_existing(*candidates: Path) -> Path:
    for p in candidates:
        if p is not None and p.exists():
            return p
    return None


sample_path = first_existing(*[r / "sample_submission.csv" for r in DATA_ROOTS])
train_path = first_existing(*[r / "train.csv" for r in DATA_ROOTS])
test_path = first_existing(*[r / "test.csv" for r in DATA_ROOTS])

if sample_path is None:
    raise FileNotFoundError(
        "Could not locate sample_submission.csv under known Kaggle roots."
    )
if train_path is None:
    raise FileNotFoundError("Could not locate train.csv under known Kaggle roots.")
if test_path is None:
    raise FileNotFoundError("Could not locate test.csv under known Kaggle roots.")

sample_sub = pd.read_csv(sample_path)
missing = [c for c in (["id"] + label_cols) if c not in sample_sub.columns]
if missing:
    raise ValueError(f"sample_submission.csv missing required columns: {missing}")

print("Using sample_submission:", str(sample_path))
print("Using train:", str(train_path))
print("Using test:", str(test_path))
print("sample_submission shape:", sample_sub.shape)



## === cell 2

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

req_train = ["id", "comment_text"] + label_cols
missing_train = [c for c in req_train if c not in train_df.columns]
if missing_train:
    raise ValueError(f"train.csv missing required columns: {missing_train}")
req_test = ["id", "comment_text"]
missing_test = [c for c in req_test if c not in test_df.columns]
if missing_test:
    raise ValueError(f"test.csv missing required columns: {missing_test}")

X_train_text = train_df["comment_text"].fillna("").astype(str)
X_test_text = test_df["comment_text"].fillna("").astype(str)
y_train = train_df[label_cols].astype(np.int8)

word_tfidf = TfidfVectorizer(
    strip_accents="unicode",
    analyzer="word",
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.9,
    sublinear_tf=True,
)
char_tfidf = TfidfVectorizer(
    strip_accents="unicode",
    analyzer="char",
    ngram_range=(3, 5),
    min_df=3,
    max_df=0.9,
    sublinear_tf=True,
)

vectorizer = FeatureUnion([("word", word_tfidf), ("char", char_tfidf)])

print("Vectorizing text...")
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)
print("X_train shape:", X_train.shape, "X_test shape:", X_test.shape)

clf = OneVsRestClassifier(
    LogisticRegression(
        C=4.0,
        solver="liblinear",
        max_iter=200,
    ),
    n_jobs=None,
)

print("Fitting classifier...")
clf.fit(X_train, y_train)

print("Predicting probabilities...")
test_pred = clf.predict_proba(X_test)  # shape (n_test, 6)

p_res = sample_sub[["id"] + label_cols].copy()
pred_df = pd.DataFrame(test_pred, columns=label_cols)
pred_df.insert(0, "id", test_df["id"].astype(str).values)

base_ids = p_res["id"].astype(str)
aligned = pd.DataFrame({"id": base_ids}).merge(
    pred_df.assign(id=pred_df["id"].astype(str)),
    on="id",
    how="left",
    validate="one_to_one",
)

vals = aligned[label_cols].astype(np.float64)
if vals.isna().any().any():
    n_miss = int(vals.isna().any(axis=1).sum())
    print(
        f"Warning: model predictions missing {n_miss} ids after alignment; filling with 0.5"
    )
    vals = vals.fillna(0.5)

p_res[label_cols] = vals.to_numpy()

p_res[label_cols] = p_res[label_cols].clip(0.0, 1.0)
p_res = p_res[["id"] + label_cols]

print("Result head:")
print(p_res.head())
print("Result shape:", p_res.shape)



## === cell 3
expected_cols = list(sample_sub.columns)
if list(p_res.columns) != expected_cols:
    for c in expected_cols:
        if c not in p_res.columns:
            p_res[c] = 0.5
    p_res = p_res[expected_cols]

assert p_res["id"].notna().all()
assert p_res.shape[0] == sample_sub.shape[0]
for c in label_cols:
    assert ((p_res[c] >= 0.0) & (p_res[c] <= 1.0)).all()



## === cell 4
out_path = "submission.csv"
p_res.to_csv(out_path, index=False)
print(f"Wrote {out_path} with shape {p_res.shape} and columns {list(p_res.columns)}")
print(p_res.head())
