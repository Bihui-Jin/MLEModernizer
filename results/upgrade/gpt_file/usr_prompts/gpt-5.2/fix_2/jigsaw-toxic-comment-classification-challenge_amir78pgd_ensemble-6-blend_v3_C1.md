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
import pandas as pd
import numpy as np

CANDIDATE_BASES = [
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge",
    "/kaggle/data/jigsaw-toxic-comment-classification-challenge",
    "/kaggle/data",
    "/kaggle/input",
    "../input/jigsaw-toxic-comment-classification-challenge",
    "../input",
]


def _find_file(filename: str) -> str:
    for base in CANDIDATE_BASES:
        path = os.path.join(base, filename)
        if os.path.exists(path):
            return path
    if os.path.exists(filename):
        return filename
    raise FileNotFoundError(
        f"Could not locate {filename}. Tried bases: {CANDIDATE_BASES}"
    )


train_path = _find_file("train.csv")
test_path = _find_file("test.csv")
sample_path = _find_file("sample_submission.csv")

print("Using paths:")
print("train:", train_path)
print("test :", test_path)
print("sample:", sample_path)



## === cell 1
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_path)

label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

train_text = train_df["comment_text"].fillna("").astype(str)
test_text = test_df["comment_text"].fillna("").astype(str)

y = train_df[label_cols].astype(np.float32).values

print(train_df.shape, test_df.shape, sample_sub.shape)
print("Label means:", dict(zip(label_cols, y.mean(axis=0).round(4))))



## === cell 2

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

tfidf = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    min_df=2,
    max_features=300000,
    strip_accents="unicode",
    lowercase=True,
)

X_train = tfidf.fit_transform(train_text)
X_test = tfidf.transform(test_text)

base_lr = LogisticRegression(
    C=4.0,
    solver="liblinear",
    max_iter=1000,
)

clf = OneVsRestClassifier(base_lr, n_jobs=None)
clf.fit(X_train, y)



## === cell 3
pred = clf.predict_proba(X_test)

pred = np.clip(pred, 0.0, 1.0)

sub = pd.DataFrame(pred, columns=label_cols)
sub.insert(0, "id", test_df["id"].values)

sub = sub[sample_sub.columns]

print(sub.head())
print("Submission shape:", sub.shape)



## === cell 4
assert list(sub.columns) == ["id"] + label_cols, "Submission columns/order mismatch."
assert sub["id"].isna().sum() == 0, "Missing ids in submission."
for c in label_cols:
    assert np.isfinite(sub[c].values).all(), f"Non-finite values in column {c}"
    assert (
        (sub[c] >= 0) & (sub[c] <= 1)
    ).all(), f"Out-of-range probabilities in column {c}"



## === cell 5
out_path = "submission.csv"
sub.to_csv(out_path, index=False)
print("Wrote:", out_path)
print("File size (bytes):", os.path.getsize(out_path))
