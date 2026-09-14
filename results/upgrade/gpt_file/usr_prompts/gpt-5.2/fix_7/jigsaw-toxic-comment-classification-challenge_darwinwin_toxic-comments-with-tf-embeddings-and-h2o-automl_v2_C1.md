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

train_text = train_df["comment_text"].fillna("").astype(str).to_numpy()
test_text = test_df["comment_text"].fillna("").astype(str).to_numpy()

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

all_text = np.concatenate([train_text, test_text], axis=0)
n_train = train_text.shape[0]

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

Xw_all = word_vectorizer.fit_transform(all_text)
Xc_all = char_vectorizer.fit_transform(all_text)

Xw_train = Xw_all[:n_train]
Xw_test = Xw_all[n_train:]
Xc_train = Xc_all[:n_train]
Xc_test = Xc_all[n_train:]

X_train = sparse.hstack([Xw_train, Xc_train], format="csr", dtype=np.float32)
X_test = sparse.hstack([Xw_test, Xc_test], format="csr", dtype=np.float32)

base_clf = LogisticRegression(
    solver="saga",
    C=4.0,
    max_iter=1000,
    n_jobs=-1,
    random_state=0,
)

clf = OneVsRestClassifier(base_clf, n_jobs=-1)

print("Vectorized shapes:", X_train.shape, X_test.shape)



## === cell 3
clf.fit(X_train, y)

test_pred = clf.predict_proba(X_test)
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
