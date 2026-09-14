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

3.7

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

BASE_CANDIDATES = [
    "../input/jigsaw-toxic-comment-classification-challenge",
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge",
    "../input",
    "/kaggle/input",
]


def _find_file(filename: str):
    for base in BASE_CANDIDATES:
        cand = os.path.join(base, filename)
        if os.path.exists(cand):
            return cand
        cand2 = os.path.join(
            base, "jigsaw-toxic-comment-classification-challenge", filename
        )
        if os.path.exists(cand2):
            return cand2
    raise FileNotFoundError(
        f"Could not find {filename} under any of: {BASE_CANDIDATES}"
    )


train_path = _find_file("train.csv")
test_path = _find_file("test.csv")
sample_path = _find_file("sample_submission.csv")

print("Using:")
print(" train:", train_path)
print(" test :", test_path)
print(" sample:", sample_path)



## === cell 1
import numpy as np
import pandas as pd

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample = pd.read_csv(sample_path)

label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

assert "id" in train.columns and "comment_text" in train.columns
assert "id" in test.columns and "comment_text" in test.columns
for c in label_cols:
    assert c in train.columns, f"Missing label col {c} in train.csv"
for c in ["id"] + label_cols:
    assert c in sample.columns, f"Missing col {c} in sample_submission.csv"

train["comment_text"] = train["comment_text"].fillna("")
test["comment_text"] = test["comment_text"].fillna("")

X_train_text = train["comment_text"].astype(str).values
X_test_text = test["comment_text"].astype(str).values
Y = train[label_cols].astype(np.float32).values

print(train.shape, test.shape, sample.shape)



## === cell 2

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression


def fit_predict_proba(vectorizer, X_tr, Y_tr, X_te, C=4.0, max_iter=300, n_jobs=1):
    """
    Fits one-vs-rest logistic regression per label and returns test probabilities.
    Kept intentionally simple and deterministic.
    """
    Xtr = vectorizer.fit_transform(X_tr)
    Xte = vectorizer.transform(X_te)

    preds = np.zeros((Xte.shape[0], Y_tr.shape[1]), dtype=np.float32)

    for j in range(Y_tr.shape[1]):
        y = Y_tr[:, j]
        if np.all(y == y[0]):
            preds[:, j] = float(y[0])
            continue
        clf = LogisticRegression(
            C=C,
            solver="liblinear",
            max_iter=max_iter,
            random_state=42,
        )
        clf.fit(Xtr, y)
        preds[:, j] = clf.predict_proba(Xte)[:, 1].astype(np.float32)

    return preds


vec1 = TfidfVectorizer(
    strip_accents="unicode",
    lowercase=True,
    analyzer="word",
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.9,
    sublinear_tf=True,
    token_pattern=r"(?u)\b\w\w+\b",
)
vec2 = TfidfVectorizer(
    strip_accents="unicode",
    lowercase=True,
    analyzer="word",
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    token_pattern=r"(?u)\b\w\w+\b",
)
vec3 = TfidfVectorizer(
    strip_accents="unicode",
    lowercase=True,
    analyzer="char",
    ngram_range=(3, 5),
    min_df=3,
    max_df=0.9,
    sublinear_tf=True,
)
vec4 = CountVectorizer(
    strip_accents="unicode",
    lowercase=True,
    analyzer="word",
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.9,
    token_pattern=r"(?u)\b\w\w+\b",
)



## === cell 3
pred1 = fit_predict_proba(vec1, X_train_text, Y, X_test_text, C=4.0, max_iter=300)
pred2 = fit_predict_proba(vec2, X_train_text, Y, X_test_text, C=3.0, max_iter=300)
pred3 = fit_predict_proba(vec3, X_train_text, Y, X_test_text, C=2.0, max_iter=300)
pred4 = fit_predict_proba(vec4, X_train_text, Y, X_test_text, C=4.0, max_iter=300)

pred = (pred1 + pred2 + pred3 + pred4) / 4.0

pred = np.clip(pred, 1e-6, 1 - 1e-6)

print("Pred shape:", pred.shape)



## === cell 4
submission = pd.DataFrame(pred, columns=label_cols)
submission.insert(0, "id", test["id"].values)

submission = submission[["id"] + label_cols]

assert submission.shape[0] == test.shape[0]
assert list(submission.columns) == ["id"] + label_cols
print(submission.head())



## === cell 5
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
