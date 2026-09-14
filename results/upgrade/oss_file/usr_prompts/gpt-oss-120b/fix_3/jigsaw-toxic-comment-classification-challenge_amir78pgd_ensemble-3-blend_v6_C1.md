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
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from joblib import (
    Parallel,
    delayed,
)  # parallelize model training without changing logic

base_path = Path("../input") / "jigsaw-toxic-comment-classification-challenge"
train_path = base_path / "train.csv"
test_path = base_path / "test.csv"
sample_sub_path = base_path / "sample_submission.csv"



## === cell 1
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

train_df["comment_text"] = train_df["comment_text"].fillna(" ")
test_df["comment_text"] = test_df["comment_text"].fillna(" ")



## === cell 2
vectorizer = TfidfVectorizer(
    max_features=200_000, ngram_range=(1, 2), stop_words="english", dtype=np.float32
)

full_vec = vectorizer.fit_transform(train_df["comment_text"])

X_train_vec, X_val_vec, y_train, y_val = train_test_split(
    full_vec, train_df[label_cols], test_size=0.1, random_state=42
)



## === cell 3
models = {}
val_preds = np.zeros((X_val_vec.shape[0], len(label_cols)), dtype=np.float32)


def train_val(col):
    lr = LogisticRegression(
        solver="sag", max_iter=1000, n_jobs=1, C=4.0, class_weight="balanced", verbose=0
    )
    lr.fit(X_train_vec, y_train[col])
    pred = lr.predict_proba(X_val_vec)[:, 1]
    return col, lr, pred


results = Parallel(n_jobs=5)(delayed(train_val)(c) for c in label_cols)

for idx, (col, model, pred) in enumerate(results):
    models[col] = model
    val_preds[:, idx] = pred

val_auc = np.mean(
    [roc_auc_score(y_val[col], val_preds[:, i]) for i, col in enumerate(label_cols)]
)
print(f"Validation mean ROC‑AUC: {val_auc:.6f}")



## === cell 4
test_vec = vectorizer.transform(test_df["comment_text"])

full_models = {}
test_preds = np.zeros((test_vec.shape[0], len(label_cols)), dtype=np.float32)


def train_full(col):
    lr = LogisticRegression(
        solver="sag", max_iter=1000, n_jobs=1, C=4.0, class_weight="balanced", verbose=0
    )
    lr.fit(full_vec, train_df[col])
    pred = lr.predict_proba(test_vec)[:, 1]
    return col, lr, pred


full_results = Parallel(n_jobs=5)(delayed(train_full)(c) for c in label_cols)

for idx, (col, model, pred) in enumerate(full_results):
    full_models[col] = model
    test_preds[:, idx] = pred



## === cell 5
submission = pd.DataFrame(test_preds, columns=label_cols)
submission.insert(0, "id", test_df["id"])
submission.to_csv("submission.csv", index=False)
print("submission.csv written successfully")
