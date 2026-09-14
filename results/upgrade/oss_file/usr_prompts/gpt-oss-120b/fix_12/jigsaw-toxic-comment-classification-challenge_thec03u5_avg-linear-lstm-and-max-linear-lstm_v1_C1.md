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

3.6

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
import multiprocessing as mp

NUM_CPUS = max(1, mp.cpu_count())
os.environ["OMP_NUM_THREADS"] = str(NUM_CPUS)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score



## === cell 1
possible_paths = [
    "../input/jigsaw-toxic-comment-classification-challenge/train.csv",
    "../input/jigsaw-toxic-comment-classification-challenge/test.csv",
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv",
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv",
    "data/jigsaw-toxic-comment-classification-challenge/train.csv",
    "data/jigsaw-toxic-comment-classification-challenge/test.csv",
    "data/train.csv",
    "data/test.csv",
    "train.csv",
    "test.csv",
]


def find_file(name):
    for p in possible_paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{name} not found in any of the expected locations")


train_path = find_file("train.csv")
test_path = find_file("test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)



## === cell 2
label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

train_text, val_text, train_labels, val_labels = train_test_split(
    train_df["comment_text"],
    train_df[label_cols],
    test_size=0.10,
    random_state=42,
)



## === cell 3
vectorizer = TfidfVectorizer(
    max_features=400000,
    ngram_range=(1, 3),
    stop_words="english",
    dtype=np.float32,
    sublinear_tf=True,
)
vectorizer.fit(train_text)

combined_text = pd.concat([train_text, val_text])
X_combined = vectorizer.transform(combined_text)

X_train = X_combined[: len(train_text)]
X_val = X_combined[len(train_text) :]

X_test = vectorizer.transform(test_df["comment_text"])



## === cell 4
models = {}
val_preds = pd.DataFrame(index=val_labels.index, columns=label_cols)

for col in label_cols:
    lr = LogisticRegression(
        C=20.0,
        solver="saga",
        max_iter=1000,
        n_jobs=NUM_CPUS,
        class_weight="balanced",
        penalty="l2",
        random_state=42,
    )
    lr.fit(X_train, train_labels[col])
    val_pred = lr.predict_proba(X_val)[:, 1]
    models[col] = lr
    val_preds[col] = val_pred

auc_scores = {col: roc_auc_score(val_labels[col], val_preds[col]) for col in label_cols}
mean_auc = np.mean(list(auc_scores.values()))
print("Validation ROC‑AUC per label:")
for k, v in auc_scores.items():
    print(f"  {k}: {v:.5f}")
print(f"Mean ROC‑AUC: {mean_auc:.5f}")



## === cell 5
test_preds = pd.DataFrame()
test_preds["id"] = test_df["id"]

for col in label_cols:
    test_preds[col] = models[col].predict_proba(X_test)[:, 1]

submission_path = os.path.join(os.getcwd(), "submission.csv")
test_preds.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
