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
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score

DATA_ROOT = "/kaggle/input/jigsaw-toxic-comment-classification-challenge"
TRAIN_PATH = os.path.join(DATA_ROOT, "train.csv")
TEST_PATH = os.path.join(DATA_ROOT, "test.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

TARGET_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

train_df["comment_text"] = train_df["comment_text"].fillna(" ")
test_df["comment_text"] = test_df["comment_text"].fillna(" ")



## === cell 1
vectorizer = TfidfVectorizer(
    max_features=200000,
    ngram_range=(1, 2),
    analyzer="char",
    min_df=2,
    dtype=np.float32,
)

X_all = vectorizer.fit_transform(train_df["comment_text"])  # single costly fit
labels_all = train_df[TARGET_COLS]

all_indices = np.arange(X_all.shape[0])
train_idx, val_idx = train_test_split(all_indices, test_size=0.2, random_state=42)

X_train = X_all[train_idx]
X_val = X_all[val_idx]
train_labels = labels_all.iloc[train_idx]
val_labels = labels_all.iloc[val_idx]



## === cell 2
base_clf = LogisticRegression(
    solver="saga",
    max_iter=1000,
    n_jobs=-1,
    class_weight="balanced",
    C=1.0,
    penalty="l2",
)
clf = OneVsRestClassifier(base_clf)

clf.fit(X_train, train_labels)

val_preds = clf.predict_proba(X_val)

auc_scores = []
for i, col in enumerate(TARGET_COLS):
    auc = roc_auc_score(val_labels[col], val_preds[:, i])
    auc_scores.append(auc)
mean_auc = sum(auc_scores) / len(auc_scores)
print(f"Mean ROC‑AUC on hold‑out: {mean_auc:.6f}")



## === cell 3
X_full = X_all
clf_full = OneVsRestClassifier(base_clf)
clf_full.fit(X_full, labels_all)

X_test = vectorizer.transform(test_df["comment_text"])
test_preds = clf_full.predict_proba(X_test)



## === cell 4
submission = pd.DataFrame(test_preds, columns=TARGET_COLS)
submission.insert(0, "id", test_df["id"])

submission_path = "model_9872_baseline_submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")
