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

No external packages required in the script and installed.

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




## === cell 1
DATA_ROOT = "/kaggle/input/jigsaw-toxic-comment-classification-challenge"
TRAIN_PATH = os.path.join(DATA_ROOT, "train.csv")
TEST_PATH = os.path.join(DATA_ROOT, "test.csv")
SUBMIT_PATH = "/kaggle/working/submission.csv"

df_train = pd.read_csv(TRAIN_PATH)
df_test = pd.read_csv(TEST_PATH)

print("Train shape:", df_train.shape)
print("Test  shape:", df_test.shape)




## === cell 2
TARGETS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

df_train["comment_text"] = df_train["comment_text"].fillna(" ")
df_test["comment_text"] = df_test["comment_text"].fillna(" ")

X = df_train["comment_text"].values
y = df_train[TARGETS].values.astype(np.float32)




## === cell 3
idx = np.arange(len(X))
idx_tr, idx_val, y_tr, y_val = train_test_split(
    idx, y, test_size=0.1, random_state=42, stratify=y[:, 0]
)

vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents="unicode",
    analyzer="word",
    token_pattern=r"\w{1,}",
    ngram_range=(1, 2),
    max_features=200000,
    stop_words="english",
    dtype=np.float32,
)

X_vec = vectorizer.fit_transform(X)

X_tr_vec = X_vec[idx_tr]
X_val_vec = X_vec[idx_val]

base_clf = LogisticRegression(
    solver="saga",
    max_iter=1000,
    C=4.0,
    n_jobs=-1,
    class_weight="balanced",
    penalty="l2",
)
clf = OneVsRestClassifier(base_clf)

clf.fit(X_tr_vec, y_tr)

val_pred = clf.predict_proba(X_val_vec)
auc_scores = [roc_auc_score(y_val[:, i], val_pred[:, i]) for i in range(len(TARGETS))]
mean_auc = np.mean(auc_scores)
print(f"Validation ROC‑AUC per class: {auc_scores}")
print(f"Mean Validation ROC‑AUC: {mean_auc:.6f}")




## === cell 4
clf_full = OneVsRestClassifier(base_clf)
clf_full.fit(X_vec, y)

X_test_vec = vectorizer.transform(df_test["comment_text"].values)
test_pred = clf_full.predict_proba(X_test_vec)




## === cell 5
submission = pd.DataFrame(test_pred, columns=TARGETS)
submission.insert(0, "id", df_test["id"])
print("Submission head:")
print(submission.head())

submission.to_csv(SUBMIT_PATH, index=False)
print(f"Submission file written to {SUBMIT_PATH}")
