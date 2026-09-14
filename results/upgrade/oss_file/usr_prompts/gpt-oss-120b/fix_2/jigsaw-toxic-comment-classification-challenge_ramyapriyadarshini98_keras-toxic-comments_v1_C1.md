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

3.7

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

# 5. Target score

0.778875022066981

# 6. Current score

0.97436

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.97436) has done: 'The fix removes the fragile custom text cleaning that caused shape errors, replaces it with straightforward TF‑IDF on the raw comment text, and switches from the broken Keras model to a scikit‑learn One‑Vs‑Rest Logistic Regression which works in the provided environment. The pipeline now loads the data, vectorises text, trains the model, evaluates ROC‑AUC locally, refits on the full training set, predicts on the test set, and writes a correctly‑formatted `Submitted.csv` file.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score

print("Files in ../input:", os.listdir("../input"))



## === cell 1
train_path = "../input/train.csv"
test_path = "../input/test.csv"
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print("Train head:")
print(train.head())
print("\nTest head:")
print(test.head())



## === cell 2
X = train["comment_text"].astype(str)
y = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]

X_test_raw = test["comment_text"].astype(str)



## === cell 3
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y["toxic"]
)



## === cell 4
tfidf = TfidfVectorizer(
    ngram_range=(1, 2), max_features=50000, min_df=2, stop_words="english"
)

X_tr_vec = tfidf.fit_transform(X_tr)
X_val_vec = tfidf.transform(X_val)



## === cell 5
base_clf = LogisticRegression(solver="sag", max_iter=1000, n_jobs=-1, random_state=42)
clf = OneVsRestClassifier(base_clf)

clf.fit(X_tr_vec, y_tr)

val_pred = clf.predict_proba(X_val_vec)
auc_scores = []
for i, col in enumerate(y.columns):
    auc = roc_auc_score(y_val[col], val_pred[:, i])
    auc_scores.append(auc)
mean_auc = np.mean(auc_scores)
print(f"Validation mean ROC‑AUC: {mean_auc:.6f}")



## === cell 6
X_full_vec = tfidf.fit_transform(X)  # re‑fit tfidf on all data
test_vec = tfidf.transform(X_test_raw)

clf.fit(X_full_vec, y)



## === cell 7
test_pred = clf.predict_proba(test_vec)



## === cell 8
submission = pd.DataFrame(
    {
        "id": test["id"],
        "toxic": test_pred[:, 0],
        "severe_toxic": test_pred[:, 1],
        "obscene": test_pred[:, 2],
        "threat": test_pred[:, 3],
        "insult": test_pred[:, 4],
        "identity_hate": test_pred[:, 5],
    }
)

submission_path = "Submitted.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
