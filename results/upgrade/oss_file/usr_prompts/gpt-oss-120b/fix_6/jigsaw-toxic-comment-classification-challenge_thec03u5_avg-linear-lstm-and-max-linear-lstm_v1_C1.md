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

# 5. Target score

0.9833102134439234

# 6. Current score

None

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from joblib import Parallel, delayed



## === cell 1
possible_paths = [
    "train.csv",
    "test.csv",
    "data/train.csv",
    "data/test.csv",
    "../input/jigsaw-toxic-comment-classification-challenge/train.csv",
    "../input/jigsaw-toxic-comment-classification-challenge/test.csv",
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv",
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv",
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
    test_size=0.10,  # 10 % for validation (more stable estimate)
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

X_train = vectorizer.transform(train_text)
X_val = vectorizer.transform(val_text)
X_test = vectorizer.transform(test_df["comment_text"])




## === cell 4
def train_one_label(col):
    lr = LogisticRegression(
        C=10.0,  # slightly weaker regularisation
        solver="saga",
        max_iter=300,
        n_jobs=1,
        class_weight="balanced",
        penalty="l2",
        random_state=42,
    )
    lr.fit(X_train, train_labels[col])
    val_pred = lr.predict_proba(X_val)[:, 1]
    return col, lr, val_pred


results = Parallel(n_jobs=len(label_cols), backend="loky")(
    delayed(train_one_label)(col) for col in label_cols
)

models = {}
val_preds = pd.DataFrame(index=val_labels.index, columns=label_cols)
for col, model, val_pred in results:
    models[col] = model
    val_preds[col] = val_pred

auc_scores = {col: roc_auc_score(val_labels[col], val_preds[col]) for col in label_cols}
mean_auc = np.mean(list(auc_scores.values()))
print("Validation ROC‑AUC per label:")
for k, v in auc_scores.items():
    print(f"  {k}: {v:.5f}")
print(f"Mean ROC‑AUC: {mean_auc:.5f}")




## === cell 5
def predict_one_label(col):
    return models[col].predict_proba(X_test)[:, 1]


test_pred_arrays = Parallel(n_jobs=len(label_cols), backend="loky")(
    delayed(predict_one_label)(col) for col in label_cols
)

test_preds = pd.DataFrame()
test_preds["id"] = test_df["id"]
for col, preds in zip(label_cols, test_pred_arrays):
    test_preds[col] = preds

submission_path = "submission.csv"
test_preds.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
