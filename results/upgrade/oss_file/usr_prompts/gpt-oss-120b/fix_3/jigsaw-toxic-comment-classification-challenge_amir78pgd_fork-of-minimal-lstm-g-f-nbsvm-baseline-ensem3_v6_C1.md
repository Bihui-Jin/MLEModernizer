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
import os, numpy as np, pandas as pd

print("Input dir contents:", os.listdir("../input"))




## === cell 1
train_path = "../input/train.csv"
test_path = "../input/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)




## === cell 2
label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
X_text = train_df["comment_text"].fillna("").astype(str)
y = train_df[label_cols].astype(float)

from sklearn.model_selection import train_test_split

indices = np.arange(len(train_df))
train_idx, val_idx = train_test_split(
    indices, test_size=0.2, random_state=42, stratify=y["toxic"]
)

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    max_features=20000, ngram_range=(1, 2), stop_words="english", min_df=2
)
X_all_vec = tfidf.fit_transform(train_df["comment_text"].fillna("").astype(str))

X_tr_vec = X_all_vec[train_idx]
X_val_vec = X_all_vec[val_idx]

y_tr = y.iloc[train_idx]
y_val = y.iloc[val_idx]

X_test_vec = tfidf.transform(test_df["comment_text"].fillna("").astype(str))




## === cell 3
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

models = {}
val_scores = []

for col in label_cols:
    clf = LogisticRegression(
        C=4.0, max_iter=1000, solver="saga", n_jobs=-1, class_weight="balanced"
    )
    clf.fit(X_tr_vec, y_tr[col].values)
    models[col] = clf
    val_pred = clf.predict_proba(X_val_vec)[:, 1]
    auc = roc_auc_score(y_val[col], val_pred)
    val_scores.append(auc)

mean_auc = np.mean(val_scores)
print(f"Validation ROC‑AUC per label: {val_scores}")
print(f"Mean validation ROC‑AUC: {mean_auc:.6f}")




## === cell 4
test_preds = {}
for col in label_cols:
    test_preds[col] = models[col].predict_proba(X_test_vec)[:, 1]

submission = pd.DataFrame({"id": test_df["id"]})
for col in label_cols:
    submission[col] = test_preds[col]




## === cell 5
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
