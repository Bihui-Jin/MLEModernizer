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

3.12

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
pass



## === cell 1
import numpy as np
import pandas as pd

import sklearnex

sklearnex.patch_sklearn()  # speeds up TF‑IDF and linear models without altering results

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score

np.random.seed(42)



## === cell 2
train_zip = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip"
test_zip = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip"
sample_zip = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip"

train_df = pd.read_csv(
    train_zip,
    compression="zip",
    usecols=[
        "id",
        "comment_text",
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ],
)

test_df = pd.read_csv(
    test_zip,
    compression="zip",
    usecols=["id", "comment_text"],
)

sample_df = pd.read_csv(sample_zip, compression="zip")
train_df.head()



## === cell 3
pass



## === cell 4
pass



## === cell 5
pass



## === cell 6
train_df.info()



## === cell 7
test_df.info()



## === cell 8
label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
train_df[label_cols] = train_df[label_cols].astype("int32")



## === cell 9
train_df.isnull().sum()



## === cell 10
test_df.isnull().sum()



## === cell 11
pass



## === cell 12
y = train_df[label_cols]



## === cell 13
X_train, X_valid, y_train, y_valid = train_test_split(
    train_df["comment_text"], y, random_state=42, train_size=0.8
)



## === cell 14
vectorizer = TfidfVectorizer(
    max_features=200000,
    min_df=2,
    ngram_range=(1, 3),
    sublinear_tf=True,
    stop_words="english",
    dtype=np.float32,  # keep matrix in float32 to save memory and speed up computation
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_valid_tfidf = vectorizer.transform(X_valid)
X_test_tfidf = vectorizer.transform(test_df["comment_text"])



## === cell 15
clf = OneVsRestClassifier(
    LogisticRegression(
        solver="saga",
        random_state=42,
        max_iter=2000,
        class_weight="balanced",
        C=6.0,
        n_jobs=-1,  # use all cores inside each binary classifier
    ),
    n_jobs=-1,  # run the six classifiers in parallel
)

clf.fit(X_train_tfidf, y_train)



## === cell 16
y_valid_pred_proba = clf.predict_proba(X_valid_tfidf)
auc_per_class = []
for i, col in enumerate(y.columns):
    auc = roc_auc_score(y_valid[col], y_valid_pred_proba[:, i])
    auc_per_class.append(auc)
mean_auc = np.mean(auc_per_class)
print(f"Mean column‑wise ROC‑AUC on validation: {mean_auc:.5f}")
print("AUC per class:")
for col, auc in zip(y.columns, auc_per_class):
    print(f"{col}: {auc:.5f}")



## === cell 17
y_test_pred_proba = clf.predict_proba(X_test_tfidf)



## === cell 18
submission = pd.DataFrame(y_test_pred_proba, columns=y.columns)
submission.insert(0, "id", test_df["id"])



## === cell 19
submission.to_csv("submission.csv", index=False)
print(submission.head())



## === cell 20
sub = pd.read_csv("submission.csv")
sub.head()
