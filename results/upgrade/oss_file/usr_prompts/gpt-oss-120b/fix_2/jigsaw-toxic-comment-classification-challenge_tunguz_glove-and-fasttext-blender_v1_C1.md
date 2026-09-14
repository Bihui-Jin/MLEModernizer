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
import os, sys
import numpy as np
import pandas as pd

print("Root input dirs:", os.listdir("../input"))




## === cell 1
def find_file(rel_path):
    candidates = [
        f"../input/{rel_path}",
        f"../input/jigsaw-toxic-comment-classification-challenge/{rel_path}",
        f"../input/data/{rel_path}",
        rel_path,  # fallback to current dir
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"Unable to locate {rel_path}")




## === cell 2
train_path = find_file("train.csv")
test_path = find_file("test.csv")
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
print("Train shape:", train.shape, "Test shape:", test.shape)



## === cell 3
categories = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
assert all(
    col in train.columns for col in categories
), "Missing label columns in train data"



## === cell 4
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=200_000,
    ngram_range=(1, 2),
    sublinear_tf=True,
)

X_train = vectorizer.fit_transform(train["comment_text"].fillna(""))
X_test = vectorizer.transform(test["comment_text"].fillna(""))



## === cell 5
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

logreg = OneVsRestClassifier(
    LogisticRegression(
        solver="saga",
        max_iter=1000,
        n_jobs=5,
        verbose=0,
        class_weight="balanced",
    )
)

y_train = train[categories].values
logreg.fit(X_train, y_train)



## === cell 6
test_pred = logreg.predict_proba(X_test)  # shape (n_test, n_labels)



## === cell 7
submission = pd.DataFrame(test_pred, columns=categories)
submission.insert(0, "id", test["id"])
print("Submission head:")
print(submission.head())



## === cell 8
output_path = "blend.csv"
submission.to_csv(output_path, index=False)
print(f"Saved submission to {output_path}")
