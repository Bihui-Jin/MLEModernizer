# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

# 5. Target score

0.9837304401872063

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, glob
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

print("Input directories:", os.listdir("../input"))



## === cell 1
train_path = "../input/jigsaw-toxic-comment-classification-challenge/train.csv"
test_path = "../input/jigsaw-toxic-comment-classification-challenge/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]



## === cell 2
tfidf = TfidfVectorizer(
    max_features=50000, ngram_range=(1, 2), stop_words="english", dtype=np.float32
)

X_train = tfidf.fit_transform(train_df["comment_text"].fillna(""))
X_test = tfidf.transform(test_df["comment_text"].fillna(""))

base_clf = LogisticRegression(
    C=4.0, max_iter=200, solver="saga", n_jobs=-1, class_weight="balanced"
)
clf = OneVsRestClassifier(base_clf)

clf.fit(X_train, train_df[label_cols])

test_preds = clf.predict_proba(X_test)  # list of arrays, one per class
pred_df = pd.DataFrame(np.column_stack(test_preds), columns=label_cols)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3212509994.py in <cell line: 0>()
     18 test_preds = clf.predict_proba(X_test)  # list of arrays, one per class
     19 # Convert list of arrays to DataFrame with proper column order
---> 20 pred_df = pd.DataFrame(np.column_stack(test_preds), columns=label_cols)
     21 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __init__(self, data, index, columns, dtype, copy)
    825                 )
    826             else:
--> 827                 mgr = ndarray_to_mgr(
    828                     data,
    829                     index,

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in ndarray_to_mgr(values, index, columns, dtype, copy, typ)
    334     )
    335 
--> 336     _check_values_indices_shape_match(values, index, columns)
    337 
    338     if typ == "array":

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in _check_values_indices_shape_match(values, index, columns)
    418         passed = values.shape
    419         implied = (len(index), len(columns))
--> 420         raise ValueError(f"Shape of passed values is {passed}, indices imply {implied}")
    421 
    422 

ValueError: Shape of passed values is (6, 153164), indices imply (6, 6)

## === cell 3
submission = pd.concat([test_df[["id"]].reset_index(drop=True), pred_df], axis=1)
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv with shape:", submission.shape)

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1490396811.py in <cell line: 0>()
      1 # Assemble submission with required columns
----> 2 submission = pd.concat([test_df[["id"]].reset_index(drop=True), pred_df], axis=1)
      3 submission.to_csv("submission.csv", index=False)
      4 print("Submission saved to submission.csv with shape:", submission.shape)

NameError: name 'pred_df' is not defined
