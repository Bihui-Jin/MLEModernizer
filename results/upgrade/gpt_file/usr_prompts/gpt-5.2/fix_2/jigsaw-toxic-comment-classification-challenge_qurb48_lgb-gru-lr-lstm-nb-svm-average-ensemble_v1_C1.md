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

0.9813539331579268

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_CANDIDATES = [
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge",
    "/kaggle/data/jigsaw-toxic-comment-classification-challenge",
    "/kaggle/input",
    "/kaggle/data",
    "../input/jigsaw-toxic-comment-classification-challenge",
    "../input",
    "../data/jigsaw-toxic-comment-classification-challenge",
    "../data",
]


def find_file(filename: str) -> str:
    for base in BASE_CANDIDATES:
        path = os.path.join(base, filename)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Could not find {filename} in any known Kaggle paths.")


train_path = find_file("train.csv")
test_path = find_file("test.csv")
sample_path = find_file("sample_submission.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_path)

labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

train_text = train_df["comment_text"].fillna("").astype(str)
test_text = test_df["comment_text"].fillna("").astype(str)



## === cell 1

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def fit_predict_multilabel(
    train_text, train_y_df, test_text, vectorizer_kwargs, lr_kwargs
):
    preds = pd.DataFrame({"id": test_df["id"].values})
    for lab in labels:
        pipe = Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(**vectorizer_kwargs)),
                ("lr", LogisticRegression(**lr_kwargs)),
            ]
        )
        y = train_y_df[lab].values
        pipe.fit(train_text, y)
        preds[lab] = pipe.predict_proba(test_text)[:, 1]
    return preds


common_lr = dict(
    solver="liblinear",
    C=3.0,
    max_iter=200,
    random_state=RANDOM_STATE,
)

ftgru = fit_predict_multilabel(
    train_text,
    train_df[labels],
    test_text,
    vectorizer_kwargs=dict(
        ngram_range=(1, 2), min_df=2, max_features=200000, strip_accents="unicode"
    ),
    lr_kwargs=common_lr,
)

ftlstm = fit_predict_multilabel(
    train_text,
    train_df[labels],
    test_text,
    vectorizer_kwargs=dict(
        ngram_range=(1, 2), min_df=3, max_features=150000, strip_accents="unicode"
    ),
    lr_kwargs=dict(**common_lr, C=2.0),
)

ggru = fit_predict_multilabel(
    train_text,
    train_df[labels],
    test_text,
    vectorizer_kwargs=dict(
        ngram_range=(1, 3), min_df=2, max_features=250000, strip_accents="unicode"
    ),
    lr_kwargs=dict(**common_lr, C=2.5),
)

glstm = fit_predict_multilabel(
    train_text,
    train_df[labels],
    test_text,
    vectorizer_kwargs=dict(
        ngram_range=(1, 1), min_df=2, max_features=100000, strip_accents="unicode"
    ),
    lr_kwargs=dict(**common_lr, C=1.5),
)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4017943071.py in <cell line: 0>()
     51         ngram_range=(1, 2), min_df=3, max_features=150000, strip_accents="unicode"
     52     ),
---> 53     lr_kwargs=dict(**common_lr, C=2.0),
     54 )
     55 

TypeError: dict() got multiple values for keyword argument 'C'

## === cell 2
for label in labels:
    a = ftgru[label].rank(pct=True)
    b = ftlstm[label].rank(pct=True)
    c = ggru[label].rank(pct=True)
    d = glstm[label].rank(pct=True)
    corr = np.corrcoef([a, b, c, d])
    print(label)
    print(corr)

submission = pd.DataFrame()
submission["id"] = test_df["id"].values

for label in labels:
    submission[label] = (
        ftgru[label].rank(pct=True) * 0.15
        + ftlstm[label].rank(pct=True) * 0.15
        + ggru[label].rank(pct=True) * 0.35
        + glstm[label].rank(pct=True) * 0.35
    )

submission = submission[["id"] + labels]

submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1130336908.py in <cell line: 0>()
      2 for label in labels:
      3     a = ftgru[label].rank(pct=True)
----> 4     b = ftlstm[label].rank(pct=True)
      5     c = ggru[label].rank(pct=True)
      6     d = glstm[label].rank(pct=True)

NameError: name 'ftlstm' is not defined
