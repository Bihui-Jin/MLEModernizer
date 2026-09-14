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
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

BASE_CANDIDATES = [
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge",
    "/kaggle/data/jigsaw-toxic-comment-classification-challenge",
    "/kaggle/input",
    "/kaggle/data",
]
DATA_DIR = None
for c in BASE_CANDIDATES:
    if os.path.exists(os.path.join(c, "train.csv")) and os.path.exists(
        os.path.join(c, "test.csv")
    ):
        DATA_DIR = c
        break

if DATA_DIR is None:
    raise FileNotFoundError(
        "Could not locate train.csv/test.csv in expected Kaggle paths. "
        "Checked: " + ", ".join(BASE_CANDIDATES)
    )

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sub_path = os.path.join(DATA_DIR, "sample_submission.csv")

usecols_train = ["id", "comment_text"] + label_cols
dtype_train = {c: np.int8 for c in label_cols}
train = pd.read_csv(train_path, usecols=usecols_train, dtype=dtype_train)
test = pd.read_csv(test_path, usecols=["id", "comment_text"])
sample_sub = pd.read_csv(sub_path)

train["comment_text"] = train["comment_text"].fillna("")
test["comment_text"] = test["comment_text"].fillna("")

missing = [c for c in label_cols if c not in train.columns]
if missing:
    raise KeyError(f"Missing label columns in train.csv: {missing}")

print("Loaded:", train.shape, test.shape, sample_sub.shape)
print("Using DATA_DIR:", DATA_DIR)



## === cell 1
word_vectorizer = TfidfVectorizer(
    strip_accents="unicode",
    analyzer="word",
    token_pattern=r"\w{1,}",
    ngram_range=(1, 2),
    max_features=200000,
    min_df=3,
    sublinear_tf=True,
)

char_vectorizer = TfidfVectorizer(
    strip_accents="unicode",
    analyzer="char",
    ngram_range=(3, 5),
    max_features=200000,
    min_df=3,
    sublinear_tf=True,
)

Xw_tr = word_vectorizer.fit_transform(train["comment_text"].values)
Xc_tr = char_vectorizer.fit_transform(train["comment_text"].values)

Xw_te = word_vectorizer.transform(test["comment_text"].values)
Xc_te = char_vectorizer.transform(test["comment_text"].values)

print("Word TFIDF:", Xw_tr.shape, Xw_te.shape, "Char TFIDF:", Xc_tr.shape, Xc_te.shape)



## === cell 2
from scipy.sparse import hstack

X_tr = hstack([Xw_tr, Xc_tr], format="csr")
X_te = hstack([Xw_te, Xc_te], format="csr")

print("Final sparse matrices:", X_tr.shape, X_te.shape)



## === cell 3
preds = np.zeros((test.shape[0], len(label_cols)), dtype=np.float64)

clf = LogisticRegression(
    solver="saga",
    penalty="l2",
    C=4.0,
    max_iter=200,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    warm_start=True,
)

for i, col in enumerate(label_cols):
    y = train[col].values
    clf.fit(X_tr, y)
    preds[:, i] = clf.predict_proba(X_te)[:, 1]
    print(f"Trained {col}: positive_rate={y.mean():.6f}")



## === cell 4
submission = sample_sub.copy()

if (len(submission) == len(test)) and submission["id"].values.tolist() == test[
    "id"
].values.tolist():
    for j, col in enumerate(label_cols):
        submission[col] = preds[:, j]
else:
    test_id_to_row = pd.Series(np.arange(test.shape[0]), index=test["id"].values)
    row_idx = test_id_to_row.reindex(submission["id"].values)
    if row_idx.isna().any():
        tmp = pd.DataFrame({"id": test["id"].values})
        for j, col in enumerate(label_cols):
            tmp[col] = preds[:, j]
        submission = submission[["id"]].merge(tmp, on="id", how="left")
    else:
        row_idx = row_idx.astype(int).values
        for j, col in enumerate(label_cols):
            submission[col] = preds[row_idx, j]

for col in label_cols:
    submission[col] = submission[col].clip(0.0, 1.0)

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)

print("Wrote:", submission_path)
print(submission.head())



## === cell 5
submission.to_csv("submissionAvg.csv", index=False)
submission.to_csv("submissionMax.csv", index=False)
submission.to_csv("submissionCondMax.csv", index=False)

print("Also wrote: submissionAvg.csv, submissionMax.csv, submissionCondMax.csv")
