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
import os, warnings
import numpy as np
import pandas as pd

warnings.simplefilter(action="ignore", category=FutureWarning)

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))




## === cell 1
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import gc  # for explicit memory cleanup




## === cell 2
train_path = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv"
test_path = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv"

df_train = pd.read_csv(
    train_path,
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
df_test = pd.read_csv(test_path, usecols=["id", "comment_text"])

print("Train shape:", df_train.shape)
print("Test shape :", df_test.shape)




## === cell 3
df_train["cleaned"] = df_train["comment_text"].astype(str).str.lower()
df_test["cleaned"] = df_test["comment_text"].astype(str).str.lower()




## === cell 4
vectorizer = TfidfVectorizer(
    max_features=50_000,  # smaller feature space for faster training
    min_df=2,  # ignore extremely rare words
    ngram_range=(1, 2),
    stop_words="english",
    dtype=np.float32,
    sublinear_tf=True,  # log‑scale term frequencies for speed
)

X_train = vectorizer.fit_transform(df_train["cleaned"])
X_test = vectorizer.transform(df_test["cleaned"])

y_train = df_train[
    ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
].values.astype(
    np.float32
)  # ensure float32 for faster LogisticRegression

print("Vectorized shapes – X_train:", X_train.shape, "X_test:", X_test.shape)

del df_train["cleaned"], df_test["cleaned"]
gc.collect()




## === cell 6
clf = OneVsRestClassifier(
    LogisticRegression(
        solver="sag",  # fast solver for large sparse data
        max_iter=1000,
        n_jobs=-1,
        class_weight="balanced",
        warm_start=True,  # reuse previous solution for second fit (kept for consistency)
        verbose=0,
    )
)

clf.fit(X_train, y_train)

test_pred = clf.predict_proba(X_test)




## === cell 7
submission = pd.DataFrame(
    {
        "id": df_test["id"],
        "toxic": test_pred[:, 0],
        "severe_toxic": test_pred[:, 1],
        "obscene": test_pred[:, 2],
        "threat": test_pred[:, 3],
        "insult": test_pred[:, 4],
        "identity_hate": test_pred[:, 5],
    }
)

submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print("Submission saved to:", submission_path)
