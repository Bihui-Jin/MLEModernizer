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
joblib==1.5.2
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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
import os, re, string, gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings("ignore")



## === cell 1
train_path = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv"
test_path = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv"
df = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)



## === cell 2
punct_regex = f"[{re.escape(string.punctuation)}]"
df["clean_text"] = (
    df["comment_text"]
    .astype(str)
    .str.lower()
    .str.replace(punct_regex, " ", regex=True)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)
df_test["clean_text"] = (
    df_test["comment_text"]
    .astype(str)
    .str.lower()
    .str.replace(punct_regex, " ", regex=True)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)



## === cell 3
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=50000,
    analyzer="word",
    dtype=np.float32,
    stop_words="english",
)
vectorizer.fit(df["clean_text"])

X = vectorizer.transform(df["clean_text"])
X_test = vectorizer.transform(df_test["clean_text"])



## === cell 4
label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = df[label_cols].values



## === cell 5
prob = pd.DataFrame({"id": df_test["id"]})
for idx, col in enumerate(label_cols):
    y_col = y[:, idx]
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y_col, test_size=0.2, random_state=42, stratify=y_col
    )
    model = LogisticRegression(
        solver="saga",
        max_iter=1000,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced",
        C=4.0,
        warm_start=True,  # reuse coefficients for the second fit
    )
    model.fit(X_tr, y_tr)
    val_pred = model.predict_proba(X_val)[:, 1]
    print(f"{col} validation ROC‑AUC: {roc_auc_score(y_val, val_pred):.5f}")

    model.fit(X, y_col)
    prob[col] = model.predict_proba(X_test)[:, 1]



## === cell 6
submission_path = "submission-LR-tfidf.csv"
prob.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
