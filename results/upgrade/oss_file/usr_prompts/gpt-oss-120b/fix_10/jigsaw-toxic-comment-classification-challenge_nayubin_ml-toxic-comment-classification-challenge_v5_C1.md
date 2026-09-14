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

3.13

# 3. Installed packages

geopandas==0.14.4
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
seaborn==0.12.2
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

if not os.path.isdir("/kaggle/input"):
    raise FileNotFoundError("Input directory not found.")
print("Input directory verified.")



## === cell 1
import subprocess
import os


def unzip_if_needed(zip_path, target_name):
    """
    Unzip only if the target CSV does not already exist.
    This avoids repetitive decompression work on subsequent runs.
    """
    if not os.path.isfile(target_name):
        subprocess.run(
            ["unzip", "-q", zip_path],
            check=True,
        )
    else:
        print(f"{target_name} already exists, skipping unzip.")


unzip_if_needed(
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip",
    "train.csv",
)
unzip_if_needed(
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip",
    "sample_submission.csv",
)
unzip_if_needed(
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip",
    "test.csv",
)



## === cell 2
import pandas as pd
import numpy as np

train_cols = [
    "id",
    "comment_text",
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]
label_dtype = np.uint8  # binary labels fit into uint8
data = pd.read_csv(
    "train.csv",
    usecols=train_cols,
    dtype={
        "toxic": label_dtype,
        "severe_toxic": label_dtype,
        "obscene": label_dtype,
        "threat": label_dtype,
        "insult": label_dtype,
        "identity_hate": label_dtype,
    },
    low_memory=False,
)

test = pd.read_csv(
    "test.csv",
    usecols=["id", "comment_text"],
    low_memory=False,
)




## === cell 3
import re

_space_pat = re.compile(r"\s+")
_punct_pat = re.compile(r"[^\w\s]")


def preprocess_series(s: pd.Series) -> pd.Series:
    s = s.str.replace(_space_pat, " ", regex=True)
    s = s.str.replace(_punct_pat, "", regex=True)
    return s.str.lower()




## === cell 4
data["comment_text"] = preprocess_series(data["comment_text"])
test["comment_text"] = preprocess_series(test["comment_text"])



## === cell 5
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import gc

X = data["comment_text"]
y = data[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y["toxic"],
)

print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

word_vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
    max_features=30000,  # <-- reduced from 50000
    dtype=np.float32,
)

char_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=2,
    sublinear_tf=True,
    max_features=30000,  # <-- reduced from 50000
    dtype=np.float32,
)

X_train_word = word_vectorizer.fit_transform(X_train)
X_train_char = char_vectorizer.fit_transform(X_train)

X_train_tfidf = hstack([X_train_word, X_train_char], format="csr")

del X_train_word, X_train_char
gc.collect()

X_val_word = word_vectorizer.transform(X_val)
X_val_char = char_vectorizer.transform(X_val)
X_val_tfidf = hstack([X_val_word, X_val_char], format="csr")
del X_val_word, X_val_char
gc.collect()

from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier

logreg = LogisticRegression(
    solver="saga",
    max_iter=300,
    n_jobs=-1,
    class_weight="balanced",
    penalty="l2",
    C=1.0,
    random_state=42,
)

model = MultiOutputClassifier(logreg, n_jobs=-1)
model.fit(X_train_tfidf, y_train)



## === cell 6
from sklearn.metrics import roc_auc_score

y_val_pred_proba = model.predict_proba(X_val_tfidf)

roc_auc = {}
for idx, label in enumerate(y_train.columns):
    auc = roc_auc_score(y_val.iloc[:, idx], y_val_pred_proba[idx][:, 1])
    roc_auc[label] = auc
    print(f"{label} AUC: {auc:.4f}")

macro_auc = np.mean(list(roc_auc.values()))
print(f"Macro Average AUC (mean of per‑label): {macro_auc:.4f}")



## === cell 7
test_word = word_vectorizer.transform(test["comment_text"])
test_char = char_vectorizer.transform(test["comment_text"])
test_tfidf = hstack([test_word, test_char], format="csr")
del test_word, test_char
gc.collect()

test_pred_proba = model.predict_proba(test_tfidf)

pred_columns = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
submission_array = np.column_stack(
    [test_pred_proba[i][:, 1] for i in range(len(pred_columns))]
)

submission = pd.DataFrame(submission_array, columns=pred_columns)
submission.insert(0, "id", test["id"])
submission = submission[["id"] + pred_columns]

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file saved as {submission_path}")
