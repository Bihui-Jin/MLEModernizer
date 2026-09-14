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

3.9

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

# 5. Target score

0.93441

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import string
import re
from statistics import mean
from pathlib import Path

from sklearnex import patch

patch()

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

pd.options.display.float_format = "{:,.3f}".format

STOP_WORDS = list(ENGLISH_STOP_WORDS)

_CLEAN_RE = re.compile(rf"[{re.escape(string.punctuation)}0-9]")


def clean_series(series: pd.Series) -> pd.Series:
    """
    Vectorized text cleaning:
    - strip punctuation & digits,
    - lower‑case,
    - collapse whitespace.
    Stop‑words are removed later by CountVectorizer.
    """
    return (
        series.str.replace(_CLEAN_RE, " ", regex=True)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def locate_file(*candidates):
    """Return the first existing path among candidates."""
    for cand in candidates:
        p = Path(cand)
        if p.is_file():
            return str(p)
    raise FileNotFoundError(f"None of the candidate files exist: {candidates}")




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/2291726913.py in <cell line: 0>()
      8 # Accelerate scikit‑learn estimators (Naive Bayes, LogisticRegression, etc.)
      9 # This does not alter the algorithmic logic, only speeds up the same computations.
---> 10 from sklearnex import patch
     11 
     12 patch()

ImportError: cannot import name 'patch' from 'sklearnex' (/usr/local/lib/python3.11/dist-packages/sklearnex/__init__.py)

## === cell 1
train_path = locate_file(
    "kaggle/data/jigsaw-toxic-comment-classification-challenge/train.csv",
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv",
    "data/jigsaw-toxic-comment-classification-challenge/train.csv",
)
test_path = locate_file(
    "kaggle/data/jigsaw-toxic-comment-classification-challenge/test.csv",
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv",
    "data/jigsaw-toxic-comment-classification-challenge/test.csv",
)
sample_sub_path = locate_file(
    "kaggle/data/jigsaw-toxic-comment-classification-challenge/sample_submission.csv",
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv",
    "data/jigsaw-toxic-comment-classification-challenge/sample_submission.csv",
)

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_submission = pd.read_csv(sample_sub_path)

print(train_df.shape, test_df.shape, sample_submission.shape)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3633316675.py in <cell line: 0>()
----> 1 train_path = locate_file(
      2     "kaggle/data/jigsaw-toxic-comment-classification-challenge/train.csv",
      3     "/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv",
      4     "data/jigsaw-toxic-comment-classification-challenge/train.csv",
      5 )

NameError: name 'locate_file' is not defined

## === cell 2
X = train_df["comment_text"]
y = train_df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=123, shuffle=True
)

X_train_clean = clean_series(X_train)
X_val_clean = clean_series(X_val)
test_clean = clean_series(test_df["comment_text"])




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2649323963.py in <cell line: 0>()
----> 1 X = train_df["comment_text"]
      2 y = train_df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]
      3 
      4 X_train, X_val, y_train, y_val = train_test_split(
      5     X, y, test_size=0.2, random_state=123, shuffle=True

NameError: name 'train_df' is not defined

## === cell 3
vectorizer = CountVectorizer(max_features=5000, stop_words=STOP_WORDS)
X_train_dtm = vectorizer.fit_transform(X_train_clean)
X_val_dtm = vectorizer.transform(X_val_clean)

X_train_dtm = X_train_dtm.astype(np.float32)
X_val_dtm = X_val_dtm.astype(np.float32)

print("DTM shapes:", X_train_dtm.shape, X_val_dtm.shape)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1862951622.py in <cell line: 0>()
----> 1 vectorizer = CountVectorizer(max_features=5000, stop_words=STOP_WORDS)
      2 X_train_dtm = vectorizer.fit_transform(X_train_clean)
      3 X_val_dtm = vectorizer.transform(X_val_clean)
      4 
      5 # Cast to float32 to reduce memory bandwidth for downstream linear models.

NameError: name 'CountVectorizer' is not defined

## === cell 4
nb_model = MultiOutputClassifier(MultinomialNB(), n_jobs=-1).fit(X_train_dtm, y_train)
lr_model = MultiOutputClassifier(
    LogisticRegression(class_weight="balanced", max_iter=3000, solver="saga", n_jobs=-1)
).fit(X_train_dtm, y_train)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2824340408.py in <cell line: 0>()
----> 1 nb_model = MultiOutputClassifier(MultinomialNB(), n_jobs=-1).fit(X_train_dtm, y_train)
      2 lr_model = MultiOutputClassifier(
      3     LogisticRegression(class_weight="balanced", max_iter=3000, solver="saga", n_jobs=-1)
      4 ).fit(X_train_dtm, y_train)
      5 

NameError: name 'MultiOutputClassifier' is not defined

## === cell 5
def calculate_roc_auc(y_true: np.ndarray, y_pred: np.ndarray) -> list:
    """Return list of ROC‑AUC for each column."""
    return [roc_auc_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]


y_val_np = y_val.to_numpy()

for model, name in [(nb_model, "NaiveBayes"), (lr_model, "LogisticRegression")]:
    probs = np.transpose(np.array(model.predict_proba(X_val_dtm))[:, :, 1])
    mean_auc = mean(calculate_roc_auc(y_val_np, probs))
    print(f"{name} Mean AUC: {mean_auc:.4f}")




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3616435846.py in <cell line: 0>()
      4 
      5 
----> 6 y_val_np = y_val.to_numpy()
      7 
      8 for model, name in [(nb_model, "NaiveBayes"), (lr_model, "LogisticRegression")]:

NameError: name 'y_val' is not defined

## === cell 6
X_test_dtm = vectorizer.transform(test_clean)
X_test_dtm = X_test_dtm.astype(np.float32)

test_probs = np.transpose(np.array(lr_model.predict_proba(X_test_dtm))[:, :, 1])

submission = pd.DataFrame(
    data=test_probs,
    columns=["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"],
)
submission.insert(0, "id", test_df["id"])

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}, shape: {submission.shape}")

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/805890843.py in <cell line: 0>()
----> 1 X_test_dtm = vectorizer.transform(test_clean)
      2 X_test_dtm = X_test_dtm.astype(np.float32)
      3 
      4 test_probs = np.transpose(np.array(lr_model.predict_proba(X_test_dtm))[:, :, 1])
      5 

NameError: name 'vectorizer' is not defined
