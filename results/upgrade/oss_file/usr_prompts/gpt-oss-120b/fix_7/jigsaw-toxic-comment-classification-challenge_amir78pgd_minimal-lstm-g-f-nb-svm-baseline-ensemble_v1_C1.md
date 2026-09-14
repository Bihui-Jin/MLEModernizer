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

3.7

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

np.random.seed(42)

print("Input contents:", os.listdir("../input"))




## === cell 1
train_path = "../input/jigsaw-toxic-comment-classification-challenge/train.csv"
test_path = "../input/jigsaw-toxic-comment-classification-challenge/test.csv"




## === cell 2
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
print("Train shape:", train_df.shape, "Test shape:", test_df.shape)




## === cell 3
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
from joblib import Parallel, delayed

train_df["comment_text"] = train_df["comment_text"].fillna(" ")
test_df["comment_text"] = test_df["comment_text"].fillna(" ")

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents="unicode",
    analyzer="word",
    token_pattern=r"\w{1,}",
    stop_words="english",
    ngram_range=(1, 2),
    max_features=50000,
)

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents="unicode",
    analyzer="char",
    stop_words=None,
    ngram_range=(3, 5),
    max_features=50000,
)


def fit_transform(vect, series):
    X = vect.fit_transform(series)
    return vect, X


(word_vectorizer, X_train_word), (char_vectorizer, X_train_char) = Parallel(n_jobs=2)(
    delayed(fit_transform)(vect, train_df["comment_text"])
    for vect in (word_vectorizer, char_vectorizer)
)

X_train = sparse.hstack([X_train_word, X_train_char]).astype(np.float32)


def transform(vect, series):
    return vect.transform(series)


X_test_word, X_test_char = Parallel(n_jobs=2)(
    delayed(transform)(vect, test_df["comment_text"])
    for vect in (word_vectorizer, char_vectorizer)
)

X_test = sparse.hstack([X_test_word, X_test_char]).astype(np.float32)

del X_train_word, X_train_char, X_test_word, X_test_char

print("Feature matrix shape:", X_train.shape)




## === cell 4
from sklearn.linear_model import LogisticRegression
import os
from joblib import Parallel, delayed

label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
outer_n_jobs = max(1, os.cpu_count() or 1)  # number of parallel label trainings


def train_one(lbl):
    lr = LogisticRegression(
        solver="sag",
        max_iter=1000,
        class_weight="balanced",
        n_jobs=1,  # use a single core per model
        random_state=42,
    )
    lr.fit(X_train, train_df[lbl])
    return lbl, lr


results = Parallel(n_jobs=outer_n_jobs)(delayed(train_one)(lbl) for lbl in label_cols)

models = dict(results)

del X_train




## === cell 5
preds = {}
for lbl in label_cols:
    preds[lbl] = models[lbl].predict_proba(X_test)[:, 1]

submission = pd.DataFrame({"id": test_df["id"]})
for lbl in label_cols:
    submission[lbl] = preds[lbl]

print("Submission preview:")
print(submission.head())




## === cell 6
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
