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
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score

DATA_ROOT = "/kaggle/input/jigsaw-toxic-comment-classification-challenge"
TRAIN_PATH = os.path.join(DATA_ROOT, "train.csv")
TEST_PATH = os.path.join(DATA_ROOT, "test.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")

label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]



## === cell 1
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

train_df["comment_text"] = train_df["comment_text"].astype(str).fillna("")
test_df["comment_text"] = test_df["comment_text"].astype(str).fillna("")



## === cell 2
X_train, X_val, y_train, y_val = train_test_split(
    train_df["comment_text"],
    train_df[label_cols],
    test_size=0.1,
    random_state=42,
    stratify=train_df["toxic"],  # stratify on a single column for reproducibility
)



## === cell 3
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents="unicode",
    analyzer="word",
    token_pattern=r"\w{1,}",
    stop_words="english",
    ngram_range=(1, 2),
    max_features=200000,
)

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents="unicode",
    analyzer="char",
    ngram_range=(3, 5),
    max_features=200000,
)

word_vectorizer.fit(X_train)
char_vectorizer.fit(X_train)

X_train_word = word_vectorizer.transform(X_train)
X_train_char = char_vectorizer.transform(X_train)
X_train_tf = np.hstack([X_train_word.toarray(), X_train_char.toarray()])

X_val_word = word_vectorizer.transform(X_val)
X_val_char = char_vectorizer.transform(X_val)
X_val_tf = np.hstack([X_val_word.toarray(), X_val_char.toarray()])

base_clf = LogisticRegression(
    solver="liblinear",
    max_iter=1000,
    C=4.0,
    random_state=42,
)
clf = OneVsRestClassifier(base_clf)

clf.fit(X_train_tf, y_train)

val_pred = clf.predict_proba(X_val_tf)
auc_scores = [
    roc_auc_score(y_val[col], val_pred[:, i]) for i, col in enumerate(label_cols)
]
print("Validation AUC per label:", dict(zip(label_cols, auc_scores)))
print("Mean Validation AUC:", np.mean(auc_scores))



## === cell 4
full_word = word_vectorizer.transform(train_df["comment_text"])
full_char = char_vectorizer.transform(train_df["comment_text"])
X_full = np.hstack([full_word.toarray(), full_char.toarray()])

clf.fit(X_full, train_df[label_cols])



## === cell 5
test_word = word_vectorizer.transform(test_df["comment_text"])
test_char = char_vectorizer.transform(test_df["comment_text"])
X_test = np.hstack([test_word.toarray(), test_char.toarray()])

test_pred = clf.predict_proba(X_test)

submission = pd.DataFrame(test_pred, columns=label_cols)
submission.insert(0, "id", test_df["id"])
submission.head()



## === cell 6
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
