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

3.9

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
tf_keras==2.18.0

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
import glob
import zipfile
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score

MAX_NUM_WORDS = 100000  # increased from 20000
VALIDATION_SPLIT = 0.2
RANDOM_STATE = 123



## === cell 1
zip_paths = glob.glob(
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge/*.zip"
)
for zp in zip_paths:
    try:
        with zipfile.ZipFile(zp, "r") as zf:
            zf.extractall("/kaggle/working")
    except Exception:
        pass  # ignore extraction errors (e.g., already extracted)




## === cell 2
def load_csv(filename):
    wk_path = os.path.join("/kaggle/working", filename)
    return pd.read_csv(wk_path if os.path.exists(wk_path) else filename)


df_train = load_csv("train.csv")
df_test = load_csv("test.csv")

train_texts = df_train["comment_text"].astype(str).values
test_texts = df_test["comment_text"].astype(str).values

train_labels = df_train[
    ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
].values



## === cell 3
vectorizer = TfidfVectorizer(
    max_features=MAX_NUM_WORDS,
    ngram_range=(1, 2),
    stop_words="english",
    token_pattern=r"(?u)\b\w\w+\b",
    sublinear_tf=True,  # use sub‑linear TF scaling for stability
)
X_all = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)



## === cell 4
X_train, X_val, y_train, y_val = train_test_split(
    X_all,
    train_labels,
    test_size=VALIDATION_SPLIT,
    random_state=RANDOM_STATE,
    shuffle=True,
)



## === cell 5
clf = OneVsRestClassifier(
    LogisticRegression(
        max_iter=2000,  # more iterations for convergence
        n_jobs=-1,
        solver="saga",
        class_weight="balanced",
        C=4.0,  # reduce regularisation strength
    )
)
clf.fit(X_train, y_train)

val_preds = clf.predict_proba(X_val)
auc_per_label = [
    roc_auc_score(y_val[:, i], val_preds[:, i]) for i in range(y_val.shape[1])
]
mean_auc = np.mean(auc_per_label)
print(f"Validation mean ROC‑AUC: {mean_auc:.5f}")



## === cell 6
test_preds = clf.predict_proba(X_test)

submission = pd.DataFrame(
    test_preds,
    columns=["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"],
)
submission.insert(0, "id", df_test["id"].values)

submission.to_csv("submission.csv", index=False)
print("Submission written to submission.csv")
