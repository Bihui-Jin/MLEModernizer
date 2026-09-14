# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

3.10

# 3. Installed packages

geopandas==0.14.4
nltk==3.9.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
seaborn==0.12.2
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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

0.66487

# 6. Current score

0.96252

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.88404) has done: 'I remove the protobuf environment setting that causes an import error, fix the model’s input shape to match Keras expectations (using a scalar string input), and ensure the training, validation, and test pipelines work with this corrected input. These minimal changes eliminate the runtime failures, allow the model to train, generate predictions, and finally write a proper `submission.csv` file, moving the solution toward the target score.'
- What this solution (achieved 0.96252) has done: 'The changes keep the same preprocessing, vectorization, model type, and validation evaluation, but eliminate the second full‑dataset re‑training which doubled the computation time. By re‑using the model already fitted on the training split we stay within the 600 s limit while preserving the original logic for cleaning, TF‑IDF construction, validation AUC calculation, and test prediction.'

# 9. Code solution

## === cell 0
import os
import re
import string

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score

import scipy.sparse as sp

try:
    from sklearnex import patch

    patch()
except ImportError:
    pass




## === cell 1
class Config:
    tfidf_vocab_size = 40000
    batch_size = 1024
    validation_split = 0.15
    epochs = 10
    best_auc_model_path = "model_best_auc.weights.h5"
    best_acc_model_path = "model_best_acc.weights.h5"
    lastest_model_path = "model_latest.weights.h5"
    labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


config = Config()



## === cell 2
train_path = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv"
test_path = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv"
sample_sub_path = (
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv"
)

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample_submission = pd.read_csv(sample_sub_path)



## === cell 3
_clean_pattern = re.compile(
    r"<.*?>|[" + re.escape(string.punctuation) + r"]|\d+", flags=re.UNICODE
)


def vectorized_clean(series: pd.Series) -> pd.Series:
    s = series.str.lower()
    s = s.str.replace(_clean_pattern, " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s


np.random.seed(42)

cleaned_comments = vectorized_clean(train["comment_text"].astype(str))

X = cleaned_comments
y = train[config.labels].astype(np.int8)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=config.validation_split, random_state=42
)



## === cell 4
tfidf_vectorizer = TfidfVectorizer(
    max_features=config.tfidf_vocab_size,
    ngram_range=(1, 2),
    stop_words="english",
    dtype=np.float32,
)

X_train_vec = tfidf_vectorizer.fit_transform(X_train)
X_val_vec = tfidf_vectorizer.transform(X_val)



## === cell 5
base_clf = LogisticRegression(
    max_iter=1000,
    n_jobs=-1,
    class_weight="balanced",
    solver="saga",
    warm_start=True,
    random_state=42,
)
model = OneVsRestClassifier(base_clf)

model.fit(X_train_vec, y_train.values)



## === cell 6
val_probs = model.predict_proba(X_val_vec)
val_auc = np.mean(
    [roc_auc_score(y_val[col], val_probs[:, i]) for i, col in enumerate(config.labels)]
)
print(f"Mean validation ROC‑AUC: {val_auc:.5f}")



## === cell 8
test_X = vectorized_clean(test["comment_text"].astype(str))
test_vec = tfidf_vectorizer.transform(test_X)
test_preds = model.predict_proba(test_vec)



## === cell 9
submission = sample_submission.copy()
submission[config.labels] = test_preds
submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
