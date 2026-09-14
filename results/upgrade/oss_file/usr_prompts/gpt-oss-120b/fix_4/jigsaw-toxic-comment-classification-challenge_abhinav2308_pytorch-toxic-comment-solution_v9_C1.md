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

3.7

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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
sklearn-pandas==2.2.0
spacy==3.8.7
spacy-legacy==3.0.12
spacy-loggers==1.0.5
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124

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

0.50776

# 6. Current score

0.96166

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.96166) has done: 'Implemented a robust file‑locating helper that searches the expected directories and falls back to a recursive glob, fixing the FileNotFoundError that blocked the whole pipeline. Re‑indexed cells to start at 1, added necessary imports, and kept the original modeling workflow unchanged. The script now loads the data, trains the TF‑IDF + LogisticRegression model, evaluates validation AUC, and writes a correctly‑formatted `submission.csv`.'

# 9. Code solution

## === cell 0
import os, re, random, warnings, glob
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def locate_file(rel_path):
    """
    Return the absolute path to `rel_path` if it exists in one of the
    expected Kaggle data locations. If not found, fall back to a recursive
    search from the current working directory.
    """
    possible_roots = [
        "./data/jigsaw-toxic-comment-classification-challenge",
        "./data/input/jigsaw-toxic-comment-classification-challenge",
        "./kaggle/data/jigsaw-toxic-comment-classification-challenge",
        "./kaggle/input/jigsaw-toxic-comment-classification-challenge",
        "./data",
        "./",
    ]
    for root in possible_roots:
        candidate = Path(root) / rel_path
        if candidate.is_file():
            return str(candidate)
    matches = list(Path(".").rglob(rel_path))
    if matches:
        return str(matches[0])
    raise FileNotFoundError(f"Could not find {rel_path} in any expected location.")


train_path = locate_file("train.csv")
test_path = locate_file("test.csv")
sample_sub_path = locate_file("sample_submission.csv")



## === cell 1
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
X = train_df["comment_text"].fillna("").astype(str)
y = train_df[label_cols].values



## === cell 2
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=SEED, stratify=y[:, 0]
)



## === cell 3
tfidf = TfidfVectorizer(
    max_features=50000,
    stop_words="english",
    ngram_range=(1, 2),
    tokenizer=lambda txt: re.sub(r"[^a-zA-Z\s]", "", txt).lower().split(),
)
X_train_vec = tfidf.fit_transform(X_train)
X_val_vec = tfidf.transform(X_val)



## === cell 4
base_clf = LogisticRegression(
    solver="saga",
    max_iter=1000,
    n_jobs=-1,
    class_weight="balanced",
    penalty="l2",
    random_state=SEED,
)
model = OneVsRestClassifier(base_clf)
model.fit(X_train_vec, y_train)



## === cell 5
val_pred = model.predict_proba(X_val_vec)
auc_scores = [
    roc_auc_score(y_val[:, i], val_pred[:, i]) for i in range(len(label_cols))
]
mean_auc = np.mean(auc_scores)
print(f"Validation ROC‑AUC per label: {auc_scores}")
print(f"Mean ROC‑AUC: {mean_auc:.5f}")



## === cell 6
test_vec = tfidf.transform(test_df["comment_text"].fillna("").astype(str))
test_pred = model.predict_proba(test_vec)

submission = pd.DataFrame(test_pred, columns=label_cols)
submission.insert(0, "id", test_df["id"])
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")
