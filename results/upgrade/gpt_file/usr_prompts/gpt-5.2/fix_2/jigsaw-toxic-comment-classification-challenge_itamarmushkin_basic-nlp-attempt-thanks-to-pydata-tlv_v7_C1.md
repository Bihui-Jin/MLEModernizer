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

gensim==4.4.0
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

0.68352

# 6. Current score

0.93944

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.93944) has done: 'I fix the pandas `str.replace` failure by explicitly setting `regex=True` and handling missing comment text, which unblocks vectorization and training. Then I ensure the model produces valid probability predictions for each of the six labels (required for ROC AUC evaluation), switching from `LinearSVC.predict()` hard labels to `LogisticRegression.predict_proba()` while keeping the same TF‑IDF feature pipeline and one-vs-rest loop. Finally, I build the submission from `sample_submission.csv` to guarantee the exact required columns/order and correct row count aligned to test `id`, and write `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os, re

from sklearn.feature_extraction import text
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression

INPUT_BASES = [
    "../input",
    "/kaggle/input",
    "/kaggle/data",
]


def _find_file(filename):
    for base in INPUT_BASES:
        cand = os.path.join(base, filename)
        if os.path.exists(cand):
            return cand
        cand2 = os.path.join(
            base, "jigsaw-toxic-comment-classification-challenge", filename
        )
        if os.path.exists(cand2):
            return cand2
    raise FileNotFoundError(f"Could not find {filename} under {INPUT_BASES}")


train_path = _find_file("train.csv")
test_path = _find_file("test.csv")
sample_path = _find_file("sample_submission.csv")

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)
sample_submission = pd.read_csv(sample_path)

print(
    "train:",
    train_data.shape,
    "test:",
    test_data.shape,
    "sample:",
    sample_submission.shape,
)

X = train_data["comment_text"]
X_test = test_data["comment_text"]

labels = train_data.columns.values[2:]  # six target columns




## === cell 1
def clean_text(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str).str.lower()
    digits = re.compile(r"\d[\d\.\$]*")
    not_allowed = re.compile(r"[^\s\w<>_]")
    s = s.str.replace(digits, "", regex=True)
    s = s.str.replace(not_allowed, "", regex=True)
    return s




## === cell 2
X = clean_text(X)
X_test = clean_text(X_test)

ys = train_data[labels].copy()

X_train, X_crossval, y_trains, y_crossvals = train_test_split(
    X, ys, test_size=0.3, random_state=20180301
)

vectorizer = text.TfidfVectorizer(max_features=1000, max_df=0.05)
vectorizer.fit(X_train)

X_train_vec = vectorizer.transform(X_train)
X_crossval_vec = vectorizer.transform(X_crossval)
X_test_vec = vectorizer.transform(X_test)

print("Vectorized shapes:", X_train_vec.shape, X_crossval_vec.shape, X_test_vec.shape)



## === cell 3
test_pred = pd.DataFrame({"id": test_data["id"].values})

for label in labels:
    y_train = y_trains[label].values
    y_crossval = y_crossvals[label].values

    model = LogisticRegression(
        solver="liblinear",
        max_iter=1000,
        random_state=20180301,
    )
    model.fit(X_train_vec, y_train)

    yh_crossval = (model.predict_proba(X_crossval_vec)[:, 1] >= 0.5).astype(int)
    print(label)
    print(classification_report(y_crossval, yh_crossval, zero_division=0))

    yh_test_proba = model.predict_proba(X_test_vec)[:, 1]
    test_pred[label] = yh_test_proba



## === cell 4
sub = sample_submission[["id"] + list(labels)].copy()

sub = sub.merge(test_pred, on="id", how="left", suffixes=("", "_pred"))

for label in labels:
    pred_col = f"{label}_pred"
    if pred_col in sub.columns:
        sub[label] = sub[pred_col]
        sub.drop(columns=[pred_col], inplace=True)

assert list(sub.columns) == ["id"] + list(labels), "Submission columns/order mismatch"
assert (
    sub.shape[0] == sample_submission.shape[0]
), "Row count mismatch vs sample submission"
assert sub["id"].isna().sum() == 0, "Missing ids in submission"
for label in labels:
    sub[label] = sub[label].astype(float).clip(0.0, 1.0)

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv:", sub.shape)
print(sub.head())
