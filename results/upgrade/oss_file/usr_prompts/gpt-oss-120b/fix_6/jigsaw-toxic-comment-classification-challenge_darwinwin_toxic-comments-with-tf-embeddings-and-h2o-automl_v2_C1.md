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

# 5. Target score

0.654157714273941

# 6. Current score

0.85714

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.95725) has done: 'The changes speed up the workflow by (1) removing an unnecessary cast to float32 that duplicated the large sparse matrix, (2) enabling multi‑core computation for LogisticRegression (n_jobs=-1) and parallelising the six independent model fits, and (3) parallelising the validation and test predictions. These adjustments keep the exact same model, features, and evaluation logic, only improving runtime without affecting accuracy.'
- What this solution (achieved 0.86111) has done: 'I slightly reduce the model capacity and increase regularisation so the AUC drops from the current ≈ 0.96 toward the target ≈ 0.65. Specifically, I lower the TF‑IDF max_features to 5 000 and set LogisticRegression C to 0.01 (stronger regularisation). These are minimal adjustments that keep the original pipeline intact while degrading predictive power enough to approach the desired score.'
- What this solution (achieved 0.85714) has done: 'I lower the model capacity and increase regularisation a bit more to reduce the AUC toward the target. Specifically, I shrink the TF‑IDF vocabulary (`max_features` = 1000, `min_df` = 5) and set LogisticRegression’s inverse‑regularisation `C` to 0.001. These tweaks keep the same pipeline and evaluation logic while making the model less expressive, which should bring the score down from 0.86 into the desired 0.65‑range.'

# 9. Code solution

## === cell 0
import os, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed  # parallel execution for speed

print("Available input directories:", os.listdir("../input"))



## === cell 1
train_path = "../input/jigsaw-toxic-comment-classification-challenge/train.csv"
test_path = "../input/jigsaw-toxic-comment-classification-challenge/test.csv"
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)



## === cell 2
target_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
X = train_df["comment_text"].fillna("")
y = train_df[target_cols]



## === cell 3
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y["toxic"]
)



## === cell 4
tfidf = TfidfVectorizer(
    max_features=1000,  # smaller vocab than before
    ngram_range=(1, 2),
    stop_words="english",
    min_df=5,  # filter out rarer terms
)
X_train_vec = tfidf.fit_transform(X_train)
X_val_vec = tfidf.transform(X_val)
X_test_vec = tfidf.transform(test_df["comment_text"].fillna(""))




## === cell 5
def fit_one(col):
    model = LogisticRegression(
        solver="saga",
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1,
        C=0.001,  # tighter regularisation than before
    )
    model.fit(X_train_vec, y_train[col])
    return col, model


models = dict(Parallel(n_jobs=-1)(delayed(fit_one)(col) for col in target_cols))




## === cell 6
def val_predict(item):
    col, model = item
    return col, model.predict_proba(X_val_vec)[:, 1]


val_preds = dict(
    Parallel(n_jobs=-1)(delayed(val_predict)((col, models[col])) for col in target_cols)
)




## === cell 7
def test_predict(item):
    col, model = item
    return col, model.predict_proba(X_test_vec)[:, 1]


test_preds = dict(
    Parallel(n_jobs=-1)(
        delayed(test_predict)((col, models[col])) for col in target_cols
    )
)

submission = pd.DataFrame({"id": test_df["id"]})
for col in target_cols:
    submission[col] = test_preds[col]



## === cell 8
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv with shape:", submission.shape)
