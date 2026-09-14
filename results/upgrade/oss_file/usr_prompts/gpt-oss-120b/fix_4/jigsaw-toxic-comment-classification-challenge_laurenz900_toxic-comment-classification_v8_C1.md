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

3.6

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
nltk==3.9.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
sklearn-pandas==2.2.0
textblob==0.19.0

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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from textblob import TextBlob
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from scipy.sparse import hstack, csr_matrix




## === cell 1
df_train = pd.read_csv("../input/train.csv")
df_predict = pd.read_csv("../input/test.csv")




## === cell 2
def add_features(df):
    df["ex_mark"] = df["comment_text"].str.count("!")
    df["ex_mark"] = (df["ex_mark"] >= 1).astype(int)

    df["qu_mark"] = df["comment_text"].str.count("\?")
    df["qu_mark"] = (df["qu_mark"] >= 1).astype(int)

    smileys_good = r"((:|;|X)-?(\)|P|D))"
    smileys_bad = r"((:|;)-?\'?(\())"
    df["smileys_good"] = (
        df["comment_text"].str.extract(smileys_good, expand=True)[0].fillna(0)
    )
    df["smileys_bad"] = (
        df["comment_text"].str.extract(smileys_bad, expand=True)[0].fillna(0)
    )
    df.loc[df["smileys_good"] != 0, "smileys_good"] = 1
    df.loc[df["smileys_bad"] != 0, "smileys_bad"] = 1
    df["smileys_good"] = df["smileys_good"].astype(int)
    df["smileys_bad"] = df["smileys_bad"].astype(int)

    df["word_count"] = df["comment_text"].str.findall(r"(?u)\b\w\w+\b").apply(len)
    df["sent_count"] = df["comment_text"].str.count(r"\.\b")
    df["link_count"] = df["comment_text"].str.count(r"\.www")
    df["quote_count"] = df["comment_text"].str.count(r"(\'|\")")
    df["comma_count"] = df["comment_text"].str.count(r"\,")

    patterns = [
        (r"a*h+a+h+a+", "haha"),
        (r"a+hh+", "ahh"),
        (r"(lo+l+\s?)+", "lol"),
        (r"a+b+c\w*", "abc"),
        (r"a+r+g+h+", "argh"),
        (r"a+w+e+s+o+m+e+", "awesome"),
        (r"\ba*f+u+c*k*\b", "fuck"),
        (r"aa+ww+", "aww"),
        (r"abdu\w*", "abdu"),
        (r"y+e*a*y+", "yeah"),
        (r"y+e*a+h+", "yeah"),
        (r"y+e{2,}s{2,}", "yeah"),
        (r"(.)\1{1,}", r"\1"),
    ]
    for pat, repl in patterns:
        df["comment_text"] = df["comment_text"].str.replace(pat, repl, regex=True)

    return df


df_train = add_features(df_train)
df_predict = add_features(df_predict)




## === cell 3
all_text = pd.concat([df_train["comment_text"], df_predict["comment_text"]])
vect = TfidfVectorizer(
    min_df=4, ngram_range=(1, 2), stop_words="english", lowercase=True
).fit(all_text)




## === cell 4
feature_cols = [
    "smileys_good",
    "smileys_bad",
    "ex_mark",
    "qu_mark",
    "word_count",
    "sent_count",
    "link_count",
    "quote_count",
    "comma_count",
]

X_train_text = vect.transform(df_train["comment_text"])
X_test_text = vect.transform(df_predict["comment_text"])

X_train_extra = csr_matrix(df_train[feature_cols].astype(np.int64).values)
X_test_extra = csr_matrix(df_predict[feature_cols].astype(np.int64).values)

train_features = hstack([X_train_text, X_train_extra])
test_features = hstack([X_test_text, X_test_extra])

Y = df_train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]




## === cell 5
model = LogisticRegression(max_iter=1000, solver="saga", random_state=0)
param_grid = {"C": [1], "random_state": [0]}

pred_df = pd.DataFrame()
pred_df["id"] = df_predict["id"]
scores = []

for col in Y.columns:
    grid = GridSearchCV(model, param_grid, scoring="roc_auc", cv=3, n_jobs=1)
    grid.fit(train_features, Y[col])
    best_score = grid.best_score_
    scores.append(best_score)
    pred_df[col] = grid.predict_proba(test_features)[:, 1]
    print(f"{col}: {best_score:.5f}")

print("Mean CV AUC across labels:", np.mean(scores))

submission = pred_df[
    ["id", "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
]
submission.to_csv("submission.csv", index=False)
