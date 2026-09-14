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
from scipy.sparse import hstack, csr_matrix

try:
    from sklearnex import patch

    patch()
except Exception:
    pass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline




## === cell 1
df_train = pd.read_csv("../input/train.csv")
df_predict = pd.read_csv("../input/test.csv")




## === cell 2
def add_features(df):
    df["ex_mark"] = df["comment_text"].str.count("!")
    df["ex_mark"] = (df["ex_mark"] >= 1).astype(int)

    df["qu_mark"] = df["comment_text"].str.count(r"\?")
    df["qu_mark"] = (df["qu_mark"] >= 1).astype(int)

    smileys_good = r"((:|;)-?(\)|P|D))"
    smileys_bad = r"((:|;)-?\'?(\())"
    df["smileys_good"] = (
        df["comment_text"].str.extract(smileys_good, expand=True)[0].fillna(0)
    )
    df["smileys_bad"] = (
        df["comment_text"].str.extract(smileys_bad, expand=True)[0].fillna(0)
    )
    df["smileys_good"] = (df["smileys_good"] != 0).astype(int)
    df["smileys_bad"] = (df["smileys_bad"] != 0).astype(int)

    df["word_count"] = df["comment_text"].str.findall(r"(?u)\b\w\w+\b").apply(len)
    df["sent_count"] = df["comment_text"].str.count(r"\.\b")
    df["link_count"] = df["comment_text"].str.count(r"\.www")
    df["quote_count"] = df["comment_text"].str.count(r"(\"|\')")
    df["comma_count"] = df["comment_text"].str.count(r",")

    df["comment_text"] = df["comment_text"].str.replace(r"h+a+h+a+", "haha", regex=True)
    df["comment_text"] = df["comment_text"].str.replace(
        r"(lol\s?lol)+", "lol", regex=True
    )
    df["comment_text"] = df["comment_text"].str.replace(r"abc\w*", "abc", regex=True)
    df["comment_text"] = df["comment_text"].str.replace(r"a+r+g+h+", "argh", regex=True)
    df["comment_text"] = df["comment_text"].str.replace(
        r"a+w+e+s+o+m+e+", "awesome", regex=True
    )

    return df


df_train = add_features(df_train)
df_predict = add_features(df_predict)




## === cell 3
vect = TfidfVectorizer(
    min_df=4, ngram_range=(1, 2), stop_words="english", lowercase=True
).fit(df_train["comment_text"])




## === cell 4
X_train = df_train[
    [
        "comment_text",
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
]
X_predict = df_predict[
    [
        "comment_text",
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
]

Y = df_train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]

X_train_vectorized = vect.transform(X_train["comment_text"])
X_predict_vectorized = vect.transform(X_predict["comment_text"])




## === cell 5
try:
    scores = chi2(X_train_vectorized, Y["toxic"])[1]
    feature_names = vect.get_feature_names_out()
    chis = pd.DataFrame(
        list(zip(feature_names, scores)), columns=["feature", "score"]
    ).sort_values(by="score")
    rel_pct = (chis["score"] < 0.3).mean()
    print("Perc. of rel. features (toxic):", rel_pct)
except Exception as e:
    print("Chi‑square diagnostics skipped:", e)




## === cell 6
extra_cols = [
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

train_extra_sparse = csr_matrix(X_train[extra_cols].astype(np.int8).values)
predict_extra_sparse = csr_matrix(X_predict[extra_cols].astype(np.int8).values)

train_features = hstack([X_train_vectorized, train_extra_sparse])
predict_features = hstack([X_predict_vectorized, predict_extra_sparse])




## === cell 7
Y_predicted = pd.DataFrame()
Y_predicted["id"] = df_predict["id"]
cv_scores = []

for col in Y.columns:
    model = LogisticRegression(
        C=1,
        max_iter=1000,
        solver="saga",
        n_jobs=-1,  # enable parallelism inside the estimator where possible
        random_state=0,
    )
    cv_score = cross_val_score(
        model,
        train_features,
        Y[col],
        cv=3,
        scoring="roc_auc",
        n_jobs=-1,
    ).mean()
    cv_scores.append(cv_score)

    model.fit(train_features, Y[col])
    Y_predicted[col] = model.predict_proba(predict_features)[:, 1]

    print(f"{col}: {cv_score:.5f}")

print("Mean CV AUC across labels:", np.mean(cv_scores))




## === cell 8
submission = Y_predicted
submission.to_csv("submission.csv", index=False)
