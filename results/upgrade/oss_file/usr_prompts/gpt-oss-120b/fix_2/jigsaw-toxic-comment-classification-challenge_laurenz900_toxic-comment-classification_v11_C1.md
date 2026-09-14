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
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings

warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from scipy.sparse import hstack



## === cell 1
df_train = pd.read_csv("../input/train.csv")
df_predict = pd.read_csv("../input/test.csv")




## === cell 2
def add_features(df):
    df["ex_mark"] = df["comment_text"].str.findall(r"!+").apply(len)
    df.loc[df["ex_mark"] > df["ex_mark"].quantile(0.9), "ex_mark"] = df[
        "ex_mark"
    ].quantile(0.9)

    df["qu_mark"] = df["comment_text"].str.findall(r"\?+").apply(len)
    df.loc[df["qu_mark"] > df["qu_mark"].quantile(0.9), "qu_mark"] = df[
        "qu_mark"
    ].quantile(0.9)

    df["star_mark"] = df["comment_text"].str.findall(r"\*+").apply(len)

    smileys_good = r"((:|;|X)-?(\)|P|D))\W"
    smileys_bad = r"((:|;)-?(\())\W"
    df["smileys_good"] = (
        df["comment_text"].str.extract(smileys_good, expand=True)[0].fillna(0)
    )
    df["smileys_bad"] = (
        df["comment_text"].str.extract(smileys_bad, expand=True)[0].fillna(0)
    )
    df.loc[df["smileys_good"] != 0, "smileys_good"] = 1
    df.loc[df["smileys_bad"] != 0, "smileys_bad"] = 1

    df["link_count"] = df["comment_text"].str.findall(r"\wwww\.").apply(len)
    df["quote_count"] = df["comment_text"].str.findall(r"('+|\"+)").apply(len)
    df.loc[df["quote_count"] > df["quote_count"].mean() * 2, "quote_count"] = (
        df["quote_count"].mean() * 2
    )
    df["comma_count"] = df["comment_text"].str.findall(r",+").apply(len)
    df.loc[df["comma_count"] > df["comma_count"].mean() * 2, "comma_count"] = (
        df["comma_count"].mean() * 2
    )

    patterns = [
        (r"a*h+a+h+a+", "haha"),
        (r"a+hh+", "ahh"),
        (r"(l+o+l+\s?)+", "lol"),
        (r"a+b+c\w*", "abc"),
        (r"a+r+g+h+", "argh"),
        (r"a+w+e+s+o+m+e+", "awesome"),
        (r"\ba*f+u+c*k*\b", "fuck"),
        (r"aa+ww+", "aww"),
        (r"y+e*a+y+", "yeah"),
        (r"y+e+a+h+", "yeah"),
        (r"y+e{2,}s{2,}", "yeah"),
        (r"ass", "azz"),
    ]
    for pat, repl in patterns:
        df["comment_text"] = df["comment_text"].str.replace(pat, repl, regex=True)

    df["comment_text"] = df["comment_text"].str.replace(r"(.)\1+", r"\1", regex=True)
    return df


df_train = add_features(df_train)
df_predict = add_features(df_predict)



## === cell 3
all_text = pd.concat([df_train["comment_text"], df_predict["comment_text"]])
bin_vect = TfidfVectorizer(
    min_df=4, ngram_range=(1, 2), stop_words="english", lowercase=True, binary=True
).fit(all_text)
vect = TfidfVectorizer(
    min_df=4, ngram_range=(1, 2), stop_words="english", lowercase=True, binary=False
).fit(all_text)



## === cell 4
X_train = df_train["comment_text"]
X_predict = df_predict["comment_text"]
Y = df_train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]

X_train_vectorized = vect.transform(X_train)
X_train_bin_vectorized = bin_vect.transform(X_train)
X_predict_vectorized = vect.transform(X_predict)
X_predict_bin_vectorized = bin_vect.transform(X_predict)

X_train_filtered = X_train_bin_vectorized
X_predict_filtered = X_predict_bin_vectorized



## === cell 6
train_features = hstack(
    [X_train_filtered, np.array(df_train["ex_mark"].astype("int64"))[:, None]]
)
predict_features = hstack(
    [X_predict_filtered, np.array(df_predict["ex_mark"].astype("int64"))[:, None]]
)

model = LogisticRegression()
params = {"C": [1], "random_state": [0]}

Y_predicted = pd.DataFrame()
Y_predicted["id"] = df_predict["id"]
scores = []

for y_col in Y.columns:
    gsCV = GridSearchCV(model, params, scoring="roc_auc", cv=3).fit(
        train_features, Y[y_col]
    )
    best_score = gsCV.best_score_
    scores.append(best_score)
    Y_predicted[y_col] = gsCV.predict_proba(predict_features)[:, 1]
    print(f"{y_col}: {best_score:.5f}")

print(f"Mean CV AUC across labels: {np.mean(scores):.5f}")



## === cell 7
submission = Y_predicted
submission.to_csv("submission.csv", index=False)
