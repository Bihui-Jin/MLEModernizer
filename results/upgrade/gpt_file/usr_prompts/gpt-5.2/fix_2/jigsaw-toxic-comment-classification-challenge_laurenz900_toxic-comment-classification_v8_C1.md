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

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectPercentile, chi2
from scipy.sparse import hstack, csr_matrix

import matplotlib.pyplot as plt
import re




## === cell 1
def _resolve_path(filename: str) -> str:
    candidates = [
        f"../input/{filename}",
        f"/kaggle/input/{filename}",
        f"/kaggle/input/jigsaw-toxic-comment-classification-challenge/{filename}",
        f"/kaggle/data/{filename}",
        f"/kaggle/data/jigsaw-toxic-comment-classification-challenge/{filename}",
    ]
    for p in candidates:
        try:
            with open(p, "rb"):
                return p
        except Exception:
            pass
    return candidates[0]


train_path = _resolve_path("train.csv")
test_path = _resolve_path("test.csv")
sample_sub_path = _resolve_path("sample_submission.csv")

df_train = pd.read_csv(train_path)
df_predict = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_sub_path)




## === cell 2
def add_features(df):
    df = df.copy()
    df["comment_text"] = df["comment_text"].fillna("")

    df["ex_mark"] = df["comment_text"].str.findall(r"\!").apply(len)
    df["ex_mark"] = (df["ex_mark"] >= 1).astype("int64")

    df["qu_mark"] = df["comment_text"].str.findall(r"\?").apply(len)
    df["qu_mark"] = (df["qu_mark"] >= 1).astype("int64")

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
    df["smileys_good"] = df["smileys_good"].astype("int64")
    df["smileys_bad"] = df["smileys_bad"].astype("int64")

    df["word_count"] = (
        df["comment_text"].str.findall(r"(?u)\b\w\w+\b").apply(len).astype("int64")
    )
    df["sent_count"] = (
        df["comment_text"].str.findall(r"\.\b").apply(len).astype("int64")
    )
    df["link_count"] = (
        df["comment_text"].str.findall(r"\.www").apply(len).astype("int64")
    )
    df["quote_count"] = (
        df["comment_text"].str.findall(r"(\'|\")").apply(len).astype("int64")
    )
    df["comma_count"] = df["comment_text"].str.findall(r"\,").apply(len).astype("int64")

    df["comment_text"] = df["comment_text"].str.replace(
        r"a*h+a+h+a+", "haha", regex=True
    )
    df["comment_text"] = df["comment_text"].str.replace(r"a+hh+", "ahh", regex=True)
    df["comment_text"] = df["comment_text"].str.replace(
        r"(lo+l+\s?)+", "lol", regex=True
    )
    df["comment_text"] = df["comment_text"].str.replace(r"a+b+c\w*", "abc", regex=True)
    df["comment_text"] = df["comment_text"].str.replace(r"a+r+g+h+", "argh", regex=True)
    df["comment_text"] = df["comment_text"].str.replace(
        r"a+w+e+s+o+m+e+", "awesome", regex=True
    )
    df["comment_text"] = df["comment_text"].str.replace(
        r"\ba*f+u+c*k*\b", "fuck", regex=True
    )
    df["comment_text"] = df["comment_text"].str.replace(r"aa+ww+", "aww", regex=True)
    df["comment_text"] = df["comment_text"].str.replace(r"abdu\w*", "abdu", regex=True)
    df["comment_text"] = df["comment_text"].str.replace(r"y+e*a+y+", "yeah", regex=True)
    df["comment_text"] = df["comment_text"].str.replace(r"y+e+a+h+", "yeah", regex=True)
    df["comment_text"] = df["comment_text"].str.replace(
        r"y+e{2,}s{2,}", "yeah", regex=True
    )

    df["comment_text"] = df["comment_text"].str.replace(r"(.)\1{1,}", r"\1", regex=True)
    return df


df_train = add_features(df_train)
df_predict = add_features(df_predict)

all_text = pd.concat([df_train["comment_text"], df_predict["comment_text"]], axis=0)

target_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
Y = df_train[target_cols]

df_train.head()



## === cell 3
vect = TfidfVectorizer(
    min_df=4, ngram_range=(1, 2), stop_words="english", lowercase=True
).fit(all_text)



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

X_train_vectorized = vect.transform(X_train["comment_text"])
X_predict_vectorized = vect.transform(X_predict["comment_text"])



## === cell 5
scores = chi2(X_train_vectorized, Y["toxic"])[1]
feature_names = vect.get_feature_names_out()
chis = pd.DataFrame(list(zip(feature_names, scores))).sort_values(by=1)
print("Perc. of rel. features (toxic p<0.3): " + str((chis[1] < 0.3).mean()))




## === cell 6
def plot_model():
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline

    model = Pipeline(
        [
            ("chi2", SelectPercentile(score_func=chi2)),
            ("lr", LogisticRegression(max_iter=1000)),
        ]
    )
    percentiles = (1, 5, 60)

    for y_col in Y.columns:
        score_means = []
        for percentile in percentiles:
            model.set_params(chi2__percentile=percentile)
            this_scores = cross_val_score(
                model, X_train_vectorized, Y[y_col], n_jobs=1, scoring="roc_auc"
            )
            score_means.append(this_scores.mean())

        plt.plot(percentiles, score_means)
        plt.title(y_col)
        plt.xlabel("Percentile")
        plt.ylabel("ROC AUC")
        plt.show()


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
X_train_extra = X_train[extra_cols].astype("int64")
X_predict_extra = X_predict[extra_cols].astype("int64")



## === cell 7
model = LogisticRegression(max_iter=1000)
params = {"C": [1], "random_state": [0]}

Y_predicted = pd.DataFrame({"id": df_predict["id"]})
scores = []

for y_col in target_cols:
    chi2_filter = SelectPercentile(score_func=chi2, percentile=60)
    X_train_filtered = chi2_filter.fit_transform(X_train_vectorized, Y[y_col])
    X_predict_filtered = chi2_filter.transform(X_predict_vectorized)

    train_features = hstack([X_train_filtered, csr_matrix(X_train_extra.values)])
    predict_features = hstack([X_predict_filtered, csr_matrix(X_predict_extra.values)])

    gsCV = GridSearchCV(model, params, scoring="roc_auc").fit(train_features, Y[y_col])
    scoreX = float(np.mean(gsCV.cv_results_["mean_test_score"]))
    scores.append(scoreX)

    Y_predicted[y_col] = gsCV.predict_proba(predict_features)[:, 1]
    print(y_col + ":" + str(scoreX))

print("mean score: " + str(float(np.mean(scores))))



## === cell 8
submission = sample_sub[["id"] + target_cols].copy()
submission = submission.merge(
    Y_predicted[["id"] + target_cols], on="id", how="left", suffixes=("_sample", "")
)

final_sub = pd.DataFrame({"id": submission["id"]})
for c in target_cols:
    final_sub[c] = submission[c].astype(float)

final_sub.to_csv("submission.csv", index=False)
print(final_sub.head())
print("Wrote submission.csv with shape:", final_sub.shape)
