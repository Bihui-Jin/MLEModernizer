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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from joblib import Parallel, delayed




## === cell 1
df_train = pd.read_csv("../input/train.csv")
df_predict = pd.read_csv("../input/test.csv")




## === cell 2
def add_features(df):
    df["ex_mark"] = (df["comment_text"].str.count("!") >= 1).astype(int)

    df["qu_mark"] = (df["comment_text"].str.count(r"\?") >= 1).astype(int)

    smileys_good_pat = r"((:|;)-?(\)|P|D))"
    smileys_bad_pat = r"((:|;)-?\'?(\())"
    df["smileys_good"] = (
        df["comment_text"].str.contains(smileys_good_pat, regex=True).astype(int)
    )
    df["smileys_bad"] = (
        df["comment_text"].str.contains(smileys_bad_pat, regex=True).astype(int)
    )

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
def _fit_vectorizer(vect, texts):
    vect.fit(texts)
    return vect


word_vect = TfidfVectorizer(
    min_df=2,
    ngram_range=(1, 2),
    stop_words="english",
    lowercase=True,
    sublinear_tf=True,
    dtype=np.float32,
)

char_vect = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=5,
    sublinear_tf=True,
    lowercase=False,
    dtype=np.float32,
)

word_vect, char_vect = Parallel(n_jobs=2)(
    delayed(_fit_vectorizer)(v, df_train["comment_text"])
    for v in (word_vect, char_vect)
)




## === cell 4
X_train_word = word_vect.transform(df_train["comment_text"])
X_predict_word = word_vect.transform(df_predict["comment_text"])
X_train_char = char_vect.transform(df_train["comment_text"])
X_predict_char = char_vect.transform(df_predict["comment_text"])

X_train_vectorized = hstack([X_train_word, X_train_char])
X_predict_vectorized = hstack([X_predict_word, X_predict_char])




## === cell 5
print("Chi‑square diagnostics skipped to reduce runtime.")




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

train_extra_sparse = csr_matrix(df_train[extra_cols].astype(np.int8).values)
predict_extra_sparse = csr_matrix(df_predict[extra_cols].astype(np.int8).values)

train_features = hstack([X_train_vectorized, train_extra_sparse])
predict_features = hstack([X_predict_vectorized, predict_extra_sparse])




## === cell 7
subset_size = 30_000
if train_features.shape[0] > subset_size:
    rng = np.random.RandomState(0)
    cv_indices = rng.choice(train_features.shape[0], size=subset_size, replace=False)
    cv_features = train_features[cv_indices]
    cv_labels = df_train.iloc[cv_indices][
        ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    ]
else:
    cv_features = train_features
    cv_labels = df_train[
        ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    ]

label_names = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


def train_and_predict(col_name):
    """Fit LogisticRegression for a single label and return CV AUC and predictions."""
    model = LogisticRegression(
        C=2,
        max_iter=1000,
        solver="saga",
        n_jobs=1,  # use a single core per model; outer Parallel handles concurrency
        class_weight="balanced",
        random_state=0,
    )
    cv_score = cross_val_score(
        model,
        cv_features,
        cv_labels[col_name],
        cv=2,
        scoring="roc_auc",
        n_jobs=1,
    ).mean()
    model.fit(train_features, df_train[col_name])
    preds = model.predict_proba(predict_features)[:, 1]
    return col_name, cv_score, preds


results = Parallel(n_jobs=2, backend="threading")(
    delayed(train_and_predict)(c) for c in label_names
)

Y_predicted = pd.DataFrame()
Y_predicted["id"] = df_predict["id"]
cv_scores = []

for col_name, cv_score, preds in results:
    cv_scores.append(cv_score)
    Y_predicted[col_name] = preds
    print(f"{col_name}: {cv_score:.5f}")

print("Mean CV AUC across labels:", np.mean(cv_scores))




## === cell 8
submission = Y_predicted
submission.to_csv("submission.csv", index=False)
