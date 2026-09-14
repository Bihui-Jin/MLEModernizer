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

# 5. Target score

0.97184

# 6. Current score

0.83609

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.83609) has done: 'The changes parallelize the six logistic‑regression trainings with `MultiOutputClassifier`, removing the explicit Python loop that refits the same estimator six times. This keeps the exact same model configuration and prediction logic while cutting total wall‑clock time dramatically. Minor I/O tweaks (using `.to_numpy` for the extra sparse features and adding a fixed random seed) avoid unnecessary data copies and ensure reproducibility without altering results.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from scipy.sparse import hstack, csr_matrix

np.random.seed(0)




## === cell 1
df_train = pd.read_csv("../input/train.csv")
df_predict = pd.read_csv("../input/test.csv")




## === cell 2
def add_features(df):
    df["ex_mark"] = (df["comment_text"].str.count("!") >= 1).astype(int)
    df["qu_mark"] = (df["comment_text"].str.count(r"\?") >= 1).astype(int)

    smileys_good_pat = r"((:|;|X)-?(\)|P|D))"
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
    df["quote_count"] = df["comment_text"].str.count(r"(\'|\")")
    df["comma_count"] = df["comment_text"].str.count(r"\,")

    patterns = {
        r"a*h+a+h+a+": "haha",
        r"a+hh+": "ahh",
        r"(lo+l+\s?)+": "lol",
        r"a+b+c\w*": "abc",
        r"a+r+g+h+": "argh",
        r"a+w+e+s+o+m+e+": "awesome",
        r"\ba*f+u+c*k*\b": "fuck",
        r"aa+ww+": "aww",
        r"abdu\w*": "abdu",
        r"y+e*a*y+": "yeah",
        r"y+e*a+h+": "yeah",
        r"y+e{2,}s{2,}": "yeah",
        r"(.)\1{1,}": r"\1",
    }
    df["comment_text"] = df["comment_text"].replace(patterns, regex=True)

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

X_train_extra = csr_matrix(df_train[feature_cols].to_numpy(np.int8))
X_test_extra = csr_matrix(df_predict[feature_cols].to_numpy(np.int8))

train_features = hstack([X_train_text, X_train_extra])
test_features = hstack([X_test_text, X_test_extra])

Y = df_train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]




## === cell 5
base_model = LogisticRegression(max_iter=1000, solver="saga", random_state=0, n_jobs=-1)

multi_clf = MultiOutputClassifier(base_model, n_jobs=-1)
multi_clf.fit(train_features, Y)

prob_arrays = multi_clf.predict_proba(test_features)  # list of (n_samples, 2) arrays

pred_df = pd.DataFrame()
pred_df["id"] = df_predict["id"]
for col, probs in zip(Y.columns, prob_arrays):
    pred_df[col] = probs[:, 1]  # probability of class 1
    print(f"{col}: model trained and predictions generated")

submission = pred_df[
    ["id", "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
]
submission.to_csv("submission.csv", index=False)
