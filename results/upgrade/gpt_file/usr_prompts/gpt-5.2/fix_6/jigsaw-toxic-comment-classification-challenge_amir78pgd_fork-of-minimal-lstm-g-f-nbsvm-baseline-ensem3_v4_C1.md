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

# 5. Code solution

## === cell 0
import os
import re
import numpy as np
import pandas as pd

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_DIR_CANDIDATES = [
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge",
    "/kaggle/data/jigsaw-toxic-comment-classification-challenge",
    "/kaggle/input",
    "/kaggle/data",
    "../input/jigsaw-toxic-comment-classification-challenge",
    "../data/jigsaw-toxic-comment-classification-challenge",
    "../input",
    "../data",
]


def _first_existing_file(relpath):
    for b in BASE_DIR_CANDIDATES:
        f = os.path.join(b, relpath)
        if os.path.exists(f):
            return f
    return None


train_path = _first_existing_file("train.csv") or _first_existing_file(
    "jigsaw-toxic-comment-classification-challenge/train.csv"
)
test_path = _first_existing_file("test.csv") or _first_existing_file(
    "jigsaw-toxic-comment-classification-challenge/test.csv"
)
sample_path = _first_existing_file("sample_submission.csv") or _first_existing_file(
    "jigsaw-toxic-comment-classification-challenge/sample_submission.csv"
)

if train_path is None or test_path is None or sample_path is None:
    raise FileNotFoundError(
        f"Could not locate required CSVs. Resolved paths: train={train_path}, test={test_path}, sample={sample_path}"
    )

print("Using paths:")
print(" train:", train_path)
print(" test :", test_path)
print(" sample:", sample_path)



## === cell 1
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

train_df = pd.read_csv(
    train_path,
    usecols=["id", "comment_text"] + label_cols,
    dtype={c: np.int8 for c in label_cols},
)
test_df = pd.read_csv(
    test_path,
    usecols=["id", "comment_text"],
)
sample_sub = pd.read_csv(sample_path, usecols=["id"] + label_cols)

train_text = train_df["comment_text"].fillna("").astype(str).values
test_text = test_df["comment_text"].fillna("").astype(str).values

y = train_df[label_cols].astype(np.int32).values

print(
    "Train shape:",
    train_df.shape,
    "Test shape:",
    test_df.shape,
    "Sample shape:",
    sample_sub.shape,
)



## === cell 2
import sklearn

_WORD_TOKEN_PATTERN = r"(?u)\b\w\w+\b"

train_text_lower = np.char.lower(train_text.astype(str))
test_text_lower = np.char.lower(test_text.astype(str))


def train_and_predict(tfidf_params, lr_params, model_name="model"):
    vec = TfidfVectorizer(**tfidf_params)
    X_tr = vec.fit_transform(
        train_text_lower if tfidf_params.get("analyzer") == "char" else train_text
    )
    X_te = vec.transform(
        test_text_lower if tfidf_params.get("analyzer") == "char" else test_text
    )

    base_lr = LogisticRegression(**lr_params)
    clf = OneVsRestClassifier(base_lr, n_jobs=-1)
    clf.fit(X_tr, y)
    proba = clf.predict_proba(X_te)  # shape: (n_test, 6)
    return proba


tfidf_1 = dict(
    strip_accents="unicode",
    analyzer="word",
    lowercase=True,
    tokenizer=None,
    preprocessor=None,
    token_pattern=_WORD_TOKEN_PATTERN,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.9,
    sublinear_tf=True,
    max_features=200000,
)
tfidf_2 = dict(
    strip_accents="unicode",
    analyzer="word",
    lowercase=True,
    tokenizer=None,
    preprocessor=None,
    token_pattern=_WORD_TOKEN_PATTERN,
    ngram_range=(1, 1),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    max_features=150000,
)
tfidf_3 = dict(
    strip_accents="unicode",
    analyzer="char",
    lowercase=False,
    preprocessor=None,
    ngram_range=(3, 5),
    min_df=2,
    max_df=0.9,
    sublinear_tf=True,
    max_features=200000,
)
tfidf_4 = dict(
    strip_accents="unicode",
    analyzer="char",
    lowercase=False,
    preprocessor=None,
    ngram_range=(4, 6),
    min_df=2,
    max_df=0.9,
    sublinear_tf=True,
    max_features=200000,
)

lr_1 = dict(
    solver="saga",
    penalty="l2",
    C=4.0,
    max_iter=200,
    random_state=RANDOM_STATE,
    n_jobs=1,
)
lr_2 = dict(
    solver="saga",
    penalty="l2",
    C=2.0,
    max_iter=200,
    random_state=RANDOM_STATE,
    n_jobs=1,
)
lr_3 = dict(
    solver="saga",
    penalty="l2",
    C=3.0,
    max_iter=150,
    random_state=RANDOM_STATE,
    n_jobs=1,
)
lr_4 = dict(
    solver="saga",
    penalty="l2",
    C=1.5,
    max_iter=150,
    random_state=RANDOM_STATE,
    n_jobs=1,
)

tasks = [
    ("model_1", tfidf_1, lr_1),
    ("model_2", tfidf_2, lr_2),
    ("model_3", tfidf_3, lr_3),
    ("model_4", tfidf_4, lr_4),
]

print(f"scikit-learn={sklearn.__version__} | running {len(tasks)} models sequentially")

results = []
for name, tfidf, lr in tasks:
    print(f"Training {name} ...")
    results.append(train_and_predict(tfidf, lr, name))

pred1, pred2, pred3, pred4 = results


def make_pred_df(pred):
    df = pd.DataFrame(pred, columns=label_cols)
    df.insert(0, "id", test_df["id"].values)
    return df


p_lstm_glove_pl = make_pred_df(pred1)
p_nbsvm = make_pred_df(pred2)
p_lstm_fast_pl = make_pred_df(pred3)
p_bi_lstm_dual = make_pred_df(pred4)

print(
    "Prepared 4 prediction frames:",
    [d.shape for d in [p_lstm_glove_pl, p_nbsvm, p_lstm_fast_pl, p_bi_lstm_dual]],
)




## === cell 3
def align_to_sample(pred_df, sample_df):
    sample_ids = sample_df["id"].values
    pred_ids = pred_df["id"].values

    sample_pos = pd.Index(sample_ids).get_indexer(pred_ids)  # -1 if not found

    out = sample_df[["id"]].copy()
    out_mat = np.full((len(sample_ids), len(label_cols)), 0.5, dtype=np.float64)

    valid = sample_pos >= 0
    if np.any(valid):
        out_mat[sample_pos[valid], :] = pred_df.loc[valid, label_cols].to_numpy(
            dtype=np.float64, copy=False
        )

    out[label_cols] = out_mat
    return out


p_lstm_glove_pl = align_to_sample(p_lstm_glove_pl, sample_sub)
p_nbsvm = align_to_sample(p_nbsvm, sample_sub)
p_lstm_fast_pl = align_to_sample(p_lstm_fast_pl, sample_sub)
p_bi_lstm_dual = align_to_sample(p_bi_lstm_dual, sample_sub)



## === cell 4
p_res = p_lstm_glove_pl.copy()
p_res[label_cols] = (
    p_nbsvm[label_cols].values
    + p_lstm_glove_pl[label_cols].values
    + p_lstm_fast_pl[label_cols].values
    + p_bi_lstm_dual[label_cols].values
) / 4.0

p_res[label_cols] = p_res[label_cols].clip(0.0, 1.0)



## === cell 5
p_res = p_res[["id"] + label_cols]
if len(p_res) != len(sample_sub):
    raise ValueError(
        f"Submission row count mismatch: got {len(p_res)} expected {len(sample_sub)}"
    )
print(p_res.head())
print("Submission columns:", list(p_res.columns))



## === cell 6
p_res.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", p_res.shape)
print(
    "File exists:",
    os.path.exists("submission.csv"),
    "Size:",
    os.path.getsize("submission.csv"),
)
