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
import glob

CANDIDATE_INPUT_DIRS = [
    "../input/jigsaw-toxic-comment-classification-challenge",
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge",
    "/kaggle/data/jigsaw-toxic-comment-classification-challenge",
    "/kaggle/data",
    "../input",
    "../data",
    "/kaggle/input",
    "/kaggle/data",
]


def find_file(filename):
    for d in CANDIDATE_INPUT_DIRS:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    for d in CANDIDATE_INPUT_DIRS:
        if os.path.exists(d):
            hits = glob.glob(os.path.join(d, "**", filename), recursive=True)
            if hits:
                return hits[0]
    raise FileNotFoundError(
        f"Could not find {filename} in candidate dirs: {CANDIDATE_INPUT_DIRS}"
    )


train_path = find_file("train.csv")
test_path = find_file("test.csv")
sub_path = find_file("sample_submission.csv")

(train_path, test_path, sub_path)



## === cell 1
import numpy as np
import pandas as pd

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample_sub = pd.read_csv(sub_path)

label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
assert all(c in train.columns for c in ["id", "comment_text"] + label_cols)
assert all(c in test.columns for c in ["id", "comment_text"])
assert (
    list(sample_sub.columns) == ["id"] + label_cols
), "Sample submission columns mismatch."

train["comment_text"] = train["comment_text"].fillna("")
test["comment_text"] = test["comment_text"].fillna("")

X_train = train["comment_text"].astype(str)
X_test = test["comment_text"].astype(str)
Y = train[label_cols].astype(np.float32).values

(train.shape, test.shape, sample_sub.shape)



## === cell 2

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.multiclass import OneVsRestClassifier


def fit_predict_proba(model_name, estimator, vectorizer):
    """
    Returns: (submission_like_df, fitted_pipeline)
    submission_like_df has columns: id + label_cols (aligned to test)
    """
    pipe = Pipeline(
        [
            ("tfidf", vectorizer),
            ("clf", OneVsRestClassifier(estimator, n_jobs=-1)),
        ]
    )
    pipe.fit(X_train, Y)
    proba = pipe.predict_proba(X_test)
    proba = np.clip(proba, 1e-6, 1 - 1e-6)

    df = pd.DataFrame(proba, columns=label_cols)
    df.insert(0, "id", test["id"].values)
    return df, pipe


vec_word = TfidfVectorizer(
    strip_accents="unicode",
    lowercase=True,
    analyzer="word",
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.9,
    sublinear_tf=True,
    norm="l2",
)

vec_char = TfidfVectorizer(
    strip_accents="unicode",
    lowercase=True,
    analyzer="char",
    ngram_range=(3, 5),
    min_df=3,
    max_df=0.9,
    sublinear_tf=True,
    norm="l2",
)

p_slgbm, _ = fit_predict_proba(
    "lr_word", LogisticRegression(solver="liblinear", C=4.0, max_iter=1000), vec_word
)

p_nbsvm, _ = fit_predict_proba("cnb_word", ComplementNB(alpha=0.5), vec_word)

svc = LinearSVC(C=2.0)
cal_svc = CalibratedClassifierCV(svc, method="sigmoid", cv=3)
p_caps_gru, _ = fit_predict_proba("cal_svc_char", cal_svc, vec_char)

p_dual_embed_pl, _ = fit_predict_proba(
    "lr_char", LogisticRegression(solver="liblinear", C=2.0, max_iter=1000), vec_char
)

p_dual_embed_mish, _ = fit_predict_proba(
    "lr_word_C1", LogisticRegression(solver="liblinear", C=1.0, max_iter=1000), vec_word
)

p_lstm_glove_tta, _ = fit_predict_proba(
    "lr_word_C8", LogisticRegression(solver="liblinear", C=8.0, max_iter=1000), vec_word
)

p_dual_embed_dehyp, _ = fit_predict_proba(
    "cnb_word_a1", ComplementNB(alpha=1.0), vec_word
)

p_lstm_fast, _ = fit_predict_proba("cnb_word_a02", ComplementNB(alpha=0.2), vec_word)

p_bi_post, _ = fit_predict_proba(
    "cal_svc_char_C1",
    CalibratedClassifierCV(LinearSVC(C=1.0), method="sigmoid", cv=3),
    vec_char,
)

p_dpcnn, _ = fit_predict_proba(
    "cal_svc_char_C3",
    CalibratedClassifierCV(LinearSVC(C=3.0), method="sigmoid", cv=3),
    vec_char,
)

p_dmcnn, _ = fit_predict_proba(
    "lr_char_C4", LogisticRegression(solver="liblinear", C=4.0, max_iter=1000), vec_char
)

p_rcn, _ = fit_predict_proba(
    "lr_char_C1", LogisticRegression(solver="liblinear", C=1.0, max_iter=1000), vec_char
)

p_attn_300d, _ = fit_predict_proba(
    "lr_char_C8", LogisticRegression(solver="liblinear", C=8.0, max_iter=1000), vec_char
)

for df in [p_slgbm, p_caps_gru, p_dual_embed_pl, p_nbsvm]:
    assert df.shape[0] == test.shape[0] and list(df.columns) == ["id"] + label_cols
    assert (df["id"].values == test["id"].values).all()

(p_caps_gru.head(), p_slgbm.head())



## === cell 3


def assert_aligned(base, other, name):
    if not (base["id"].values == other["id"].values).all():
        raise ValueError(f"ID alignment failed vs {name}")


base = p_caps_gru
for name, df in [
    ("p_slgbm", p_slgbm),
    ("p_dual_embed_pl", p_dual_embed_pl),
    ("p_dual_embed_mish", p_dual_embed_mish),
    ("p_lstm_glove_tta", p_lstm_glove_tta),
    ("p_dual_embed_dehyp", p_dual_embed_dehyp),
    ("p_lstm_fast", p_lstm_fast),
    ("p_nbsvm", p_nbsvm),
    ("p_bi_post", p_bi_post),
    ("p_dpcnn", p_dpcnn),
    ("p_dmcnn", p_dmcnn),
    ("p_rcn", p_rcn),
    ("p_attn_300d", p_attn_300d),
]:
    assert_aligned(base, df, name)

p_res = p_caps_gru.copy()
p_res[label_cols] = (
    p_slgbm[label_cols]
    + p_caps_gru[label_cols]
    + p_dual_embed_pl[label_cols]
    + p_dual_embed_mish[label_cols]
    + p_lstm_glove_tta[label_cols]
    + p_dual_embed_dehyp[label_cols]
    + p_lstm_fast[label_cols]
    + p_nbsvm[label_cols]
    + p_bi_post[label_cols] * 3
    + p_dpcnn[label_cols] * 3
    + p_dmcnn[label_cols] * 3
    + p_rcn[label_cols] * 3
    + p_attn_300d[label_cols] * 3
) / 23.0

p_res[label_cols] = p_res[label_cols].clip(1e-6, 1 - 1e-6)

p_res.head()



## === cell 4
submission = sample_sub[["id"]].copy()
submission = submission.merge(p_res, on="id", how="left", validate="one_to_one")

for c in label_cols:
    if c not in submission.columns:
        submission[c] = 0.5
submission[label_cols] = submission[label_cols].fillna(0.5).clip(1e-6, 1 - 1e-6)

submission = submission[["id"] + label_cols]

assert submission.shape[0] == sample_sub.shape[0]
assert list(submission.columns) == list(sample_sub.columns)

submission.head()



## === cell 5
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv", submission.shape)
print(submission.describe().T)
