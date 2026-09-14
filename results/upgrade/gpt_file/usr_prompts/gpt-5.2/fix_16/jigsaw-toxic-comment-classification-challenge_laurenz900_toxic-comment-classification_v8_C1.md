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
import os
import re
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import chi2
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from scipy.sparse import hstack, csr_matrix

import matplotlib.pyplot as plt

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

np.random.seed(0)




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

target_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
usecols_train = ["id", "comment_text"] + target_cols
usecols_test = ["id", "comment_text"]

dtype_train = {c: "int8" for c in target_cols}
dtype_train.update({"id": "string", "comment_text": "string"})
dtype_test = {"id": "string", "comment_text": "string"}

df_train = pd.read_csv(
    train_path, usecols=usecols_train, engine="c", low_memory=False, dtype=dtype_train
)
df_predict = pd.read_csv(
    test_path, usecols=usecols_test, engine="c", low_memory=False, dtype=dtype_test
)
sample_sub = pd.read_csv(
    sample_sub_path, engine="c", low_memory=False, dtype={"id": "string"}
)



## === cell 2
_RE_SMILEY_GOOD = re.compile(r"((:|;|X)-?(\)|P|D))")
_RE_SMILEY_BAD = re.compile(r"((:|;)-?\'?(\())")

_REPL = [
    (re.compile(r"a*h+a+h+a+"), "haha"),
    (re.compile(r"a+hh+"), "ahh"),
    (re.compile(r"(lo+l+\s?)+"), "lol"),
    (re.compile(r"a+b+c\w*"), "abc"),
    (re.compile(r"a+r+g+h+"), "argh"),
    (re.compile(r"a+w+e+s+o+m+e+"), "awesome"),
    (re.compile(r"\ba*f+u+c*k*\b"), "fuck"),
    (re.compile(r"aa+ww+"), "aww"),
    (re.compile(r"abdu\w*"), "abdu"),
    (re.compile(r"y+e*a+y+"), "yeah"),
    (re.compile(r"y+e+a+h+"), "yeah"),
    (re.compile(r"y+e{2,}s{2,}"), "yeah"),
    (re.compile(r"(.)\1{1,}"), r"\1"),
]


def add_features(df):
    df = df.copy(deep=False)
    s = df["comment_text"]
    if not pd.api.types.is_string_dtype(s):
        s = s.astype(str)
    s = s.fillna("")

    df["ex_mark"] = (s.str.count("!") >= 1).astype("int64")
    df["qu_mark"] = (s.str.count(r"\?") >= 1).astype("int64")
    df["smileys_good"] = s.str.contains(_RE_SMILEY_GOOD, regex=True).astype("int64")
    df["smileys_bad"] = s.str.contains(_RE_SMILEY_BAD, regex=True).astype("int64")
    df["word_count"] = s.str.count(r"(?u)\b\w\w+\b").astype("int64")
    df["sent_count"] = s.str.count(r"\.\b").astype("int64")
    df["link_count"] = s.str.count(r"\.www").astype("int64")
    df["quote_count"] = s.str.count(r"(\'|\")").astype("int64")
    df["comma_count"] = s.str.count(",").astype("int64")

    txt = s
    for rgx, rep in _REPL:
        txt = txt.str.replace(rgx, rep, regex=True)
    df["comment_text"] = txt
    return df


df_train = add_features(df_train)
df_predict = add_features(df_predict)

Y = df_train[target_cols]
df_train.head()



## === cell 3
vect = TfidfVectorizer(
    min_df=4, ngram_range=(1, 2), stop_words="english", lowercase=True
)
vect.fit(df_train["comment_text"])



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
chis = pd.DataFrame({"feature": feature_names, "p": scores}).sort_values(
    by="p", ascending=True
)
print("Perc. of rel. features (toxic p<0.3): " + str((chis["p"] < 0.3).mean()))




## === cell 6
def plot_model():
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.feature_selection import SelectPercentile

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
from sklearn.metrics import roc_auc_score
from sklearn.utils.extmath import safe_sparse_dot
from scipy.special import expit

LR_KW = dict(max_iter=1000, C=1, random_state=0, n_jobs=-1, solver="saga")
base_model = LogisticRegression(**LR_KW)

Y_predicted = pd.DataFrame({"id": df_predict["id"]})
scores = []

train_extra_sp = csr_matrix(X_train_extra.values)
predict_extra_sp = csr_matrix(X_predict_extra.values)

X_train_full = hstack([X_train_vectorized, train_extra_sp], format="csr")
X_predict_full = hstack([X_predict_vectorized, predict_extra_sp], format="csr")

n_text_features = X_train_vectorized.shape[1]
n_extra_features = train_extra_sp.shape[1]
extra_idx = np.arange(
    n_text_features, n_text_features + n_extra_features, dtype=np.int64
)

cv = StratifiedKFold(n_splits=5, shuffle=False)
split_idx_by_label = {}
Y_np = {}
for y_col in target_cols:
    y_arr = Y[y_col].to_numpy(copy=False)
    Y_np[y_col] = y_arr
    split_idx_by_label[y_col] = tuple(cv.split(y_arr, y_arr))

k = int(np.ceil(0.60 * n_text_features))
if k < 1:
    k = 1

col_sum = np.asarray(X_train_vectorized.sum(axis=0)).ravel()
n_samples = X_train_vectorized.shape[0]
eps = 1e-12

topk_idx_by_label = {}

Y_mat = Y[target_cols].to_numpy(dtype=np.float64, copy=False)  # 0/1
pos_feat_sum = safe_sparse_dot(
    X_train_vectorized.T, Y_mat, dense_output=True
)  # (n_feat, 6)
pos_counts = Y_mat.sum(axis=0)  # (6,)
total_feat_sum = col_sum  # (n_feat,)

for j, y_col in enumerate(target_cols):
    pos = float(pos_counts[j])
    neg = float(n_samples - pos)

    a = pos_feat_sum[:, j]
    b = total_feat_sum - a

    exp_a = total_feat_sum * (pos / n_samples)
    exp_b = total_feat_sum * (neg / n_samples)

    chi2_stats = (a - exp_a) ** 2 / (exp_a + eps) + (b - exp_b) ** 2 / (exp_b + eps)

    topk_idx = np.argpartition(chi2_stats, -k)[-k:]
    topk_idx_by_label[y_col] = topk_idx  # order irrelevant

train_features_by_label = {}
predict_features_by_label = {}
for y_col in target_cols:
    topk_idx = topk_idx_by_label[y_col]
    cols = np.concatenate([topk_idx, extra_idx])
    train_features_by_label[y_col] = X_train_full[:, cols].tocsr()
    predict_features_by_label[y_col] = X_predict_full[:, cols].tocsr()

for y_col in target_cols:
    train_features = train_features_by_label[y_col]
    predict_features = predict_features_by_label[y_col]
    y = Y_np[y_col]
    fold_scores = []

    fold_data = []
    for tr_idx, va_idx in split_idx_by_label[y_col]:
        fold_data.append(
            (train_features[tr_idx], train_features[va_idx], y[tr_idx], y[va_idx])
        )

    for X_tr, X_va, y_tr, y_va in fold_data:
        m = clone(base_model)
        m.fit(X_tr, y_tr)

        decision = safe_sparse_dot(X_va, m.coef_.T, dense_output=True).ravel() + float(
            m.intercept_[0]
        )
        p = expit(decision)
        fold_scores.append(roc_auc_score(y_va, p))

    scoreX = float(np.mean(fold_scores))
    scores.append(scoreX)

    model = clone(base_model)
    model.fit(train_features, y)

    decision_test = safe_sparse_dot(
        predict_features, model.coef_.T, dense_output=True
    ).ravel() + float(model.intercept_[0])
    Y_predicted[y_col] = expit(decision_test)
    print(y_col + ":" + str(scoreX))

print("mean score: " + str(float(np.mean(scores))))



## === cell 8
final_sub = pd.DataFrame({"id": sample_sub["id"]})

pred_aligned = sample_sub[["id"]].merge(Y_predicted, on="id", how="left")[target_cols]

for c in target_cols:
    final_sub[c] = pred_aligned[c].astype(float).to_numpy()

final_sub.to_csv("submission.csv", index=False)
print(final_sub.head())
print("Wrote submission.csv with shape:", final_sub.shape)
