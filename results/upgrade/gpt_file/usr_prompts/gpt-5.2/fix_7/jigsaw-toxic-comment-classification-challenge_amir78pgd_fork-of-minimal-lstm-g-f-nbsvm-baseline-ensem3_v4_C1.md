# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.983346815228054

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

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

train_text = train_df["comment_text"].fillna("").astype(str).to_numpy()
test_text = test_df["comment_text"].fillna("").astype(str).to_numpy()
y = train_df[label_cols].astype(np.int32).to_numpy()

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

train_text_lower = np.char.lower(train_text)
test_text_lower = np.char.lower(test_text)

all_word = np.concatenate([train_text, test_text], axis=0)
all_char = np.concatenate([train_text_lower, test_text_lower], axis=0)
n_train = len(train_text)
n_test = len(test_text)


def train_and_predict(tfidf_params, lr_params, model_name="model"):
    analyzer = tfidf_params.get("analyzer")
    corpus = all_char if analyzer == "char" else all_word

    vec = TfidfVectorizer(**tfidf_params)
    X_all = vec.fit_transform(corpus)
    X_tr = X_all[:n_train]
    X_te = X_all[n_train : n_train + n_test]

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

preds = [pred1, pred2, pred3, pred4]
test_ids = test_df["id"].to_numpy()

print("Prepared 4 prediction matrices:", [p.shape for p in preds])




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2262352453.py in <cell line: 0>()
      4 
      5 # Speed: precompute lowercase variants once (same as before).
----> 6 train_text_lower = np.char.lower(train_text)
      7 test_text_lower = np.char.lower(test_text)
      8 

/usr/local/lib/python3.11/dist-packages/numpy/core/defchararray.py in lower(a)
   1133     """
   1134     a_arr = numpy.asarray(a)
-> 1135     return _vec_string(a_arr, a_arr.dtype, 'lower')
   1136 
   1137 

TypeError: string operation on non-string array

## === cell 3
def align_to_sample(pred_ids, pred_mat, sample_df):
    sample_ids = sample_df["id"].to_numpy()

    sample_pos = pd.Index(sample_ids).get_indexer(pred_ids)  # -1 if not found

    out_mat = np.full((len(sample_ids), len(label_cols)), 0.5, dtype=np.float64)

    valid = sample_pos >= 0
    if np.any(valid):
        out_mat[sample_pos[valid], :] = pred_mat[valid, :].astype(
            np.float64, copy=False
        )

    out = sample_df[["id"]].copy()
    out[label_cols] = out_mat
    return out


sample_ids = sample_sub["id"].to_numpy()
ids_match = (len(sample_ids) == len(test_ids)) and np.array_equal(sample_ids, test_ids)
if ids_match:
    p_lstm_glove_pl = pd.DataFrame(pred1, columns=label_cols)
    p_lstm_glove_pl.insert(0, "id", test_ids)

    p_nbsvm = pd.DataFrame(pred2, columns=label_cols)
    p_nbsvm.insert(0, "id", test_ids)

    p_lstm_fast_pl = pd.DataFrame(pred3, columns=label_cols)
    p_lstm_fast_pl.insert(0, "id", test_ids)

    p_bi_lstm_dual = pd.DataFrame(pred4, columns=label_cols)
    p_bi_lstm_dual.insert(0, "id", test_ids)
else:
    p_lstm_glove_pl = align_to_sample(test_ids, pred1, sample_sub)
    p_nbsvm = align_to_sample(test_ids, pred2, sample_sub)
    p_lstm_fast_pl = align_to_sample(test_ids, pred3, sample_sub)
    p_bi_lstm_dual = align_to_sample(test_ids, pred4, sample_sub)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/297852506.py in <cell line: 0>()
     21 # Correctness preserved by verifying exact id equality; otherwise fall back to original alignment.
     22 sample_ids = sample_sub["id"].to_numpy()
---> 23 ids_match = (len(sample_ids) == len(test_ids)) and np.array_equal(sample_ids, test_ids)
     24 if ids_match:
     25     # Build aligned DataFrames directly (same semantics as align_to_sample when ids match).

NameError: name 'test_ids' is not defined

## === cell 4
p_res = sample_sub[["id"]].copy()
avg = (
    p_nbsvm[label_cols].to_numpy(dtype=np.float64, copy=False)
    + p_lstm_glove_pl[label_cols].to_numpy(dtype=np.float64, copy=False)
    + p_lstm_fast_pl[label_cols].to_numpy(dtype=np.float64, copy=False)
    + p_bi_lstm_dual[label_cols].to_numpy(dtype=np.float64, copy=False)
) / 4.0
avg = np.clip(avg, 0.0, 1.0)
p_res[label_cols] = avg



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/985534926.py in <cell line: 0>()
      2 p_res = sample_sub[["id"]].copy()
      3 avg = (
----> 4     p_nbsvm[label_cols].to_numpy(dtype=np.float64, copy=False)
      5     + p_lstm_glove_pl[label_cols].to_numpy(dtype=np.float64, copy=False)
      6     + p_lstm_fast_pl[label_cols].to_numpy(dtype=np.float64, copy=False)

NameError: name 'p_nbsvm' is not defined

## === cell 5
p_res = p_res[["id"] + label_cols]
if len(p_res) != len(sample_sub):
    raise ValueError(
        f"Submission row count mismatch: got {len(p_res)} expected {len(sample_sub)}"
    )
print(p_res.head())
print("Submission columns:", list(p_res.columns))



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3967149803.py in <cell line: 0>()
----> 1 p_res = p_res[["id"] + label_cols]
      2 if len(p_res) != len(sample_sub):
      3     raise ValueError(
      4         f"Submission row count mismatch: got {len(p_res)} expected {len(sample_sub)}"
      5     )

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'] not in index"

## === cell 6
p_res.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", p_res.shape)
print(
    "File exists:",
    os.path.exists("submission.csv"),
    "Size:",
    os.path.getsize("submission.csv"),
)

## --- ERROR in outputing the csv:
Invalid submission: Submission is missing the following columns: {'insult', 'toxic', 'threat', 'severe_toxic', 'identity_hate', 'obscene'}
