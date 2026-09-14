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
pass



## === cell 2
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

train_text = train_df["comment_text"].fillna("").astype(str).tolist()
test_text = test_df["comment_text"].fillna("").astype(str).tolist()

y = train_df[label_cols].astype(np.int32).values

print(
    "Train shape:",
    train_df.shape,
    "Test shape:",
    test_df.shape,
    "Sample shape:",
    sample_sub.shape,
)



## === cell 3
from functools import lru_cache
from joblib import Parallel, delayed
import sklearn

_word_vec = TfidfVectorizer(strip_accents="unicode", analyzer="word")
_word_pre = _word_vec.build_preprocessor()
_word_tok = _word_vec.build_tokenizer()


@lru_cache(maxsize=300_000)
def _cached_word_tokens(doc: str):
    return _word_tok(_word_pre(doc))


_char_vec = TfidfVectorizer(strip_accents="unicode", analyzer="char")
_char_pre = _char_vec.build_preprocessor()


@lru_cache(maxsize=300_000)
def _cached_char_pre(doc: str):
    return _char_pre(doc)


def train_and_predict(tfidf_params, lr_params, model_name="model"):
    vec = TfidfVectorizer(**tfidf_params)
    X_tr = vec.fit_transform(train_text)
    X_te = vec.transform(test_text)

    base_lr = LogisticRegression(**lr_params)
    clf = OneVsRestClassifier(base_lr, n_jobs=-1)
    clf.fit(X_tr, y)
    proba = clf.predict_proba(X_te)  # shape: (n_test, 6)
    return proba


tfidf_1 = dict(
    strip_accents="unicode",
    analyzer="word",
    tokenizer=_cached_word_tokens,
    preprocessor=None,
    token_pattern=None,  # required when custom tokenizer is set; avoids extra regex work
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.9,
    sublinear_tf=True,
    max_features=200000,
)
tfidf_2 = dict(
    strip_accents="unicode",
    analyzer="word",
    tokenizer=_cached_word_tokens,
    preprocessor=None,
    token_pattern=None,
    ngram_range=(1, 1),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    max_features=150000,
)
tfidf_3 = dict(
    strip_accents="unicode",
    analyzer="char",
    preprocessor=_cached_char_pre,
    ngram_range=(3, 5),
    min_df=2,
    max_df=0.9,
    sublinear_tf=True,
    max_features=200000,
)
tfidf_4 = dict(
    strip_accents="unicode",
    analyzer="char",
    preprocessor=_cached_char_pre,
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

cpu = os.cpu_count() or 4
n_outer_jobs = min(4, max(1, cpu // 2))

print(
    f"scikit-learn={sklearn.__version__} | outer parallel jobs={n_outer_jobs} | cpu={cpu}"
)
results = Parallel(n_jobs=n_outer_jobs, backend="loky", verbose=0)(
    delayed(train_and_predict)(tfidf, lr, name) for (name, tfidf, lr) in tasks
)
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




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
_RemoteTraceback                          Traceback (most recent call last)
_RemoteTraceback: 
"""
Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/joblib/externals/loky/process_executor.py", line 453, in _process_worker
    call_item = call_queue.get(block=True, timeout=timeout)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.11/multiprocessing/queues.py", line 122, in get
    return _ForkingPickler.loads(res)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: Can't get attribute '_cached_word_tokens' on <module 'joblib.externals.loky.backend.popen_loky_posix' from '/usr/local/lib/python3.11/dist-packages/joblib/externals/loky/backend/popen_loky_posix.py'>
"""

The above exception was the direct cause of the following exception:

BrokenProcessPool                         Traceback (most recent call last)
/tmp/ipykernel_11/214925724.py in <cell line: 0>()
    134     f"scikit-learn={sklearn.__version__} | outer parallel jobs={n_outer_jobs} | cpu={cpu}"
    135 )
--> 136 results = Parallel(n_jobs=n_outer_jobs, backend="loky", verbose=0)(
    137     delayed(train_and_predict)(tfidf, lr, name) for (name, tfidf, lr) in tasks
    138 )

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in __call__(self, iterable)
   2070         next(output)
   2071 
-> 2072         return output if self.return_generator else list(output)
   2073 
   2074     def __repr__(self):

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _get_outputs(self, iterator, pre_dispatch)
   1680 
   1681             with self._backend.retrieval_context():
-> 1682                 yield from self._retrieve()
   1683 
   1684         except GeneratorExit:

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _retrieve(self)
   1782             # worker traceback.
   1783             if self._aborting:
-> 1784                 self._raise_error_fast()
   1785                 break
   1786 

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _raise_error_fast(self)
   1857         # called directly or if the generator is gc'ed.
   1858         if error_job is not None:
-> 1859             error_job.get_result(self.timeout)
   1860 
   1861     def _warn_exit_early(self):

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in get_result(self, timeout)
    756             # callback thread, and is stored internally. It's just waiting to
    757             # be returned.
--> 758             return self._return_or_raise()
    759 
    760         # For other backends, the main thread needs to run the retrieval step.

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _return_or_raise(self)
    771         try:
    772             if self.status == TASK_ERROR:
--> 773                 raise self._result
    774             return self._result
    775         finally:

BrokenProcessPool: A task has failed to un-serialize. Please ensure that the arguments of the function are all picklable.

## === cell 4
def align_to_sample(pred_df, sample_df):
    pred_df = pred_df[["id"] + label_cols].copy()
    merged = sample_df[["id"]].merge(
        pred_df, on="id", how="left", validate="one_to_one"
    )
    for c in label_cols:
        merged[c] = merged[c].astype(float).fillna(0.5)
    return merged


p_lstm_glove_pl = align_to_sample(p_lstm_glove_pl, sample_sub)
p_nbsvm = align_to_sample(p_nbsvm, sample_sub)
p_lstm_fast_pl = align_to_sample(p_lstm_fast_pl, sample_sub)
p_bi_lstm_dual = align_to_sample(p_bi_lstm_dual, sample_sub)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2094107271.py in <cell line: 0>()
      9 
     10 
---> 11 p_lstm_glove_pl = align_to_sample(p_lstm_glove_pl, sample_sub)
     12 p_nbsvm = align_to_sample(p_nbsvm, sample_sub)
     13 p_lstm_fast_pl = align_to_sample(p_lstm_fast_pl, sample_sub)

NameError: name 'p_lstm_glove_pl' is not defined

## === cell 5
p_res = p_lstm_glove_pl.copy()
p_res[label_cols] = (
    p_nbsvm[label_cols].values
    + p_lstm_glove_pl[label_cols].values
    + p_lstm_fast_pl[label_cols].values
    + p_bi_lstm_dual[label_cols].values
) / 4.0

p_res[label_cols] = p_res[label_cols].clip(0.0, 1.0)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/740829947.py in <cell line: 0>()
----> 1 p_res = p_lstm_glove_pl.copy()
      2 p_res[label_cols] = (
      3     p_nbsvm[label_cols].values
      4     + p_lstm_glove_pl[label_cols].values
      5     + p_lstm_fast_pl[label_cols].values

NameError: name 'p_lstm_glove_pl' is not defined

## === cell 6
p_res = p_res[["id"] + label_cols]
if len(p_res) != len(sample_sub):
    raise ValueError(
        f"Submission row count mismatch: got {len(p_res)} expected {len(sample_sub)}"
    )
print(p_res.head())
print("Submission columns:", list(p_res.columns))



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3967149803.py in <cell line: 0>()
----> 1 p_res = p_res[["id"] + label_cols]
      2 if len(p_res) != len(sample_sub):
      3     raise ValueError(
      4         f"Submission row count mismatch: got {len(p_res)} expected {len(sample_sub)}"
      5     )

NameError: name 'p_res' is not defined

## === cell 7
p_res.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", p_res.shape)
print(
    "File exists:",
    os.path.exists("submission.csv"),
    "Size:",
    os.path.getsize("submission.csv"),
)

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/733620044.py in <cell line: 0>()
----> 1 p_res.to_csv("submission.csv", index=False)
      2 print("Wrote submission.csv with shape:", p_res.shape)
      3 print(
      4     "File exists:",
      5     os.path.exists("submission.csv"),

NameError: name 'p_res' is not defined
