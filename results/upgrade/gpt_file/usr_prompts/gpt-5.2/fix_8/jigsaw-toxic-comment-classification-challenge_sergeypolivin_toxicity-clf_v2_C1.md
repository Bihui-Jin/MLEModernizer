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

3.12

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
seaborn==0.12.2
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

0.97502

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'Your code likely doesn’t produce a valid Kaggle submission because it unpacks the .zip files into `/kaggle/working/` but then reads from `/kaggle/working/train.csv` (while the extracted files typically land inside a nested competition folder). I make the smallest change needed to robustly locate/read the CSVs from either the input directory or any nested extracted path in working, ensuring the pipeline runs end-to-end and writes `submission.csv`. I also set `random_state` on the LogisticRegression to keep results deterministic (stability helps you iterate toward the target score without score jitter). Core modeling logic (TF-IDF word+char + OneVsRest LogisticRegression) is unchanged.'
- What this solution (achieved 0.5) has done: 'Main bottlenecks are (1) fitting two large TF‑IDF vectorizers on ~560k rows twice (word+char) and then stacking, and (2) training LogisticRegression twice (full fit + a separate “sanity-check” fit). To stay within 600s without changing the learning logic, the script below removes redundant unzip/disk work, reads only required columns with efficient dtypes, uses deterministic Intel-accelerated scikit-learn where available, and skips the second expensive training run (replacing it with a cheap AUC sanity-check computed from the already-fit full model). It also ensures the submission columns/order exactly match `sample_submission.csv` (fixing your “missing columns” error) and avoids slow in-notebook plotting/display overhead that doesn’t affect the final result.'

# 9. Code solution

## === cell 0
import os
import shutil
import gc

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

try:
    from sklearnex import patch_sklearn

    patch_sklearn()  # safe no-op if already patched
except Exception:
    pass

DATA_DIR = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/"
OUTPUT_DIR = "/kaggle/working/"
RANDOM_STATE = 42

os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 4))
os.environ.setdefault("MKL_NUM_THREADS", str(os.cpu_count() or 4))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(os.cpu_count() or 4))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(os.cpu_count() or 4))
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(os.cpu_count() or 4))

stopwords = "english"

cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]



## === cell 1
os.listdir(DATA_DIR)



## === cell 2
os.listdir(OUTPUT_DIR)




## === cell 3
def unpack_zipfile(filename):
    """Unpacks zip-file by name from DATA_DIR to OUTPUT_DIR."""
    out_csv = os.path.join(OUTPUT_DIR, filename.replace(".zip", ""))
    if os.path.exists(out_csv):
        return
    try:
        shutil.unpack_archive(
            filename=DATA_DIR + filename,
            extract_dir=OUTPUT_DIR,
            format="zip",
        )
    except Exception as e:
        print(e)
    else:
        print(f"Archive file '{filename}' has been unpacked successfully.")




## === cell 4
for csv_name, zip_name in [
    ("train.csv", "train.csv.zip"),
    ("test.csv", "test.csv.zip"),
    ("sample_submission.csv", "sample_submission.csv.zip"),
]:
    if not os.path.exists(os.path.join(DATA_DIR, csv_name)):
        unpack_zipfile(filename=zip_name)



## === cell 5
os.listdir(OUTPUT_DIR)




## === cell 6
def _find_file_recursive(root_dir, filename):
    for r, _, files in os.walk(root_dir):
        if filename in files:
            return os.path.join(r, filename)
    return None


def read_csv_fallback(name, usecols=None, dtype=None):
    in_path = os.path.join(DATA_DIR, name)
    out_path = os.path.join(OUTPUT_DIR, name)

    read_kwargs = dict(usecols=usecols, dtype=dtype)
    try:
        read_kwargs["engine"] = "pyarrow"
    except Exception:
        pass

    if os.path.exists(in_path):
        return pd.read_csv(in_path, **read_kwargs)
    if os.path.exists(out_path):
        return pd.read_csv(out_path, **read_kwargs)

    found = _find_file_recursive(DATA_DIR, name)
    if found is not None:
        return pd.read_csv(found, **read_kwargs)

    found = _find_file_recursive(OUTPUT_DIR, name)
    if found is not None:
        return pd.read_csv(found, **read_kwargs)

    raise FileNotFoundError(f"Could not locate {name} under {OUTPUT_DIR} or {DATA_DIR}")


train_df = read_csv_fallback(
    "train.csv",
    usecols=["id", "comment_text"] + cols,
    dtype={c: "int8" for c in cols},
)
test_df = read_csv_fallback(
    "test.csv",
    usecols=["id", "comment_text"],
)
sample_sub = read_csv_fallback(
    "sample_submission.csv",
)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ArrowInvalid                              Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/arrow_parser_wrapper.py in read(self)
    265         try:
--> 266             table = pyarrow_csv.read_csv(
    267                 self.src,

/usr/local/lib/python3.11/dist-packages/pyarrow/_csv.pyx in pyarrow._csv.read_csv()

/usr/local/lib/python3.11/dist-packages/pyarrow/_csv.pyx in pyarrow._csv.read_csv()

/usr/local/lib/python3.11/dist-packages/pyarrow/error.pxi in pyarrow.lib.pyarrow_internal_check_status()

/usr/local/lib/python3.11/dist-packages/pyarrow/error.pxi in pyarrow.lib.check_status()

ArrowInvalid: CSV parse error: Expected 8 columns, got 2: Archive-5, have 44 printed pages of A4 size.

The above exception was the direct cause of the following exception:

ParserError                               Traceback (most recent call last)
/tmp/ipykernel_11/2549068166.py in <cell line: 0>()
     34 
     35 
---> 36 train_df = read_csv_fallback(
     37     "train.csv",
     38     usecols=["id", "comment_text"] + cols,

/tmp/ipykernel_11/2549068166.py in read_csv_fallback(name, usecols, dtype)
     19 
     20     if os.path.exists(in_path):
---> 21         return pd.read_csv(in_path, **read_kwargs)
     22     if os.path.exists(out_path):
     23         return pd.read_csv(out_path, **read_kwargs)

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    624 
    625     with parser:
--> 626         return parser.read(nrows)
    627 
    628 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read(self, nrows)
   1909             try:
   1910                 # error: "ParserBase" has no attribute "read"
-> 1911                 df = self._engine.read()  # type: ignore[attr-defined]
   1912             except Exception:
   1913                 self.close()

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/arrow_parser_wrapper.py in read(self)
    271             )
    272         except pa.ArrowInvalid as e:
--> 273             raise ParserError(e) from e
    274 
    275         dtype_backend = self.kwds["dtype_backend"]

ParserError: CSV parse error: Expected 8 columns, got 2: Archive-5, have 44 printed pages of A4 size.

## === cell 7
test_labels = None
test_labels_info = None



## === cell 8
(train_df.shape, test_df.shape, sample_sub.shape)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1610769299.py in <cell line: 0>()
----> 1 (train_df.shape, test_df.shape, sample_sub.shape)
      2 

NameError: name 'train_df' is not defined

## === cell 9
label_counts = train_df[cols].sum(axis=0).astype(int)
label_counts



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/585966304.py in <cell line: 0>()
----> 1 label_counts = train_df[cols].sum(axis=0).astype(int)
      2 label_counts
      3 

NameError: name 'train_df' is not defined

## === cell 10
pass



## === cell 11
(test_df.shape,)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1235265255.py in <cell line: 0>()
----> 1 (test_df.shape,)
      2 

NameError: name 'test_df' is not defined

## === cell 12
corpus_train = train_df["comment_text"].fillna("").astype(str).to_numpy()
corpus_train[:2]



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/336414024.py in <cell line: 0>()
      1 # --- Speed: convert text columns once to contiguous numpy arrays.
      2 # Preserves exact text content (NaNs -> ""), identical to prior logic.
----> 3 corpus_train = train_df["comment_text"].fillna("").astype(str).to_numpy()
      4 corpus_train[:2]
      5 

NameError: name 'train_df' is not defined

## === cell 13
target_train = train_df[cols].to_numpy()
target_train[:2]



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1899448731.py in <cell line: 0>()
----> 1 target_train = train_df[cols].to_numpy()
      2 target_train[:2]
      3 

NameError: name 'train_df' is not defined

## === cell 14
corpus_test = test_df["comment_text"].fillna("").astype(str).to_numpy()
corpus_test[:2]



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3260613937.py in <cell line: 0>()
----> 1 corpus_test = test_df["comment_text"].fillna("").astype(str).to_numpy()
      2 corpus_test[:2]
      3 

NameError: name 'test_df' is not defined

## === cell 15
vectorizer = TfidfVectorizer(
    stop_words=stopwords,
    strip_accents="unicode",
    lowercase=True,
    ngram_range=(1, 2),
    analyzer="word",
    min_df=3,
    max_df=0.9,
    sublinear_tf=True,
)

vectorizer_char = TfidfVectorizer(
    strip_accents="unicode",
    lowercase=True,
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=3,
    max_df=0.9,
    sublinear_tf=True,
)



## === cell 16
from scipy.sparse import hstack

corpus_all = np.concatenate([corpus_train, corpus_test], axis=0)
n_train = corpus_train.shape[0]

features_all_word = vectorizer.fit_transform(corpus_all)
features_all_char = vectorizer_char.fit_transform(corpus_all)

features_train = hstack(
    [features_all_word[:n_train], features_all_char[:n_train]],
    format="csr",
)
features_test = hstack(
    [features_all_word[n_train:], features_all_char[n_train:]],
    format="csr",
)

del corpus_all, features_all_word, features_all_char
gc.collect()

(features_train.shape, features_test.shape)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2543797191.py in <cell line: 0>()
      5 # a second pass of tokenization for vocabulary building and improves cache locality.
      6 # Core feature logic (TF-IDF settings, analyzers, n-grams) remains unchanged.
----> 7 corpus_all = np.concatenate([corpus_train, corpus_test], axis=0)
      8 n_train = corpus_train.shape[0]
      9 

NameError: name 'corpus_train' is not defined

## === cell 17
gc.collect()



## === cell 18
base_estimator = LogisticRegression(
    class_weight="balanced",
    max_iter=2000,
    solver="saga",
    random_state=RANDOM_STATE,
    n_jobs=1,  # deterministic per-estimator; OVR parallelism handled by OneVsRestClassifier
)

classifier = OneVsRestClassifier(
    estimator=base_estimator,
    n_jobs=-1,
)



## === cell 19
classifier.fit(features_train, target_train)



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2234104921.py in <cell line: 0>()
----> 1 classifier.fit(features_train, target_train)
      2 

NameError: name 'features_train' is not defined

## === cell 20
test_ids = test_df["id"].astype(str).to_numpy()
test_ids[:2]



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2909459561.py in <cell line: 0>()
----> 1 test_ids = test_df["id"].astype(str).to_numpy()
      2 test_ids[:2]
      3 

NameError: name 'test_df' is not defined

## === cell 21
try:
    idx_tr, idx_va = train_test_split(
        np.arange(features_train.shape[0]),
        test_size=0.1,
        random_state=RANDOM_STATE,
    )
    X_va_vec = features_train[idx_va]
    y_va = target_train[idx_va]

    va_proba = classifier.predict_proba(X_va_vec)

    aucs = []
    for i, c in enumerate(cols):
        if len(np.unique(y_va[:, i])) < 2:
            continue
        aucs.append(roc_auc_score(y_va[:, i], va_proba[:, i]))
    if len(aucs) > 0:
        print(f"Validation mean ROC AUC (sanity-check, split): {np.mean(aucs):.5f}")
except Exception as e:
    print("Sanity-check skipped due to:", repr(e))



## === cell 22
try:
    va_pred = (va_proba >= 0.5).astype(np.int8)
    f1_micro = f1_score(va_pred, y_va, average="micro")
    f1_macro = f1_score(va_pred, y_va, average="macro")
    f1_weighted = f1_score(va_pred, y_va, average="weighted")
    print(f"F1-score (micro, validation split): {f1_micro:.4f}")
    print(f"F1-score (macro, validation split): {f1_macro:.4f}")
    print(f"F1-score (weighted, validation split): {f1_weighted:.4f}")
except Exception as e:
    print("F1-score skipped due to:", repr(e))



## === cell 23
proba_predictions_test = classifier.predict_proba(features_test)



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3891617953.py in <cell line: 0>()
----> 1 proba_predictions_test = classifier.predict_proba(features_test)
      2 

NameError: name 'features_test' is not defined

## === cell 24
sample_sub.head()



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1275749787.py in <cell line: 0>()
----> 1 sample_sub.head()
      2 

NameError: name 'sample_sub' is not defined

## === cell 25
submission = pd.DataFrame(proba_predictions_test, columns=cols)
submission.insert(0, "id", test_ids)
submission = submission.reindex(columns=sample_sub.columns)
submission.head()



## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/818133003.py in <cell line: 0>()
----> 1 submission = pd.DataFrame(proba_predictions_test, columns=cols)
      2 submission.insert(0, "id", test_ids)
      3 submission = submission.reindex(columns=sample_sub.columns)
      4 submission.head()
      5 

NameError: name 'proba_predictions_test' is not defined

## === cell 26
sample_sub.info()



## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/676720337.py in <cell line: 0>()
----> 1 sample_sub.info()
      2 

NameError: name 'sample_sub' is not defined

## === cell 27
submission.info()



## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2614940646.py in <cell line: 0>()
----> 1 submission.info()
      2 

NameError: name 'submission' is not defined

## === cell 28
out_path = os.path.join(OUTPUT_DIR, "submission.csv")
submission.to_csv(out_path, index=False)
print(f"The submission has been successfully saved to: {out_path}")



## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3342574884.py in <cell line: 0>()
      1 out_path = os.path.join(OUTPUT_DIR, "submission.csv")
----> 2 submission.to_csv(out_path, index=False)
      3 print(f"The submission has been successfully saved to: {out_path}")
      4 

NameError: name 'submission' is not defined

## === cell 29
with open(out_path, "r", encoding="utf-8") as f:
    for _ in range(5):
        print(f.readline().rstrip("\n"))

## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3203098282.py in <cell line: 0>()
----> 1 with open(out_path, "r", encoding="utf-8") as f:
      2     for _ in range(5):
      3         print(f.readline().rstrip("\n"))

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/submission.csv'
