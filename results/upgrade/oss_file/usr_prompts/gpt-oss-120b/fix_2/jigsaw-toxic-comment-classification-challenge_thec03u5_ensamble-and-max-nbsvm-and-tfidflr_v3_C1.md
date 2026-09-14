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

3.6

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

0.9813460050965214

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np, pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score



## === cell 1
nbsvm_path = Path("../input/nbsvm/submissionNBSVM.csv")
tfidflr_path = Path("../input/tfidf-and-lr/word_submission.csv")

try:
    p_nbsvm = pd.read_csv(nbsvm_path)
    p_tfidflr = pd.read_csv(tfidflr_path)
    external_available = True
except FileNotFoundError:
    p_nbsvm = None
    p_tfidflr = None
    external_available = False

label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]



## === cell 2
if external_available:
    p_res = p_nbsvm.copy()
    p_res[label_cols] = (p_tfidflr[label_cols] + p_nbsvm[label_cols]) / 2

    p_res2 = p_nbsvm.copy()
    for col in label_cols:
        temp_df = pd.concat([p_tfidflr[col], p_nbsvm[col]], axis=1)
        p_res2[col] = temp_df.max(axis=1)

    p_res3 = p_nbsvm.copy()
    for col in label_cols:
        temp_df = pd.concat([p_tfidflr[col], p_nbsvm[col]], axis=1)
        p_res3[col] = temp_df.apply(
            lambda r: np.max(r) if np.mean(r) > 0.5 else np.mean(r), axis=1
        )

    p_res.to_csv("submissionAvg.csv", index=False)
    p_res2.to_csv("submissionMax.csv", index=False)
    p_res3.to_csv("submissionCondMax.csv", index=False)
else:
    train_path = Path("data/jigsaw-toxic-comment-classification-challenge/train.csv")
    test_path = Path("data/jigsaw-toxic-comment-classification-challenge/test.csv")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    tfidf = TfidfVectorizer(
        max_features=200000, ngram_range=(1, 2), stop_words="english", dtype=np.float32
    )
    X_all = tfidf.fit_transform(train_df["comment_text"])
    X_test = tfidf.transform(test_df["comment_text"])

    pred_df = pd.DataFrame({"id": test_df["id"]})

    for col in label_cols:
        y = train_df[col].values
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_all, y, test_size=0.1, random_state=42, stratify=y
        )
        clf = LogisticRegression(
            solver="saga",
            max_iter=1000,
            n_jobs=-1,
            class_weight="balanced",
            C=4.0,
            penalty="l2",
            verbose=0,
        )
        clf.fit(X_tr, y_tr)
        val_pred = clf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, val_pred)
        print(f"Validation AUC for {col}: {auc:.5f}")
        clf.fit(X_all, y)
        test_pred = clf.predict_proba(X_test)[:, 1]
        pred_df[col] = test_pred

    pred_df.to_csv("submission.csv", index=False)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/101165594.py in <cell line: 0>()
     24     train_path = Path("data/jigsaw-toxic-comment-classification-challenge/train.csv")
     25     test_path = Path("data/jigsaw-toxic-comment-classification-challenge/test.csv")
---> 26     train_df = pd.read_csv(train_path)
     27     test_df = pd.read_csv(test_path)
     28 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: 'data/jigsaw-toxic-comment-classification-challenge/train.csv'

## === cell 3
print("Submission files generated:")
for f in Path(".").glob("submission*.csv"):
    print("-", f.name)
