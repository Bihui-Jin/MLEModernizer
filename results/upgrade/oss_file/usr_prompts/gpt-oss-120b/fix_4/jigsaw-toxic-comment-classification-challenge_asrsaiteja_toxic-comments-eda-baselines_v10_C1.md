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

3.7

# 3. Installed packages

geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
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

0.90828

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
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score

data_dir = os.path.join("data", "jigsaw-toxic-comment-classification-challenge")

data_paths = {
    "train.csv": os.path.join(data_dir, "train.csv"),
    "test.csv": os.path.join(data_dir, "test.csv"),
    "sample_submission.csv": os.path.join(data_dir, "sample_submission.csv"),
}

train_df = pd.read_csv(data_paths["train.csv"])
test_df = pd.read_csv(data_paths["test.csv"])
sub_df = pd.read_csv(data_paths["sample_submission.csv"])

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("Columns in Train:", train_df.columns.tolist())



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/33205733.py in <cell line: 0>()
     22 
     23 # Load datasets
---> 24 train_df = pd.read_csv(data_paths["train.csv"])
     25 test_df = pd.read_csv(data_paths["test.csv"])
     26 sub_df = pd.read_csv(data_paths["sample_submission.csv"])

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

## === cell 1
drop_col = ["id"]
text_col = ["comment_text"]
label_col = [c for c in train_df.columns if c not in text_col + drop_col]

print("Label columns:", label_col)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/798828967.py in <cell line: 0>()
      2 drop_col = ["id"]
      3 text_col = ["comment_text"]
----> 4 label_col = [c for c in train_df.columns if c not in text_col + drop_col]
      5 
      6 print("Label columns:", label_col)

NameError: name 'train_df' is not defined

## === cell 2
labels_per_comment = train_df[label_col].sum(axis=1)
train_df["is_clean"] = (labels_per_comment == 0).astype(int)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/959867292.py in <cell line: 0>()
      1 # Simple exploratory counts (optional, now safe)
----> 2 labels_per_comment = train_df[label_col].sum(axis=1)
      3 train_df["is_clean"] = (labels_per_comment == 0).astype(int)
      4 

NameError: name 'train_df' is not defined

## === cell 3
X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    train_df["comment_text"],
    train_df[label_col],
    test_size=0.2,
    random_state=2019,
    stratify=train_df[label_col].idxmax(axis=1),  # simple stratification
)

X_test_raw = test_df["comment_text"]

tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    strip_accents="unicode",
    analyzer="word",
    use_idf=True,
    smooth_idf=True,
    sublinear_tf=True,
    stop_words="english",
)
X_train = tfidf.fit_transform(X_train_raw)
X_val = tfidf.transform(X_val_raw)
X_test = tfidf.transform(X_test_raw)

print(
    "Data dimensions -> Train:",
    X_train.shape,
    "Val:",
    X_val.shape,
    "Test:",
    X_test.shape,
)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3449309100.py in <cell line: 0>()
      1 # Train/validation split
      2 X_train_raw, X_val_raw, y_train, y_val = train_test_split(
----> 3     train_df["comment_text"],
      4     train_df[label_col],
      5     test_size=0.2,

NameError: name 'train_df' is not defined

## === cell 4
preds_train = np.zeros(y_train.shape)
preds_valid = np.zeros(y_val.shape)
preds_test = np.zeros((len(test_df), len(label_col)))

train_rocs = []
valid_rocs = []

for i, label in enumerate(label_col):
    pos_prior = y_train[label].mean()
    class_prior = np.array([1.0 - pos_prior, pos_prior])

    model = MultinomialNB(alpha=0.1, fit_prior=True, class_prior=class_prior)
    model.fit(X_train, y_train[label])

    preds_train[:, i] = model.predict_proba(X_train)[:, 1]
    train_roc = roc_auc_score(y_train[label], preds_train[:, i])
    train_rocs.append(train_roc)

    preds_valid[:, i] = model.predict_proba(X_val)[:, 1]
    valid_roc = roc_auc_score(y_val[label], preds_valid[:, i])
    valid_rocs.append(valid_roc)

    preds_test[:, i] = model.predict_proba(X_test)[:, 1]

print("\nMean column-wise ROC AUC on Train:", np.mean(train_rocs))
print("Mean column-wise ROC AUC on Validation:", np.mean(valid_rocs))



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/978881149.py in <cell line: 0>()
      1 # Train a separate MultinomialNB for each label
----> 2 preds_train = np.zeros(y_train.shape)
      3 preds_valid = np.zeros(y_val.shape)
      4 preds_test = np.zeros((len(test_df), len(label_col)))
      5 

NameError: name 'y_train' is not defined

## === cell 5
sub_df.iloc[:, 1:] = preds_test
sub_df.head()



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4072898075.py in <cell line: 0>()
      1 # Build submission file
----> 2 sub_df.iloc[:, 1:] = preds_test
      3 sub_df.head()
      4 

NameError: name 'preds_test' is not defined

## === cell 6
sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)
print(f"Submission saved to {sub_path}")

from IPython.display import FileLink

FileLink(sub_path)

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4279921618.py in <cell line: 0>()
      1 # Save submission
      2 sub_path = "submission.csv"
----> 3 sub_df.to_csv(sub_path, index=False)
      4 print(f"Submission saved to {sub_path}")
      5 

NameError: name 'sub_df' is not defined
