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
Given pairs of phrases (an `anchor` and a `target` phrase), build a model to rate how similar they are.  

## Metric
Pearson correlation coefficient.

## Submission Format
For each `id` (representing a pair of phrases) in the test set, you must predict the similarity `score`. The file should contain a header and have the following format:

```
id,score
4112d61851461f60,0
09e418c93a776564,0.25
36baf228038e314b,1
etc.

```

## Dataset
The scores are in the 0-1 range with increments of 0.25 with the following meanings:

- **1.0** - Very close match. This is typically an exact match except possibly for differences in conjugation, quantity (e.g. singular vs. plural), and addition or removal of stopwords (e.g. "the", "and", "or").
- **0.75** - Close synonym, e.g. "mobile phone" vs. "cellphone". This also includes abbreviations, e.g. "TCP" -> "transmission control protocol".
- **0.5** - Synonyms which don't have the same meaning (same function, same properties). This includes broad-narrow (hyponym) and narrow-broad (hypernym) matches.
- **0.25** - Somewhat related, e.g. the two phrases are in the same high level domain but are not synonyms. This also includes antonyms.
- **0.0** - Unrelated.

Files
-----

- **train.csv** - the training set, containing phrases, contexts, and their similarity scores
- **test.csv** - the test set set, identical in structure to the training set but without the score
- **sample_submission.csv** - a sample submission file in the correct format

Columns
-------

- `id` - a unique identifier for a pair of phrases
- `anchor` - the first phrase
- `target` - the second phrase
- `context` - the [CPC classification (version 2021.05)](https://en.wikipedia.org/wiki/Cooperative_Patent_Classification), which indicates the subject within which the similarity is to be scored
- `score` - the similarity. This is sourced from a combination of one or more manual expert ratings.

# 2. Python version

3.10

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (118 lines)
            sample_submission.csv (3649 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (3649 lines)
            test.csv.zip (86.4 kB)
            train.csv (32826 lines)
            train.csv.zip (790.5 kB)
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
        input/
            description.md (118 lines)
            sample_submission.csv (3649 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (3649 lines)
            test.csv.zip (86.4 kB)
            train.csv (32826 lines)
            train.csv.zip (790.5 kB)
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
        working/
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
```

-> data/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> data/test.csv has 3648 rows and 4 columns.
The columns are: id, anchor, target, context

-> data/train.csv has 32825 rows and 5 columns.
The columns are: id, anchor, target, context, score

-> data/us-patent-phrase-to-phrase-matching/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> data/us-patent-phrase-to-phrase-matching/test.csv has 3648 rows and 4 columns.
The columns are: id, anchor, target, context

-> data/us-patent-phrase-to-phrase-matching/train.csv has 32825 rows and 5 columns.
The columns are: id, anchor, target, context, score

-> input/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> (stopped after 10 files for performance)

# 5. Target score

0.8084755399348172

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr



## === cell 1
base_path = Path("data") / "us-patent-phrase-to-phrase-matching"
train_path = base_path / "train.csv"
test_path = base_path / "test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3152433875.py in <cell line: 0>()
      4 test_path = base_path / "test.csv"
      5 
----> 6 train_df = pd.read_csv(train_path)
      7 test_df = pd.read_csv(test_path)
      8 

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

FileNotFoundError: [Errno 2] No such file or directory: 'data/us-patent-phrase-to-phrase-matching/train.csv'

## === cell 2
def make_input(df):
    return (
        df["context"].fillna("")
        + " "
        + df["anchor"].fillna("")
        + " "
        + df["target"].fillna("")
    )


train_df["inputs"] = make_input(train_df)
test_df["inputs"] = make_input(test_df)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1016281640.py in <cell line: 0>()
     10 
     11 
---> 12 train_df["inputs"] = make_input(train_df)
     13 test_df["inputs"] = make_input(test_df)
     14 

NameError: name 'train_df' is not defined

## === cell 3
np.random.seed(42)
unique_anchors = train_df["anchor"].unique()
np.random.shuffle(unique_anchors)
val_ratio = 0.25
val_anchors = set(unique_anchors[: int(len(unique_anchors) * val_ratio)])

is_val = train_df["anchor"].isin(val_anchors)
train_split = train_df[~is_val]
val_split = train_df[is_val]

X_train, y_train = train_split["inputs"].values, train_split["score"].values
X_val, y_val = val_split["inputs"].values, val_split["score"].values



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2838183339.py in <cell line: 0>()
      1 # Create a reproducible train/validation split that mimics the original anchor‑based split
      2 np.random.seed(42)
----> 3 unique_anchors = train_df["anchor"].unique()
      4 np.random.shuffle(unique_anchors)
      5 val_ratio = 0.25

NameError: name 'train_df' is not defined

## === cell 4
vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), analyzer="char")
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

model = Ridge(alpha=1.0)
model.fit(X_train_vec, y_train)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2622083204.py in <cell line: 0>()
      1 # TF‑IDF vectorization + Ridge regression (single‑label regression)
      2 vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), analyzer="char")
----> 3 X_train_vec = vectorizer.fit_transform(X_train)
      4 X_val_vec = vectorizer.transform(X_val)
      5 

NameError: name 'X_train' is not defined

## === cell 5
val_preds = model.predict(X_val_vec)
pearson, _ = pearsonr(y_val, val_preds)
print(f"Validation Pearson correlation: {pearson:.6f}")



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/379275060.py in <cell line: 0>()
      1 # Evaluate Pearson correlation on validation set
----> 2 val_preds = model.predict(X_val_vec)
      3 pearson, _ = pearsonr(y_val, val_preds)
      4 print(f"Validation Pearson correlation: {pearson:.6f}")
      5 

NameError: name 'model' is not defined

## === cell 6
X_test_vec = vectorizer.transform(test_df["inputs"].values)
test_preds = model.predict(X_test_vec)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1165211545.py in <cell line: 0>()
      1 # Predict on the official test set
----> 2 X_test_vec = vectorizer.transform(test_df["inputs"].values)
      3 test_preds = model.predict(X_test_vec)
      4 

NameError: name 'test_df' is not defined

## === cell 7
submission = pd.DataFrame({"id": test_df["id"], "score": test_preds})
submission_path = Path("submission.csv")
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path.resolve()}")

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3572908702.py in <cell line: 0>()
      1 # Prepare and write submission file
----> 2 submission = pd.DataFrame({"id": test_df["id"], "score": test_preds})
      3 submission_path = Path("submission.csv")
      4 submission.to_csv(submission_path, index=False)
      5 print(f"Submission saved to {submission_path.resolve()}")

NameError: name 'test_df' is not defined
