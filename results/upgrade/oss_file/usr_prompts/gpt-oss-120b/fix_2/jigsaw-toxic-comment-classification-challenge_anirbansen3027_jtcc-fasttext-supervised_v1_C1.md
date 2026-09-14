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

3.9

# 3. Installed packages

fasttext==0.9.3
geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tqdm==4.67.1

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

0.79019

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import unicodedata
from fasttext import train_supervised
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from statistics import mean



## === cell 2
train_text = pd.read_csv("train.csv")
test_text = pd.read_csv("test.csv")
sample_submission = pd.read_csv("sample_submission.csv")
print(train_text.shape, test_text.shape, sample_submission.shape)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1732716958.py in <cell line: 0>()
----> 1 train_text = pd.read_csv("train.csv")
      2 test_text = pd.read_csv("test.csv")
      3 sample_submission = pd.read_csv("sample_submission.csv")
      4 print(train_text.shape, test_text.shape, sample_submission.shape)
      5 

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

FileNotFoundError: [Errno 2] No such file or directory: 'train.csv'

## === cell 3
y_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]




## === cell 4
def clean_it(text, normalize=True):
    s = (
        str(text)
        .replace(",", " ")
        .replace('"', "")
        .replace("'", " ' ")
        .replace(".", " . ")
        .replace("(", " ( ")
        .replace(")", " ) ")
        .replace("!", " ! ")
        .replace("?", " ? ")
        .replace(":", " ")
        .replace(";", " ")
        .lower()
    )
    s = s.replace("\n", " ")
    if normalize:
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("utf-8")
    return s


def clean_df(
    data, cleanit=False, shuffleit=False, encodeit=False, label_prefix="__class__"
):
    df = data[["comment_text"]].copy(deep=True)
    for col in y_cols:
        df[col] = label_prefix + data[col].astype(str) + " "
    if cleanit:
        df["comment_text"] = df["comment_text"].apply(lambda x: clean_it(x, encodeit))
    if shuffleit:
        df = df.sample(frac=1).reset_index(drop=True)
    return df




## === cell 5
train_split, val_split = train_test_split(train_text, shuffle=True, random_state=123)

df_train_cleaned = clean_df(train_split, cleanit=True, shuffleit=True)
df_val_cleaned = clean_df(val_split, cleanit=True, shuffleit=True, label_prefix="")



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1015963263.py in <cell line: 0>()
      1 # train/validation split
----> 2 train_split, val_split = train_test_split(train_text, shuffle=True, random_state=123)
      3 
      4 df_train_cleaned = clean_df(train_split, cleanit=True, shuffleit=True)
      5 df_val_cleaned = clean_df(val_split, cleanit=True, shuffleit=True, label_prefix="")

NameError: name 'train_text' is not defined

## === cell 6
model_dict = {}
all_val_preds = []

train_file = "/kaggle/working/final_train.csv"  # temporary file used for every label

for col in y_cols:
    df_train_cleaned[[col, "comment_text"]].to_csv(
        train_file, header=False, index=False, columns=[col, "comment_text"]
    )

    model = train_supervised(
        input=train_file,
        label="__class__",
        lr=1.0,
        epoch=2,
        loss="ova",
        wordNgrams=2,
        dim=200,
        thread=2,
        verbose=0,
    )
    model_dict[col] = model

    val_preds = []
    for text in tqdm(df_val_cleaned["comment_text"].values, desc=f"Validating {col}"):
        prob = model.predict(text, k=2)[1][1]
        val_preds.append(prob)
    all_val_preds.append(val_preds)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2776130538.py in <cell line: 0>()
      7 for col in y_cols:
      8     # Prepare training file for the current label
----> 9     df_train_cleaned[[col, "comment_text"]].to_csv(
     10         train_file, header=False, index=False, columns=[col, "comment_text"]
     11     )

NameError: name 'df_train_cleaned' is not defined

## === cell 7
val_pred_array = np.transpose(np.array(all_val_preds))
y_val_actuals = df_val_cleaned[y_cols].astype(int).to_numpy()
mean_auc = mean(
    [
        roc_auc_score(y_val_actuals[:, i], val_pred_array[:, i])
        for i in range(len(y_cols))
    ]
)
print("Mean validation AUC:", mean_auc)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4038121408.py in <cell line: 0>()
      1 # Validation AUC
      2 val_pred_array = np.transpose(np.array(all_val_preds))
----> 3 y_val_actuals = df_val_cleaned[y_cols].astype(int).to_numpy()
      4 mean_auc = mean(
      5     [

NameError: name 'df_val_cleaned' is not defined

## === cell 8
df_test = pd.merge(test_text, sample_submission, on="id")
df_test_cleaned = clean_df(df_test, cleanit=True, shuffleit=True, label_prefix="")



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1003207941.py in <cell line: 0>()
      1 # Prepare test data
----> 2 df_test = pd.merge(test_text, sample_submission, on="id")
      3 df_test_cleaned = clean_df(df_test, cleanit=True, shuffleit=True, label_prefix="")
      4 

NameError: name 'test_text' is not defined

## === cell 9
all_test_preds = []
for col in tqdm(y_cols, desc="Testing"):
    model = model_dict[col]
    test_preds = []
    for text in df_test_cleaned["comment_text"].values:
        prob = model.predict(text, k=2)[1][1]
        test_preds.append(prob)
    all_test_preds.append(test_preds)

test_pred_array = np.transpose(np.array(all_test_preds))
df_test[y_cols] = test_pred_array
df_test.drop(columns=["comment_text"], inplace=True)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1480252738.py in <cell line: 0>()
      2 all_test_preds = []
      3 for col in tqdm(y_cols, desc="Testing"):
----> 4     model = model_dict[col]
      5     test_preds = []
      6     for text in df_test_cleaned["comment_text"].values:

KeyError: 'toxic'

## === cell 10
df_test.to_csv("sample_submission.csv", index=False)
print("Submission file written:", "sample_submission.csv", " rows:", df_test.shape[0])

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2093116211.py in <cell line: 0>()
      1 # Write submission
----> 2 df_test.to_csv("sample_submission.csv", index=False)
      3 print("Submission file written:", "sample_submission.csv", " rows:", df_test.shape[0])

NameError: name 'df_test' is not defined
