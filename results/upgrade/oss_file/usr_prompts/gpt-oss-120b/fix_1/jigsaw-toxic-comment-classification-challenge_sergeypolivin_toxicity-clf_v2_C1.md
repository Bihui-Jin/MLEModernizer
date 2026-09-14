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

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import nltk
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords as nltk_stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

DATA_DIR = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/"
OUTPUT_DIR = "/kaggle/working/"
RANDOM_STATE = 42

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)

nltk.download('stopwords')
stopwords = list(nltk_stopwords.words('english'))


## === cell 1
os.listdir(DATA_DIR)


## === cell 2
os.listdir(OUTPUT_DIR)


## === cell 3
def unpack_zipfile(filename):
    """Unpacks zip-file by name from DATA_DIR to OUTPUT_DIR."""
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
unpack_zipfile(filename="train.csv.zip")
unpack_zipfile(filename="test.csv.zip")
unpack_zipfile(filename="sample_submission.csv.zip")
unpack_zipfile(filename="test_labels.csv.zip")


## === cell 5
os.listdir(OUTPUT_DIR)


## === cell 6
train_df = pd.read_csv(OUTPUT_DIR + "train.csv")
test_df = pd.read_csv(OUTPUT_DIR + "test.csv")
sample_sub = pd.read_csv(OUTPUT_DIR + "sample_submission.csv")


## === cell 7
test_labels = pd.read_csv(OUTPUT_DIR + "test_labels.csv")
test_labels_info = test_labels[test_labels.toxic != -1]
test_labels_info.tail()


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/4207045601.py in <cell line: 0>()
----> 1 test_labels = pd.read_csv(OUTPUT_DIR + "test_labels.csv")
      2 test_labels_info = test_labels[test_labels.toxic != -1]
      3 test_labels_info.tail()

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

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/test_labels.csv'

## === cell 8
train_df.sample(n=5, random_state=RANDOM_STATE)


## === cell 9
train_df.info()


## === cell 10
cols = train_df.columns[2:]
df = pd.Series()
for col in cols:
    temp = train_df[col].value_counts()[1]
    df[col] = temp


## === cell 11
df.sort_values(ascending=False).plot(kind='bar')
plt.title("Labels occurrences (train set)", fontsize=15)
plt.xlabel("Classes (comment toxicity degree)")
plt.ylabel("Number of examples")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


## === cell 12
test_df.sample(n=5, random_state=RANDOM_STATE)


## === cell 13
test_df.info()


## === cell 14
corpus_train = train_df["comment_text"].values.astype("U")
corpus_train[:5]


## === cell 15
target_train = train_df[cols].values
target_train[:5]


## === cell 16
corpus_test = test_df["comment_text"].values.astype("U")
corpus_test[:5]


## === cell 17
vectorizer = TfidfVectorizer(stop_words=stopwords)


## === cell 18
features_train = vectorizer.fit_transform(corpus_train)
features_train.shape


## === cell 19
features_test = vectorizer.transform(corpus_test)
features_test.shape


## === cell 20
base_estimator = LogisticRegression(
    class_weight="balanced",
    max_iter=10000,
)

classifier = OneVsRestClassifier(
    estimator=base_estimator,
)


## === cell 21
%%time
classifier.fit(features_train, target_train);


## === cell 22
test_ids = test_labels_info["id"].values.astype("U")
test_ids


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3554639875.py in <cell line: 0>()
----> 1 test_ids = test_labels_info["id"].values.astype("U")
      2 test_ids

NameError: name 'test_labels_info' is not defined

## === cell 23
for _ in range(20):
    test_id = np.random.choice(test_ids)
    true_labels = test_labels.loc[test_labels.id == test_id][cols].values
    test_comment = test_df.loc[test_df.id == test_id]["comment_text"].values

    test_comment_encoded = vectorizer.transform(test_comment)
    predictions = classifier.predict(test_comment_encoded)

    print(f"\nText: {test_comment}\n")
    print(f"Labels: {cols.values}")
    print(f"True labels: {true_labels}")
    print(f"Pred labels: {predictions}\n")

    print(f"Accuracy: {(true_labels == predictions).sum()}/{6}\n")
    print(80 * "*")


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3414447420.py in <cell line: 0>()
      1 for _ in range(20):
----> 2     test_id = np.random.choice(test_ids)
      3     true_labels = test_labels.loc[test_labels.id == test_id][cols].values
      4     test_comment = test_df.loc[test_df.id == test_id]["comment_text"].values
      5 

NameError: name 'test_ids' is not defined

## === cell 24
predictions_train = classifier.predict(features_train)
f1_micro = f1_score(predictions_train, target_train, average='micro')
f1_macro = f1_score(predictions_train, target_train, average='macro')
f1_weighted = f1_score(predictions_train, target_train, average='weighted')
print(f"F1-score (micro): {f1_micro:.4f}")
print(f"F1-score (macro): {f1_macro:.4f}")
print(f"F1-score (weighted): {f1_weighted:.4f}")


## === cell 25
proba_predictions_test = classifier.predict_proba(features_test)


## === cell 26
sample_sub.head()


## === cell 27
submission = pd.DataFrame(
    proba_predictions_test, 
    columns=cols, 
    index=test_df.id
).reset_index()

submission.head()


## === cell 28
sample_sub.info()


## === cell 29
submission.info()


## === cell 30
submission.to_csv('submission.csv', index=False)

print("The submission has been successfully saved.")


## === cell 31
!head submission.csv -n 5
