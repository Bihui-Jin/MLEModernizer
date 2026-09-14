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

0.97495

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
%pip install contractions
import numpy as np
import pandas as pd


## === cell 1
train = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip')
test = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip')
test_label = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/test_labels.csv.zip')
sample_submission = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip')


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/146318017.py in <cell line: 0>()
      1 train = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip')
      2 test = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip')
----> 3 test_label = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/test_labels.csv.zip')
      4 sample_submission = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip')

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
    792             # "Union[str, BaseBuffer]"; expected "Union[Union[str, PathLike[str]],
    793             # ReadBuffer[bytes], WriteBuffer[bytes]]"
--> 794             handle = _BytesZipFile(
    795                 handle, ioargs.mode, **compression_args  # type: ignore[arg-type]
    796             )

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in __init__(self, file, mode, archive_name, **kwargs)
   1035         # error: Incompatible types in assignment (expression has type "ZipFile",
   1036         # base class "_BufferedWriter" defined the type as "BytesIO")
-> 1037         self.buffer: zipfile.ZipFile = zipfile.ZipFile(  # type: ignore[assignment]
   1038             file, mode, **kwargs
   1039         )

/usr/lib/python3.11/zipfile.py in __init__(self, file, mode, compression, allowZip64, compresslevel, strict_timestamps, metadata_encoding)
   1293             while True:
   1294                 try:
-> 1295                     self.fp = io.open(file, filemode)
   1296                 except OSError:
   1297                     if filemode in modeDict:

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/jigsaw-toxic-comment-classification-challenge/test_labels.csv.zip'

## === cell 2
display(train.shape)
display(train.loc[40:50])


## === cell 3
import re


def remove_URL(text: str) -> str:
    """
        Remove URLs from a sample string
    """
    return re.sub(r'https?://\S+|www\.\S+', ' ', text)


def remove_html(text: str) -> str:
    """
        Remove the html in sample text
    """
    html = re.compile(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    return re.sub(html, ' ', text)


def remove_non_ascii(text: str) -> str:
    """
        Remove non-ASCII characters 
    """
    return re.sub(r'[^\x00-\x7f]', r' ', text)


def remove_special_characters(text: str) -> str:
    """
        Remove special special characters, including symbols, emojis, and other graphic characters
    """
    emoji_pattern = re.compile(
        '['
        u'\U0001F600-\U0001F64F'  # emoticons
        u'\U0001F300-\U0001F5FF'  # symbols & pictographs
        u'\U0001F680-\U0001F6FF'  # transport & map symbols
        u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
        u'\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251'
        ']+',
        flags=re.UNICODE)
    return emoji_pattern.sub(r' ', text)


def remove_punct(text: str) -> str:
    """
        Remove the punctuation
    """
    return re.sub(r'[]!"$%&\'()*+,./:;=#@?[\\^_`{|}~-]+', ' ', text)


def remove_linebreak(text: str) -> str:
    r"""
        Remove linebreak ('\n', '\r')
    """
    return re.sub(r'/\\n+|\\r+|\n+|\r+/', r' ', text)


## === cell 4
from contractions import fix as contractions_fix


def preprocessing_text(text: str) -> str:
    text = remove_non_ascii(text)
    text = remove_special_characters(text)
    text = text.lower()
    text = contractions_fix(text)
    text = remove_URL(text)
    text = remove_html(text)
    text = remove_linebreak(text)
    text = remove_punct(text)
    return text


## === cell 5
from sklearn.model_selection import train_test_split
train, valid = train_test_split(train, train_size=0.8, random_state=42)


## === cell 6
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize

vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=word_tokenize,
               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1, preprocessor=preprocessing_text)
trn_term_doc = vec.fit_transform(train['comment_text'])
val_term_doc = vec.transform(valid['comment_text'])
test_term_doc = vec.transform(test['comment_text'])


## === cell 7
x = trn_term_doc
val_x = val_term_doc


## === cell 8
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


## === cell 9
def pr(y_i, y):
    p = x[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)

def get_mdl(y):
    y = y.values
    r = np.log(pr(1,y) / pr(0,y))
    model = LogisticRegression(C = 4, solver = 'liblinear')
    x_nb = x.multiply(r)
    return model.fit(x_nb, y), r


## === cell 10
classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train_labels = train.drop(['id', 'comment_text'], axis = 1)
valid_labels = valid.drop(['id', 'comment_text'], axis = 1)


## === cell 11
model = []
for i, col in enumerate(classes):
    print(col)
    model.append(get_mdl(train_labels[col]))
    preds= model[i][0].predict_proba(val_x.multiply(model[i][1]))[:,1].reshape(-1, 1)
    print(roc_auc_score(valid_labels, preds))


## === cell 12
preds = np.zeros((len(test), len(classes)))

for i, col in enumerate(classes):
    print(col)
    preds[:, i] = model[i][0].predict_proba(test_term_doc.multiply(model[i][1]))[:, 1]


## === cell 13
submid = pd.DataFrame({'id': sample_submission["id"]})
submission = pd.concat([submid, pd.DataFrame(preds, columns = classes)], axis=1)
submission.to_csv('submission.csv', index=False)


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2243877603.py in <cell line: 0>()
----> 1 submid = pd.DataFrame({'id': sample_submission["id"]})
      2 submission = pd.concat([submid, pd.DataFrame(preds, columns = classes)], axis=1)
      3 submission.to_csv('submission.csv', index=False)

NameError: name 'sample_submission' is not defined
