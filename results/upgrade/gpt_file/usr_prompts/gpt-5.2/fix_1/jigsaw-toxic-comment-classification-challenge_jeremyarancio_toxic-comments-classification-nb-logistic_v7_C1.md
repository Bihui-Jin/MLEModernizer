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

3.10

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

0.87504

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
import re
import string

import nltk
from nltk.corpus import stopwords                  # module for stop words that come with NLTK

nltk.download('stopwords') #stopwords used to preprocess the corpus

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


## === cell 1
stopwords_english = stopwords.words('english') # a list of English stopwords


## === cell 2


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 3
train_data = pd.read_csv("/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip")
test_data = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip')
test_label_data = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/test_labels.csv.zip')
samp_subm = pd.read_csv("/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip")


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3167987616.py in <cell line: 0>()
      1 train_data = pd.read_csv("/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip")
      2 test_data = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip')
----> 3 test_label_data = pd.read_csv('/kaggle/input/jigsaw-toxic-comment-classification-challenge/test_labels.csv.zip')
      4 samp_subm = pd.read_csv("/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip")

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

## === cell 4
train_data.info()


## === cell 5
train_data.isna().sum()


## === cell 6
train_data.head(10)


## === cell 7
for i in range(10):
    print(train_data['comment_text'][i])
    print('---------------')


## === cell 8
test_data.head(10)


## === cell 9
samp_subm


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4072632689.py in <cell line: 0>()
      1 # Submission data set
----> 2 samp_subm

NameError: name 'samp_subm' is not defined

## === cell 10

def preprocess(corpus):
    
    '''
    From a string, make text lowercase, remove hyperlinks, word containing numbers.
    Input : a list of strings
    Output : a list of tokens stored in a generator (yield)
    '''

    for text in corpus:

        text = text.lower()                                               # Lowercase
        text = re.sub(r'https?://[^\s\n\r]+', '', text)                   # Remove links
        text = re.sub('\w*\d\w*', '', text)                               # Remove words containing numbers
    
        yield ' '.join([word for word in text.split(' ')]) # Return a generator 


## === cell 11
%%time

clean_comments = list(preprocess(train_data['comment_text']))


## === cell 12
for i in range(10):
    print(clean_comments[i])
    print('------------')


## === cell 13
%%time
test_clean_comments = list(preprocess(test_data["comment_text"]))


## === cell 14
target = train_data[['toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate']]
target.head()


## === cell 15
target.sum(axis=0) / target.shape[0]


## === cell 16
def probNB(bow,target,cat):

    '''
    Naive Bayes probability for each word
    Inputs :
    bow : bag of words (with doc in rows and words in columns)
    target : classification vector (filled with 1 and 0)
    cat : 1 or 0, in target
    Output : 
    Vector of Naive Bayes probabilities with smoothing (n_words,1)
    '''

    p = np.array(bow[target==cat].sum(axis=0))

    return np.transpose((p+1) / (p.sum() + bow.shape[1]))
    


## === cell 17
def get_model(bow,target):

    '''
    Function that return the log likelihood of a document
    Inputs :
    bow : bag of words (n_doc,n_words)
    target : classification of comments (n_doc,1)
    Output : 
    Return a vector of Log Likelihood for each comment (Naïve Bayes) (n_doc,1)
    '''

    log = np.log(probNB(bow,target,1)/probNB(bow,target,0))
    m = bow.dot(log)
    model = LogisticRegression(C=4).fit(m,target)
    return model , log


## === cell 18
tfidf_vec = TfidfVectorizer(min_df=3,max_df=0.9)
tfidf = tfidf_vec.fit_transform(clean_comments)
tfidf_test = tfidf_vec.transform(test_clean_comments)

df_classification = pd.DataFrame() #We store probabilities into a Dataframe
df_classification['Comments'] = test_data['comment_text']

for i,j in enumerate(target.columns):
    print('fit', j)
    model,log = get_model(tfidf,target[j])
    df_classification[j] = model.predict_proba(tfidf_test.dot(log))[:,1]


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3386366296.py in <cell line: 0>()
     10 for i,j in enumerate(target.columns):
     11     print('fit', j)
---> 12     model,log = get_model(tfidf,target[j])
     13     df_classification[j] = model.predict_proba(tfidf_test.dot(log))[:,1]

/tmp/ipykernel_11/1195362066.py in get_model(bow, target)
     10     '''
     11 
---> 12     log = np.log(probNB(bow,target,1)/probNB(bow,target,0))
     13     m = bow.dot(log)
     14     model = LogisticRegression(C=4).fit(m,target)

/tmp/ipykernel_11/368263604.py in probNB(bow, target, cat)
     11     '''
     12 
---> 13     p = np.array(bow[target==cat].sum(axis=0))
     14 
     15     return np.transpose((p+1) / (p.sum() + bow.shape[1]))

/usr/local/lib/python3.11/dist-packages/scipy/sparse/_index.py in __getitem__(self, key)
     28     """
     29     def __getitem__(self, key):
---> 30         index, new_shape = self._validate_indices(key)
     31 
     32         # 1D array

/usr/local/lib/python3.11/dist-packages/scipy/sparse/_index.py in _validate_indices(self, key)
    281                         f"bool index {i} has shape {mid_shape} instead of {ix.shape}"
    282                     )
--> 283                 index.extend(ix.nonzero())
    284                 array_indices.extend(range(index_ndim, tmp_ndim))
    285                 index_ndim = tmp_ndim

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

AttributeError: 'Series' object has no attribute 'nonzero'

## === cell 19
keys = target.columns
submid = pd.DataFrame({"id" : samp_subm["id"]})
submission = pd.concat([submid,df_classification[keys]],axis=1)
submission.to_csv('submission.csv', index=False)
print('Done!')


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1750298288.py in <cell line: 0>()
      1 keys = target.columns
----> 2 submid = pd.DataFrame({"id" : samp_subm["id"]})
      3 submission = pd.concat([submid,df_classification[keys]],axis=1)
      4 submission.to_csv('submission.csv', index=False)
      5 print('Done!')

NameError: name 'samp_subm' is not defined
