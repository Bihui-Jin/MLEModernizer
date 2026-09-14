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
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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

0.9598777576795992

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 1

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 2
!pip install tensorflow --upgrade

## === cell 3
import numpy as np
np.random.seed(42)
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt 
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['OMP_NUM_THREADS'] = '4'

## === cell 4
train = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv')
test = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv')
local_sumb = pd.read_csv('../input/saved-relations/gru_we_submission.csv')
subm = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv')


## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3319261522.py in <cell line: 0>()
      1 train = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv')
      2 test = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv')
----> 3 local_sumb = pd.read_csv('../input/saved-relations/gru_we_submission.csv')
      4 subm = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv')

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

FileNotFoundError: [Errno 2] No such file or directory: '../input/saved-relations/gru_we_submission.csv'

## === cell 6
test.head()

## === cell 7
subm.head()

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2995130811.py in <cell line: 0>()
----> 1 subm.head()

NameError: name 'subm' is not defined

## === cell 8
train.head()

## === cell 9
text = train['comment_text']

## === cell 10
text[0]

## === cell 11
train['comment_text'][0]

## === cell 12
lens = train.comment_text.str.len()
lens.mean(), lens.std(), lens.max()

## === cell 13
lens = test.comment_text.str.len()
lens.mean(), lens.std(), lens.max()

## === cell 14
lens.hist();

## === cell 15
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train['none'] = 1-train[label_cols].max(axis=1) ## each colum may have the value of one ( Labled ) . 1- calc the max # if has no lable max = 0 then col = 1 -0 = 0
train.describe()

## === cell 16
len(train),len(test)

## === cell 17
COMMENT = 'comment_text'
train[COMMENT].fillna("unknown", inplace=True)
test[COMMENT].fillna("unknown", inplace=True)

## === cell 19
import re, string
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()
def clean(s): return re_tok.sub(r' \1 ', s)

## === cell 20
words = []
for t in text:
    words.extend(tokenize(t))
print(words[:100])
vocab = list(set(words))
print(len(words), len(vocab))

## === cell 21
train['comment_text'][0]

## === cell 22
clean(train['comment_text'][0])

## === cell 23
def full_one_hot_word_embedding(vtrain_data,vtest_data):
    train_labels = vtrain_data[label_cols]
    txt_train_data = [clean(txt) for txt in train['comment_text']]
    txt_test_data = [clean(txt) for txt in test['comment_text']]
    
    vocab_size = 10000
    encoded_txt_train_data = [keras.preprocessing.text.one_hot(d, vocab_size) for d in txt_train_data]
    encoded_txt_test_data = [keras.preprocessing.text.one_hot(d, vocab_size) for d in txt_test_data]

    ptxt_train_data = keras.preprocessing.sequence.pad_sequences(encoded_txt_train_data,
                                                            padding='post',
                                                            maxlen=5000)

    ptxt_test_data = keras.preprocessing.sequence.pad_sequences(encoded_txt_test_data,
                                                           padding='post',
                                                           maxlen=5000)
    partial_x_train = ptxt_train_data[:100]
    partial_y_train = train_labels[:100]
    return (partial_x_train,partial_y_train,ptxt_test_data)

## === cell 24
max_features = 30000
maxlen = 5000
embed_size = 300
vocab_size = 10000

def get_model():
    inp = keras.layers.Input(shape=(maxlen, ))
    x = keras.layers.Embedding(vocab_size, 16)(inp)
    x = keras.layers.SpatialDropout1D(0.2)(x)
    x = keras.layers.GRU(80, return_sequences=True)(x)
    avg_pool = keras.layers.GlobalAveragePooling1D()(x)
    max_pool = keras.layers.GlobalMaxPooling1D()(x)
    conc = keras.layers.concatenate([avg_pool, max_pool])
    outp = keras.layers.Dense(6, activation="sigmoid")(conc)
    
    model = keras.models.Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

## === cell 25
x_train,y_train,test_data = full_one_hot_word_embedding(train[:100],test[:100]) 

## === cell 26
x_train.shape,y_train.shape

## === cell 27
model1 = get_model()

## === cell 28
batch_size = 32
epochs = 1

X_tra, X_val, y_tra, y_val = train_test_split(x_train, y_train, train_size=0.95, random_state=233)

X_tra = X_tra.astype(np.float32)
X_val = X_val.astype(np.float32)
y_tra = y_tra.values.astype(np.float32)
y_val = y_val.values.astype(np.float32)

## === cell 29
hist = model1.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),
                  verbose=1)

## === cell 30
model_json = model1.to_json()
with open("my_model1.json", "w") as json_file:
    json_file.write(model_json)
    
model1.save_weights('my_model1_weights.h5')

## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3349329645.py in <cell line: 0>()
      4     json_file.write(model_json)
      5 
----> 6 model1.save_weights('my_model1_weights.h5')

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py in save_weights(model, filepath, overwrite, max_shard_size, **kwargs)
    237     filepath_str = str(filepath)
    238     if max_shard_size is None and not filepath_str.endswith(".weights.h5"):
--> 239         raise ValueError(
    240             "The filename must end in `.weights.h5`. "
    241             f"Received: filepath={filepath_str}"

ValueError: The filename must end in `.weights.h5`. Received: filepath=my_model1_weights.h5

## === cell 31
json_file = open('my_model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model1 = tf.keras.models.model_from_json(loaded_model_json)
model1.load_weights("my_model1_weights.h5")
print("Loaded model from disk")

## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3208044393.py in <cell line: 0>()
      6 model1 = tf.keras.models.model_from_json(loaded_model_json)
      7 # load weights into new model
----> 8 model1.load_weights("my_model1_weights.h5")
      9 print("Loaded model from disk")

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in __init__(self, name, mode, driver, libver, userblock_size, swmr, rdcc_nslots, rdcc_nbytes, rdcc_w0, track_order, fs_strategy, fs_persist, fs_threshold, fs_page_size, page_buf_size, min_meta_keep, min_raw_keep, locking, alignment_threshold, alignment_interval, meta_block_size, **kwds)
    562                                  fs_persist=fs_persist, fs_threshold=fs_threshold,
    563                                  fs_page_size=fs_page_size)
--> 564                 fid = make_fid(name, mode, userblock_size, fapl, fcpl, swmr=swmr)
    565 
    566             if isinstance(libver, tuple):

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in make_fid(name, mode, userblock_size, fapl, fcpl, swmr)
    236         if swmr and swmr_support:
    237             flags |= h5f.ACC_SWMR_READ
--> 238         fid = h5f.open(name, flags, fapl=fapl)
    239     elif mode == 'r+':
    240         fid = h5f.open(name, h5f.ACC_RDWR, fapl=fapl)

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/h5f.pyx in h5py.h5f.open()

FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = 'my_model1_weights.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

## === cell 32
def column(matrix, i):
    return [row[i] for row in matrix]

## === cell 34
submission = local_sumb

## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3933867121.py in <cell line: 0>()
----> 1 submission = local_sumb

NameError: name 'local_sumb' is not defined

## === cell 36
submission.to_csv('submission.csv', index=False)

## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1690294540.py in <cell line: 0>()
----> 1 submission.to_csv('submission.csv', index=False)

NameError: name 'submission' is not defined

## === cell 37
submission

## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/493289180.py in <cell line: 0>()
----> 1 submission

NameError: name 'submission' is not defined
