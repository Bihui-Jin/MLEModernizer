# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.9

# 2. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
tf_keras==2.18.0

# 3. Data file paths

```
/
    kaggle/
        data/
            description.md (98 lines)
            sample_submission.csv (1959 lines)
            sample_submission.csv.zip (7.4 kB)
            test.csv (1959 lines)
            test.csv.zip (133.3 kB)
            train.csv (17622 lines)
            train.csv.zip (1.2 MB)
            train.zip (1.2 MB)
            spooky-author-identification/
                description.md (98 lines)
                sample_submission.csv (1959 lines)
                ... and 6 other files
                spooky-author-identification/
        input/
            description.md (98 lines)
            sample_submission.csv (1959 lines)
            sample_submission.csv.zip (7.4 kB)
            test.csv (1959 lines)
            test.csv.zip (133.3 kB)
            train.csv (17622 lines)
            train.csv.zip (1.2 MB)
            train.zip (1.2 MB)
            spooky-author-identification/
                description.md (98 lines)
                sample_submission.csv (1959 lines)
                ... and 6 other files
                spooky-author-identification/
        working/
            spooky-author-identification/
                description.md (98 lines)
                sample_submission.csv (1959 lines)
                ... and 6 other files
                spooky-author-identification/
```

-> data/sample_submission.csv has 1958 rows and 4 columns.
The columns are: id, EAP, HPL, MWS

-> data/spooky-author-identification/sample_submission.csv has 1958 rows and 4 columns.
The columns are: id, EAP, HPL, MWS

-> data/spooky-author-identification/test.csv has 1958 rows and 2 columns.
The columns are: id, text

-> data/spooky-author-identification/train.csv has 17621 rows and 3 columns.
The columns are: id, text, author

-> data/test.csv has 1958 rows and 2 columns.
The columns are: id, text

-> data/train.csv has 17621 rows and 3 columns.
The columns are: id, text, author

-> input/sample_submission.csv has 1958 rows and 4 columns.
The columns are: id, EAP, HPL, MWS

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0
"""
Title: Text classification with Transformer
Author: [Apoorv Nandan](https://twitter.com/NandanApoorv)
Date created: 2020/05/10
Last modified: 2020/05/10
Description: Implement a Transformer block as a Keras layer and use it for text classification.
"""
"""
## Setup
"""

import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

try:
    import google.protobuf  # noqa: F401
    from importlib.metadata import version as _pkg_version

    _pb_ver = _pkg_version("protobuf")
except Exception:
    _pb_ver = None


def _major(v):
    try:
        return int(str(v).split(".", 1)[0])
    except Exception:
        return None


if _major(_pb_ver) is not None and _major(_pb_ver) >= 6:
    import sys
    import subprocess

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "protobuf<5"]
    )

    import importlib

    importlib.invalidate_caches()
    for _m in list(sys.modules):
        if _m.startswith("google.protobuf"):
            del sys.modules[_m]

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from keras.utils import to_categorical

from tensorflow.keras.preprocessing import text, sequence

from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

"""
## Implement multi head self attention as a Keras layer
"""

print("Tensorflow Version: ", tf.__version__)
print("Eager mode enabled: ", tf.executing_eagerly())
print("GPU available: ", tf.test.is_gpu_available())

notebookstart = time.time()


## === cell 1
class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output


"""
## Implement a Transformer block as a layer
"""


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


"""
## Implement embedding layer
Two seperate embedding layers, one for tokens, one for token index (positions).
"""


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


## === cell 2
MAXLEN = 250
VOCABLEN = 10000
NCHANNEL = 3
BATCHSIZE = 32
EPOCHS = 60

TEXTCOL = "text"
TARGETCOL = "author"
CHARS_TO_REMOVE = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'


## === cell 3
train = pd.read_csv("zip://../input/spooky-author-identification/train.zip::train.csv")
test = pd.read_csv("zip://../input/spooky-author-identification/test.zip::test.csv")
testdex = test.id
submission = pd.read_csv(
    "zip://../input/spooky-author-identification/sample_submission.zip::sample_submission.csv"
)
sub_cols = submission.columns

xtrain = train[TEXTCOL].astype(str)
xtest = test[TEXTCOL].astype(str)

label_mapper = {name: i for i, name in enumerate(set(train[TARGETCOL].values))}
num_label = np.vectorize(label_mapper.get)(train[TARGETCOL].values)
y_train = to_categorical(num_label)

tokenizer = text.Tokenizer(
    filters=CHARS_TO_REMOVE,
    lower=False,
    num_words=VOCABLEN,
)
tokenizer.fit_on_texts(list(xtrain) + list(xtest))

xtrain = tokenizer.texts_to_sequences(xtrain)
xtest = tokenizer.texts_to_sequences(xtest)

xtrain = sequence.pad_sequences(xtrain, maxlen=MAXLEN)
xtest = sequence.pad_sequences(xtest, maxlen=MAXLEN)

print("X Shape: {}".format(xtrain.shape))
print("y Shape: {}".format(xtrain.shape))

X_train, X_val, y_train, y_val = train_test_split(
    xtrain, y_train, test_size=0.33, random_state=42
)

print("X_train Shape: {}".format(X_train.shape))
print("y_train Shape: {}".format(y_train.shape))

del train, test, xtrain


## --- ERROR in cell 3, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/545908709.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      1[0m [0;31m# Pandas cannot auto-read ZIPs that contain multiple files; explicitly select the CSV inside each ZIP.[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 2[0;31m [0mtrain[0m [0;34m=[0m [0mpd[0m[0;34m.[0m[0mread_csv[0m[0;34m([0m[0;34m"zip://../input/spooky-author-identification/train.zip::train.csv"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      3[0m [0mtest[0m [0;34m=[0m [0mpd[0m[0;34m.[0m[0mread_csv[0m[0;34m([0m[0;34m"zip://../input/spooky-author-identification/test.zip::test.csv"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0mtestdex[0m [0;34m=[0m [0mtest[0m[0;34m.[0m[0mid[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m submission = pd.read_csv(

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py[0m in [0;36mread_csv[0;34m(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)[0m
[1;32m   1024[0m     [0mkwds[0m[0;34m.[0m[0mupdate[0m[0;34m([0m[0mkwds_defaults[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1025[0m [0;34m[0m[0m
[0;32m-> 1026[0;31m     [0;32mreturn[0m [0m_read[0m[0;34m([0m[0mfilepath_or_buffer[0m[0;34m,[0m [0mkwds[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1027[0m [0;34m[0m[0m
[1;32m   1028[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py[0m in [0;36m_read[0;34m(filepath_or_buffer, kwds)[0m
[1;32m    618[0m [0;34m[0m[0m
[1;32m    619[0m     [0;31m# Create the parser.[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 620[0;31m     [0mparser[0m [0;34m=[0m [0mTextFileReader[0m[0;34m([0m[0mfilepath_or_buffer[0m[0;34m,[0m [0;34m**[0m[0mkwds[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    621[0m [0;34m[0m[0m
[1;32m    622[0m     [0;32mif[0m [0mchunksize[0m [0;32mor[0m [0miterator[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py[0m in [0;36m__init__[0;34m(self, f, engine, **kwds)[0m
[1;32m   1618[0m [0;34m[0m[0m
[1;32m   1619[0m         [0mself[0m[0;34m.[0m[0mhandles[0m[0;34m:[0m [0mIOHandles[0m [0;34m|[0m [0;32mNone[0m [0;34m=[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1620[0;31m         [0mself[0m[0;34m.[0m[0m_engine[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_make_engine[0m[0;34m([0m[0mf[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0mengine[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1621[0m [0;34m[0m[0m
[1;32m   1622[0m     [0;32mdef[0m [0mclose[0m[0;34m([0m[0mself[0m[0;34m)[0m [0;34m->[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py[0m in [0;36m_make_engine[0;34m(self, f, engine)[0m
[1;32m   1878[0m                 [0;32mif[0m [0;34m"b"[0m [0;32mnot[0m [0;32min[0m [0mmode[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1879[0m                     [0mmode[0m [0;34m+=[0m [0;34m"b"[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1880[0;31m             self.handles = get_handle(
[0m[1;32m   1881[0m                 [0mf[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1882[0m                 [0mmode[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/io/common.py[0m in [0;36mget_handle[0;34m(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)[0m
[1;32m    726[0m [0;34m[0m[0m
[1;32m    727[0m     [0;31m# open URLs[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 728[0;31m     ioargs = _get_filepath_or_buffer(
[0m[1;32m    729[0m         [0mpath_or_buf[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    730[0m         [0mencoding[0m[0;34m=[0m[0mencoding[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/io/common.py[0m in [0;36m_get_filepath_or_buffer[0;34m(filepath_or_buffer, encoding, compression, mode, storage_options)[0m
[1;32m    428[0m [0;34m[0m[0m
[1;32m    429[0m         [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 430[0;31m             file_obj = fsspec.open(
[0m[1;32m    431[0m                 [0mfilepath_or_buffer[0m[0;34m,[0m [0mmode[0m[0;34m=[0m[0mfsspec_mode[0m[0;34m,[0m [0;34m**[0m[0;34m([0m[0mstorage_options[0m [0;32mor[0m [0;34m{[0m[0;34m}[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    432[0m             ).open()

[0;32m/usr/local/lib/python3.11/dist-packages/fsspec/core.py[0m in [0;36mopen[0;34m(urlpath, mode, compression, encoding, errors, protocol, newline, expand, **kwargs)[0m
[1;32m    489[0m     """
[1;32m    490[0m     [0mexpand[0m [0;34m=[0m [0mDEFAULT_EXPAND[0m [0;32mif[0m [0mexpand[0m [0;32mis[0m [0;32mNone[0m [0;32melse[0m [0mexpand[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 491[0;31m     out = open_files(
[0m[1;32m    492[0m         [0murlpath[0m[0;34m=[0m[0;34m[[0m[0murlpath[0m[0;34m][0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    493[0m         [0mmode[0m[0;34m=[0m[0mmode[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fsspec/core.py[0m in [0;36mopen_files[0;34m(urlpath, mode, compression, encoding, errors, name_function, num, protocol, newline, auto_mkdir, expand, **kwargs)[0m
[1;32m    293[0m       [0mhttps[0m[0;34m:[0m[0;34m//[0m[0mfilesystem[0m[0;34m-[0m[0mspec[0m[0;34m.[0m[0mreadthedocs[0m[0;34m.[0m[0mio[0m[0;34m/[0m[0men[0m[0;34m/[0m[0mlatest[0m[0;34m/[0m[0mapi[0m[0;34m.[0m[0mhtml[0m[0;31m#other-known-implementations[0m[0;34m[0m[0;34m[0m[0m
[1;32m    294[0m     """
[0;32m--> 295[0;31m     fs, fs_token, paths = get_fs_token_paths(
[0m[1;32m    296[0m         [0murlpath[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    297[0m         [0mmode[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fsspec/core.py[0m in [0;36mget_fs_token_paths[0;34m(urlpath, mode, num, name_function, storage_options, protocol, expand)[0m
[1;32m    665[0m         [0minkwargs[0m[0;34m[[0m[0;34m"fo"[0m[0;34m][0m [0;34m=[0m [0murls[0m[0;34m[0m[0;34m[0m[0m
[1;32m    666[0m     [0mpaths[0m[0;34m,[0m [0mprotocol[0m[0;34m,[0m [0m_[0m [0;34m=[0m [0mchain[0m[0;34m[[0m[0;36m0[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 667[0;31m     [0mfs[0m [0;34m=[0m [0mfilesystem[0m[0;34m([0m[0mprotocol[0m[0;34m,[0m [0;34m**[0m[0minkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    668[0m     [0;32mif[0m [0misinstance[0m[0;34m([0m[0murlpath[0m[0;34m,[0m [0;34m([0m[0mlist[0m[0;34m,[0m [0mtuple[0m[0;34m,[0m [0mset[0m[0;34m)[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    669[0m         pchains = [

[0;32m/usr/local/lib/python3.11/dist-packages/fsspec/registry.py[0m in [0;36mfilesystem[0;34m(protocol, **storage_options)[0m
[1;32m    320[0m [0;34m[0m[0m
[1;32m    321[0m     [0mcls[0m [0;34m=[0m [0mget_filesystem_class[0m[0;34m([0m[0mprotocol[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 322[0;31m     [0;32mreturn[0m [0mcls[0m[0;34m([0m[0;34m**[0m[0mstorage_options[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    323[0m [0;34m[0m[0m
[1;32m    324[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fsspec/spec.py[0m in [0;36m__call__[0;34m(cls, *args, **kwargs)[0m
[1;32m     82[0m             [0;32mreturn[0m [0mcls[0m[0;34m.[0m[0m_cache[0m[0;34m[[0m[0mtoken[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[1;32m     83[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 84[0;31m             [0mobj[0m [0;34m=[0m [0msuper[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0m__call__[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m,[0m [0;34m**[0m[0mstrip_tokenize_options[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     85[0m             [0;31m# Setting _fs_token here causes some static linters to complain.[0m[0;34m[0m[0;34m[0m[0m
[1;32m     86[0m             [0mobj[0m[0;34m.[0m[0m_fs_token_[0m [0;34m=[0m [0mtoken[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fsspec/implementations/zip.py[0m in [0;36m__init__[0;34m(self, fo, mode, target_protocol, target_options, compression, allowZip64, compresslevel, **kwargs)[0m
[1;32m     60[0m         [0mself[0m[0;34m.[0m[0mforce_zip_64[0m [0;34m=[0m [0mallowZip64[0m[0;34m[0m[0;34m[0m[0m
[1;32m     61[0m         [0mself[0m[0;34m.[0m[0mof[0m [0;34m=[0m [0mfo[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 62[0;31m         [0mself[0m[0;34m.[0m[0mfo[0m [0;34m=[0m [0mfo[0m[0;34m.[0m[0m__enter__[0m[0;34m([0m[0;34m)[0m  [0;31m# the whole instance is a context[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     63[0m         self.zip = zipfile.ZipFile(
[1;32m     64[0m             [0mself[0m[0;34m.[0m[0mfo[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fsspec/core.py[0m in [0;36m__enter__[0;34m(self)[0m
[1;32m    103[0m [0;34m[0m[0m
[1;32m    104[0m         [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 105[0;31m             [0mf[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mfs[0m[0;34m.[0m[0mopen[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mpath[0m[0;34m,[0m [0mmode[0m[0;34m=[0m[0mmode[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    106[0m         [0;32mexcept[0m [0mFileNotFoundError[0m [0;32mas[0m [0me[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    107[0m             [0;32mif[0m [0mhas_magic[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mpath[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fsspec/spec.py[0m in [0;36mopen[0;34m(self, path, mode, block_size, cache_options, compression, **kwargs)[0m
[1;32m   1347[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1348[0m             [0mac[0m [0;34m=[0m [0mkwargs[0m[0;34m.[0m[0mpop[0m[0;34m([0m[0;34m"autocommit"[0m[0;34m,[0m [0;32mnot[0m [0mself[0m[0;34m.[0m[0m_intrans[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1349[0;31m             f = self._open(
[0m[1;32m   1350[0m                 [0mpath[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1351[0m                 [0mmode[0m[0;34m=[0m[0mmode[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fsspec/implementations/local.py[0m in [0;36m_open[0;34m(self, path, mode, block_size, **kwargs)[0m
[1;32m    208[0m         [0;32mif[0m [0mself[0m[0;34m.[0m[0mauto_mkdir[0m [0;32mand[0m [0;34m"w"[0m [0;32min[0m [0mmode[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    209[0m             [0mself[0m[0;34m.[0m[0mmakedirs[0m[0;34m([0m[0mself[0m[0;34m.[0m[0m_parent[0m[0;34m([0m[0mpath[0m[0;34m)[0m[0;34m,[0m [0mexist_ok[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 210[0;31m         [0;32mreturn[0m [0mLocalFileOpener[0m[0;34m([0m[0mpath[0m[0;34m,[0m [0mmode[0m[0;34m,[0m [0mfs[0m[0;34m=[0m[0mself[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    211[0m [0;34m[0m[0m
[1;32m    212[0m     [0;32mdef[0m [0mtouch[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mpath[0m[0;34m,[0m [0mtruncate[0m[0;34m=[0m[0;32mTrue[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fsspec/implementations/local.py[0m in [0;36m__init__[0;34m(self, path, mode, autocommit, fs, compression, **kwargs)[0m
[1;32m    385[0m         [0mself[0m[0;34m.[0m[0mcompression[0m [0;34m=[0m [0mget_compression[0m[0;34m([0m[0mpath[0m[0;34m,[0m [0mcompression[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    386[0m         [0mself[0m[0;34m.[0m[0mblocksize[0m [0;34m=[0m [0mio[0m[0;34m.[0m[0mDEFAULT_BUFFER_SIZE[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 387[0;31m         [0mself[0m[0;34m.[0m[0m_open[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    388[0m [0;34m[0m[0m
[1;32m    389[0m     [0;32mdef[0m [0m_open[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/fsspec/implementations/local.py[0m in [0;36m_open[0;34m(self)[0m
[1;32m    390[0m         [0;32mif[0m [0mself[0m[0;34m.[0m[0mf[0m [0;32mis[0m [0;32mNone[0m [0;32mor[0m [0mself[0m[0;34m.[0m[0mf[0m[0;34m.[0m[0mclosed[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    391[0m             [0;32mif[0m [0mself[0m[0;34m.[0m[0mautocommit[0m [0;32mor[0m [0;34m"w"[0m [0;32mnot[0m [0;32min[0m [0mself[0m[0;34m.[0m[0mmode[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 392[0;31m                 [0mself[0m[0;34m.[0m[0mf[0m [0;34m=[0m [0mopen[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mpath[0m[0;34m,[0m [0mmode[0m[0;34m=[0m[0mself[0m[0;34m.[0m[0mmode[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    393[0m                 [0;32mif[0m [0mself[0m[0;34m.[0m[0mcompression[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    394[0m                     [0mcompress[0m [0;34m=[0m [0mcompr[0m[0;34m[[0m[0mself[0m[0;34m.[0m[0mcompression[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m

[0;31mFileNotFoundError[0m: [Errno 2] No such file or directory: '/kaggle/working/train.csv'

## === cell 4
embed_dim = 32  # Embedding size for each token
num_heads = 4  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer

inputs = layers.Input(shape=(MAXLEN,))
embedding_layer = TokenAndPositionEmbedding(MAXLEN, VOCABLEN, embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(32, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(NCHANNEL, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
