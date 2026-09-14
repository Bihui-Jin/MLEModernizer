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

-0.3844

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Embedding,
    Bidirectional,
    LSTM,
    Flatten,
    Lambda,
)
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
base_dir = os.path.join("input", "us-patent-phrase-to-phrase-matching")
train_path = os.path.join(base_dir, "train.csv")
test_path = os.path.join(base_dir, "test.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/1360243419.py in <cell line: 0>()
      4 test_path = os.path.join(base_dir, "test.csv")
      5 
----> 6 train = pd.read_csv(train_path)
      7 test = pd.read_csv(test_path)
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

FileNotFoundError: [Errno 2] No such file or directory: 'input/us-patent-phrase-to-phrase-matching/train.csv'

## === cell 2
X_train, X_val, y_train, y_val = train_test_split(
    train[["anchor", "target"]], train["score"], test_size=0.25, random_state=42
)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2264538843.py in <cell line: 0>()
      1 # Train‑validation split
      2 X_train, X_val, y_train, y_val = train_test_split(
----> 3     train[["anchor", "target"]], train["score"], test_size=0.25, random_state=42
      4 )
      5 

NameError: name 'train' is not defined

## === cell 3
vocab_size = 7905
embedding_dim = 16
max_length = 4
trunc_type = "post"
oov_tok = "<OOV>"




## === cell 4
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(
    pd.concat([X_train["anchor"], X_train["target"]]).astype(str).values
)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3062765490.py in <cell line: 0>()
      2 tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
      3 tokenizer.fit_on_texts(
----> 4     pd.concat([X_train["anchor"], X_train["target"]]).astype(str).values
      5 )
      6 

NameError: name 'X_train' is not defined

## === cell 5
def encode_column(col):
    seq = tokenizer.texts_to_sequences(col.astype(str).values)
    return pad_sequences(seq, maxlen=max_length, truncating=trunc_type)


padded_anchor_train = encode_column(X_train["anchor"])
padded_target_train = encode_column(X_train["target"])
padded_anchor_val = encode_column(X_val["anchor"])
padded_target_val = encode_column(X_val["target"])




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4200440870.py in <cell line: 0>()
      5 
      6 # Encode train and validation sets
----> 7 padded_anchor_train = encode_column(X_train["anchor"])
      8 padded_target_train = encode_column(X_train["target"])
      9 padded_anchor_val = encode_column(X_val["anchor"])

NameError: name 'X_train' is not defined

## === cell 6
def initialize_base_network():
    inp = Input(shape=(max_length,))
    x = Embedding(vocab_size, embedding_dim, input_length=max_length)(inp)
    x = Bidirectional(LSTM(32, return_sequences=True))(x)
    x = Flatten()(x)
    return Model(inputs=inp, outputs=x)


base_network = initialize_base_network()




## === cell 7
def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, _ = shapes
    return (shape1[0], 1)




## === cell 8
input_1 = Input(shape=(max_length,), name="anchor_input")
input_2 = Input(shape=(max_length,), name="target_input")

vec_1 = base_network(input_1)
vec_2 = base_network(input_2)

distance = Lambda(
    euclidean_distance, output_shape=eucl_dist_output_shape, name="output"
)([vec_1, vec_2])

siamese_model = Model([input_1, input_2], distance)




## === cell 9
siamese_model.compile(optimizer=RMSprop(), loss="mse")




## === cell 10
history = siamese_model.fit(
    [padded_anchor_train, padded_target_train],
    y_train.values,
    epochs=10,
    batch_size=64,
    validation_data=([padded_anchor_val, padded_target_val], y_val.values),
    verbose=2,
)




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3543183628.py in <cell line: 0>()
      1 history = siamese_model.fit(
----> 2     [padded_anchor_train, padded_target_train],
      3     y_train.values,
      4     epochs=10,
      5     batch_size=64,

NameError: name 'padded_anchor_train' is not defined

## === cell 11
def encode_test(df):
    a_seq = encode_column(df["anchor"])
    t_seq = encode_column(df["target"])
    return a_seq, t_seq


test_anchor_seq, test_target_seq = encode_test(test)
test_pred = siamese_model.predict([test_anchor_seq, test_target_seq])




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/22373913.py in <cell line: 0>()
      5 
      6 
----> 7 test_anchor_seq, test_target_seq = encode_test(test)
      8 test_pred = siamese_model.predict([test_anchor_seq, test_target_seq])
      9 

NameError: name 'test' is not defined

## === cell 12
test_pred = test_pred.reshape(-1)
test["score"] = test_pred




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/540596234.py in <cell line: 0>()
----> 1 test_pred = test_pred.reshape(-1)
      2 test["score"] = test_pred
      3 
      4 

NameError: name 'test_pred' is not defined

## === cell 13
submission = test[["id", "score"]]
submission.to_csv("submission.csv", index=False)

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3044386032.py in <cell line: 0>()
----> 1 submission = test[["id", "score"]]
      2 submission.to_csv("submission.csv", index=False)

NameError: name 'test' is not defined
