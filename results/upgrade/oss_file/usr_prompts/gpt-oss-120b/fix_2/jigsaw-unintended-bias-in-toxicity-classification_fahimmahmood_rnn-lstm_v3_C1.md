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
Build a model that recognizes toxicity and minimizes unintended bias with respect to mentions of identities.

## Metric
We combine several submetrics: An overall ROC-AUC for the full evaluation set, along with the ROC-AUCs on three specific subsets of the test set capturing different aspects of bias.

The final model score looks like:

$$
\text { score }=w_0 A U C_{\text {overall }}+\sum_{a=1}^A w_a M_p\left(m_{s, a}\right)
$$
where:
$A=$ number of submetrics $(3)$
$m_{s, a}=$ bias metric for identity subgroup $s$ using submetric $a$
$w_a=$ a weighting for the relative importance of each submetric; all four $w$ values set to 0.25

Overall AUC: This is the ROC-AUC for the full evaluation set.

### Bias AUCs
To measure unintended bias, we again calculate the ROC-AUC, this time on three specific subsets of the test set for each identity, each capturing a different aspect of unintended bias. 

**Subgroup AUC**: Here, we restrict the data set to only the examples that mention the specific identity subgroup. *A low value in this metric means the model does a poor job of distinguishing between toxic and non-toxic comments that mention the identity*.

**BPSN (Background Positive, Subgroup Negative) AUC**: Here, we restrict the test set to the non-toxic examples that mention the identity and the toxic examples that do not. *A low value in this metric means that the model confuses non-toxic examples that mention the identity with toxic examples that do not*, likely meaning that the model predicts higher toxicity scores than it should for non-toxic examples mentioning the identity.

**BNSP (Background Negative, Subgroup Positive) AUC**: Here, we restrict the test set to the toxic examples that mention the identity and the non-toxic examples that do not. *A low value here means that the model confuses toxic examples that mention the identity with non-toxic examples that do not*, likely meaning that the model predicts lower toxicity scores than it should for toxic examples mentioning the identity.

#### Generalized Mean of Bias AUCs
To combine the per-identity Bias AUCs into one overall measure, we calculate their generalized mean as defined below:

$$
M_p\left(m_s\right)=\left(\frac{1}{N} \sum_{s=1}^N m_s^p\right)^{\frac{1}{p}}
$$

where:
$M_p=$ the $p$ th power-mean function
$m_s=$ the bias metric $m$ calulated for subgroup $S$
$N=$ number of identity subgroups

For this competition, we use a $p$ value of -5 to encourage competitors to improve the model for the identity subgroups with the lowest model performance.

## Submission Format
```
id,prediction
7000000,0.0
7000001,0.0
etc.

```

## Dataset
The text of the individual comment is found in the `comment_text` column. Each comment in Train has a toxicity label (`target`), and models should predict the `target` toxicity for the Test data. This attribute (and all others) are fractional values which represent the fraction of human raters who believed the attribute applied to the given comment. For evaluation, test set examples with `target >= 0.5` will be considered to be in the positive class (toxic).

The data also has several additional toxicity subtype attributes. Models do not need to predict these attributes for the competition, they are included as an additional avenue for research. Subtype attributes are:

- severe_toxicity
- obscene
- threat
- insult
- identity_attack
- sexual_explicit

Additionally, a subset of comments have been labelled with a variety of identity attributes, representing the identities that are *mentioned* in the comment. The columns corresponding to identity attributes are listed below. Only identities shown below will be included in the evaluation calculation.

- **male**
- **female**
- **homosexual_gay_or_lesbian**
- **christian**
- **jewish**
- **muslim**
- **black**
- **white**
- **psychiatric_or_mental_illness**

### Files
- **train.csv** - the training set, which includes toxicity labels and subgroups
- **test.csv** - the test set, which does **not** include toxicity labels or subgroups
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.9

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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (286 lines)
            sample_submission.csv (97321 lines)
            sample_submission.csv.zip (230.8 kB)
            test.csv (205781 lines)
            test.csv.zip (12.5 MB)
            train.csv (3820210 lines)
            train.csv.zip (285.9 MB)
            jigsaw-unintended-bias-in-toxicity-classification/
                description.md (286 lines)
                sample_submission.csv (97321 lines)
                ... and 5 other files
                jigsaw-unintended-bias-in-toxicity-classification/
        input/
            description.md (286 lines)
            sample_submission.csv (97321 lines)
            sample_submission.csv.zip (230.8 kB)
            test.csv (205781 lines)
            test.csv.zip (12.5 MB)
            train.csv (3820210 lines)
            train.csv.zip (285.9 MB)
            jigsaw-unintended-bias-in-toxicity-classification/
                description.md (286 lines)
                sample_submission.csv (97321 lines)
                ... and 5 other files
                jigsaw-unintended-bias-in-toxicity-classification/
        working/
            jigsaw-unintended-bias-in-toxicity-classification/
                description.md (286 lines)
                sample_submission.csv (97321 lines)
                ... and 5 other files
                jigsaw-unintended-bias-in-toxicity-classification/
```

-> data/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv has 97320 rows and 2 columns.
The columns are: id, prediction

-> data/jigsaw-unintended-bias-in-toxicity-classification/test.csv has 205780 rows and 2 columns.
The columns are: id, comment_text

-> data/jigsaw-unintended-bias-in-toxicity-classification/train.csv has 3820209 rows and 45 columns.
The columns are: id, target, comment_text, severe_toxicity, obscene, identity_attack, insult, threat, asian, atheist, bisexual, black, buddhist, christian, female... and 30 more columns

-> data/sample_submission.csv has 97320 rows and 2 columns.
The columns are: id, prediction

-> data/test.csv has 205780 rows and 2 columns.
The columns are: id, comment_text

-> data/train.csv has 3820209 rows and 45 columns.
The columns are: id, target, comment_text, severe_toxicity, obscene, identity_attack, insult, threat, asian, atheist, bisexual, black, buddhist, christian, female... and 30 more columns

-> (stopped after 10 files for performance)

# 5. Target score

0.51726

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
import re, string, pathlib
from sklearn.model_selection import train_test_split
import multiprocessing

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Activation, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.callbacks import EarlyStopping



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
base_path = pathlib.Path("./data/jigsaw-unintended-bias-in-toxicity-classification")
train_path = base_path / "train.csv"
test_path = base_path / "test.csv"
sample_sub_path = base_path / "sample_submission.csv"


def load_csv(p):
    return pd.read_csv(p)


with multiprocessing.Pool() as pool:
    train, test, sub = pool.map(load_csv, [train_path, test_path, sample_sub_path])



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
RemoteTraceback                           Traceback (most recent call last)
RemoteTraceback: 
"""
Traceback (most recent call last):
  File "/usr/lib/python3.11/multiprocessing/pool.py", line 125, in worker
    result = (True, func(*args, **kwds))
                    ^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.11/multiprocessing/pool.py", line 48, in mapstar
    return list(map(*args))
           ^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_56/1697086224.py", line 9, in load_csv
    return pd.read_csv(p)
           ^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py", line 1026, in read_csv
    return _read(filepath_or_buffer, kwds)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py", line 620, in _read
    parser = TextFileReader(filepath_or_buffer, **kwds)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py", line 1620, in __init__
    self._engine = self._make_engine(f, self.engine)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py", line 1880, in _make_engine
    self.handles = get_handle(
                   ^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/pandas/io/common.py", line 873, in get_handle
    handle = open(
             ^^^^^
FileNotFoundError: [Errno 2] No such file or directory: 'data/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv'
"""

The above exception was the direct cause of the following exception:

FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_56/1697086224.py in <cell line: 0>()
     11 
     12 with multiprocessing.Pool() as pool:
---> 13     train, test, sub = pool.map(load_csv, [train_path, test_path, sample_sub_path])
     14 

/usr/lib/python3.11/multiprocessing/pool.py in map(self, func, iterable, chunksize)
    365         in a list that is returned.
    366         '''
--> 367         return self._map_async(func, iterable, mapstar, chunksize).get()
    368 
    369     def starmap(self, func, iterable, chunksize=None):

/usr/lib/python3.11/multiprocessing/pool.py in get(self, timeout)
    772             return self._value
    773         else:
--> 774             raise self._value
    775 
    776     def _set(self, i, obj):

/usr/lib/python3.11/multiprocessing/pool.py in worker()
    123         job, i, func, args, kwds = task
    124         try:
--> 125             result = (True, func(*args, **kwds))
    126         except Exception as e:
    127             if wrap_exception and func is not _helper_reraises_exception:

/usr/lib/python3.11/multiprocessing/pool.py in mapstar()
     46 
     47 def mapstar(args):
---> 48     return list(map(*args))
     49 
     50 def starmapstar(args):

/tmp/ipykernel_56/1697086224.py in load_csv()
      7 
      8 def load_csv(p):
----> 9     return pd.read_csv(p)
     10 
     11 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv()
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read()
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__()
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine()
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle()
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: 'data/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv'

## === cell 2
train["target"] = np.where(train["target"] > 0.5, 1.0, 0.0)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/1065706362.py in <cell line: 0>()
      1 # Binarize target as in original notebook
----> 2 train["target"] = np.where(train["target"] > 0.5, 1.0, 0.0)
      3 

NameError: name 'train' is not defined

## === cell 3
alphanumeric = lambda x: re.sub(r"\w*\d\w*", " ", str(x))
punc_lower = lambda x: re.sub(f"[{re.escape(string.punctuation)}]", " ", str(x).lower())
remove_n = lambda x: re.sub("\n", " ", str(x))
remove_non_ascii = lambda x: re.sub(r"[^\x00-\x7f]", " ", str(x))


def clean_series(s):
    return s.map(alphanumeric).map(punc_lower).map(remove_n).map(remove_non_ascii)


train["comment_text"] = clean_series(train["comment_text"])
test["comment_text"] = clean_series(test["comment_text"])



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/2722275564.py in <cell line: 0>()
     10 
     11 
---> 12 train["comment_text"] = clean_series(train["comment_text"])
     13 test["comment_text"] = clean_series(test["comment_text"])
     14 

NameError: name 'train' is not defined

## === cell 4
toxic_train = train[train["target"] > 0.5].iloc[0:3000, :]
neutral_train = train[train["target"] <= 0.5].iloc[0:7000, :]
balanced_train = pd.concat([toxic_train, neutral_train], axis=0).reset_index(drop=True)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/47225765.py in <cell line: 0>()
      1 # Build a small balanced subset (same logic as original)
----> 2 toxic_train = train[train["target"] > 0.5].iloc[0:3000, :]
      3 neutral_train = train[train["target"] <= 0.5].iloc[0:7000, :]
      4 balanced_train = pd.concat([toxic_train, neutral_train], axis=0).reset_index(drop=True)
      5 

NameError: name 'train' is not defined

## === cell 5
X = balanced_train["comment_text"].astype(str)
Y = balanced_train["target"].astype(float)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/2850905531.py in <cell line: 0>()
----> 1 X = balanced_train["comment_text"].astype(str)
      2 Y = balanced_train["target"].astype(float)
      3 

NameError: name 'balanced_train' is not defined

## === cell 6
X_train, X_val, Y_train, Y_val = train_test_split(
    X, Y, test_size=0.15, random_state=42, stratify=Y
)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/364242818.py in <cell line: 0>()
      1 X_train, X_val, Y_train, Y_val = train_test_split(
----> 2     X, Y, test_size=0.15, random_state=42, stratify=Y
      3 )
      4 

NameError: name 'X' is not defined

## === cell 7
max_words = 100000
max_len = 250

tok = Tokenizer(num_words=max_words, oov_token="<OOV>")
tok.fit_on_texts(X_train)

train_seq = tok.texts_to_sequences(X_train)
X_train_seq = sequence.pad_sequences(train_seq, maxlen=max_len)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/3078356130.py in <cell line: 0>()
      3 
      4 tok = Tokenizer(num_words=max_words, oov_token="<OOV>")
----> 5 tok.fit_on_texts(X_train)
      6 
      7 train_seq = tok.texts_to_sequences(X_train)

NameError: name 'X_train' is not defined

## === cell 8
def RNN():
    inputs = Input(name="inputs", shape=[max_len])
    x = Embedding(max_words, 50, input_length=max_len)(inputs)
    x = LSTM(64)(x)
    x = Dense(256, name="FC1")(x)
    x = Activation("relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(1, name="out_layer")(x)
    outputs = Activation("sigmoid")(x)
    return Model(inputs=inputs, outputs=outputs)




## === cell 9
model = RNN()
model.compile(loss="binary_crossentropy", optimizer=RMSprop(), metrics=["accuracy"])



## === cell 10
model.fit(
    X_train_seq,
    Y_train,
    batch_size=2048,
    epochs=5,
    validation_split=0.2,
    callbacks=[
        EarlyStopping(
            monitor="val_loss", min_delta=1e-4, patience=2, restore_best_weights=True
        )
    ],
    verbose=2,
)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/2192007087.py in <cell line: 0>()
      1 model.fit(
----> 2     X_train_seq,
      3     Y_train,
      4     batch_size=2048,
      5     epochs=5,

NameError: name 'X_train_seq' is not defined

## === cell 11
test_seq = tok.texts_to_sequences(test["comment_text"].astype(str))
test_seq_pad = sequence.pad_sequences(test_seq, maxlen=max_len)

preds = model.predict(test_seq_pad, batch_size=2048).flatten()



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/1092857595.py in <cell line: 0>()
      1 # Prepare test data using the same tokenizer
----> 2 test_seq = tok.texts_to_sequences(test["comment_text"].astype(str))
      3 test_seq_pad = sequence.pad_sequences(test_seq, maxlen=max_len)
      4 
      5 preds = model.predict(test_seq_pad, batch_size=2048).flatten()

NameError: name 'test' is not defined

## === cell 12
sub["prediction"] = preds
sub.to_csv("submission.csv", index=False)

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/1520321993.py in <cell line: 0>()
----> 1 sub["prediction"] = preds
      2 sub.to_csv("submission.csv", index=False)

NameError: name 'preds' is not defined
