# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Predict the sentiment of phrases.

## Metric
Classification accuracy.

## Submission Format
For each phrase in the test set, predict a label for the sentiment. Your submission should have a header and look like the following:

```
PhraseId,Sentiment
156061,2
156062,2
156063,2
...
```

## Dataset
The dataset is comprised of tab-separated files with phrases. Each phrase has a PhraseId. Each sentence has a SentenceId.

The sentiment labels are:

0 - negative

1 - somewhat negative

2 - neutral

3 - somewhat positive

4 - positive

# 2. Python version

3.7

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
            description.md (72 lines)
            sampleSubmission.csv (46819 lines)
            sampleSubmission.csv.zip (146.0 kB)
            test.tsv (46819 lines)
            test.tsv.zip (1.1 MB)
            train.tsv (109243 lines)
            train.tsv.zip (2.7 MB)
            movie-review-sentiment-analysis-kernels-only/
                description.md (72 lines)
                sampleSubmission.csv (46819 lines)
                ... and 5 other files
                movie-review-sentiment-analysis-kernels-only/
        input/
            description.md (72 lines)
            sampleSubmission.csv (46819 lines)
            sampleSubmission.csv.zip (146.0 kB)
            test.tsv (46819 lines)
            test.tsv.zip (1.1 MB)
            train.tsv (109243 lines)
            train.tsv.zip (2.7 MB)
            movie-review-sentiment-analysis-kernels-only/
                description.md (72 lines)
                sampleSubmission.csv (46819 lines)
                ... and 5 other files
                movie-review-sentiment-analysis-kernels-only/
        working/
            movie-review-sentiment-analysis-kernels-only/
                description.md (72 lines)
                sampleSubmission.csv (46819 lines)
                ... and 5 other files
                movie-review-sentiment-analysis-kernels-only/
```

-> data/movie-review-sentiment-analysis-kernels-only/sampleSubmission.csv has 46818 rows and 2 columns.
Here is some information about the columns:
PhraseId (int64) has range: 29.00 - 156030.00, 0 nan values
Sentiment (int64) has 1 unique values: [2]

-> data/sampleSubmission.csv has 46818 rows and 2 columns.
Here is some information about the columns:
PhraseId (int64) has range: 29.00 - 156030.00, 0 nan values
Sentiment (int64) has 1 unique values: [2]

-> (stopped after 10 files for performance)

# 5. Target score

0.5882

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
seed = 0

import random
import numpy as np

random.seed(seed)
np.random.seed(seed)


## === cell 1
import pandas as pd

train = pd.read_csv('../input/train.tsv',  sep="\t")
test = pd.read_csv('../input/test.tsv',  sep="\t")


## === cell 2
train.head()


## === cell 3
train['Sentiment'].value_counts()


## === cell 4
def format_data(train, test, max_features, maxlen):
    """
    Convert data to proper format.
    1) Shuffle
    2) Lowercase
    3) Sentiments to Categorical
    4) Tokenize and Fit
    5) Convert to sequence (format accepted by the network)
    6) Pad
    7) Voila!
    """
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    from keras.utils import to_categorical
    
    train = train.sample(frac=1).reset_index(drop=True)
    train['Phrase'] = train['Phrase'].apply(lambda x: x.lower())
    test['Phrase'] = test['Phrase'].apply(lambda x: x.lower())

    X = train['Phrase']
    test_X = test['Phrase']
    Y = to_categorical(train['Sentiment'].values)

    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(X))

    X = tokenizer.texts_to_sequences(X)
    X = pad_sequences(X, maxlen=maxlen)
    test_X = tokenizer.texts_to_sequences(test_X)
    test_X = pad_sequences(test_X, maxlen=maxlen)

    return X, Y, test_X


## === cell 5
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")


def format_data(train, test, max_features, maxlen):
    """
    Convert data to proper format.
    1) Shuffle
    2) Lowercase
    3) Sentiments to Categorical
    4) Tokenize and Fit
    5) Convert to sequence (format accepted by the network)
    6) Pad
    7) Voila!

    Fix: Avoid importing TF/Keras preprocessing in this TF 2.18 + protobuf 6.x
    environment, which can crash with:
        AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'
    Implement minimal local equivalents for Tokenizer, pad_sequences, and
    to_categorical to keep identical interfaces and array shapes.
    """
    import numpy as np

    class _Tokenizer:
        def __init__(self, num_words=None):
            self.num_words = num_words
            self.word_index = {}

        def fit_on_texts(self, texts):
            counts = {}
            for t in texts:
                for w in str(t).split():
                    counts[w] = counts.get(w, 0) + 1
            sorted_items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            self.word_index = {w: i + 1 for i, (w, _) in enumerate(sorted_items)}

        def texts_to_sequences(self, texts):
            seqs = []
            for t in texts:
                seq = []
                for w in str(t).split():
                    idx = self.word_index.get(w)
                    if idx is None:
                        continue
                    if self.num_words is not None and idx >= self.num_words:
                        continue
                    seq.append(idx)
                seqs.append(seq)
            return seqs

    def _pad_sequences(
        sequences, maxlen, dtype="int32", padding="pre", truncating="pre", value=0
    ):
        x = np.full((len(sequences), maxlen), value, dtype=dtype)
        for i, s in enumerate(sequences):
            s = list(s)
            if len(s) == 0:
                continue
            if truncating == "pre":
                trunc = s[-maxlen:]
            else:
                trunc = s[:maxlen]
            trunc = np.asarray(trunc, dtype=dtype)
            if padding == "pre":
                x[i, -len(trunc) :] = trunc
            else:
                x[i, : len(trunc)] = trunc
        return x

    def _to_categorical(y):
        y = np.asarray(y, dtype="int64").ravel()
        num_classes = int(np.max(y)) + 1 if y.size else 0
        out = np.zeros((y.shape[0], num_classes), dtype="float32")
        if y.size:
            out[np.arange(y.shape[0]), y] = 1.0
        return out

    train = train.sample(frac=1).reset_index(drop=True)
    train["Phrase"] = train["Phrase"].apply(lambda x: str(x).lower())
    test["Phrase"] = test["Phrase"].apply(lambda x: str(x).lower())

    X = train["Phrase"]
    test_X = test["Phrase"]
    Y = _to_categorical(train["Sentiment"].values)

    tokenizer = _Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(X))

    X = tokenizer.texts_to_sequences(X)
    X = _pad_sequences(X, maxlen=maxlen)
    test_X = tokenizer.texts_to_sequences(test_X)
    test_X = _pad_sequences(test_X, maxlen=maxlen)

    return X, Y, test_X


maxlen = 50
max_features = 1000

X, Y, test_X = format_data(train, test, max_features, maxlen)


## === cell 6
X


## === cell 7
Y


## === cell 8
test_X


## === cell 9
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.25, random_state=seed)


## === cell 10
import os
import sys
import subprocess

try:
    import google.protobuf as _pb
    from packaging import version as _v

    _pb_ver = getattr(_pb, "__version__", "0")
except Exception:
    _pb_ver = "0"
    _v = None

if _v is not None and _v.parse(_pb_ver) >= _v.parse("5"):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
    )
    os.execv(sys.executable, [sys.executable] + sys.argv)

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import tensorflow as tf  # ensures protobuf implementation is selected before keras is imported

from keras.layers import Input, Dense, Embedding, Flatten
from keras.layers import SpatialDropout1D
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.models import Sequential


## === cell 11
model = Sequential()

model.add(Embedding(max_features, 150, input_length=maxlen))

model.add(SpatialDropout1D(0.25))

model.add(Conv1D(filters=16, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())

model.add(Dense(5, activation='sigmoid'))


## === cell 12
epochs = 5
batch_size = 32


## === cell 13
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=epochs, batch_size=batch_size, verbose=1)


## === cell 14
sub = pd.read_csv('../input/../input/sampleSubmission.csv')

sub['Sentiment'] = model.predict_classes(test_X, batch_size=batch_size, verbose=1)
sub.to_csv('sub_cnn.csv', index=False)
