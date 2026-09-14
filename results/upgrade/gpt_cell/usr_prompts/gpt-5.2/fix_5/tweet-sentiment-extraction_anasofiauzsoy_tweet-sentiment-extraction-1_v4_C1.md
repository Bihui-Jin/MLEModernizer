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
Predict the word or phrase from tweets that exemplifies the labelled sentiment.

## Metric
Word-level Jaccard score.

## Submission Format
For each ID in the test set, you must predict the string that best supports the sentiment for the tweet in question. Note that the selected text _needs_ to be **quoted** and **complete** (include punctuation, etc. - the above code splits ONLY on whitespace) to work correctly. The file should contain a header and have the following format:
```
textID,selected_text
2,"very good"
5,"I don't care"
6,"bad"
8,"it was, yes"
etc.
```

## Dataset
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

- `textID` - unique ID for each piece of text
- `text` - the text of the tweet
- `sentiment` - the general sentiment of the tweet
- `selected_text` - [train only] the text that supports the tweet's sentiment

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
            description.md (130 lines)
            sample_submission.csv (2750 lines)
            sample_submission.csv.zip (18.6 kB)
            test.csv (2750 lines)
            test.csv.zip (114.6 kB)
            train.csv (24733 lines)
            train.csv.zip (1.2 MB)
            tweet-sentiment-extraction/
                description.md (130 lines)
                sample_submission.csv (2750 lines)
                ... and 5 other files
                tweet-sentiment-extraction/
        input/
            description.md (130 lines)
            sample_submission.csv (2750 lines)
            sample_submission.csv.zip (18.6 kB)
            test.csv (2750 lines)
            test.csv.zip (114.6 kB)
            train.csv (24733 lines)
            train.csv.zip (1.2 MB)
            tweet-sentiment-extraction/
                description.md (130 lines)
                sample_submission.csv (2750 lines)
                ... and 5 other files
                tweet-sentiment-extraction/
        working/
            tweet-sentiment-extraction/
                description.md (130 lines)
                sample_submission.csv (2750 lines)
                ... and 5 other files
                tweet-sentiment-extraction/
```

-> data/sample_submission.csv has 2749 rows and 2 columns.
The columns are: textID, selected_text

-> data/test.csv has 2749 rows and 3 columns.
The columns are: textID, text, sentiment

-> data/train.csv has 24732 rows and 4 columns.
The columns are: textID, text, selected_text, sentiment

-> data/tweet-sentiment-extraction/sample_submission.csv has 2749 rows and 2 columns.
The columns are: textID, selected_text

-> data/tweet-sentiment-extraction/test.csv has 2749 rows and 3 columns.
The columns are: textID, text, sentiment

-> data/tweet-sentiment-extraction/train.csv has 24732 rows and 4 columns.
The columns are: textID, text, selected_text, sentiment

-> input/sample_submission.csv has 2749 rows and 2 columns.
The columns are: textID, selected_text

-> (stopped after 10 files for performance)

# 5. Target score

0.36886

# 6. Current score

0.30124

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.29179) has done: 'Diagnosis: The crash happens when importing TensorFlow, and the traceback (`'MessageFactory' object has no attribute 'GetPrototype'`) is a known incompatibility between TensorFlow 2.18 and the installed `protobuf==6.33.0`. TensorFlow 2.18 expects protobuf 4.x/5.x APIs, so protobuf 6 breaks the import at runtime. Since we cannot change installed packages, the minimal deterministic workaround is to force TensorFlow to use the pure-Python protobuf implementation, which avoids the C++ API mismatch that triggers this error.

Patch summary: Modify only cell 1 by setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` (and version `2`) *before* importing TensorFlow. This keeps the core logic unchanged and unblocks TensorFlow import so later cells can run.

Updated cells: Only cell 1 is changed.

Compatibility notes for cell k+1: `tf` and `plt` still be defined exactly as expected for subsequent cells; no interface or variable names change.

Assumptions: Setting the protobuf implementation to `python` is permitted in this environment and does not require any additional packages or file changes; any small performance difference is acceptable since it only affects protobuf parsing during imports/graph serialization, not the model architecture or training semantics.'
- What this solution (achieved 0.29659) has done: 'Diagnosis: The crash happens while importing TensorFlow in cell 1 due to an incompatibility between TensorFlow 2.18 and the installed `protobuf` (6.x). Setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` does not resolve this particular breakage, and the error `'MessageFactory' object has no attribute 'GetPrototype'` is a known symptom of the protobuf major-version mismatch. The minimal deterministic fix is to force TensorFlow to use the pure-Python protobuf runtime *and* ensure TensorFlow imports after those environment variables are set, without changing any model/training logic.

Patch summary: Update cell 1 to set the protobuf env vars *before* importing TensorFlow, and add `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` plus disable C++ fast-path via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=2` (kept) while also setting `TF_CPP_MIN_LOG_LEVEL` to avoid noisy logs. No other cells are changed.

Updated cells: Only cell 1 is modified below.

Compatibility notes for cell k+1: `tf` and `plt` are still imported under the same names, so downstream cells referencing TensorFlow/Matplotlib remain compatible.

Assumptions: The environment allows using the pure-Python protobuf implementation via environment variables (no need to downgrade packages), and later cells rely on `tensorflow as tf` being importable with the same API.'
- What this solution (achieved 0.29644) has done: 'The crash happens during `import tensorflow as tf` because TensorFlow 2.18.0 is incompatible with the installed `protobuf==6.33.0`, triggering `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. Your attempted workaround (`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python"`) does not fix this incompatibility in this environment. The minimal deterministic fix is to downgrade `protobuf` at runtime to a TensorFlow-compatible version (<=4.25.x) before importing TensorFlow, then import TensorFlow normally. This change is localized to the failing cell and preserves all downstream variables (`tf`, `plt`) exactly as expected by later cells.'
- What this solution (achieved 0.30124) has done: 'Diagnosis: The crash is caused by using `dataset[field][index]` inside a loop over `enumerate(dataset[field])`. After filtering rows (`data = data[pd.notnull(...)]`), the DataFrame/Series index is no longer a 0..n-1 RangeIndex, so `dataset[field][index]` performs label-based lookup and can raise `KeyError` (e.g., missing label 17674).  
Patch summary: In `clean_text`, switch to position-based scalar assignment via `.iloc` so the loop index matches positional rows regardless of the Series’ index labels. Also make the `str.replace` calls explicit with `regex=False/True` to keep behavior stable under pandas 2.x and avoid unintended regex interpretation.  
Updated cells: Only cell 9 is modified.  
Compatibility notes for cell k+1: `data` remains a DataFrame with the same columns (`text`, `selected_text`) cleaned in-place; downstream access like `data.text[34]` continues to work (and now won’t have been interrupted by the earlier KeyError).  
Assumptions: The intent is to clean the text values in-place without changing row order or dropping additional rows; maintaining the loop-based per-row cleanup is acceptable as core logic.'

# 9. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import os

import sys
import subprocess

try:
    import google.protobuf  # noqa: F401
    from google.protobuf import __version__ as _pb_version
except Exception:
    _pb_version = None


def _pb_major(ver):
    try:
        return int(str(ver).split(".")[0])
    except Exception:
        return None


if _pb_major(_pb_version) is not None and _pb_major(_pb_version) >= 6:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
    )
    if "google.protobuf" in sys.modules:
        del sys.modules["google.protobuf"]

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
import matplotlib.pyplot as plt


## === cell 3
data = pd.read_csv("../input/tweet-sentiment-extraction/train.csv")


## === cell 4
data.head()


## === cell 5
data = data[pd.notnull(data.selected_text)]


## === cell 6
data


## === cell 7
sum(data.text.str.startswith("http"))


## === cell 8
data.text[24]


## === cell 9
import string
import re


def clean_text(dataset, field):
    dataset[field] = dataset[field].str.lower()
    dataset[field] = dataset[field].str.replace("'", "", regex=False)

    for i, strin in enumerate(dataset[field].tolist()):
        val = strin.strip()
        val = re.sub(r"^https?:\/\/.*[\r\n]*", "", val)
        dataset[field].iloc[i] = val

    dataset[field] = dataset[field].str.replace(
        "[{}]".format(string.punctuation), "", regex=True
    )


clean_text(data, "text")
clean_text(data, "selected_text")


## === cell 10
data.text[34]


## === cell 11
sum(data.text.str.startswith("http"))


## === cell 12
from sklearn.model_selection import train_test_split

train, validation = train_test_split(data, test_size = 0.25)
print(len(data), len(train), len(validation))


## === cell 13
train.head()


## === cell 14
validation.head()


## === cell 15
vocab_size = 4000
embedding_dim = 16
max_length = 50
trunc_type='post'
pad_type='post'
oov_tok = ""

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train.text)
word_index = tokenizer.word_index


## === cell 16
word_index


## === cell 17
training_sequences = tokenizer.texts_to_sequences(np.array(train.text))
training_padded = pad_sequences(training_sequences,truncating=trunc_type, padding=pad_type)

max_length = len(training_padded[0])

validation_sequences = tokenizer.texts_to_sequences(np.array(validation.text))
validation_padded = pad_sequences(validation_sequences, padding=pad_type, maxlen = max_length)


## === cell 18
training_selected_sequences = tokenizer.texts_to_sequences(np.array(train.selected_text))
validation_selected_sequences = tokenizer.texts_to_sequences(np.array(validation.selected_text))


## === cell 19
training_padded[4]


## === cell 20
training_selected_sequences[4]


## === cell 21
def get_list(padded, sequence):
    return np.array([1 if x in sequence else 0 for x in padded])


## === cell 22
get_list(training_padded[4], training_selected_sequences[4])


## === cell 23
train_y = np.array([get_list(i,j) for i,j in zip(training_padded, training_selected_sequences)])
validate_y = np.array([get_list(i,j) for i,j in zip(validation_padded, validation_selected_sequences)])


## === cell 24
train_y


## === cell 25
np.array(train.sentiment).shape


## === cell 26
training_padded.shape


## === cell 27
rev_word_index = {v: k for k, v in word_index.items()}


## === cell 28
def get_phrase(index): 
    return np.array([rev_word_index[i] for i in train_x[index][train_y.astype(bool)[index]]])


## === cell 29
train_x = np.copy(training_padded)
validate_x = np.copy(validation_padded)


## === cell 30
len(np.array(train.sentiment))


## === cell 31
len(training_padded)


## === cell 32
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout



## === cell 33
plt.style.use('dark_background')

def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  


## === cell 39
print(training_padded[(train.sentiment == "positive")].shape)
print(training_padded[(train.sentiment == "neutral")].shape)
training_padded[(train.sentiment == "negative")].shape


## === cell 40
train_positive_x = training_padded[(train.sentiment == "positive")]
train_neutral_x = training_padded[(train.sentiment == "neutral")]
train_negative_x = training_padded[(train.sentiment == "negative")]
train_positive_y = train_y[(train.sentiment == "positive")]
train_neutral_y = train_y[(train.sentiment == "neutral")]
train_negative_y = train_y[(train.sentiment == "negative")]

validate_positive_x = validation_padded[(validation.sentiment == "positive")]
validate_neutral_x = validation_padded[(validation.sentiment == "neutral")]
validate_negative_x = validation_padded[(validation.sentiment == "negative")]
validate_positive_y = validate_y[(validation.sentiment == "positive")]
validate_neutral_y = validate_y[(validation.sentiment == "neutral")]
validate_negative_y = validate_y[(validation.sentiment == "negative")]


## === cell 41
positive_model = Sequential()
positive_model.add(Embedding(vocab_size, 16, input_length=max_length))
positive_model.add(Dropout(0.5))
positive_model.add(Bidirectional(tf.keras.layers.LSTM(embedding_dim)))
positive_model.add(Dropout(0.3))
positive_model.add(Dense(max_length * 2, activation='sigmoid'))
positive_model.add(Dense(max_length, activation='sigmoid'))
positive_model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.001), metrics=['accuracy'])
positive_history = positive_model.fit(train_positive_x, train_positive_y, epochs=20, verbose=2,
                   validation_data = (validate_positive_x, validate_positive_y))


## === cell 42
plot_graphs(positive_history, "accuracy")
plot_graphs(positive_history, "loss")


## === cell 43
neutral_model = Sequential()
neutral_model.add(Embedding(vocab_size, 16, input_length=max_length))
neutral_model.add(Bidirectional(tf.keras.layers.GRU(32)))
neutral_model.add(Dropout(0.5))
neutral_model.add(Dense(max_length, activation='sigmoid'))
neutral_model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.001), metrics=['accuracy'])
neutral_history = neutral_model.fit(train_neutral_x, train_neutral_y, epochs=25, verbose=2,
                   validation_data = (validate_neutral_x, validate_neutral_y))


## === cell 44
plot_graphs(neutral_history, "accuracy")
plot_graphs(neutral_history, "loss")


## === cell 45
print(train_negative_x.shape)
print(max_length)


## === cell 46
negative_model = Sequential()
negative_model.add(Embedding(vocab_size, 16, input_length=max_length))
negative_model.add(Dropout(0.5))
negative_model.add(Bidirectional(tf.keras.layers.GRU(32)))
negative_model.add(Dropout(0.5))
negative_model.add(Dense(max_length, activation='sigmoid'))
negative_model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.001), metrics=['accuracy'])
negative_history = negative_model.fit(train_negative_x, train_negative_y, epochs=25, verbose=2,
                   validation_data = (validate_negative_x, validate_negative_y))


## === cell 47
plot_graphs(negative_history, "accuracy")
plot_graphs(negative_history, "loss")


## === cell 48
test = pd.read_csv("../input/tweet-sentiment-extraction/test.csv")


## === cell 49
test


## === cell 50
clean_text(test, 'text')


## === cell 51
data.selected_text[34]


## === cell 52
test_sequences = tokenizer.texts_to_sequences(np.array(test.text))
test_padded = pad_sequences(test_sequences,truncating=trunc_type, maxlen = max_length,padding=pad_type)


## === cell 53
def get_phrase(array_x, array_y, index): 
    return np.array([rev_word_index[i] for i in array_x[index][array_y.astype(bool)[index]]])


## === cell 54
for i,j in enumerate(validate_positive_y[:10]):
    print(get_phrase(validate_negative_x, validate_negative_y, i))


## === cell 55
test_padded[2].astype(bool).astype(int)[np.newaxis]


## === cell 56
preds = []
for index, item in enumerate(test_padded):
    if test.sentiment[index] == "positive":
        p = np.round(positive_model.predict(item[np.newaxis]))
        preds.append(p)
    elif test.sentiment[index] == "negative":
        p = np.round(negative_model.predict(item[np.newaxis]))
        preds.append(p)
    else:
        preds.append(test_padded[index].astype(bool).astype(int)[np.newaxis])


## === cell 57
preds


## === cell 58
x = np.round(neutral_model.predict((training_padded[3])[np.newaxis]))


## === cell 59
np.array([rev_word_index[i] for i in train_x[3][x.astype(bool)[0]]if i != 0])


## === cell 60
list(train.text)[3]


## === cell 61
list(train.selected_text)[3]


## === cell 62
list(train.sentiment)[3]


## === cell 63
preds[0]


## === cell 64
def get_phrase(array_x, array_y, index): 
    return np.array([rev_word_index[i] for i in array_x[index][array_y.astype(bool)[index][0]]if i != 0])


## === cell 65
preds[2][0]


## === cell 66
test_padded[2][np.array(preds).astype(bool)[2][0]]


## === cell 67
[rev_word_index[i] for i in test_padded[2][np.array(preds).astype(bool)[2][0]] if i != 0]


## === cell 68
test["prediction"] = np.zeros(len(test))

for index, item in enumerate(preds):
    test['prediction'][index] = str(get_phrase(test_padded, np.array(preds), index))


## === cell 69
test[test.sentiment == "neutral"]


## === cell 70
test.prediction = test.prediction.str.replace("[", "")
test.prediction = test.prediction.str.replace("]", "")
test.prediction = test.prediction.str.replace("'", "")
test.prediction = test.prediction.str.replace("", "")


## === cell 71
test[:25]


## === cell 72
evaluation = test.textID.copy().to_frame()


## === cell 73
evaluation['selected_text'] = test['prediction']
evaluation


## === cell 74
evaluation.to_csv("submission.csv", index=False)


## === cell 75
test.text[34]


## === cell 76
test.prediction[34]


## === cell 77
data.text[56]


## === cell 78
data.selected_text[56]


## === cell 79
data[:30]


## === cell 80
train[train.sentiment == "neutral"]


## === cell 81
equals = [i==j for i,j in zip(train.text[train.sentiment == "neutral"], train.selected_text[train.sentiment == "neutral"])]


## === cell 82
equals


## === cell 83
len(train.text[train.sentiment == "neutral"])


## === cell 84
equals


## === cell 85
list(train.selected_text[train.sentiment == "neutral"])[1] == list(train.text[train.sentiment == "neutral"])[1]


## === cell 86
list(train.text[train.sentiment == "neutral"])[1]


## === cell 87
list(train.selected_text[train.sentiment == "neutral"])[1]
