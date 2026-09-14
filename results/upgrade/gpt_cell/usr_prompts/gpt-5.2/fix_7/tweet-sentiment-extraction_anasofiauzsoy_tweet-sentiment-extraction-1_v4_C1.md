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

0.43922

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
- What this solution (achieved 0.43922) has done: 'Diagnosis: The crash originates in cell 62 when calling `get_phrase(test_padded, preds_arr, index)`. `preds_arr` is an `object` array mixing `None` and `(1, max_length)` numpy arrays, so `preds_arr.astype(bool)` triggers ambiguous truth-value conversions and results in invalid indexing; this then cascades into “setting an array element with a sequence.” The fix is to avoid `astype(bool)` on the mixed `object` array and instead extract the per-row prediction safely, handling `None` and squeezing the `(1, max_length)` array into a 1D boolean mask.

Patch summary: Update only cell 62 to build `pred_texts` by checking `preds[index]` directly, converting non-None predictions into a 1D boolean mask (`pred_mask`) and using it to select token ids from `test_padded[index]`. This preserves the existing model outputs, rounding behavior, and “neutral uses full raw text” logic, while making indexing deterministic and valid.

Updated cells: cell 62 only.

Compatibility notes for cell k+1: Cell 63 expects `test` to exist and `test["prediction"]` to be created; the patch keeps `test` as a DataFrame and still sets `test["prediction"]` exactly as before.

Assumptions: For non-neutral rows, `preds[index]` is either `None` or a numpy array shaped `(1, max_length)` with 0/1 values (from `np.round(model.predict(...))`); `rev_word_index` may not contain index 0 and token id 0 should be skipped as padding.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
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



## === cell 2
data = pd.read_csv("../input/tweet-sentiment-extraction/train.csv")



## === cell 3
data.head()



## === cell 4
data = data[pd.notnull(data.selected_text)]



## === cell 5
data



## === cell 6
sum(data.text.str.startswith("http"))



## === cell 7
data.text[24]



## === cell 8
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



## === cell 9
data.text[34]



## === cell 10
sum(data.text.str.startswith("http"))



## === cell 11
from sklearn.model_selection import train_test_split

train, validation = train_test_split(data, test_size=0.25)
print(len(data), len(train), len(validation))



## === cell 12
train.head()



## === cell 13
validation.head()



## === cell 14
vocab_size = 4000
embedding_dim = 16
max_length = 50
trunc_type = "post"
pad_type = "post"
oov_tok = ""

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train.text)
word_index = tokenizer.word_index



## === cell 15
word_index



## === cell 16
training_sequences = tokenizer.texts_to_sequences(np.array(train.text))
training_padded = pad_sequences(
    training_sequences, truncating=trunc_type, padding=pad_type
)

max_length = len(training_padded[0])

validation_sequences = tokenizer.texts_to_sequences(np.array(validation.text))
validation_padded = pad_sequences(
    validation_sequences, padding=pad_type, maxlen=max_length
)



## === cell 17
training_selected_sequences = tokenizer.texts_to_sequences(
    np.array(train.selected_text)
)
validation_selected_sequences = tokenizer.texts_to_sequences(
    np.array(validation.selected_text)
)



## === cell 18
training_padded[4]



## === cell 19
training_selected_sequences[4]




## === cell 20
def get_list(padded, sequence):
    return np.array([1 if x in sequence else 0 for x in padded])




## === cell 21
get_list(training_padded[4], training_selected_sequences[4])



## === cell 22
train_y = np.array(
    [get_list(i, j) for i, j in zip(training_padded, training_selected_sequences)]
)
validate_y = np.array(
    [get_list(i, j) for i, j in zip(validation_padded, validation_selected_sequences)]
)



## === cell 23
train_y



## === cell 24
np.array(train.sentiment).shape



## === cell 25
training_padded.shape



## === cell 26
rev_word_index = {v: k for k, v in word_index.items()}




## === cell 27
def get_phrase(index):
    return np.array(
        [rev_word_index[i] for i in train_x[index][train_y.astype(bool)[index]]]
    )




## === cell 28
train_x = np.copy(training_padded)
validate_x = np.copy(validation_padded)



## === cell 29
len(np.array(train.sentiment))



## === cell 30
len(training_padded)



## === cell 31
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout



## === cell 32
plt.style.use("dark_background")


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history["val_" + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, "val_" + string])
    plt.show()




## === cell 33
print(training_padded[(train.sentiment == "positive")].shape)
print(training_padded[(train.sentiment == "neutral")].shape)
training_padded[(train.sentiment == "negative")].shape



## === cell 34
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



## === cell 35
positive_model = Sequential()
positive_model.add(Embedding(vocab_size, 16, input_length=max_length))
positive_model.add(Dropout(0.5))
positive_model.add(Bidirectional(tf.keras.layers.LSTM(embedding_dim)))
positive_model.add(Dropout(0.3))
positive_model.add(Dense(max_length * 2, activation="sigmoid"))
positive_model.add(Dense(max_length, activation="sigmoid"))
positive_model.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=["accuracy"],
)
positive_history = positive_model.fit(
    train_positive_x,
    train_positive_y,
    epochs=20,
    verbose=2,
    validation_data=(validate_positive_x, validate_positive_y),
)



## === cell 36
plot_graphs(positive_history, "accuracy")
plot_graphs(positive_history, "loss")



## === cell 37
neutral_model = Sequential()
neutral_model.add(Embedding(vocab_size, 16, input_length=max_length))
neutral_model.add(Bidirectional(tf.keras.layers.GRU(32)))
neutral_model.add(Dropout(0.5))
neutral_model.add(Dense(max_length, activation="sigmoid"))
neutral_model.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=["accuracy"],
)
neutral_history = neutral_model.fit(
    train_neutral_x,
    train_neutral_y,
    epochs=25,
    verbose=2,
    validation_data=(validate_neutral_x, validate_neutral_y),
)



## === cell 38
plot_graphs(neutral_history, "accuracy")
plot_graphs(neutral_history, "loss")



## === cell 39
print(train_negative_x.shape)
print(max_length)



## === cell 40
negative_model = Sequential()
negative_model.add(Embedding(vocab_size, 16, input_length=max_length))
negative_model.add(Dropout(0.5))
negative_model.add(Bidirectional(tf.keras.layers.GRU(32)))
negative_model.add(Dropout(0.5))
negative_model.add(Dense(max_length, activation="sigmoid"))
negative_model.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=["accuracy"],
)
negative_history = negative_model.fit(
    train_negative_x,
    train_negative_y,
    epochs=25,
    verbose=2,
    validation_data=(validate_negative_x, validate_negative_y),
)



## === cell 41
plot_graphs(negative_history, "accuracy")
plot_graphs(negative_history, "loss")



## === cell 42
test = pd.read_csv("../input/tweet-sentiment-extraction/test.csv")



## === cell 43
test



## === cell 44
test["text_raw"] = test["text"].astype(str)

clean_text(test, "text")



## === cell 45
data.selected_text[34]



## === cell 46
test_sequences = tokenizer.texts_to_sequences(np.array(test.text))
test_padded = pad_sequences(
    test_sequences, truncating=trunc_type, maxlen=max_length, padding=pad_type
)




## === cell 47
def get_phrase(array_x, array_y, index):
    return np.array(
        [rev_word_index[i] for i in array_x[index][array_y.astype(bool)[index]]]
    )




## === cell 48
for i, j in enumerate(validate_positive_y[:10]):
    print(get_phrase(validate_negative_x, validate_negative_y, i))



## === cell 49
test_padded[2].astype(bool).astype(int)[np.newaxis]



## === cell 50
preds = []
for index, item in enumerate(test_padded):
    if test.sentiment[index] == "positive":
        p = np.round(positive_model.predict(item[np.newaxis], verbose=0))
        preds.append(p)
    elif test.sentiment[index] == "negative":
        p = np.round(negative_model.predict(item[np.newaxis], verbose=0))
        preds.append(p)
    else:
        preds.append(None)



## === cell 51
preds



## === cell 52
x = np.round(neutral_model.predict((training_padded[3])[np.newaxis], verbose=0))



## === cell 53
np.array([rev_word_index[i] for i in train_x[3][x.astype(bool)[0]] if i != 0])



## === cell 54
list(train.text)[3]



## === cell 55
list(train.selected_text)[3]



## === cell 56
list(train.sentiment)[3]



## === cell 57
preds[0]




## === cell 58
def get_phrase(array_x, array_y, index):
    return np.array(
        [
            rev_word_index[i]
            for i in array_x[index][array_y.astype(bool)[index][0]]
            if i != 0
        ]
    )




## === cell 59
preds[2][0] if preds[2] is not None else None



## === cell 60
(
    test_padded[2][np.array(preds, dtype=object).astype(bool)[2][0]]
    if preds[2] is not None
    else None
)



## === cell 61
(
    [
        rev_word_index[i]
        for i in test_padded[2][np.array(preds, dtype=object).astype(bool)[2][0]]
        if i != 0
    ]
    if preds[2] is not None
    else None
)



## === cell 62
pred_texts = []

for index in range(len(test)):
    if test.sentiment.iloc[index] == "neutral":
        pred_texts.append(test["text_raw"].iloc[index])
    else:
        pred = preds[index]
        if pred is None:
            phrase_tokens = []
        else:
            pred_mask = np.asarray(pred).astype(bool).squeeze()
            token_ids = test_padded[index][pred_mask]
            phrase_tokens = [rev_word_index[i] for i in token_ids if i != 0]
        pred_texts.append(" ".join(list(phrase_tokens)).strip())

test["prediction"] = pred_texts


## === cell 63
test[test.sentiment == "neutral"]



## === cell 64
test.prediction = test.prediction.fillna("").astype(str).str.strip()



## === cell 65
test[:25]



## === cell 66
evaluation = test.textID.copy().to_frame()



## === cell 67
evaluation["selected_text"] = test["prediction"]
evaluation



## === cell 68
evaluation.to_csv("submission.csv", index=False)



## === cell 69
test.text[34]



## === cell 70
test.prediction[34]



## === cell 71
data.text[56]



## === cell 72
data.selected_text[56]



## === cell 73
data[:30]



## === cell 74
train[train.sentiment == "neutral"]



## === cell 75
equals = [
    i == j
    for i, j in zip(
        train.text[train.sentiment == "neutral"],
        train.selected_text[train.sentiment == "neutral"],
    )
]



## === cell 76
equals



## === cell 77
len(train.text[train.sentiment == "neutral"])



## === cell 78
equals



## === cell 79
list(train.selected_text[train.sentiment == "neutral"])[1] == list(
    train.text[train.sentiment == "neutral"]
)[1]



## === cell 80
list(train.text[train.sentiment == "neutral"])[1]



## === cell 81
list(train.selected_text[train.sentiment == "neutral"])[1]
