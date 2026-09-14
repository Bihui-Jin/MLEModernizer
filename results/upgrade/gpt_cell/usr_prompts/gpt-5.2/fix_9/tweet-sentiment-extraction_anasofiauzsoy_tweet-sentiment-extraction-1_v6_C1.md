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

0.54102

# 6. Current score

0.41097

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.40049) has done: 'Diagnosis: The crash happens on `import tensorflow as tf` in cell 1, and the traceback (`'MessageFactory' object has no attribute 'GetPrototype'`) is a known incompatibility between TensorFlow 2.18 and `protobuf>=5` (you have `protobuf==6.33.0`). TensorFlow expects the Python implementation of protobuf and/or an older protobuf API, so importing TF triggers the failure. The minimal deterministic fix is to force protobuf to use the pure-Python implementation **before** importing TensorFlow, which avoids the missing method in the C++ backend. This change stays localized to cell 1 and preserves all downstream interfaces (the `tf` symbol still exists for cell 3+).

Patch summary: In cell 1, set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` (and version=2 for stability) via `os.environ` before importing TensorFlow. Keep the rest of the imports unchanged.

Updated cells: Only cell 1 is modified.

Compatibility notes for cell k+1: Cell 3 and later cells still have `tf` available exactly as expected; this patch only changes protobuf backend selection at import-time and does not alter model code or training semantics.

Assumptions: The environment allows setting `os.environ` at runtime and TensorFlow 2.18 functions correctly with the Python protobuf implementation in this container.'
- What this solution (achieved 0.39419) has done: 'Diagnosis: The crash happens while importing TensorFlow in cell 1 due to an incompatibility between TensorFlow 2.18.0 and the installed `protobuf==6.33.0`; TensorFlow expects protobuf’s C++ implementation and APIs that are not present/compatible in this setup. The current workaround forces the pure-Python protobuf implementation, which actually makes the incompatibility surface as `MessageFactory.GetPrototype` missing. The minimal fix is to avoid forcing the Python protobuf implementation and instead force the C++ implementation so TensorFlow can use the compatible fast path.

Patch summary: In cell 1, change the environment variables to request protobuf’s `cpp` implementation (and do not pin a Python implementation version) before importing TensorFlow. Keep the rest of the imports and logic unchanged so downstream cells still work the same.

Updated cells: Only cell 1 is modified.

Compatibility notes for cell k+1: TensorFlow (`tf`) and `matplotlib.pyplot as plt` remain defined exactly as before, so cell 3 (and later cells) can continue to use them without interface changes.

Assumptions: The runtime has the protobuf C++ backend available (common in Kaggle-like images); if it does not, TensorFlow import would still fail, but this is the safest deterministic fix given the installed package set and the observed error.'
- What this solution (achieved 0.40134) has done: 'The crash happens while importing TensorFlow because the cell forces protobuf to use the C++ (“cpp”) implementation, but the installed `protobuf==6.33.0` in this environment does not provide the required `google.protobuf.pyext._message` module, causing `ImportError`. The minimal fix is to stop forcing the C++ protobuf runtime and instead explicitly select the pure-Python protobuf implementation, which is compatible and avoids the missing extension. This change is localized to cell 1 and preserves the rest of the notebook’s logic and interfaces (`tf` and `plt` still import the same). No other cells need modification.'
- What this solution (achieved 0.39419) has done: 'The crash happens during `import tensorflow as tf` because TensorFlow’s generated protobuf code expects the C++ protobuf runtime API (`MessageFactory.GetPrototype`), but your environment is forcing the pure-Python protobuf implementation via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python"`, which is incompatible with the installed `protobuf==6.33.0`. The minimal fix is to stop forcing the Python protobuf runtime and instead force the C++ runtime (`"cpp"`), while keeping the rest of the cell unchanged. This resolves the missing method and allows TensorFlow to import normally. No other cells are modified, and `tf`/`plt` remain defined for subsequent cells.'
- What this solution (achieved 0.40061) has done: 'Diagnosis: The crash happens in cell 1 when importing TensorFlow because the notebook forces `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="cpp"`, which makes protobuf try to load the C++ extension (`google.protobuf.pyext._message`) that is not available in this environment, resulting in `ImportError: cannot import name '_message'`. TensorFlow 2.18 works with the default Python protobuf runtime here, so forcing the C++ implementation is unnecessary and breaks the import.  

Patch summary: In cell 1, stop forcing the protobuf C++ backend and instead explicitly select the Python implementation (or leave it unset) before importing TensorFlow, ensuring TensorFlow imports successfully without changing any model/training logic.  

Updated cells: Only cell 1 is modified.  

Compatibility notes for cell k+1: `tf` and `plt` are still imported with the same names and remain available to subsequent cells; no interfaces or variables used later are changed.  

Assumptions: This environment lacks the protobuf C++ extension module (`google.protobuf.pyext._message`), and using the Python protobuf implementation is acceptable and compatible with TensorFlow 2.18.'
- What this solution (achieved 0.39879) has done: 'Diagnosis: The crash happens during `import tensorflow as tf` in cell 1 due to an incompatibility between TensorFlow 2.18.0 and the installed `protobuf==6.33.0`, which triggers `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. The attempted environment overrides (`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`) don’t resolve this because the newer protobuf API removed/changed `MessageFactory.GetPrototype` that TensorFlow expects. The minimal fix is to pin protobuf to a TensorFlow-compatible major version (typically `<5`) at runtime before importing TensorFlow, then proceed with the same imports.

Patch summary: In cell 1 only, install a compatible protobuf version (e.g., `<5`) using pip, then restart-import protobuf/TensorFlow in-process. Keep the existing environment variable lines and the rest of the cell logic unchanged, only adding the minimal pip/install + import ordering needed to prevent the TensorFlow import crash.

Updated cells: Only cell 1 is changed.

Compatibility notes for cell k+1: This fix preserves `tf` and `plt` being imported and available exactly as before, so later cells that expect TensorFlow to be imported work unchanged.

Assumptions: The runtime allows `pip` installs (as typical in notebook environments) and there are no other pinned dependencies preventing protobuf from being downgraded to a TensorFlow-compatible version.'
- What this solution (achieved 0.40037) has done: 'Diagnosis: The crash happens because `get_phrase` (defined in cell 39) only accepts a single argument (`index`), but `check_negative` calls it with three arguments (`validate_negative_x, np.array(val_neg_preds), index`). Since we must not edit earlier non-buggy cells, the minimal fix is to change `check_negative` to call `get_phrase` with only the `index` it expects. This preserves the existing function signatures and avoids altering model logic or training/inference semantics; it only fixes the incorrect call.

Patch summary: Update `check_negative` in cell 65 to call `get_phrase(index)` instead of passing extra arguments.

Updated cells: cell 65 only.

Compatibility notes for cell k+1: Cell 66 (`validate_negative_x`) is unaffected; all variables remain defined as before.

Assumptions: `get_phrase(index)` is intended to use the global `train_x/train_y` it references (as currently implemented), and the purpose of `check_negative` is debugging/printing rather than strict evaluation.'
- What this solution (achieved 0.41097) has done: 'I make three minimal changes aimed at improving Jaccard without changing your model architecture or training loops: (1) fix the text cleaning bug so it actually cleans non-empty strings (your current `if not strin:` prevents cleaning almost all rows), (2) align the loss with your `softmax` output by switching `BinaryCrossentropy(from_logits=True)` to `from_logits=False` (same loss family, but correct semantics), and (3) make prediction decoding select the best contiguous span from the model’s softmax probabilities instead of rounding independent positions (this keeps the same model outputs, but post-processes them in a way that better matches word-level Jaccard). These are small, localized edits that preserve your overall approach (tokenize → padded sequence → per-position selection via LSTM) while typically boosting score from the ~0.40 range toward your 0.54 target.'

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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import sys
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "protobuf<5"])

import tensorflow as tf
import matplotlib.pyplot as plt



## === cell 2
data = pd.read_csv("../input/tweet-sentiment-extraction/train.csv")



## === cell 3
data.head()



## === cell 4
print(len(data.text), len(data.textID))



## === cell 5
data.text[data.textID == "a88287bbda"]



## === cell 6
print(len(data.text), len(data.textID))



## === cell 7
sum(data.text.str.startswith("http"))



## === cell 8
data.text[24]



## === cell 9
data.text[data.textID == "a88287bbda"]



## === cell 10
len(data["text"])



## === cell 11
import string
import re


def clean_text(dataset, field):
    print(len(dataset[field]))
    for index, strin in enumerate(dataset[field].astype(str).tolist()):
        if strin is None:
            strin = ""
        strin = strin.lower()
        strin = strin.replace("\n", " ")
        strin = strin.strip()
        strin = re.sub(r"^https?:\/\/.*[\r\n]*", "", strin)
        strin = re.sub(rf"[{re.escape(string.punctuation)}]", "", strin)
        dataset.at[dataset.index[index], field] = strin


clean_text(data, "text")
clean_text(data, "selected_text")



## === cell 12
print(len(data.text), len(data.textID))



## === cell 13
data.text[24]



## === cell 14
len(data.text)



## === cell 15
len(data.textID)



## === cell 16
data.text[data.textID == "a88287bbda"]



## === cell 17
data = data[pd.notnull(data.selected_text)]



## === cell 18
print(data.text[data.textID == "a88287bbda"])
print(len(data.text), len(data.textID))



## === cell 19
from sklearn.model_selection import train_test_split

train, validation = train_test_split(data, test_size=0.25, random_state=42)
print(len(data), len(train), len(validation))



## === cell 20
train



## === cell 21
validation.head()



## === cell 22
print(train.text[train.textID == "a88287bbda"])



## === cell 23
list(data.text[data.textID == "a88287bbda"])



## === cell 24
vocab_size = 10000
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



## === cell 25
word_index



## === cell 26
training_sequences = tokenizer.texts_to_sequences(np.array(train.text))
training_padded = pad_sequences(
    training_sequences, truncating=trunc_type, padding=pad_type
)

max_length = len(training_padded[0])

validation_sequences = tokenizer.texts_to_sequences(np.array(validation.text))
validation_padded = pad_sequences(
    validation_sequences, padding=pad_type, maxlen=max_length
)



## === cell 27
training_selected_sequences = tokenizer.texts_to_sequences(
    np.array(train.selected_text)
)
validation_selected_sequences = tokenizer.texts_to_sequences(
    np.array(validation.selected_text)
)



## === cell 28
training_padded[4]



## === cell 29
training_selected_sequences[4]




## === cell 30
def get_list(padded, sequence):
    return np.array([1 if x in sequence else 0 for x in padded])




## === cell 31
get_list(training_padded[4], training_selected_sequences[4])



## === cell 32
train_y = np.array(
    [get_list(i, j) for i, j in zip(training_padded, training_selected_sequences)]
)
validate_y = np.array(
    [get_list(i, j) for i, j in zip(validation_padded, validation_selected_sequences)]
)



## === cell 33
train_y



## === cell 34
np.array(train.sentiment).shape



## === cell 35
training_padded.shape



## === cell 36
rev_word_index = {v: k for k, v in word_index.items()}




## === cell 37
def get_phrase(index):
    return np.array(
        [rev_word_index[i] for i in train_x[index][train_y.astype(bool)[index]]]
    )




## === cell 38
train_x = np.copy(training_padded)
validate_x = np.copy(validation_padded)



## === cell 39
len(np.array(train.sentiment))



## === cell 40
len(training_padded)



## === cell 41
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout



## === cell 42
plt.style.use("dark_background")


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history["val_" + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, "val_" + string])
    plt.show()




## === cell 43
print(training_padded[(train.sentiment == "positive")].shape)
print(training_padded[(train.sentiment == "neutral")].shape)
training_padded[(train.sentiment == "negative")].shape



## === cell 44
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



## === cell 45
l = tf.keras.losses.BinaryCrossentropy(from_logits=False)



## === cell 46
positive_model = Sequential()
positive_model.add(Embedding(vocab_size, 16, input_length=max_length))
positive_model.add(Dropout(0.5))
positive_model.add(Bidirectional(LSTM(20)))
positive_model.add(Dropout(0.5))
positive_model.add(Dense(max_length, activation="softmax"))
positive_model.compile(
    loss=l, optimizer=tf.keras.optimizers.Adam(0.001), metrics=["accuracy"]
)
positive_history = positive_model.fit(
    train_positive_x.astype(float),
    train_positive_y.astype(float),
    epochs=40,
    verbose=2,
    validation_data=(
        validate_positive_x.astype(float),
        validate_positive_y.astype(float),
    ),
)



## === cell 47
plot_graphs(positive_history, "accuracy")
plot_graphs(positive_history, "loss")



## === cell 48
print(train_negative_x.shape)
print(max_length)



## === cell 49
negative_model = Sequential()
negative_model.add(Embedding(vocab_size, 16, input_length=max_length))
negative_model.add(Dropout(0.5))
negative_model.add(Bidirectional(LSTM(20)))
negative_model.add(Dropout(0.5))
negative_model.add(Dense(max_length, activation="softmax"))
negative_model.compile(
    loss=l, optimizer=tf.keras.optimizers.Adam(0.001), metrics=["accuracy"]
)
negative_history = negative_model.fit(
    train_negative_x.astype(float),
    train_negative_y.astype(float),
    epochs=40,
    verbose=2,
    validation_data=(
        validate_negative_x.astype(float),
        validate_negative_y.astype(float),
    ),
)



## === cell 50
train_negative_x.shape



## === cell 51
train_negative_y[0].shape



## === cell 52
train_negative_y



## === cell 53
plot_graphs(negative_history, "accuracy")
plot_graphs(negative_history, "loss")



## === cell 54
val_neg_preds = [
    np.round(negative_model.predict(item[np.newaxis], verbose=0))
    for item in validate_negative_x
]




## === cell 55
def check_negative(index):
    print(list(validation.text[validation.sentiment == "negative"])[index])
    print(get_phrase(index))


check_negative(7)



## === cell 56
validate_negative_x



## === cell 57
validate_negative_y



## === cell 58
np.array(val_neg_preds)



## === cell 59
test = pd.read_csv("../input/tweet-sentiment-extraction/test.csv")



## === cell 60
test



## === cell 61
clean_text(test, "text")



## === cell 62
test



## === cell 63
test_sequences = tokenizer.texts_to_sequences(np.array(test.text))
test_padded = pad_sequences(
    test_sequences, truncating=trunc_type, maxlen=max_length, padding=pad_type
)




## === cell 64
def get_phrase(array_x, array_y, index):
    return np.array(
        [rev_word_index[i] for i in array_x[index][array_y.astype(bool)[index]]]
    )




## === cell 65
for i, j in enumerate(validate_positive_y[:10]):
    print(get_phrase(validate_negative_x, validate_negative_y, i))



## === cell 66
test_padded[2].astype(bool).astype(int)[np.newaxis]




## === cell 67
def best_span_from_probs(x_tokens, probs, min_len=1):
    """
    x_tokens: (max_length,) token ids (padded with 0)
    probs: (max_length,) probabilities from softmax
    Returns boolean mask (max_length,) for the chosen contiguous span.
    """
    x_tokens = np.asarray(x_tokens)
    probs = np.asarray(probs)

    valid_pos = np.where(x_tokens != 0)[0]
    if len(valid_pos) == 0:
        return np.zeros_like(x_tokens, dtype=bool)

    lo, hi = valid_pos[0], valid_pos[-1]
    p = probs[lo : hi + 1]
    n = len(p)

    p_center = p - p.mean()
    best_sum = -1e18
    best_i = 0
    best_j = 0
    cur_sum = 0.0
    cur_i = 0
    for j in range(n):
        if cur_sum <= 0:
            cur_sum = p_center[j]
            cur_i = j
        else:
            cur_sum += p_center[j]
        if cur_sum > best_sum:
            best_sum = cur_sum
            best_i = cur_i
            best_j = j

    if best_j - best_i + 1 < min_len:
        k = int(np.argmax(p))
        best_i, best_j = k, k

    mask = np.zeros_like(x_tokens, dtype=bool)
    mask[lo + best_i : lo + best_j + 1] = True
    return mask




## === cell 68
preds = []
for index, item in enumerate(test_padded):
    sent = test.sentiment.iloc[index]
    if sent == "positive":
        prob = positive_model.predict(item[np.newaxis], verbose=0)[0]  # (max_length,)
        mask = best_span_from_probs(item, prob)[np.newaxis, :]  # (1, max_length)
        preds.append(mask.astype(int))
    elif sent == "negative":
        prob = negative_model.predict(item[np.newaxis], verbose=0)[0]
        mask = best_span_from_probs(item, prob)[np.newaxis, :]
        preds.append(mask.astype(int))
    else:
        preds.append(item.astype(bool).astype(int)[np.newaxis])



## === cell 69
preds



## === cell 70
list(train.text)[3]



## === cell 71
list(train.selected_text)[3]



## === cell 72
list(train.sentiment)[3]



## === cell 73
data



## === cell 74
preds[0]




## === cell 75
def get_phrase(array_x, array_y, index):
    return np.array(
        [
            rev_word_index[i]
            for i in array_x[index][array_y.astype(bool)[index][0]]
            if i != 0
        ]
    )




## === cell 76
preds[2][0]



## === cell 77
test_padded[2][np.array(preds).astype(bool)[2][0]]



## === cell 78
[
    rev_word_index[i]
    for i in test_padded[2][np.array(preds).astype(bool)[2][0]]
    if i != 0
]



## === cell 79
test["prediction"] = np.zeros(len(test), dtype=object)

for index, item in enumerate(preds):
    test.at[test.index[index], "prediction"] = str(
        get_phrase(test_padded, np.array(preds), index)
    )



## === cell 80
test.prediction[test.sentiment == "neutral"] = test.text[test.sentiment == "neutral"]



## === cell 81
test[test.sentiment == "neutral"]



## === cell 82
test.prediction = test.prediction.str.replace("[", "", regex=False)
test.prediction = test.prediction.str.replace("]", "", regex=False)
test.prediction = test.prediction.str.replace("'", "", regex=False)
test.prediction = test.prediction.str.replace("", "", regex=False)



## === cell 83
test[:25]



## === cell 84
evaluation = test.textID.copy().to_frame()



## === cell 85
evaluation["selected_text"] = test["prediction"]
evaluation



## === cell 86
evaluation.to_csv("submission.csv", index=False)



## === cell 87
test.text[34]



## === cell 88
test.prediction[34]



## === cell 89
data.text[56]



## === cell 90
data.selected_text[56]



## === cell 91
data[:30]



## === cell 92
train[train.sentiment == "neutral"]



## === cell 93
equals = [
    i == j
    for i, j in zip(
        train.text[train.sentiment == "neutral"],
        train.selected_text[train.sentiment == "neutral"],
    )
]



## === cell 94
equals



## === cell 95
len(train.text[train.sentiment == "neutral"])



## === cell 96
equals



## === cell 97
list(train.selected_text[train.sentiment == "neutral"])[1] == list(
    train.text[train.sentiment == "neutral"]
)[1]



## === cell 98
list(train.text[train.sentiment == "neutral"])[1]



## === cell 99
list(train.selected_text[train.sentiment == "neutral"])[1]
