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

0.42978

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.42978) has done: 'I fix the TensorFlow import crash caused by the protobuf 6.x incompatibility by pinning the pure-Python protobuf implementation before importing TensorFlow (this is required for TF2.18 in many Kaggle images). Then I fix the text cleaning logic bug (it currently never cleans because it checks `if not strin`) and remove unsafe chained assignment so the preprocessing actually applies, which should improve tokenization and score without changing the model/training core. I also fix the `get_phrase` signature mismatch and make prediction-to-text conversion robust (handle empty selections and keep neutrals as full text), ensuring the pipeline runs end-to-end and writes a valid `submission.csv` with the correct columns. These are minimal, directly relevant fixes that should move the score upward toward the target while preserving the original modeling approach.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(42)
tf.random.set_seed(42)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
sum(data.text.str.startswith("http", na=False))



## === cell 8
data.text.iloc[24]



## === cell 9
data.text[data.textID == "a88287bbda"]



## === cell 10
len(data["text"])



## === cell 11
import string
import re

_punct_re = re.compile(r"[{}]".format(re.escape(string.punctuation)))


def _clean_one(s: str) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return s
    s = str(s).lower()
    s = s.replace("'", "")
    s = s.replace("\n", " ")
    s = s.strip()
    s = re.sub(r"^https?:\/\/.*[\r\n]*", "", s)
    s = _punct_re.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def clean_text(dataset: pd.DataFrame, field: str):
    print(len(dataset[field]))
    dataset[field] = dataset[field].apply(_clean_one)


clean_text(data, "text")
clean_text(data, "selected_text")



## === cell 12
print(len(data.text), len(data.textID))



## === cell 13
data.text.iloc[24]



## === cell 14
len(data.text)



## === cell 15
len(data.textID)



## === cell 16
data.text[data.textID == "a88287bbda"]



## === cell 17
data = data[pd.notnull(data.selected_text)].reset_index(drop=True)



## === cell 18
print(data.text[data.textID == "a88287bbda"])
print(len(data.text), len(data.textID))



## === cell 19
from sklearn.model_selection import train_test_split

train, validation = train_test_split(
    data, test_size=0.25, random_state=42, shuffle=True
)
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
tokenizer.fit_on_texts(train.text.astype(str).values)
word_index = tokenizer.word_index



## === cell 25
word_index



## === cell 26
training_sequences = tokenizer.texts_to_sequences(train.text.astype(str).values)
training_padded = pad_sequences(
    training_sequences, truncating=trunc_type, padding=pad_type
)

max_length = len(training_padded[0])

validation_sequences = tokenizer.texts_to_sequences(validation.text.astype(str).values)
validation_padded = pad_sequences(
    validation_sequences, padding=pad_type, maxlen=max_length
)



## === cell 27
training_selected_sequences = tokenizer.texts_to_sequences(
    train.selected_text.astype(str).values
)
validation_selected_sequences = tokenizer.texts_to_sequences(
    validation.selected_text.astype(str).values
)



## === cell 28
training_padded[4]



## === cell 29
training_selected_sequences[4]




## === cell 30
def get_list(padded, sequence):
    seq_set = set(sequence)
    return np.array(
        [1 if x in seq_set and x != 0 else 0 for x in padded], dtype=np.int32
    )




## === cell 31
get_list(training_padded[4], training_selected_sequences[4])



## === cell 32
train_y = np.array(
    [get_list(i, j) for i, j in zip(training_padded, training_selected_sequences)],
    dtype=np.int32,
)
validate_y = np.array(
    [get_list(i, j) for i, j in zip(validation_padded, validation_selected_sequences)],
    dtype=np.int32,
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
def get_phrase(array_x, array_y, index):
    token_ids = array_x[index]
    mask = array_y.astype(bool)[index]
    chosen = [int(t) for t, m in zip(token_ids, mask) if m and int(t) != 0]
    words = [
        rev_word_index.get(t, "") for t in chosen if rev_word_index.get(t, "") != ""
    ]
    return np.array(words)




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
l = tf.keras.losses.BinaryCrossentropy(from_logits=True)



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
plot_graphs(negative_history, "accuracy")
plot_graphs(negative_history, "loss")



## === cell 51
val_neg_preds = [
    np.round(negative_model.predict(item[np.newaxis], verbose=0))
    for item in validate_negative_x
]


def check_negative(index):
    print(list(validation.text[validation.sentiment == "negative"])[index])
    print(get_phrase(validate_negative_x, np.array(val_neg_preds)[:, 0, :], index))





## === cell 52
test = pd.read_csv("../input/tweet-sentiment-extraction/test.csv")
clean_text(test, "text")



## === cell 53
test_sequences = tokenizer.texts_to_sequences(test.text.astype(str).values)
test_padded = pad_sequences(
    test_sequences, truncating=trunc_type, maxlen=max_length, padding=pad_type
)



## === cell 54
preds = []
for index, item in enumerate(test_padded):
    if test.sentiment.iloc[index] == "positive":
        p = np.round(positive_model.predict(item[np.newaxis], verbose=0))
        preds.append(p[0])  # shape (max_length,)
    elif test.sentiment.iloc[index] == "negative":
        p = np.round(negative_model.predict(item[np.newaxis], verbose=0))
        preds.append(p[0])  # shape (max_length,)
    else:
        preds.append(item.astype(bool).astype(int))

preds = np.array(preds, dtype=np.int32)




## === cell 55
def phrase_to_string(words_arr):
    s = " ".join([w for w in list(words_arr) if w is not None and w != ""]).strip()
    return s


pred_texts = []
for i in range(len(test)):
    if test.sentiment.iloc[i] == "neutral":
        pred_texts.append(test.text.iloc[i])
    else:
        words = get_phrase(test_padded, preds, i)
        s = phrase_to_string(words)
        if s == "":
            s = test.text.iloc[i]
        pred_texts.append(s)

test["prediction"] = pred_texts



## === cell 56
evaluation = test[["textID"]].copy()
evaluation["selected_text"] = test["prediction"].astype(str)

evaluation.to_csv("submission.csv", index=False)

print(evaluation.head())
print("Wrote submission.csv with shape:", evaluation.shape)
