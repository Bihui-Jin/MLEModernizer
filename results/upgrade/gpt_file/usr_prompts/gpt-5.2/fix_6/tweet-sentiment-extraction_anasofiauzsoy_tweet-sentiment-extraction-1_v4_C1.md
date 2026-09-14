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

0.36886

# 6. Current score

0.43247

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.42944) has done: 'I fix the TensorFlow import crash by forcing the Python protobuf runtime (this resolves the `MessageFactory.GetPrototype` issue in this environment) and then keep the same modeling/training logic intact. Next, I fix the pandas `KeyError` in `clean_text` by removing chained/label indexing and using vectorized string operations, which also stabilizes preprocessing for both train and test. Finally, I make prediction/selection text reconstruction more correct (convert token-mask to a contiguous span and map back to the original tweet text), which should legitimately increase the word-level Jaccard toward your target without changing the model architecture or training loops. The script run end-to-end and write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.43009) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by setting the protobuf environment variables *before* importing TensorFlow, and by forcing TensorFlow to use the pure-Python protobuf runtime (this is a known compatibility issue in some Kaggle images). I also make the data paths robust by falling back to `/kaggle/input/...` if the relative `../input/...` path doesn’t exist, without changing any modeling/training logic. Since your current score (0.42944) is already well above the target (0.36886) and within the allowed tolerance band, I avoid any score-altering changes and only address stability/correctness so it runs end-to-end and writes `submission.csv`.'
- What this solution (achieved 0.42776) has done: 'I fix the TensorFlow import crash by setting the protobuf implementation to the pure-Python runtime before importing TensorFlow, and additionally forcing the Python protobuf module (a known workaround for the `MessageFactory.GetPrototype` issue in some Kaggle TF images). I keep the modeling/training/prediction logic identical so the score behavior stays essentially the same (your current score is already within the ±10% band around the target). I also make the data-path fallback a bit more robust (still using the same files) and ensure the submission is always written as `submission.csv` with the required columns. No architecture, training loop, or postprocessing logic is changed beyond import/runtime stability.'
- What this solution (achieved 0.42558) has done: 'I fix the TensorFlow import crash (`MessageFactory` missing `GetPrototype`) by forcing the pure-Python protobuf runtime *and* applying a small compatibility monkey-patch for newer protobuf versions before importing TensorFlow; this is score-neutral and only affects runtime stability. I also adjust the cell numbering to start at 1 (your provided notebook starts at cell 0) while keeping the same execution order and core model/training logic unchanged. Finally, I keep the existing data-path fallback and ensure `submission.csv` is always written with the required `textID, selected_text` columns.'
- What this solution (achieved 0.43247) has done: 'I fix the TensorFlow import crash by applying the protobuf compatibility patch earlier and more completely: in this environment the `MessageFactory.GetPrototype` attribute is accessed on instances, so we need to patch the instance method as well as the class method before importing TensorFlow. I keep your model/training/prediction logic unchanged to avoid moving the score further away from your target (your current score is already above target and within the ±10% band). I also renumber cells to start at 1 (as required) while preserving the original execution order, and keep the same `submission.csv` output with the required columns.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    from google.protobuf import message_factory as _message_factory

    if hasattr(_message_factory, "MessageFactory"):
        _MF = _message_factory.MessageFactory

        if not hasattr(_MF, "GetPrototype") and hasattr(_MF, "GetMessageClass"):
            _MF.GetPrototype = _MF.GetMessageClass

        _inst = _MF()
        if not hasattr(_inst, "GetPrototype") and hasattr(_inst, "GetMessageClass"):
            def _get_prototype(self, desc):
                return self.GetMessageClass(desc)

            _MF.GetPrototype = _get_prototype
except Exception:
    pass

try:
    import google.protobuf.internal.api_implementation as _api_impl

    _api_impl.Type = lambda: "python"
except Exception:
    pass

import numpy as np
import pandas as pd

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:5]:
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
TRAIN_CANDIDATES = [
    "../input/tweet-sentiment-extraction/train.csv",
    "/kaggle/input/tweet-sentiment-extraction/train.csv",
    "/kaggle/input/train.csv",
]
TEST_CANDIDATES = [
    "../input/tweet-sentiment-extraction/test.csv",
    "/kaggle/input/tweet-sentiment-extraction/test.csv",
    "/kaggle/input/test.csv",
]


def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of these paths exist: {paths}")


TRAIN_PATH = _first_existing(TRAIN_CANDIDATES)
TEST_PATH = _first_existing(TEST_CANDIDATES)

data = pd.read_csv(TRAIN_PATH)



## === cell 3
data.head()



## === cell 4
data = data[pd.notnull(data.selected_text)].reset_index(drop=True)



## === cell 5
data



## === cell 6
sum(data.text.str.startswith("http"))



## === cell 7
data.text.iloc[24]



## === cell 8
import string
import re


def clean_text(dataset, field):
    s = dataset[field].astype(str).str.lower()
    s = s.str.replace("'", "", regex=False)
    s = s.str.strip()
    s = s.str.replace(r"^https?:\/\/.*[\r\n]*", "", regex=True)
    s = s.str.replace("[{}]".format(re.escape(string.punctuation)), "", regex=True)
    dataset[field] = s


clean_text(data, "text")
clean_text(data, "selected_text")



## === cell 9
data.text.iloc[34]



## === cell 10
sum(data.text.str.startswith("http"))



## === cell 11
from sklearn.model_selection import train_test_split

train, validation = train_test_split(
    data, test_size=0.25, random_state=42, shuffle=True
)
train = train.reset_index(drop=True)
validation = validation.reset_index(drop=True)
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
    train_x = np.copy(training_padded)
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
from tensorflow.keras.layers import Embedding, Dense, Bidirectional, Dropout



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
pass



## === cell 34
pass



## === cell 35
pass



## === cell 36
pass



## === cell 37
pass



## === cell 38
print(training_padded[(train.sentiment == "positive")].shape)
print(training_padded[(train.sentiment == "neutral")].shape)
training_padded[(train.sentiment == "negative")].shape



## === cell 39
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



## === cell 40
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



## === cell 41
plot_graphs(positive_history, "accuracy")
plot_graphs(positive_history, "loss")



## === cell 42
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



## === cell 43
plot_graphs(neutral_history, "accuracy")
plot_graphs(neutral_history, "loss")



## === cell 44
print(train_negative_x.shape)
print(max_length)



## === cell 45
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



## === cell 46
plot_graphs(negative_history, "accuracy")
plot_graphs(negative_history, "loss")



## === cell 47
test = pd.read_csv(TEST_PATH)



## === cell 48
test.head()



## === cell 49
test_raw = test.copy()

clean_text(test, "text")



## === cell 50
data.selected_text.iloc[34]



## === cell 51
test_sequences = tokenizer.texts_to_sequences(np.array(test.text))
test_padded = pad_sequences(
    test_sequences, truncating=trunc_type, maxlen=max_length, padding=pad_type
)




## === cell 52
def mask_to_span(mask_1d):
    idx = np.where(mask_1d.astype(bool))[0]
    if len(idx) == 0:
        return None
    return int(idx.min()), int(idx.max())


def tokens_from_ids(ids_1d):
    toks = []
    for i in ids_1d:
        if i == 0:
            continue
        w = rev_word_index.get(int(i))
        if w is not None:
            toks.append(w)
    return toks


def reconstruct_selected_text(raw_text, clean_text_str, token_ids, mask_1d):
    span = mask_to_span(mask_1d)
    if span is None:
        return raw_text  # fallback
    lo, hi = span
    chosen_clean_tokens = tokens_from_ids(token_ids[lo : hi + 1])
    if len(chosen_clean_tokens) == 0:
        return raw_text

    raw_tokens = str(raw_text).split()
    raw_clean_tokens = [
        re.sub(
            "[{}]".format(re.escape(string.punctuation)), "", t.lower().replace("'", "")
        )
        for t in raw_tokens
    ]

    target = chosen_clean_tokens
    best_i, best_len = 0, 1
    for i in range(len(raw_clean_tokens)):
        k = 0
        while (
            (i + k < len(raw_clean_tokens))
            and (k < len(target))
            and (raw_clean_tokens[i + k] == target[k])
        ):
            k += 1
        if k > best_len:
            best_len = k
            best_i = i
        if best_len == len(target):
            break

    if best_len <= 1 and len(target) > 1:
        return " ".join(target)

    end = min(best_i + max(best_len, 1), len(raw_tokens))
    return " ".join(raw_tokens[best_i:end])




## === cell 53
def debug_preview(idx=0):
    return test_raw.loc[idx, "text"], test.loc[idx, "text"], test_padded[idx]


debug_preview(0)



## === cell 54
test_padded[2].astype(bool).astype(int)[np.newaxis]



## === cell 55
pred_masks = []
for index, item in enumerate(test_padded):
    sent = test.loc[index, "sentiment"]
    if sent == "positive":
        p = np.round(positive_model.predict(item[np.newaxis], verbose=0))
        pred_masks.append(p[0])
    elif sent == "negative":
        p = np.round(negative_model.predict(item[np.newaxis], verbose=0))
        pred_masks.append(p[0])
    else:
        pred_masks.append(item.astype(bool).astype(int))

pred_masks = np.array(pred_masks)



## === cell 56
pred_masks.shape



## === cell 57
x = np.round(neutral_model.predict((training_padded[3])[np.newaxis], verbose=0))



## === cell 58
np.array([rev_word_index[i] for i in train_x[3][x.astype(bool)[0]] if i != 0])[:20]



## === cell 59
list(train.text)[3]



## === cell 60
list(train.selected_text)[3]



## === cell 61
list(train.sentiment)[3]



## === cell 62
pred_masks[0][:10]



## === cell 63
pred_texts = []
for i in range(len(test_raw)):
    raw_t = test_raw.loc[i, "text"]
    clean_t = test.loc[i, "text"]
    tok_ids = test_padded[i]
    mask = pred_masks[i]
    pred_texts.append(reconstruct_selected_text(raw_t, clean_t, tok_ids, mask))



## === cell 64
pred_texts[0]



## === cell 65
evaluation = test_raw[["textID"]].copy()
evaluation["selected_text"] = pred_texts
evaluation["selected_text"] = evaluation["selected_text"].fillna("").astype(str)

evaluation.head()



## === cell 66
evaluation.to_csv("submission.csv", index=False)



## === cell 67
print(evaluation.shape)
print(evaluation.columns.tolist())
print(pd.read_csv("submission.csv").head())



## === cell 68
pass



## === cell 69
pass



## === cell 70
pass



## === cell 71
pass



## === cell 72
pass



## === cell 73
pass



## === cell 74
pass



## === cell 75
pass



## === cell 76
pass



## === cell 77
pass



## === cell 78
pass



## === cell 79
pass



## === cell 80
pass



## === cell 81
pass



## === cell 82
pass



## === cell 83
pass



## === cell 84
pass



## === cell 85
pass



## === cell 86
pass
