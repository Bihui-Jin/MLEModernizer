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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
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
tqdm==4.67.1

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

0.11932

# 6. Current score

0.36558

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.36035) has done: 'I fix the immediate runtime crash caused by an incompatibility between TensorFlow 2.18 and protobuf 6 by forcing the pure-Python protobuf implementation before TensorFlow imports. Then I correct label/target construction and decoding logic: your model predicts start/end positions over the *token sequence*, but the code currently creates labels over *character indices* and slices strings by character offsets, which harms correctness and score. I keep the same model architecture and training loop, but build y1/y2 over token positions, add safe handling when selected_text isn’t found, and reconstruct predicted spans back into text via tokenizer tokens with punctuation detokenization. These changes are directly aligned with the Jaccard metric and should nudge the score (currently 0.135) upward toward (and likely beyond) the target band without changing the modeling approach.'
- What this solution (achieved 0.36558) has done: 'I fix the immediate runtime crash caused by the TensorFlow 2.18 + protobuf 6 incompatibility by ensuring the pure-Python protobuf implementation is set before *any* TensorFlow-related import, and by importing TensorFlow only after that. I also remove the unconditional `EarlyStopping` callback (it changes convergence behavior and violates your “no early stopping” constraint) while keeping the same model, loss, and training loop otherwise. Finally, I keep your existing token-span label construction and decoding logic intact (score-neutral) and ensure the script always writes a valid `submission.csv` with the correct columns.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

import numpy as np
import pandas as pd
import string
from tqdm import tqdm
import gc
import time




## === cell 1
def clean_data(text):
    regular_punct = list(string.punctuation)
    for punc in regular_punct:
        text = text.replace(punc, f" {punc} ")
    return text


def drop_empty_rows(df):
    nan_value = float("NaN")
    df.replace("", nan_value, inplace=True)
    df.dropna(inplace=True)


def data_process(path):
    train_df = pd.read_csv(os.path.join(path, "train.csv"))
    test_df = pd.read_csv(os.path.join(path, "test.csv"))

    train_df["text"] = train_df["text"].astype(str).str.lower()
    train_df["selected_text"] = train_df["selected_text"].astype(str).str.lower()
    train_df["sentiment"] = train_df["sentiment"].astype(str).str.lower()
    drop_empty_rows(train_df)

    test_df["text"] = test_df["text"].astype(str).str.lower()
    test_df["sentiment"] = test_df["sentiment"].astype(str).str.lower()
    drop_empty_rows(test_df)

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)




## === cell 2
import tensorflow as tf
import tensorflow.keras.backend as K


def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c) + 1e-12)


def jaccard_distance(y_true, y_pred, smooth=100):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Dense,
    Embedding,
    GRU,
    Bidirectional,
    Concatenate,
)
from tensorflow.keras.layers import (
    GlobalMaxPooling1D,
    GlobalAveragePooling1D,
    SpatialDropout1D,
)
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau




## === cell 4
def build_model(params):
    inputs = Input(shape=(params["input_size"],))
    x = Embedding(params["num_words"], params["embed_dim"], trainable=True)(inputs)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(GRU(params["gru_units"], return_sequences=True))(x)
    x = Concatenate()([GlobalMaxPooling1D()(x), GlobalAveragePooling1D()(x)])
    x = Dense(params["gru_units"], activation="relu")(x)
    y1 = Dense(params["input_size"], activation="softmax", name="y1")(x)
    y2 = Dense(params["input_size"], activation="softmax", name="y2")(x)
    model = Model(inputs=inputs, outputs=[y1, y2])
    return model




## === cell 5
params = {}
params["batch_size"] = 1024
params["gru_units"] = 128
params["epochs"] = 100
params["input_size"] = 128
params["embed_dim"] = 128
params["num_words"] = 50000



## === cell 6
dataset_path = "../input/tweet-sentiment-extraction/"
if not os.path.exists(dataset_path):
    dataset_path = "/kaggle/input/tweet-sentiment-extraction/"
    if not os.path.exists(dataset_path):
        dataset_path = "/kaggle/data/tweet-sentiment-extraction/"

train_df, test_df = data_process(dataset_path)



## === cell 7
train_text = train_df["text"].apply(lambda x: clean_data(x)).values
train_selected_text = train_df["selected_text"].apply(lambda x: clean_data(x)).values
train_sentiment = train_df["sentiment"].values



## === cell 8
test_text = test_df["text"].apply(lambda x: clean_data(x)).values
test_sentiment = test_df["sentiment"].values



## === cell 9
train_size = len(train_text)
test_size = len(test_text)
print(f"Train data: {train_size} - Test data: {test_size}")



## === cell 10
tokenizer = text.Tokenizer(
    num_words=params["num_words"], filters="", lower=False, oov_token="<unk>"
)
total_text = list(train_text) + list(test_text)
tokenizer.fit_on_texts(total_text)

id2word = {v: k for k, v in tokenizer.word_index.items()}



## === cell 11
total_train_len = len(train_text)
x_train = []
y1_train = []
y2_train = []

for i in tqdm(range(total_train_len), desc="Building training tensors"):
    text1 = train_text[i].strip()
    text2 = train_selected_text[i].strip()

    full_tokens = text1.split()
    sel_tokens = text2.split()

    start_tok = 0
    end_tok = 0

    found = False
    if len(sel_tokens) > 0 and len(full_tokens) >= len(sel_tokens):
        for s in range(0, len(full_tokens) - len(sel_tokens) + 1):
            if full_tokens[s : s + len(sel_tokens)] == sel_tokens:
                start_tok = s
                end_tok = s + len(sel_tokens) - 1
                found = True
                break

    if not found:
        start_tok = 0
        end_tok = max(0, len(full_tokens) - 1)

    x = tokenizer.texts_to_sequences([text1])[0]

    y1 = np.zeros((len(x),), dtype=np.float32)
    y2 = np.zeros((len(x),), dtype=np.float32)

    start_tok = int(np.clip(start_tok, 0, max(0, len(x) - 1)))
    end_tok = int(np.clip(end_tok, 0, max(0, len(x) - 1)))

    y1[start_tok] = 1.0
    y2[end_tok] = 1.0

    x = sequence.pad_sequences([x], maxlen=params["input_size"])[0]
    y1 = sequence.pad_sequences([y1], maxlen=params["input_size"], dtype="float32")[0]
    y2 = sequence.pad_sequences([y2], maxlen=params["input_size"], dtype="float32")[0]

    x_train.append(x)
    y1_train.append(y1)
    y2_train.append(y2)



## === cell 12
model = build_model(params)
model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(1e-3))
model.summary()

x_train = np.array(x_train, dtype=np.int32)
y1_train = np.array(y1_train, dtype=np.float32)
y2_train = np.array(y2_train, dtype=np.float32)

callbacks = [
    ReduceLROnPlateau(patience=5, monitor="loss", factor=0.1),
]

model.fit(
    x_train,
    [y1_train, y2_train],
    batch_size=params["batch_size"],
    epochs=params["epochs"],
    callbacks=callbacks,
    verbose=1,
)




## === cell 13
def detokenize(tokens):
    ans = " ".join(tokens).strip()
    for punc in list(string.punctuation):
        ans = ans.replace(f" {punc} ", f"{punc}")
        ans = ans.replace(f" {punc}", f"{punc}")
        ans = ans.replace(f"{punc} ", f"{punc}")
    ans = " ".join(ans.split())
    return ans


test_x = tokenizer.texts_to_sequences(test_text)
test_x = sequence.pad_sequences(test_x, maxlen=params["input_size"])
print(len(test_x))

test_y1, test_y2 = model.predict(test_x, batch_size=2048, verbose=1)
print(len(test_y1), len(test_y2))

answers = []
for i in range(len(test_text)):
    start = int(np.argmax(test_y1[i]))
    end = int(np.argmax(test_y2[i]))
    if end < start:
        end = start

    seq = tokenizer.texts_to_sequences([test_text[i].strip()])[0]

    pad_len = max(0, params["input_size"] - len(seq))
    start_u = start - pad_len
    end_u = end - pad_len

    if start_u < 0 or end_u < 0 or start_u >= len(seq) or end_u >= len(seq):
        sel_ids = seq
    else:
        end_u = min(end_u, len(seq) - 1)
        sel_ids = seq[start_u : end_u + 1]

    sel_tokens = [id2word.get(tid, "") for tid in sel_ids if tid != 0]
    sel_tokens = [t for t in sel_tokens if t not in ("", "<unk>")]

    ans = detokenize(sel_tokens) if len(sel_tokens) > 0 else ""
    if ans == "":
        ans = test_text[i].strip()

    answers.append(ans)

submission = pd.read_csv(os.path.join(dataset_path, "sample_submission.csv"))
submission["selected_text"] = answers

submission = submission[["textID", "selected_text"]]
assert len(submission) == len(test_df), "Submission length must match test.csv rows."

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
