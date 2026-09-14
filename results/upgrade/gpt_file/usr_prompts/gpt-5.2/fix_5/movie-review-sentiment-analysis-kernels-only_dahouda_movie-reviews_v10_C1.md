# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

3.9

# 3. Installed packages

No external packages required in the script and installed.

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

# 5. Code solution

## === cell 0
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.max_colwidth", None)

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

seed = 101
np.random.seed(seed)
tf.random.set_seed(seed)

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


print("TensorFlow:", tf.__version__)
print("Devices:", tf.config.list_physical_devices())



## === cell 1
BASE_PATH = "/kaggle/input/movie-review-sentiment-analysis-kernels-only"

train_path = os.path.join(BASE_PATH, "train.tsv.zip")
test_path = os.path.join(BASE_PATH, "test.tsv.zip")
sub_path = os.path.join(BASE_PATH, "sampleSubmission.csv")

train = pd.read_csv(
    train_path, sep="\t", encoding="utf-8", usecols=["Phrase", "Sentiment"]
)
test = pd.read_csv(
    test_path, sep="\t", encoding="utf-8", usecols=["PhraseId", "Phrase"]
)
sub = pd.read_csv(sub_path)

print(train.shape)
print(test.shape)



## === cell 2

pass



## === cell 3
X = train["Phrase"].astype(str)
temp = test["Phrase"].astype(str)

num_classes = train["Sentiment"].nunique()
y = to_categorical(train["Sentiment"].values, num_classes=num_classes).astype("float32")

print("Number of classes:", num_classes)
print("X shape:", X.shape, "y shape:", y.shape)



## === cell 4
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=train["Sentiment"].values, random_state=seed
)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)



## === cell 5
pass



## === cell 6
max_features = 15000
tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
temp_seq = tokenizer.texts_to_sequences(temp)

max_words = 50
X_train_pad = sequence.pad_sequences(X_train_seq, maxlen=max_words).astype(
    "int32", copy=False
)
X_test_pad = sequence.pad_sequences(X_test_seq, maxlen=max_words).astype(
    "int32", copy=False
)
temp_pad = sequence.pad_sequences(temp_seq, maxlen=max_words).astype(
    "int32", copy=False
)

print(X_train_pad.shape, X_test_pad.shape, temp_pad.shape)



## === cell 7
print("X_train_pad dtype/shape:", X_train_pad.dtype, X_train_pad.shape)



## === cell 8
print("X_test_pad dtype/shape:", X_test_pad.dtype, X_test_pad.shape)



## === cell 9
batch_size = 64
epochs = 25


def get_model(max_features_local, embed_dim, embedding_matrix):
    K.clear_session()
    tf.random.set_seed(seed)
    np.random.seed(seed)

    model = Sequential()
    model.add(
        Embedding(
            max_features_local,
            embed_dim,
            input_length=X_train_pad.shape[1],
            weights=[embedding_matrix],
        )
    )

    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.0))

    model.add(Dense(100, activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return model




## === cell 10
embed_dim = 100

word_index = tokenizer.word_index
num_words = min(max_features, len(word_index) + 1)

rng = np.random.default_rng(seed)
embedding_matrix = rng.normal(loc=0.0, scale=0.05, size=(num_words, embed_dim)).astype(
    "float32"
)

max_features_effective = embedding_matrix.shape[0]
print("Using random embedding matrix with shape:", embedding_matrix.shape)



## === cell 11
model = get_model(max_features_effective, embed_dim, embedding_matrix)

AUTOTUNE = tf.data.AUTOTUNE

options = tf.data.Options()
options.experimental_deterministic = False

train_ds = tf.data.Dataset.from_tensor_slices((X_train_pad, y_train))
shuffle_buf = min(len(X_train_pad), 20000)
train_ds = train_ds.shuffle(
    buffer_size=shuffle_buf, seed=seed, reshuffle_each_iteration=True
)
train_ds = train_ds.batch(batch_size, drop_remainder=False)
train_ds = train_ds.cache()
train_ds = train_ds.prefetch(AUTOTUNE)
train_ds = train_ds.with_options(options)

val_ds = tf.data.Dataset.from_tensor_slices((X_test_pad, y_test))
val_ds = val_ds.batch(batch_size, drop_remainder=False)
val_ds = val_ds.cache()
val_ds = val_ds.prefetch(AUTOTUNE)
val_ds = val_ds.with_options(options)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    verbose=2,
)



## === cell 12
test_options = tf.data.Options()
test_options.experimental_deterministic = False

test_ds = (
    tf.data.Dataset.from_tensor_slices(temp_pad)
    .batch(batch_size)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)
test_ds = test_ds.with_options(test_options)

probs = model.predict(test_ds, verbose=0)
preds = np.argmax(probs, axis=1).astype(int)

submission = pd.DataFrame({"PhraseId": test["PhraseId"].values, "Sentiment": preds})
submission.to_csv("submission.csv", index=False)

print(submission.head())



## === cell 13
print("submission.csv rows:", submission.shape[0])
print("Unique sentiments predicted:", sorted(submission["Sentiment"].unique().tolist()))
print(submission.head())
