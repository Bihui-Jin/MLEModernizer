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

3.7

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
import numpy as np  # linear algebra
import pandas as pd  # data processing
import os
import time

print(os.listdir("../input"))



## === cell 1
if "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION" in os.environ:
    pass

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split

np.random.seed(42)
tf.random.set_seed(42)

try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass



## === cell 2
train = pd.read_csv(
    "../input/movie-review-sentiment-analysis-kernels-only/train.tsv", sep="\t"
)
test = pd.read_csv(
    "../input/movie-review-sentiment-analysis-kernels-only/test.tsv", sep="\t"
)
sub = pd.read_csv(
    "../input/movie-review-sentiment-analysis-kernels-only/sampleSubmission.csv",
    sep=",",
)



## === cell 3
x = train["Phrase"].astype(str)
y = train["Sentiment"].astype(int)



## === cell 4
embedding_dim = 300
maxlen = 300
glove_path = "../input/gloveembeddings/glove.6B.300d.txt"

MAX_VOCAB = 60000

vectorize = tf.keras.layers.TextVectorization(
    max_tokens=MAX_VOCAB,
    standardize=None,
    split="whitespace",
    output_mode="int",
    output_sequence_length=maxlen,
)

t0 = time.time()
vectorize.adapt(tf.data.Dataset.from_tensor_slices(x.values).batch(4096))
print("Vectorizer adapt seconds:", round(time.time() - t0, 3))

vocab = vectorize.get_vocabulary()  # index 0 reserved for padding, 1 for OOV
vocab_size = len(vocab)
print(vocab_size)

y = to_categorical(y, num_classes=5)



## === cell 5
t0 = time.time()
pad_seq = vectorize(x.values).numpy().astype(np.int32, copy=False)
print("Train vectorization seconds:", round(time.time() - t0, 3))



## === cell 6
t0 = time.time()

effective_vocab = min(vocab_size, MAX_VOCAB)
embedding_matrix = np.zeros((vocab_size, embedding_dim), dtype=np.float32)

inv_idx = {}
for idx in range(2, effective_vocab):
    w = vocab[idx]
    if w:
        inv_idx[w] = idx

needed_total = len(inv_idx)

if os.path.exists(glove_path):
    found = 0
    remaining = needed_total
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(" ", 1)
            if len(parts) != 2:
                continue
            w, rest = parts
            idx = inv_idx.get(w)
            if idx is None:
                continue
            vec = np.fromstring(rest, sep=" ", dtype=np.float32)
            if vec.shape[0] != embedding_dim:
                continue
            embedding_matrix[idx] = vec
            found += 1
            remaining -= 1
            if remaining == 0:
                break
    print(f"Loaded {found} in-vocab GloVe vectors (filtered, early-exit).")
else:
    print(f"GloVe file not found at: {glove_path}")
    print(
        "Proceeding with randomly initialized embedding matrix (trainable=False as in original)."
    )
    rng = np.random.RandomState(42)
    embedding_matrix[1:] = rng.normal(
        loc=0.0, scale=0.05, size=(vocab_size - 1, embedding_dim)
    ).astype(np.float32)

print("Embedding prep seconds:", round(time.time() - t0, 3))



## === cell 7
x_train, x_test, y_train, y_test = train_test_split(
    pad_seq, y, test_size=0.3, random_state=42, stratify=np.argmax(y, axis=1)
)



## === cell 8
model = Sequential()



## === cell 9
model.add(
    Embedding(
        vocab_size,
        embedding_dim,
        input_length=maxlen,
        weights=[embedding_matrix],
        trainable=False,
    )
)



## === cell 10
model.add(LSTM(75))



## === cell 11
model.add(Dense(128, activation="relu"))



## === cell 12
model.add(Dense(5, activation="softmax"))



## === cell 13
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])



## === cell 14
batch_size = 32

shuffle_buf = min(len(x_train), 20000)

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.shuffle(
    buffer_size=shuffle_buf, seed=42, reshuffle_each_iteration=True
)
train_ds = train_ds.batch(batch_size, drop_remainder=False)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
val_ds = (
    val_ds.batch(batch_size, drop_remainder=False).cache().prefetch(tf.data.AUTOTUNE)
)

history = model.fit(
    train_ds,
    epochs=4,
    validation_data=val_ds,
    verbose=2,
)



## === cell 15
_ = test.head(0)



## === cell 16
pass



## === cell 17
_ = test.head(0)



## === cell 18
testing_phrase = test["Phrase"].astype(str)



## === cell 19
t0 = time.time()
pad_test_seq = vectorize(testing_phrase.values).numpy().astype(np.int32, copy=False)
print("Test vectorization seconds:", round(time.time() - t0, 3))



## === cell 20
test_ds = (
    tf.data.Dataset.from_tensor_slices(pad_test_seq)
    .batch(1024)
    .prefetch(tf.data.AUTOTUNE)
)
proba = model.predict(test_ds, verbose=0)
predict = np.argmax(proba, axis=1).astype(int)



## === cell 21
predict[0]



## === cell 22
test["Sentiment"] = predict



## === cell 23
_ = test.head(0)



## === cell 24
submission = test[["PhraseId", "Sentiment"]].copy()
submission = submission.sort_values("PhraseId").reset_index(drop=True)



## === cell 25
_ = submission.head(0)



## === cell 26
submission.to_csv("submission.csv", index=False)
print("Wrote submission to submission.csv with shape:", submission.shape)
print(submission.dtypes)
