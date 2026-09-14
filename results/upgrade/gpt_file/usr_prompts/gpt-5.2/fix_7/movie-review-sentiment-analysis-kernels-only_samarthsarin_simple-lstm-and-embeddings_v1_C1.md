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

print(os.listdir("../input"))



## === cell 1
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

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
train[["Phrase", "Sentiment"]].head()



## === cell 5
embedding_dim = 300
maxlen = 300
glove_path = "../input/gloveembeddings/glove.6B.300d.txt"

token = Tokenizer()
y = to_categorical(y, num_classes=5)



## === cell 6
token.fit_on_texts(x)



## === cell 7
seqs = token.texts_to_sequences(x)
pad_seq = pad_sequences(seqs, maxlen=maxlen).astype(np.int32, copy=False)



## === cell 8
vocab_size = len(token.word_index) + 1
print(vocab_size)



## === cell 9
embedding_matrix = np.zeros((vocab_size, embedding_dim), dtype=np.float32)

word_index = token.word_index
needed_words = set(word_index.keys())

if os.path.exists(glove_path):
    found = 0
    with open(glove_path, "r", encoding="utf-8", newline="") as f:
        for line in f:
            parts = line.rstrip().split(" ", embedding_dim)
            if len(parts) != embedding_dim + 1:
                continue
            w = parts[0]
            if w not in needed_words:
                continue
            vec = np.fromstring(parts[1], sep=" ", dtype=np.float32)
            if vec.shape[0] != embedding_dim:
                continue
            embedding_matrix[word_index[w]] = vec
            found += 1
    print(f"Loaded {found} in-vocab GloVe vectors (filtered).")
else:
    print(f"GloVe file not found at: {glove_path}")
    print(
        "Proceeding with randomly initialized embedding matrix (trainable=False as in original)."
    )
    rng = np.random.RandomState(42)
    embedding_matrix[1:] = rng.normal(
        loc=0.0, scale=0.05, size=(vocab_size - 1, embedding_dim)
    ).astype(np.float32)



## === cell 10
x_train, x_test, y_train, y_test = train_test_split(
    pad_seq, y, test_size=0.3, random_state=42, stratify=np.argmax(y, axis=1)
)



## === cell 11
model = Sequential()



## === cell 12
model.add(
    Embedding(
        vocab_size,
        embedding_dim,
        input_length=maxlen,
        weights=[embedding_matrix],
        trainable=False,
    )
)



## === cell 13
model.add(LSTM(75))



## === cell 14
model.add(Dense(128, activation="relu"))



## === cell 15
model.add(Dense(5, activation="softmax"))



## === cell 16
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])



## === cell 17
batch_size = 32

data_opts = tf.data.Options()
data_opts.experimental_deterministic = True

train_ds = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .with_options(data_opts)
    .shuffle(
        buffer_size=min(len(x_train), 100_000), seed=42, reshuffle_each_iteration=True
    )
    .batch(batch_size, drop_remainder=False)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)
val_ds = (
    tf.data.Dataset.from_tensor_slices((x_test, y_test))
    .with_options(data_opts)
    .batch(batch_size, drop_remainder=False)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)

history = model.fit(
    train_ds,
    epochs=4,
    validation_data=val_ds,
    verbose=2,
)



## === cell 18
test.head()



## === cell 19
pass



## === cell 20
test.head()



## === cell 21
testing_phrase = test["Phrase"].astype(str)



## === cell 22
test_seqs = token.texts_to_sequences(testing_phrase)
pad_test_seq = pad_sequences(test_seqs, maxlen=maxlen).astype(np.int32, copy=False)



## === cell 23
pred_ds = (
    tf.data.Dataset.from_tensor_slices(pad_test_seq)
    .batch(1024)
    .prefetch(tf.data.AUTOTUNE)
)
proba = model.predict(pred_ds, verbose=0)
predict = np.argmax(proba, axis=1).astype(int)



## === cell 24
predict[0]



## === cell 25
test["Sentiment"] = predict



## === cell 26
test.head()



## === cell 27
submission = test[["PhraseId", "Sentiment"]].copy()

submission = submission.sort_values("PhraseId").reset_index(drop=True)



## === cell 28
submission.head()



## === cell 29
submission.to_csv("submission.csv", index=False)
print("Wrote submission to submission.csv with shape:", submission.shape)
print(submission.dtypes)
