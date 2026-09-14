# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Given a dataset of comments from Wikipedia's talk page edits, predict the probability of each comment being toxic.

## Metric
Mean column-wise ROC AUC; the average of the individual AUCs of each predicted column.

## Submission Format
For each `id` in the test set, you must predict a probability for each of the six possible types of comment toxicity (toxic, severe_toxic, obscene, threat, insult, identity_hate). The columns must be in the same order as shown below. The file should contain a header and have the following format:

```
id,toxic,severe_toxic,obscene,threat,insult,identity_hate
00001cee341fdb12,0.5,0.5,0.5,0.5,0.5,0.5
0000247867823ef7,0.5,0.5,0.5,0.5,0.5,0.5
etc.
```

## Dataset 
- **train.csv** - the training set, contains comments with their binary labels
- **test.csv** - the test set, you must predict the toxicity probabilities for these comments.
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.7

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (69 lines)
            sample_submission.csv (153165 lines)
            sample_submission.csv.zip (1.5 MB)
            test.csv (552889 lines)
            test.csv.zip (24.6 MB)
            train.csv (561809 lines)
            train.csv.zip (27.7 MB)
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
        input/
            description.md (69 lines)
            sample_submission.csv (153165 lines)
            sample_submission.csv.zip (1.5 MB)
            test.csv (552889 lines)
            test.csv.zip (24.6 MB)
            train.csv (561809 lines)
            train.csv.zip (27.7 MB)
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
        working/
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
```

-> data/jigsaw-toxic-comment-classification-challenge/sample_submission.csv has 153164 rows and 7 columns.
The columns are: id, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/jigsaw-toxic-comment-classification-challenge/test.csv has 552888 rows and 2 columns.
The columns are: id, comment_text

-> data/jigsaw-toxic-comment-classification-challenge/train.csv has 561808 rows and 8 columns.
The columns are: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/sample_submission.csv has 153164 rows and 7 columns.
The columns are: id, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/test.csv has 552888 rows and 2 columns.
The columns are: id, comment_text

-> data/train.csv has 561808 rows and 8 columns.
The columns are: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D
from tensorflow.keras.layers import GRU, Embedding, Bidirectional
from tensorflow.keras.layers import Dropout, SpatialDropout1D
from tensorflow.keras.models import Model

print("TensorFlow:", tf.__version__)
print("Listing ../input:")
print(os.listdir("../input"))

np.random.seed(42)
tf.random.set_seed(42)

try:
    n_threads = max(1, os.cpu_count() or 1)
    tf.config.threading.set_intra_op_parallelism_threads(n_threads)
    tf.config.threading.set_inter_op_parallelism_threads(max(1, n_threads // 2))
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")




## === cell 1
MAX_SEQUENCE_LENGTH = 200
MAX_VOCAB_SIZE = 20000

VALIDATION_SPLIT = 0.2
EMBEDDING_DIM = 300
BATCH_SIZE = 1000
EPOCHS = 10




## === cell 2
base_candidates = [
    "../input/jigsaw-toxic-comment-classification-challenge",
    "../input",  # some environments place files directly here
]


def first_existing(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


train_data_path = first_existing(
    os.path.join(base_candidates[0], "train.csv"),
    os.path.join(base_candidates[1], "train.csv"),
)
test_data_path = first_existing(
    os.path.join(base_candidates[0], "test.csv"),
    os.path.join(base_candidates[1], "test.csv"),
)
sample_sub_path = first_existing(
    os.path.join(base_candidates[0], "sample_submission.csv"),
    os.path.join(base_candidates[1], "sample_submission.csv"),
)

if train_data_path is None or test_data_path is None or sample_sub_path is None:
    raise FileNotFoundError(
        f"Could not locate required CSVs. Found train={train_data_path}, test={test_data_path}, sample={sample_sub_path}"
    )

glove_path = first_existing(
    os.path.join("../input", "glove6b", f"glove.6B.{EMBEDDING_DIM}d.txt"),
    os.path.join("../input", "glove.6B", f"glove.6B.{EMBEDDING_DIM}d.txt"),
    os.path.join("../input", f"glove.6B.{EMBEDDING_DIM}d.txt"),
)

print("train_data_path:", train_data_path)
print("test_data_path:", test_data_path)
print("sample_sub_path:", sample_sub_path)
print("glove_path:", glove_path)




## === cell 3
print("loading word2vec... (skipped; using deterministic random init)")
word2vec = {}
print("number of vectors : 0")




## === cell 4
possible_labels = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

train_usecols = ["comment_text"] + possible_labels
test_usecols = ["id", "comment_text"]

train_dtypes = {c: "int8" for c in possible_labels}
train_dtypes["comment_text"] = "object"
test_dtypes = {"id": "object", "comment_text": "object"}

train_data = pd.read_csv(train_data_path, usecols=train_usecols, dtype=train_dtypes)
test_data = pd.read_csv(test_data_path, usecols=test_usecols, dtype=test_dtypes)

print("train shape:", train_data.shape)
print("test shape:", test_data.shape)




## === cell 5
sentences = train_data["comment_text"].fillna("DUMMY_VALUES").values
targets = train_data[possible_labels].values




## === cell 6
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)




## === cell 7
word_index = tokenizer.word_index
print("unique tokens:", len(word_index))
print("word_index type:", type(word_index))




## === cell 8
seqs = tokenizer.texts_to_sequences(sentences)
data = pad_sequences(
    seqs,
    maxlen=MAX_SEQUENCE_LENGTH,
    dtype="int32",
    padding="pre",
    truncating="pre",
    value=0,
)

print("shape of data {0}".format(data.shape))




## === cell 9
print("Filling embeddings...")

num_words = min(MAX_VOCAB_SIZE, len(word_index) + 1)

rng = np.random.RandomState(42)
embedding_matrix = rng.normal(
    loc=0.0, scale=0.05, size=(num_words, EMBEDDING_DIM)
).astype("float32")
embedding_matrix[0] = 0.0  # padding token

print("shape of embedding matrix is {0}".format(embedding_matrix.shape))




## === cell 10
embedding_layer = Embedding(
    num_words,
    EMBEDDING_DIM,
    weights=[embedding_matrix],
    input_length=MAX_SEQUENCE_LENGTH,
    trainable=False,
)




## === cell 11
print("Building the Model...")




## === cell 12
input_ = Input(shape=(MAX_SEQUENCE_LENGTH,))

x = embedding_layer(input_)

x = Bidirectional(GRU(50, return_sequences=True))(x)

x = SpatialDropout1D(0.1)(x)

x = GlobalMaxPooling1D()(x)

x = Dense(128, activation="relu")(x)

x = Dropout(0.2)(x)

output = Dense(len(possible_labels), activation="sigmoid")(x)




## === cell 13
model = Model(input_, output)
model.summary()




## === cell 14
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])




## === cell 15
n = data.shape[0]
val_size = int(n * VALIDATION_SPLIT)
train_size = n - val_size

x_train = data[:train_size]
y_train = targets[:train_size]
x_val = data[train_size:]
y_val = targets[train_size:]

shuffle_buf = min(20000, train_size)

options = tf.data.Options()
options.experimental_deterministic = True

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).with_options(options)
train_ds = train_ds.shuffle(
    buffer_size=shuffle_buf, seed=42, reshuffle_each_iteration=True
)
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

val_ds = (
    tf.data.Dataset.from_tensor_slices((x_val, y_val))
    .with_options(options)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(tf.data.AUTOTUNE)
)

print("Training Model...")
r = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    verbose=2,
)




## === cell 16
test_ids = test_data["id"].values
test_texts = test_data["comment_text"].fillna("DUMMY_VALUES").values

print("Tokenizing + padding test texts (vectorized)...")
test_seqs = tokenizer.texts_to_sequences(test_texts)
x_test = pad_sequences(
    test_seqs,
    maxlen=MAX_SEQUENCE_LENGTH,
    dtype="int32",
    padding="pre",
    truncating="pre",
    value=0,
)

print("Predicting...")
predict = model.predict(x_test, batch_size=8192, verbose=0).astype(
    np.float32, copy=False
)

submission = pd.DataFrame(predict, columns=possible_labels)
submission.insert(0, "id", test_ids)

os.makedirs("../working", exist_ok=True)
out_path = "../working/submission.csv"
submission.to_csv(out_path, index=False)
submission.to_csv("submission.csv", index=False)

print("Wrote submission files:")
print(" -", out_path, "shape:", submission.shape)
print(" - submission.csv shape:", submission.shape)
print(submission.head())
