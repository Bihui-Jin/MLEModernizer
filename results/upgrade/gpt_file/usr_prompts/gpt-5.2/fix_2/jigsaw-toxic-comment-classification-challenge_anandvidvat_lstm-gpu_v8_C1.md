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
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D
from tensorflow.keras.layers import GRU, Embedding, Bidirectional
from tensorflow.keras.layers import Dropout, SpatialDropout1D
from tensorflow.keras.models import Model

import os

print("TensorFlow:", tf.__version__)
print("Listing ../input:")
print(os.listdir("../input"))

np.random.seed(42)
tf.random.set_seed(42)



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
"""
loading word2vectors from GloVe (optional)
Fix: handle missing glove file gracefully so pipeline runs end-to-end.
"""
print("loading word2vec...")

word2vec = {}
if glove_path is not None:
    with open(glove_path, encoding="utf8") as fs:
        for line in fs:
            values = line.rstrip().split(" ")
            word = values[0]
            vec = np.asarray(values[1:], dtype="float32")
            if vec.shape[0] == EMBEDDING_DIM:
                word2vec[word] = vec
    print("number of vectors : {0}".format(len(word2vec)))
else:
    print(
        "GloVe file not found; continuing without pre-trained vectors (random init embeddings)."
    )



## === cell 4
"""
loading training data
"""
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

print("train shape:", train_data.shape)
print("test shape:", test_data.shape)



## === cell 5
sentences = train_data["comment_text"].fillna("DUMMY_VALUES").values
possible_labels = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]
targets = train_data[possible_labels].values



## === cell 6
"""
converting sentences into integer sequences
"""
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)



## === cell 7
word_index = tokenizer.word_index
print("unique tokens:", len(word_index))
print("word_index type:", type(word_index))



## === cell 8
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print("shape of data {0}".format(data.shape))



## === cell 9
print("Filling embeddings...")

num_words = min(MAX_VOCAB_SIZE, len(word_index) + 1)

if len(word2vec) > 0:
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM), dtype="float32")
    for word, i in word_index.items():
        if i < MAX_VOCAB_SIZE:
            embedding_vector = word2vec.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
else:
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
print("Training Model...")
r = model.fit(
    data,
    targets,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=VALIDATION_SPLIT,
    verbose=2,
)



## === cell 16
test_sentences = test_data["comment_text"].fillna("DUMMY_VALUES").values
test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_feed = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

predict = model.predict(test_feed, batch_size=2048, verbose=1)

submission = pd.read_csv(sample_sub_path)

if submission.shape[0] != test_data.shape[0]:
    submission = pd.DataFrame({"id": test_data["id"].values})
    for c in possible_labels:
        submission[c] = 0.5

submission[possible_labels] = predict
submission = submission[["id"] + possible_labels]

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
