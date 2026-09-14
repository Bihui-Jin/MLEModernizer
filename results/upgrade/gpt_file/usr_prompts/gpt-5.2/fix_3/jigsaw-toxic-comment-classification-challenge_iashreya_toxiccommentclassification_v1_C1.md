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

# 5. Target score

0.4998176292104164

# 6. Current score

0.88585

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.88585) has done: 'I fix the import/runtime issues by switching from the broken standalone `keras` import to `tensorflow.keras`, and by consolidating all required imports (including `tqdm`, `random`, `matplotlib`) into the first cell so later cells don’t fail with `NameError`. I remove the dependency on the missing GloVe file by falling back to randomly-initialized embeddings while keeping the same Embedding+CNN model architecture and training loops intact. I also fix a major logic bug: the tokenizer must be fit on the training text and then reused on the test text (the original code incorrectly refit a new tokenizer on test). Finally, I ensure the submission contains probabilities (no 0/1 thresholding) in the exact required column order and is written to `submission.csv`.'
- What this solution (achieved 0.88585) has done: 'I fix the TensorFlow import crash happening at startup (`MessageFactory` / protobuf incompatibility) by forcing TensorFlow to use the pure-Python protobuf implementation before importing `tensorflow`. This change is score-neutral (it doesn’t alter model logic) but unblocks the entire pipeline so training/inference can run end-to-end and write `submission.csv`. I also keep seeds and paths intact, and make no modeling/training changes since your current score (0.88585) is already far above the target band and the priority is correctness/stability.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import random
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Embedding,
    Flatten,
    Conv1D,
    MaxPooling1D,
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tqdm.auto import tqdm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
TRAIN_PATH = "../input/jigsaw-toxic-comment-classification-challenge/train.csv"
TEST_PATH = "../input/jigsaw-toxic-comment-classification-challenge/test.csv"
SAMPLE_SUB_PATH = (
    "../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv"
)

if not os.path.exists(TRAIN_PATH):
    TRAIN_PATH = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv"
    TEST_PATH = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv"
    SAMPLE_SUB_PATH = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv"



## === cell 2
training_set = pd.read_csv(TRAIN_PATH)



## === cell 3
training_set = training_set.drop(["id"], axis=1)



## === cell 4
print("Number of training records :", len(training_set))
print("Columns :")
for i in training_set.columns:
    print("\t" + i)



## === cell 5
columns = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

count_ones = []
for i in columns:
    count_ones.append(training_set[training_set[i] == 1][i].count())
y_pos = np.arange(len(columns))
plt.figure(figsize=(10, 3))
plt.bar(y_pos, count_ones, align="center", alpha=0.5)
plt.xticks(y_pos, columns)
plt.ylabel("Number of Ones")
plt.title("Number of Ones")
plt.tight_layout()
plt.close()

count_zeros = []
for i in columns:
    count_zeros.append(training_set[training_set[i] == 0][i].count())
y_pos = np.arange(len(columns))
plt.figure(figsize=(10, 3))
plt.bar(y_pos, count_zeros, align="center", alpha=0.5)
plt.xticks(y_pos, columns)
plt.ylabel("Number of Zeros")
plt.title("Number of Zeros")
plt.tight_layout()
plt.close()



## === cell 6
for i in range(1):
    j = random.randint(0, min(10000, len(training_set) - 1))
    print(training_set.values[j])



## === cell 7
EMBED_DIM = 300
glove_path = "../input/glove-embeddings/glove.6B.300d.txt"

embedding_matrix = None
glove_embeddings = {}

if os.path.exists(glove_path):
    with open(glove_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in tqdm(f, desc="Loading GloVe"):
            temp = line.rstrip().split(" ")
            word = temp[0]
            embeds = np.asarray(temp[1:], dtype="float32")
            glove_embeddings[word] = embeds
    print("Loaded GloVe vectors:", len(glove_embeddings))
else:
    print(
        "GloVe file not found; using randomly initialized embeddings (trainable=False as in original)."
    )



## === cell 8
x = training_set["comment_text"].fillna("").astype(str)
y = training_set[columns].astype(np.float32)



## === cell 9
MAX_FEATURES = 20000
MAXLEN = 40

token = Tokenizer(num_words=MAX_FEATURES)
token.fit_on_texts(x)
seq = token.texts_to_sequences(x)

padded_seq = pad_sequences(seq, maxlen=MAXLEN)

vocab_size = len(token.word_index) + 1
print("vocab_size:", vocab_size)



## === cell 10
embeddings = np.random.normal(0, 0.05, size=(vocab_size, EMBED_DIM)).astype(np.float32)
embeddings[0] = 0.0  # padding token

if glove_embeddings:
    for word, i in tqdm(token.word_index.items(), desc="Building embedding matrix"):
        if i >= vocab_size:
            continue
        vec = glove_embeddings.get(word)
        if vec is not None and vec.shape[0] == EMBED_DIM:
            embeddings[i] = vec




## === cell 11
def build_model(vocab_size, embeddings, maxlen=40, embed_dim=300):
    model = Sequential()
    model.add(
        Embedding(
            vocab_size,
            embed_dim,
            weights=[embeddings],
            input_length=maxlen,
            trainable=False,
        )
    )
    model.add(Conv1D(128, 5, activation="relu"))
    model.add(MaxPooling1D(5))
    model.add(Conv1D(128, 5, activation="relu"))
    model.add(MaxPooling1D(3))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="Adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model




## === cell 12
model1 = build_model(vocab_size, embeddings, maxlen=MAXLEN, embed_dim=EMBED_DIM)
model1.summary()



## === cell 13
model1.fit(
    padded_seq,
    training_set["toxic"].values,
    epochs=3,
    batch_size=32,
    validation_split=0.2,
    verbose=2,
)



## === cell 14
model2 = build_model(vocab_size, embeddings, maxlen=MAXLEN, embed_dim=EMBED_DIM)
model2.summary()



## === cell 15
model2.fit(
    padded_seq,
    training_set["severe_toxic"].values,
    epochs=2,
    batch_size=32,
    validation_split=0.2,
    verbose=2,
)



## === cell 16
model3 = build_model(vocab_size, embeddings, maxlen=MAXLEN, embed_dim=EMBED_DIM)
model3.summary()



## === cell 17
model3.fit(
    padded_seq,
    training_set["obscene"].values,
    epochs=2,
    batch_size=32,
    validation_split=0.2,
    verbose=2,
)



## === cell 18
model4 = build_model(vocab_size, embeddings, maxlen=MAXLEN, embed_dim=EMBED_DIM)
model4.summary()



## === cell 19
model4.fit(
    padded_seq,
    training_set["threat"].values,
    epochs=1,
    batch_size=32,
    validation_split=0.2,
    verbose=2,
)



## === cell 20
model5 = build_model(vocab_size, embeddings, maxlen=MAXLEN, embed_dim=EMBED_DIM)
model5.summary()



## === cell 21
model5.fit(
    padded_seq,
    training_set["insult"].values,
    epochs=2,
    batch_size=32,
    validation_split=0.2,
    verbose=2,
)



## === cell 22
model6 = build_model(vocab_size, embeddings, maxlen=MAXLEN, embed_dim=EMBED_DIM)
model6.summary()



## === cell 23
model6.fit(
    padded_seq,
    training_set["identity_hate"].values,
    epochs=1,
    batch_size=32,
    validation_split=0.2,
    verbose=2,
)



## === cell 24
test_set = pd.read_csv(TEST_PATH)



## === cell 25
x_test = test_set["comment_text"].fillna("").astype(str)
test_seq = token.texts_to_sequences(x_test)
test_padded_seq = pad_sequences(test_seq, maxlen=MAXLEN)



## === cell 26
toxic = model1.predict(test_padded_seq, batch_size=1024, verbose=0).reshape(-1)
severe_toxic = model2.predict(test_padded_seq, batch_size=1024, verbose=0).reshape(-1)
obscene = model3.predict(test_padded_seq, batch_size=1024, verbose=0).reshape(-1)
threat = model4.predict(test_padded_seq, batch_size=1024, verbose=0).reshape(-1)
insult = model5.predict(test_padded_seq, batch_size=1024, verbose=0).reshape(-1)
identity_hate = model6.predict(test_padded_seq, batch_size=1024, verbose=0).reshape(-1)



## === cell 27
toxic = np.clip(toxic.astype(np.float32), 0.0, 1.0)
severe_toxic = np.clip(severe_toxic.astype(np.float32), 0.0, 1.0)
obscene = np.clip(obscene.astype(np.float32), 0.0, 1.0)
threat = np.clip(threat.astype(np.float32), 0.0, 1.0)
insult = np.clip(insult.astype(np.float32), 0.0, 1.0)
identity_hate = np.clip(identity_hate.astype(np.float32), 0.0, 1.0)



## === cell 28
ids = test_set["id"].astype(str).values

df = pd.DataFrame(
    {
        "id": ids,
        "toxic": toxic,
        "severe_toxic": severe_toxic,
        "obscene": obscene,
        "threat": threat,
        "insult": insult,
        "identity_hate": identity_hate,
    }
)

df = df[["id"] + columns]
print(df.head())
print("submission shape:", df.shape)



## === cell 29
df.to_csv("submission.csv", index=False)
print("Wrote submission.csv")
