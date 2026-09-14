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

3.8

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

0.5264495508318882

# 6. Current score

0.96058

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.96058) has done: 'I fix the environment/import crash by switching from standalone `keras` to `tensorflow.keras`, which avoids the protobuf `MessageFactory` error in Kaggle and restores `Tokenizer` / `pad_sequences`. I also remove the missing GloVe dependency (file not present) by keeping the same Embedding+LSTM architecture but letting the embedding train from scratch, so the pipeline can run end-to-end. I correct the test preprocessing to reuse the *training* tokenizer (not refit on test), and fix submission generation by using `test.csv` ids (not nonexistent `test_labels`) and matching the sample submission column order. Finally, I ensure the script writes a valid `my_submission.csv` in the working directory.'

# 9. Code solution

## === cell 0
from __future__ import print_function, division
from builtins import range

import os
import sys
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, Input
from tensorflow.keras.layers import LSTM, GlobalMaxPool1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam

SEED = 123
np.random.seed(SEED)
tf.random.set_seed(SEED)

for dirname, _, filenames in os.walk("/kaggle/input"):
    if (
        "jigsaw-toxic-comment-classification-challenge" in dirname
        or dirname == "/kaggle/input"
    ):
        for filename in filenames[:20]:
            print(os.path.join(dirname, filename))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
MAX_SEQUENCE_LENGTH = 100
MAX_VOCAB_SIZE = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 128
EPOCHS = 5

possible_labels = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

TRAIN_PATH = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip"
TEST_PATH = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip"
SAMPLE_SUB_PATH = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip"



## === cell 2
print("Loading in comments...")
train = pd.read_csv(TRAIN_PATH)

sentences = train["comment_text"].fillna("DUMMY_VALUE").astype(str).values
targets = train[possible_labels].values.astype(np.float32)

print("Train shape:", train.shape)
print("Targets shape:", targets.shape)



## === cell 3
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)

word2idx = tokenizer.word_index
print("Found %s unique tokens." % len(word2idx))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print("Shape of data tensor:", data.shape)



## === cell 4
num_words = min(MAX_VOCAB_SIZE, len(word2idx) + 1)

embedding_layer = Embedding(
    input_dim=num_words,
    output_dim=EMBEDDING_DIM,
    input_length=MAX_SEQUENCE_LENGTH,
    trainable=True,
)



## === cell 5
input_ = Input(shape=(MAX_SEQUENCE_LENGTH,))
x = embedding_layer(input_)
x = LSTM(15, return_sequences=True)(x)
x = GlobalMaxPool1D()(x)
output = Dense(len(possible_labels), activation="sigmoid")(x)

model = Model(input_, output)

model.compile(
    loss="binary_crossentropy",
    optimizer=Adam(learning_rate=0.01),
    metrics=["accuracy"],
)

model.summary()



## === cell 6
print("Training model...")
r = model.fit(
    data,
    targets,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=VALIDATION_SPLIT,
    shuffle=True,
    verbose=2,
)



## === cell 7
print("Loading test comments...")
test = pd.read_csv(TEST_PATH)
sentences_test = test["comment_text"].fillna("DUMMY_VALUE").astype(str).values

sequences_test = tokenizer.texts_to_sequences(sentences_test)
data_test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)
print("Test tensor shape:", data_test.shape)



## === cell 8
y_pred = model.predict(data_test, batch_size=1024, verbose=1)

y_pred = np.clip(y_pred, 0.0, 1.0)

print("Pred shape:", y_pred.shape)



## === cell 9
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
sub = sample_sub.copy()
sub["id"] = test["id"].values

for i, c in enumerate(possible_labels):
    sub[c] = y_pred[:, i]

expected_cols = ["id"] + possible_labels
sub = sub[expected_cols]
print(sub.head())
print("Submission shape:", sub.shape)



## === cell 10
SUB_PATH = "my_submission.csv"
sub.to_csv(SUB_PATH, index=False)
print(f"Your submission was successfully saved to {SUB_PATH}!")
