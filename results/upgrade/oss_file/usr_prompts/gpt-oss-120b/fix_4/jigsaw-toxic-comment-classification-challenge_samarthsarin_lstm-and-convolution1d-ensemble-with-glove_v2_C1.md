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

0.779438091833117

# 6. Current score

0.96653

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.96653) has done: 'The changes introduce deterministic seeding, convert the training/validation splits into efficient `tf.data.Dataset` pipelines with shuffling, batching, and prefetching, and use a dataset for test‑time prediction. These adjustments keep the exact model architecture, training epochs, and batch sizes while removing Python‑level overhead during fitting and inference, which reduces runtime enough to stay under the 600‑second limit without altering any core logic or result accuracy.'
- What this solution (achieved 0.96653) has done: 'I set the `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION` environment variable before importing TensorFlow to avoid the protobuf “MessageFactory” error, and renumber the cells so they start at 1 as required. No other logic is changed, preserving the model and its performance (which already exceeds the target score).'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

print("Files in ../input:", os.listdir("../input"))




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

df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = df[list_classes].values.astype(np.float32)




## === cell 2
MAX_LEN = 100
token = Tokenizer()
token.fit_on_texts(df["comment_text"].astype(str).values)

seq = token.texts_to_sequences(df["comment_text"].astype(str).values)
pad_seq = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")

test_seq = token.texts_to_sequences(test_df["comment_text"].astype(str).values)
test_pad_seq = pad_sequences(
    test_seq, maxlen=MAX_LEN, padding="post", truncating="post"
)

vocab_size = len(token.word_index) + 1
print("Vocab size:", vocab_size)




## === cell 3
EMBED_DIM = 100

model = Sequential()
model.add(
    Embedding(
        input_dim=vocab_size,
        output_dim=EMBED_DIM,
        input_length=MAX_LEN,
        mask_zero=False,
    )
)  # random init, trainable
model.add(Bidirectional(LSTM(64, return_sequences=False)))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(len(list_classes), activation="sigmoid"))

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[tf.keras.metrics.AUC(name="auc")],
)

model.summary()




## === cell 4
X_train, X_val, y_train, y_val = train_test_split(
    pad_seq, y, test_size=0.2, random_state=SEED
)

train_dataset = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .shuffle(10000, seed=SEED, reshuffle_each_iteration=True)
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)

val_dataset = (
    tf.data.Dataset.from_tensor_slices((X_val, y_val))
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)

history = model.fit(
    train_dataset,
    epochs=2,
    validation_data=val_dataset,
    verbose=2,
)




## === cell 5
test_dataset = (
    tf.data.Dataset.from_tensor_slices(test_pad_seq)
    .batch(256)
    .prefetch(tf.data.AUTOTUNE)
)
preds = model.predict(test_dataset, verbose=0)

sample_submission = pd.read_csv(SAMPLE_SUB_PATH)
sample_submission[list_classes] = preds
sample_submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv, shape:", sample_submission.shape)
