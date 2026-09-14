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
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

print("Files in ../input:", os.listdir("../input"))




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
    pad_seq, y, test_size=0.2, random_state=42
)

history = model.fit(
    X_train, y_train, epochs=2, batch_size=32, validation_data=(X_val, y_val), verbose=2
)




## === cell 5
preds = model.predict(test_pad_seq, batch_size=256)

sample_submission = pd.read_csv(SAMPLE_SUB_PATH)
sample_submission[list_classes] = preds
sample_submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv, shape:", sample_submission.shape)
