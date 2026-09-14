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

0.7678853274770532

# 6. Current score

0.96372

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.96372) has done: 'I remove the NLTK dependency (it’s not available/configured here and is causing import/runtime errors) and keep the same Keras tokenization/padding flow. I also fix Keras/TensorFlow compatibility by importing from `tensorflow.keras` and replacing `CuDNNLSTM` with standard `LSTM` (GPU/CPU-safe) while preserving the Bidirectional LSTM architecture. Because the original loss/activation setup is invalid for multilabel AUC (softmax + categorical_crossentropy on 6 independent binary labels), I switch to `sigmoid` + `binary_crossentropy`, which matches the competition metric and should move the score toward the target. Finally, I remove the missing GloVe file dependency and use a randomly-initialized Embedding so the notebook runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.96372) has done: 'I fix the TensorFlow/Keras import crash by using `tensorflow.keras` consistently (the `MessageFactory/GetPrototype` error comes from the standalone `keras` package mismatch). This also restore missing symbols like `Tokenizer`, which caused downstream `NameError`s and prevented training/inference. I keep the same model architecture and training loop, but ensure determinism seeds are set through TensorFlow as well. Finally, I ensure the code always writes a valid `submission.csv` with the required columns and row alignment.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import random

random.seed(42)
np.random.seed(42)

print("Listing ../input (if exists):")
try:
    print(os.listdir("../input"))
except Exception as e:
    print("Could not list ../input:", e)

DATA_DIR_CANDIDATES = [
    "../input/jigsaw-toxic-comment-classification-challenge",
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge",
    "../input",
    "/kaggle/input",
]
DATA_DIR = None
for d in DATA_DIR_CANDIDATES:
    if os.path.exists(os.path.join(d, "train.csv")) and os.path.exists(
        os.path.join(d, "test.csv")
    ):
        DATA_DIR = d
        break

if DATA_DIR is None:
    DATA_DIR = "../input/jigsaw-toxic-comment-classification-challenge"

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sub_path = os.path.join(DATA_DIR, "sample_submission.csv")

print("Using DATA_DIR:", DATA_DIR)
print("Train exists:", os.path.exists(train_path))
print("Test exists:", os.path.exists(test_path))
print("Sample submission exists:", os.path.exists(sub_path))



## === cell 1
import tensorflow as tf

tf.random.set_seed(42)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, Bidirectional, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

print("tf version:", getattr(tf, "__version__", "unknown"))
print("tf.keras version:", getattr(tf.keras, "__version__", "unknown"))



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
df = pd.read_csv(train_path)
print(df.shape)
df.head()



## === cell 3
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = df[list_classes].values.astype("float32")
x = df["comment_text"].fillna("").astype(str)

print("X size:", x.shape, "Y shape:", y.shape)



## === cell 4
maxlen = 100
max_features = 50000  # cap vocab for speed/memory; preserves core tokenization approach

token = Tokenizer(num_words=max_features, lower=True, oov_token="<OOV>")
token.fit_on_texts(x)

seq = token.texts_to_sequences(x)
pad_seq = pad_sequences(seq, maxlen=maxlen)

vocab_size = min(max_features, len(token.word_index) + 1)
print("Padded train shape:", pad_seq.shape)
print("Vocab size:", vocab_size)



## === cell 5
embed_dim = 50

model = Sequential()
model.add(
    Embedding(
        input_dim=vocab_size,
        output_dim=embed_dim,
        input_length=maxlen,
        trainable=True,
    )
)
model.add(Bidirectional(LSTM(50)))
model.add(Dense(50, activation="relu"))
model.add(Dense(6, activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()



## === cell 6
history = model.fit(
    pad_seq, y, epochs=2, batch_size=32, validation_split=0.1, verbose=2
)



## === cell 7
test = pd.read_csv(test_path)
print(test.shape)
test.head()



## === cell 8
x_test = test["comment_text"].fillna("").astype(str)
test_seq = token.texts_to_sequences(x_test)
test_pad_seq = pad_sequences(test_seq, maxlen=maxlen)

print("Padded test shape:", test_pad_seq.shape)



## === cell 9
predict = model.predict(test_pad_seq, batch_size=1024, verbose=1)
print("Prediction shape:", predict.shape)
print("First row:", predict[0])



## === cell 10
sample_submission = pd.read_csv(sub_path)

sample_submission = sample_submission[["id"] + list_classes].copy()

if len(sample_submission) != len(test) or not sample_submission["id"].equals(
    test["id"]
):
    sample_submission = pd.DataFrame({"id": test["id"].values})
    for c in list_classes:
        sample_submission[c] = 0.5  # placeholder, will be overwritten below

sample_submission[list_classes] = predict
sample_submission[list_classes] = (
    sample_submission[list_classes].clip(0.0, 1.0).astype("float32")
)

out_path = "submission.csv"
sample_submission.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(sample_submission.head())
print(sample_submission.tail())
