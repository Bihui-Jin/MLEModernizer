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

0.5592681993821013

# 6. Current score

0.95636

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.95823) has done: 'I fix the TensorFlow/Keras import error by using a consistent `tf.keras` import path (the mixed `keras` vs `tf.keras` is what typically triggers the protobuf `MessageFactory` crash in Kaggle). Then I correct the model’s final activation from `softmax` to `sigmoid` for proper multi-label probability outputs, which should substantially improve ROC-AUC while keeping the same basic architecture and training loop. I also ensure the padding uses the intended truncation setting and that the prediction array shape matches the six label columns before writing `submission.csv`. All paths stay the same and the script run end-to-end producing a valid CSV submission.'
- What this solution (achieved 0.95636) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by ensuring the runtime uses the pure-Python protobuf implementation before importing TensorFlow, which is a common Kaggle environment incompatibility. This is a stability-only change that does not alter the model, features, or training loop semantics. I also keep all file paths and submission formatting identical, ensuring `submission.csv` is written with the required columns in the correct order.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
vocab_size = 20000
max_length = 120
embedding_dim = 50
trunc_type = "post"
padding_type = "post"
oov_tok = "<OOV>"



## === cell 3
train = pd.read_csv(
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip"
)
test = pd.read_csv(
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip"
)



## === cell 4
train.isnull().sum()



## === cell 5
test.isnull().sum()



## === cell 6
label = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]



## === cell 7
y = train[label].values
test_list = test["comment_text"].fillna("_na_").values
train_sentences = train["comment_text"].fillna("_na_").values
train_sentences



## === cell 8
y.shape



## === cell 9
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(list(train_sentences))



## === cell 10
word_index = tokenizer.word_index



## === cell 11
train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_padded = pad_sequences(
    train_sequences, padding=padding_type, truncating=trunc_type, maxlen=max_length
)

test_sequences = tokenizer.texts_to_sequences(test_list)
test_padded = pad_sequences(
    test_sequences, padding=padding_type, truncating=trunc_type, maxlen=max_length
)

print("train sequences: ", len(train_sequences[0]))
print("train padded: ", len(train_padded[0]))
print(len(train_sequences[1]))
print(len(train_padded[1]))



## === cell 12
model = tf.keras.Sequential(
    [
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(24, activation="relu"),
        tf.keras.layers.Dense(6, activation="sigmoid"),
    ]
)

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()



## === cell 13
train_padded.shape



## === cell 14
num_epochs = 5
history = model.fit(train_padded, y, epochs=num_epochs)



## === cell 15
test_pred = model.predict(test_padded, verbose=2)

if test_pred.ndim != 2 or test_pred.shape[1] != len(label):
    raise ValueError(
        f"Unexpected prediction shape: {test_pred.shape}, expected (n_samples, {len(label)})"
    )

sample_submission = pd.read_csv(
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip"
)
sample_submission[label] = test_pred
sample_submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sample_submission.shape)
print(sample_submission.head())
