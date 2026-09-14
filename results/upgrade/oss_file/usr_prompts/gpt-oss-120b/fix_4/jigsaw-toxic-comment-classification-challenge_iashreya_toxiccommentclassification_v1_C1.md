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

0.90084

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.90084) has done: 'The changes increase the training and prediction batch sizes so each model processes far fewer steps per epoch, cutting total runtime dramatically while keeping the same model architecture, loss, and single‑epoch training. Larger batches use the same data and loss calculations, so the learned parameters remain equivalent up to minor floating‑point differences that do not affect overall accuracy. No core logic, layers, or training loops are altered.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_path = "../input/jigsaw-toxic-comment-classification-challenge/train.csv"
training_set = pd.read_csv(train_path)
training_set = training_set.drop(["id"], axis=1)

columns = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]



## === cell 2
print("Number of training records :", len(training_set))
print("Columns :", list(training_set.columns))



## === cell 3
max_words = 20000
max_len = 40
token = Tokenizer(num_words=max_words, oov_token="<OOV>")
token.fit_on_texts(training_set["comment_text"])
seq = token.texts_to_sequences(training_set["comment_text"])
padded_seq = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
vocab_size = min(max_words, len(token.word_index) + 1)



## === cell 4
embed_dim = 300
embeddings = np.random.normal(size=(vocab_size, embed_dim)).astype(np.float32)


def build_model(vocab_size, embed_dim, max_len):
    model = Sequential()
    model.add(
        Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
            weights=[embeddings],
            input_length=max_len,
            trainable=True,
        )
    )
    model.add(Conv1D(128, 5, activation="relu"))
    model.add(MaxPooling1D(5))
    model.add(Conv1D(128, 5, activation="relu"))
    model.add(MaxPooling1D(3))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model




## === cell 5
train_batch_size = 1024

models = {}
input_dataset = (
    tf.data.Dataset.from_tensor_slices(padded_seq)
    .batch(train_batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

for col in columns:
    print(f"Training model for {col} …")
    tf.keras.backend.clear_session()  # free memory between models
    model = build_model(vocab_size, embed_dim, max_len)

    label_dataset = (
        tf.data.Dataset.from_tensor_slices(training_set[col].values)
        .batch(train_batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    train_dataset = tf.data.Dataset.zip((input_dataset, label_dataset))

    model.fit(train_dataset, epochs=1, verbose=0)
    models[col] = model



## === cell 6
test_path = "../input/jigsaw-toxic-comment-classification-challenge/test.csv"
test_set = pd.read_csv(test_path)
x_test = test_set["comment_text"]

test_seq = token.texts_to_sequences(x_test)
test_padded_seq = pad_sequences(
    test_seq, maxlen=max_len, padding="post", truncating="post"
)



## === cell 7
preds = {}
for col in columns:
    preds[col] = (
        models[col].predict(test_padded_seq, batch_size=4096, verbose=0).reshape(-1)
    )



## === cell 8
submission = pd.DataFrame(
    {
        "id": test_set["id"],
        "toxic": preds["toxic"],
        "severe_toxic": preds["severe_toxic"],
        "obscene": preds["obscene"],
        "threat": preds["threat"],
        "insult": preds["insult"],
        "identity_hate": preds["identity_hate"],
    }
)



## === cell 9
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")
