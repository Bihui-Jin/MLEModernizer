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

# 5. Code solution

## === cell 0
import os, warnings, random

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Embedding,
    Dropout,
    Conv1D,
    GlobalMaxPooling1D,
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import roc_auc_score

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

tf.config.optimizer.set_jit(True)

gpus = tf.config.list_physical_devices("GPU")
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    has_tpu = True
except Exception:
    has_tpu = False

if gpus or has_tpu:
    from tensorflow.keras import mixed_precision

    policy = mixed_precision.Policy("mixed_float16")
    mixed_precision.set_global_policy(policy)
else:
    pass



## === cell 1
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
except Exception:
    if gpus:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
print("REPLICAS:", strategy.num_replicas_in_sync)



## === cell 2
AUTO = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 4096  # raised from 2048
TOTAL_BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync
print("Total Batch Size:", TOTAL_BATCH_SIZE)



## === cell 3
train_path = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv"
test_path = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv"

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

print("Train shape:", df_train.shape)
print("Test shape :", df_test.shape)



## === cell 4
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df_train["comment_text"].astype(str))

print("Vocabulary Size =>", len(tokenizer.word_index))



## === cell 5
MAX_LEN = 100
train_seq = tokenizer.texts_to_sequences(df_train["comment_text"].astype(str))
test_seq = tokenizer.texts_to_sequences(df_test["comment_text"].astype(str))

train_seq = pad_sequences(train_seq, maxlen=MAX_LEN, padding="post")
test_seq = pad_sequences(test_seq, maxlen=MAX_LEN, padding="post")

train_seq = train_seq.astype(np.int32)
test_seq = test_seq.astype(np.int32)

print("train_seq shape:", train_seq.shape)
print("test_seq  shape:", test_seq.shape)



## === cell 6
y_train = df_train[
    ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
].values.astype(
    np.float16
)  # pre‑cast once to avoid per‑sample casting
print("y_train shape:", y_train.shape)



## === cell 7
vocabulary = len(tokenizer.word_index) + 1


def generate_model(filters, dropout):
    inp = Input(shape=(MAX_LEN,))
    emb = Embedding(vocabulary, 100, dtype="float16")(inp)
    conv = Conv1D(filters=int(round(filters)), kernel_size=7, padding="same")(emb)
    drop = Dropout(dropout)(conv)
    pool = GlobalMaxPooling1D()(drop)
    dense = Dense(128, activation="relu")(pool)
    out = Dense(6, activation="sigmoid")(dense)
    model = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )
    return model




## === cell 8
from sklearn.model_selection import train_test_split

train_idx, val_idx = train_test_split(
    np.arange(len(train_seq)), test_size=0.2, random_state=42
)

train_ds = (
    tf.data.Dataset.from_tensor_slices((train_seq[train_idx], y_train[train_idx]))
    .shuffle(10000, seed=42)
    .batch(TOTAL_BATCH_SIZE, drop_remainder=True)
    .cache()  # cache shuffled batches for all epochs
    .prefetch(AUTO)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((train_seq[val_idx], y_train[val_idx]))
    .batch(TOTAL_BATCH_SIZE, drop_remainder=True)
    .cache()  # cache validation batches
    .prefetch(AUTO)
)

with strategy.scope():
    model = generate_model(filters=352, dropout=0.06675)

es = EarlyStopping(
    monitor="val_loss", mode="min", patience=2, verbose=1, min_delta=1e-5
)

ckpt_path = "/kaggle/working/model.keras"
mc = ModelCheckpoint(
    ckpt_path, monitor="val_loss", verbose=1, save_best_only=True, mode="min"
)

model.fit(
    train_ds,
    epochs=10,
    validation_data=val_ds,
    callbacks=[es, mc],
    verbose=1,
)



## === cell 9
test_pred = model.predict(test_seq, batch_size=TOTAL_BATCH_SIZE)



## === cell 10
submission = pd.DataFrame(
    {
        "id": df_test["id"],
        "toxic": test_pred[:, 0],
        "severe_toxic": test_pred[:, 1],
        "obscene": test_pred[:, 2],
        "threat": test_pred[:, 3],
        "insult": test_pred[:, 4],
        "identity_hate": test_pred[:, 5],
    }
)
submission.head()



## === cell 11
submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
