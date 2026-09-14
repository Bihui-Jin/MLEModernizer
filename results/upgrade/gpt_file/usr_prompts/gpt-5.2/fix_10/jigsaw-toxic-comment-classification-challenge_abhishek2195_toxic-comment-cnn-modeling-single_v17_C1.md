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
import numpy as np
import pandas as pd
import os

os.environ["PYTHONHASHSEED"] = "42"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
np.random.seed(42)

base = "/kaggle/input/jigsaw-toxic-comment-classification-challenge"
print("Using dataset folder:", base)
print("Files:", sorted(os.listdir(base))[:10])



## === cell 1
import tensorflow as tf

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
from tensorflow.keras.layers import TextVectorization

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

print("TensorFlow:", tf.__version__)

tf.keras.utils.set_random_seed(42)



## === cell 2
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print("Running on TPU ", tpu.master())
except Exception:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS:", strategy.num_replicas_in_sync)



## === cell 3
AUTO = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 64
TOTAL_BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync
print("Total Batch Size:", TOTAL_BATCH_SIZE)



## === cell 4
train_path = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv"
test_path = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv"
sub_path = (
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv"
)

label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
df_train = pd.read_csv(train_path, usecols=["id", "comment_text"] + label_cols)
df_test = pd.read_csv(test_path, usecols=["id", "comment_text"])
df_sub = pd.read_csv(sub_path)

print("Train shape:", df_train.shape)
print("Test shape:", df_test.shape)
print("Sample submission shape:", df_sub.shape)
df_train.head(2)



## === cell 5
df_train["comment_text"] = df_train["comment_text"].fillna("")
df_test["comment_text"] = df_test["comment_text"].fillna("")

df_train["cleaned"] = df_train["comment_text"].astype(str)
df_test["cleaned"] = df_test["comment_text"].astype(str)

df_test.head(2)



## === cell 6
MAXLEN = 100



## === cell 7
text_vec = TextVectorization(
    standardize="lower_and_strip_punctuation",
    split="whitespace",
    output_mode="int",
    output_sequence_length=MAXLEN,
)

text_adapt_opts = tf.data.Options()
text_adapt_opts.experimental_deterministic = True
text_ds = (
    tf.data.Dataset.from_tensor_slices(df_train["cleaned"].values)
    .with_options(text_adapt_opts)
    .batch(32768)  # identical vocab, less overhead
)
text_vec.adapt(text_ds)

vocabulary = int(text_vec.vocabulary_size())
print("Vocabulary Size=>", vocabulary)



## === cell 8
pass



## === cell 9
pass



## === cell 10
print("Vocabulary Size=>", vocabulary)



## === cell 11
print("Train rows:", len(df_train), "Test rows:", len(df_test), "MAXLEN:", MAXLEN)



## === cell 12
y_train = df_train[label_cols].values.astype(np.float32)
print(y_train.shape)



## === cell 13
pass



## === cell 14
n = len(df_train)
val_n = int(0.2 * n)
train_n = n - val_n

x_tr_text = df_train["cleaned"].values[:train_n]
y_tr = y_train[:train_n]
x_val_text = df_train["cleaned"].values[train_n:]
y_val = y_train[train_n:]


def _vectorize_numpy(text_array, batch=65536):
    ds = tf.data.Dataset.from_tensor_slices(text_array).batch(batch)
    out = []
    for b in ds:
        out.append(text_vec(b))
    return tf.concat(out, axis=0).numpy()


x_tr_seq = _vectorize_numpy(x_tr_text)
x_val_seq = _vectorize_numpy(x_val_text)
x_test_seq = _vectorize_numpy(df_test["cleaned"].values)

train_dataset = (
    tf.data.Dataset.from_tensor_slices((x_tr_seq, y_tr))
    .shuffle(min(train_n, 100_000), seed=42, reshuffle_each_iteration=True)
    .batch(TOTAL_BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)

val_dataset = (
    tf.data.Dataset.from_tensor_slices((x_val_seq, y_val))
    .batch(TOTAL_BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)

test_dataset = (
    tf.data.Dataset.from_tensor_slices(x_test_seq)
    .batch(TOTAL_BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)



## === cell 15
print(train_dataset)
print(val_dataset)
print(test_dataset)



## === cell 16
pass




## === cell 17
def generate_model(filters, dropout):
    input_1 = Input(shape=(MAXLEN,))
    embedding_1 = Embedding(vocabulary, 100)(input_1)
    conv_1 = Conv1D(filters=int(round(filters)), kernel_size=7, padding="same")(
        embedding_1
    )
    dropout_1 = Dropout(dropout)(conv_1)
    pool_1 = GlobalMaxPooling1D()(dropout_1)

    dense = Dense(128, activation="relu")(pool_1)
    output = Dense(6, activation="sigmoid")(dense)

    model = Model(inputs=[input_1], outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )
    return model




## === cell 18
pass



## === cell 19
pass



## === cell 20
pass



## === cell 21
with strategy.scope():
    model = generate_model(352, 0.06675)

es = EarlyStopping(
    monitor="val_loss", mode="min", verbose=1, patience=5, min_delta=1e-5
)

ckpt_path = "/kaggle/working/model.keras"
mc = ModelCheckpoint(
    ckpt_path,
    monitor="val_loss",
    verbose=1,
    save_best_only=True,
    mode="min",
)

model.fit(
    train_dataset,
    epochs=100,
    verbose=1,
    validation_data=val_dataset,
    callbacks=[es, mc],
)

if os.path.exists(ckpt_path):
    model = tf.keras.models.load_model(ckpt_path)
    print("Loaded best checkpoint:", ckpt_path)
else:
    print("Checkpoint not found; using last-epoch model weights.")



## === cell 22
pass



## === cell 23
final_pred = model.predict(test_dataset, verbose=1)



## === cell 24
prob = df_sub.copy()
prob[label_cols] = final_pred.astype(np.float32)

prob = prob[["id"] + label_cols]
prob.head()



## === cell 25
print(prob.head())
print(prob.shape)



## === cell 26
out_path = "submission-CNN-single-2-100-100-7-opt-opt.csv"
prob.to_csv(out_path, index=False)
print("Wrote submission:", out_path)
print("Submission columns:", list(prob.columns))
print("Any NaNs:", prob.isna().any().any())
