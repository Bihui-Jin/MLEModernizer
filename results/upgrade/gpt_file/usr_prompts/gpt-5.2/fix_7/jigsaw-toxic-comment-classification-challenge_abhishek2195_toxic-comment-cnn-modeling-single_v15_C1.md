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
import os
import random
import warnings

import numpy as np
import pandas as pd

warnings.simplefilter(action="ignore", category=FutureWarning)

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

print("Kaggle input root:", "/kaggle/input")



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
    TextVectorization,
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

tf.keras.utils.set_random_seed(SEED)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass


def _binary_auc(y_true, y_score):
    y_true = np.asarray(y_true).astype(np.int32)
    y_score = np.asarray(y_score).astype(np.float64)

    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan

    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)

    sorted_scores = y_score[order]
    i = 0
    N = len(sorted_scores)
    while i < N:
        j = i
        while j + 1 < N and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg_rank = 0.5 * (i + 1 + j + 1)
            ranks[order[i : j + 1]] = avg_rank
        i = j + 1

    sum_ranks_pos = ranks[pos].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def roc_auc_score_multilabel(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    aucs = []
    for k in range(y_true.shape[1]):
        aucs.append(_binary_auc(y_true[:, k], y_score[:, k]))
    return float(np.nanmean(aucs))




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

print("REPLICAS: ", strategy.num_replicas_in_sync)



## === cell 3
AUTO = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 64
TOTAL_BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync
print("Total Batch Size:", TOTAL_BATCH_SIZE)



## === cell 4
train_path = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv"
test_path = "/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv"
sample_path = (
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv"
)

label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
df_train = pd.read_csv(train_path, usecols=["id", "comment_text"] + label_cols)
df_test = pd.read_csv(test_path, usecols=["id", "comment_text"])
df_sample = pd.read_csv(sample_path)

print("train:", df_train.shape, "test:", df_test.shape, "sample:", df_sample.shape)
df_train.head()



## === cell 5
df_train["comment_text"] = df_train["comment_text"].fillna("").astype(str)
df_test["comment_text"] = df_test["comment_text"].fillna("").astype(str)

df_train["cleaned"] = df_train["comment_text"]
df_test["cleaned"] = df_test["comment_text"]

df_test.head()



## === cell 6
MAX_WORDS = 200000
MAXLEN = 100

vectorizer = TextVectorization(
    max_tokens=MAX_WORDS,
    output_mode="int",
    output_sequence_length=MAXLEN,
    standardize="lower_and_strip_punctuation",
    split="whitespace",
)

train_text_np = df_train["cleaned"].values  # numpy array of python strings
ADAPT_BATCH = 8192
for i in range(0, len(train_text_np), ADAPT_BATCH):
    vectorizer.adapt(train_text_np[i : i + ADAPT_BATCH])

vocabulary = int(vectorizer.vocabulary_size())
print("Vectorizer vocabulary_size =>", vocabulary)



## === cell 7
y_train = df_train[label_cols].values.astype(np.float32)
print(y_train.shape)



## === cell 8
n = len(df_train)
val_frac = 0.2
val_size = int(n * val_frac)

rng = np.random.RandomState(SEED)
perm = rng.permutation(n)
val_idx = perm[:val_size]
tr_idx = perm[val_size:]

x_tr_text = df_train.loc[tr_idx, "cleaned"].values
y_tr = y_train[tr_idx]
x_val_text = df_train.loc[val_idx, "cleaned"].values
y_val = y_train[val_idx]

options = tf.data.Options()
options.experimental_deterministic = True


def vectorize_in_batches(texts, batch_size=16384):
    out = np.empty((len(texts), MAXLEN), dtype=np.int32)
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        out[i : i + len(batch)] = vectorizer(batch).numpy()
    return out


x_tr = vectorize_in_batches(x_tr_text)
x_val = vectorize_in_batches(x_val_text)
x_test = vectorize_in_batches(df_test["cleaned"].values)


def make_train_ds_int(x_int, labels):
    ds = tf.data.Dataset.from_tensor_slices((x_int, labels)).with_options(options)
    ds = ds.shuffle(50000, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(TOTAL_BATCH_SIZE, drop_remainder=False)
    ds = ds.cache()
    ds = ds.prefetch(AUTO)
    return ds


def make_eval_ds_int(x_int, labels=None):
    if labels is None:
        ds = tf.data.Dataset.from_tensor_slices(x_int).with_options(options)
        ds = ds.batch(TOTAL_BATCH_SIZE, drop_remainder=False)
        ds = ds.cache()
        ds = ds.prefetch(AUTO)
        return ds
    ds = tf.data.Dataset.from_tensor_slices((x_int, labels)).with_options(options)
    ds = ds.batch(TOTAL_BATCH_SIZE, drop_remainder=False)
    ds = ds.cache()
    ds = ds.prefetch(AUTO)
    return ds


train_dataset = make_train_ds_int(x_tr, y_tr)
val_dataset = make_eval_ds_int(x_val, y_val)
test_dataset = make_eval_ds_int(x_test, labels=None)

print(train_dataset)
print(val_dataset)
print(test_dataset)




## === cell 9
def generate_model(filters, dropout):
    input_1 = Input(shape=(MAXLEN,), dtype=tf.int32)
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




## === cell 10
with strategy.scope():
    model = generate_model(331, 0.05)

es = EarlyStopping(
    monitor="val_loss", mode="min", verbose=1, patience=5, min_delta=1e-5
)

mc = ModelCheckpoint(
    "/kaggle/working/model.keras",
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



## === cell 11
ckpt_path = "/kaggle/working/model.keras"
if os.path.exists(ckpt_path):
    model = tf.keras.models.load_model(ckpt_path)



## === cell 12
final_pred = model.predict(test_dataset, verbose=1)



## === cell 13
submission = pd.DataFrame(final_pred, columns=label_cols)
submission.insert(0, "id", df_test["id"].values)

submission = submission[df_sample.columns.tolist()]

submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print("Wrote:", submission_path)
print(submission.head())
