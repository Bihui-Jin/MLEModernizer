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

0.9659288367986162

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.47862) has done: 'I remove the failing `pip install` steps (no internet in Kaggle runtime) and also drop the optional imports that trigger the `MessageFactory.GetPrototype` protobuf error, since they are not used in the training/inference path. I fix the broken data loading by reading the provided `/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv` and `test.csv` instead of non-existent JSONs, and create the expected `cleaned` text column from `comment_text`. I keep your CNN model, tokenization, padding length, and training loop semantics the same, only fixing the missing imports and ensuring the submission is written with the exact required columns and a `.csv` suffix. Finally, I ensure the predictions align 1:1 with `sample_submission.csv` `id` ordering to avoid a silent submission misalignment bug.'
- What this solution (achieved 0.5) has done: 'The timeout is overwhelmingly driven by training cost (very large train set, heavy TextVectorization+Conv1D, and an inefficient `tf.data` pipeline that caches massive datasets in memory after vectorization). I keep the exact same model, loss, and training loop, but make the input pipeline faster and less memory-stressed by: (1) moving vectorization to happen *before* batching (so the map can parallelize better and avoid Python overhead), (2) removing `.cache()` on the full train/val/test pipelines (caching millions of already-vectorized sequences is both slow and memory-heavy and can cause stalls), and (3) enabling graph execution for the vectorization map via `@tf.function` and setting tf.data options to reduce overhead while preserving determinism. These changes preserve evaluation semantics and accuracy (they do not alter tokens, vocabulary, model, or training schedule), but significantly reduce pipeline overhead and memory pressure so training can finish within 600 seconds more reliably.'

# 9. Code solution

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



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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

dtypes_train = {
    "id": "string",
    "comment_text": "string",
    "toxic": "int8",
    "severe_toxic": "int8",
    "obscene": "int8",
    "threat": "int8",
    "insult": "int8",
    "identity_hate": "int8",
}
dtypes_test = {"id": "string", "comment_text": "string"}

df_train = pd.read_csv(
    train_path,
    usecols=["id", "comment_text"] + label_cols,
    dtype=dtypes_train,
    engine="c",
)
df_test = pd.read_csv(
    test_path, usecols=["id", "comment_text"], dtype=dtypes_test, engine="c"
)
df_sub = pd.read_csv(sub_path, dtype={"id": "string"}, engine="c")

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
    .batch(32768)
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


@tf.function
def _vec_x(x):
    return text_vec(x)


@tf.function
def _vec_xy(x, y):
    return text_vec(x), y


def _make_text_ds(texts, labels=None, training=False):
    if labels is None:
        ds = tf.data.Dataset.from_tensor_slices(texts)
    else:
        ds = tf.data.Dataset.from_tensor_slices((texts, labels))

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    opts.experimental_optimization.autotune = True
    ds = ds.with_options(opts)

    if training:
        ds = ds.shuffle(
            min(len(texts), 100_000), seed=42, reshuffle_each_iteration=True
        )

    if labels is None:
        ds = ds.map(_vec_x, num_parallel_calls=AUTO)
        ds = ds.batch(TOTAL_BATCH_SIZE, drop_remainder=False)
    else:
        ds = ds.map(_vec_xy, num_parallel_calls=AUTO)
        ds = ds.batch(TOTAL_BATCH_SIZE, drop_remainder=False)

    ds = ds.prefetch(AUTO)
    return ds


train_dataset = _make_text_ds(x_tr_text, y_tr, training=True)
val_dataset = _make_text_ds(x_val_text, y_val, training=False)
test_dataset = _make_text_ds(df_test["cleaned"].values, labels=None, training=False)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1878303108.py in <cell line: 0>()
     52 
     53 
---> 54 train_dataset = _make_text_ds(x_tr_text, y_tr, training=True)
     55 val_dataset = _make_text_ds(x_val_text, y_val, training=False)
     56 test_dataset = _make_text_ds(df_test["cleaned"].values, labels=None, training=False)

/tmp/ipykernel_11/1878303108.py in _make_text_ds(texts, labels, training)
     33     opts.experimental_deterministic = True
     34     # Reduce tf.data iterator overhead without changing meaning.
---> 35     opts.experimental_optimization.autotune = True
     36     ds = ds.with_options(opts)
     37 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/util/options.py in __setattr__(self, name, value)
     59       object.__setattr__(self, name, value)
     60     else:
---> 61       raise AttributeError("Cannot set the property {} on {}.".format(
     62           name,
     63           type(self).__name__))

AttributeError: Cannot set the property autotune on OptimizationOptions.

## === cell 15
print(train_dataset)
print(val_dataset)
print(test_dataset)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3229313522.py in <cell line: 0>()
----> 1 print(train_dataset)
      2 print(val_dataset)
      3 print(test_dataset)
      4 

NameError: name 'train_dataset' is not defined

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



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3867373551.py in <cell line: 0>()
     16 
     17 model.fit(
---> 18     train_dataset,
     19     epochs=100,
     20     verbose=1,

NameError: name 'train_dataset' is not defined

## === cell 22
pass



## === cell 23
final_pred = model.predict(test_dataset, verbose=1)



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1674632870.py in <cell line: 0>()
----> 1 final_pred = model.predict(test_dataset, verbose=1)
      2 

NameError: name 'test_dataset' is not defined

## === cell 24
prob = df_sub.copy()
prob[label_cols] = final_pred.astype(np.float32)

prob = prob[["id"] + label_cols]
prob.head()



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/610124933.py in <cell line: 0>()
      1 prob = df_sub.copy()
----> 2 prob[label_cols] = final_pred.astype(np.float32)
      3 
      4 prob = prob[["id"] + label_cols]
      5 prob.head()

NameError: name 'final_pred' is not defined

## === cell 25
print(prob.head())
print(prob.shape)



## === cell 26
out_path = "submission-CNN-single-2-100-100-7-opt-opt.csv"
prob.to_csv(out_path, index=False)
print("Wrote submission:", out_path)
print("Submission columns:", list(prob.columns))
print("Any NaNs:", prob.isna().any().any())
