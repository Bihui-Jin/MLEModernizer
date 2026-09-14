# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.9648118536001448

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.47147) has done: 'I remove the failing `pip install` cells and the unused BayesianOptimization/iterative-stratification imports that trigger the protobuf `GetPrototype` error, keeping the core CNN model and training loop intact. I also fix the data loading paths to use the provided Jigsaw `train.csv`/`test.csv`, and create a minimal `cleaned` text column from `comment_text` so the existing tokenizer pipeline runs unchanged. Finally, I ensure the submission is written with the exact required columns/order and a `.csv` suffix. These changes are score-neutral with respect to the intended model, but they make the notebook run end-to-end and produce a valid submission file.'

# 9. Code solution

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
    tf.config.threading.set_intra_op_parallelism_threads(
        tf.config.threading.get_intra_op_parallelism_threads() or 0
    )
    tf.config.threading.set_inter_op_parallelism_threads(
        tf.config.threading.get_inter_op_parallelism_threads() or 0
    )
except Exception:
    pass




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

train_text_ds = tf.data.Dataset.from_tensor_slices(df_train["cleaned"].values).batch(
    32768
)
vectorizer.adapt(train_text_ds)

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

train_text_all = df_train["cleaned"].to_numpy(dtype=object, copy=False)
x_tr_text = train_text_all[tr_idx]
x_val_text = train_text_all[val_idx]
x_test_text = df_test["cleaned"].to_numpy(dtype=object, copy=False)

y_tr = y_train[tr_idx]
y_val = y_train[val_idx]

options = tf.data.Options()
options.experimental_deterministic = True


def _make_text_ds(texts, labels=None, training=False, cache_ds=False):
    if labels is None:
        ds = tf.data.Dataset.from_tensor_slices(texts).with_options(options)
        ds = ds.batch(TOTAL_BATCH_SIZE, drop_remainder=False)
        ds = ds.map(vectorizer, num_parallel_calls=AUTO)
        ds = ds.prefetch(AUTO)
        return ds

    ds = tf.data.Dataset.from_tensor_slices((texts, labels)).with_options(options)

    if training:
        ds = ds.shuffle(50000, seed=SEED, reshuffle_each_iteration=False)

    ds = ds.batch(TOTAL_BATCH_SIZE, drop_remainder=False)
    ds = ds.map(lambda t, y: (vectorizer(t), y), num_parallel_calls=AUTO)
    if cache_ds:
        ds = ds.cache()
    ds = ds.prefetch(AUTO)
    return ds


train_dataset = _make_text_ds(x_tr_text, y_tr, training=True, cache_ds=True)
val_dataset = _make_text_ds(x_val_text, y_val, training=False, cache_ds=True)
test_dataset = _make_text_ds(x_test_text, labels=None, training=False, cache_ds=False)

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

submission = pd.DataFrame(final_pred, columns=label_cols)
submission.insert(0, "id", df_test["id"].values)

submission = submission[df_sample.columns.tolist()]

submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print("Wrote:", submission_path)
print(submission.head())
