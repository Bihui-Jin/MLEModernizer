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

from tensorflow.keras.preprocessing.text import Tokenizer
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

tf.keras.utils.set_random_seed(SEED)


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
df_train["comment_text"] = df_train["comment_text"].fillna("")
df_test["comment_text"] = df_test["comment_text"].fillna("")

df_train["cleaned"] = df_train["comment_text"].astype(str)
df_test["cleaned"] = df_test["comment_text"].astype(str)

df_test.head()




## === cell 6
MAX_WORDS = 200000  # chosen to keep accuracy strong while cutting runtime/memory
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token=None)
tokenizer.fit_on_texts(df_train["cleaned"].values)

print("Raw Vocabulary Size=>", len(tokenizer.word_index))
print("Effective Vocabulary Cap (num_words)=>", MAX_WORDS)




## === cell 7
MAXLEN = 100


def texts_to_padded_sequences(tok, texts, maxlen=100, chunk_size=50000):
    n = len(texts)
    out = np.zeros((n, maxlen), dtype=np.int32)
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        seqs = tok.texts_to_sequences(texts[start:end])
        for i, s in enumerate(seqs):
            if not s:
                continue
            if len(s) > maxlen:
                s = s[-maxlen:]
            out[start + i, : len(s)] = s
    return out


train_seq = texts_to_padded_sequences(
    tokenizer, df_train["cleaned"].values, maxlen=MAXLEN
)
test_seq = texts_to_padded_sequences(
    tokenizer, df_test["cleaned"].values, maxlen=MAXLEN
)

vocabulary = min(MAX_WORDS, len(tokenizer.word_index) + 1)
print("Embedding Vocabulary Size=>", vocabulary)
print("Shape of train_sequence=>", train_seq.shape, train_seq.dtype)
print("Shape of test_sequence=>", test_seq.shape, test_seq.dtype)




## === cell 8
y_train = df_train[label_cols].values.astype(np.float32)
print(y_train.shape)




## === cell 9
n = train_seq.shape[0]
val_frac = 0.2
val_size = int(n * val_frac)
train_size = n - val_size

x_tr, x_val = train_seq[:train_size], train_seq[train_size:]
y_tr, y_val = y_train[:train_size], y_train[train_size:]

options = tf.data.Options()
options.experimental_deterministic = True

train_dataset = (
    tf.data.Dataset.from_tensor_slices((x_tr, y_tr))
    .with_options(options)
    .shuffle(42, seed=SEED, reshuffle_each_iteration=True)
    .batch(TOTAL_BATCH_SIZE, drop_remainder=False)
    .cache()
    .prefetch(AUTO)
)
val_dataset = (
    tf.data.Dataset.from_tensor_slices((x_val, y_val))
    .with_options(options)
    .batch(TOTAL_BATCH_SIZE, drop_remainder=False)
    .cache()
    .prefetch(AUTO)
)
test_dataset = (
    tf.data.Dataset.from_tensor_slices(test_seq)
    .with_options(options)
    .batch(TOTAL_BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)

print(train_dataset)
print(val_dataset)
print(test_dataset)




## === cell 10
def generate_model(filters, dropout):
    input_1 = Input(shape=(100,))
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




## === cell 11
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




## === cell 12
ckpt_path = "/kaggle/working/model.keras"
if os.path.exists(ckpt_path):
    model = tf.keras.models.load_model(ckpt_path)

train_pred = model.predict(
    tf.data.Dataset.from_tensor_slices(train_seq).batch(4096).prefetch(AUTO),
    verbose=1,
)
print(
    "In-sample Evaluation ROC-AUC Score:\n",
    roc_auc_score_multilabel(y_train, train_pred),
)




## === cell 13
final_pred = model.predict(test_dataset, verbose=1)




## === cell 14
submission = pd.DataFrame(final_pred, columns=label_cols)
submission.insert(0, "id", df_test["id"].values)

submission = submission[df_sample.columns.tolist()]

submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print("Wrote:", submission_path)
print(submission.head())
