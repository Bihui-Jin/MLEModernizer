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
import numpy as np  # linear algebra
import pandas as pd  # data processing
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:10]:
        print(os.path.join(dirname, filename))



## === cell 1
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import tensorflow as tf

try:
    import tf_keras as keras  # Kaggle often provides this compatible package
except Exception:
    keras = tf.keras

Model = keras.Model
layers = keras.layers
callbacks = keras.callbacks

from sklearn.metrics import roc_auc_score  # (kept; not used to save time)

print("TF version:", tf.__version__)
print("Using keras module:", getattr(keras, "__name__", str(keras)))



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

BATCH_SIZE = 128
TOTAL_BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync
print("Total Batch Size:", TOTAL_BATCH_SIZE)




## === cell 4
def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of these paths exist: {paths}")


train_path = _first_existing(
    [
        "/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv",
        "/kaggle/input/train.csv",
    ]
)

test_path = _first_existing(
    [
        "/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv",
        "/kaggle/input/test.csv",
    ]
)

sample_sub_path = _first_existing(
    [
        "/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv",
        "/kaggle/input/sample_submission.csv",
    ]
)

print("train_path:", train_path)
print("test_path:", test_path)
print("sample_sub_path:", sample_sub_path)



## === cell 5
df_train = pd.read_csv(train_path)
print("Shape=>", df_train.shape)
df_train.head()



## === cell 6
df_test = pd.read_csv(test_path)
print("Shape=>", df_test.shape)
df_test.head()



## === cell 7
df_train["cleaned"] = df_train["comment_text"].fillna("").astype(str)
df_test["cleaned"] = df_test["comment_text"].fillna("").astype(str)



## === cell 8
MAXLEN = 100

vectorize_layer = layers.TextVectorization(
    standardize="lower_and_strip_punctuation",
    split="whitespace",
    output_mode="int",
    output_sequence_length=MAXLEN,
)

adapt_ds = (
    tf.data.Dataset.from_tensor_slices(df_train["cleaned"].values)
    .batch(4096)
    .prefetch(AUTO)
)
vectorize_layer.adapt(adapt_ds)

vocabulary = vectorize_layer.vocabulary_size()
print("Vocabulary Size=>", vocabulary)



## === cell 9
label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y_train = df_train[label_cols].values.astype(np.float32)
print(y_train.shape)



## === cell 10
tf.random.set_seed(42)
np.random.seed(42)



## === cell 11
texts_train = df_train["cleaned"].values
texts_test = df_test["cleaned"].values



## === cell 12
steps_per_epoch = int(np.ceil(len(df_train) / TOTAL_BATCH_SIZE))
print("steps_per_epoch:", steps_per_epoch)



## === cell 13
with strategy.scope():
    input_1 = layers.Input(shape=(MAXLEN,), dtype=tf.int32)
    embedding_1 = layers.Embedding(vocabulary, 100)(input_1)
    conv_1 = layers.Conv1D(filters=100, kernel_size=3, padding="same")(embedding_1)
    dropout_1 = layers.Dropout(0.2)(conv_1)
    pool_1 = layers.GlobalMaxPooling1D()(dropout_1)

    dense = layers.Dense(128, activation="relu")(pool_1)
    output = layers.Dense(6, activation="sigmoid")(dense)

    model = Model(inputs=[input_1], outputs=output)

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )

model.summary()



## === cell 14
es = callbacks.EarlyStopping(
    monitor="val_loss",
    mode="min",
    verbose=1,
    patience=5,
    min_delta=1e-5,
    restore_best_weights=True,
)

mc = callbacks.ModelCheckpoint(
    "/kaggle/working/model.keras",
    monitor="val_loss",
    verbose=1,
    save_best_only=True,
    mode="min",
)



## === cell 15
val_size = int(0.1 * len(df_train))
train_size = len(df_train) - val_size

x_train_part = texts_train[:train_size]
y_train_part = y_train[:train_size]
x_val_part = texts_train[train_size:]
y_val_part = y_train[train_size:]


def _make_text_ds(texts, labels=None, training=False):
    """
    Keeps the existing vectorize-then-batch pipeline that avoids ragged batch issues.
    """
    if labels is None:
        ds = tf.data.Dataset.from_tensor_slices(texts)
        ds = ds.batch(4096)
        ds = ds.map(vectorize_layer, num_parallel_calls=AUTO)
        ds = ds.unbatch()
        ds = ds.batch(TOTAL_BATCH_SIZE, drop_remainder=False)
        ds = ds.prefetch(AUTO)
        return ds

    ds = tf.data.Dataset.from_tensor_slices((texts, labels))
    if training:
        ds = ds.shuffle(
            min(len(texts), 100_000), seed=42, reshuffle_each_iteration=True
        )
        ds = ds.repeat()

    ds = ds.batch(4096)
    ds = ds.map(lambda t, y: (vectorize_layer(t), y), num_parallel_calls=AUTO)
    ds = ds.cache()
    ds = ds.unbatch()
    ds = ds.batch(TOTAL_BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTO)
    return ds


train_ds_fit = _make_text_ds(x_train_part, y_train_part, training=True)
val_ds_fit = _make_text_ds(x_val_part, y_val_part, training=False)

steps_per_epoch_fit = int(np.ceil(train_size / TOTAL_BATCH_SIZE))
val_steps = int(np.ceil(val_size / TOTAL_BATCH_SIZE))

model.fit(
    train_ds_fit,
    epochs=100,
    steps_per_epoch=steps_per_epoch_fit,
    validation_data=val_ds_fit,
    validation_steps=val_steps,
    verbose=1,
    callbacks=[es, mc],
)



## === cell 16
print("Skipped in-sample ROC-AUC to meet 600s timeout.")



## === cell 17
test_ds_pred = _make_text_ds(texts_test, labels=None, training=False)
final_pred = model.predict(test_ds_pred, verbose=1)
print("final_pred shape:", final_pred.shape)



## === cell 18
prob = pd.DataFrame({"id": df_test["id"].values})
for i, col in enumerate(label_cols):
    prob[col] = final_pred[:, i]
prob = prob[["id"] + label_cols]
prob.head()



## === cell 19
out_path = "/kaggle/working/submission.csv"
prob.to_csv(out_path, index=False)

expected_cols = ["id"] + label_cols
assert prob.columns.tolist() == expected_cols, f"Bad columns: {prob.columns.tolist()}"

print("Wrote submission to:", out_path)
print(prob.shape)
print(prob.columns.tolist())
print(prob.head(2).to_string(index=False))
