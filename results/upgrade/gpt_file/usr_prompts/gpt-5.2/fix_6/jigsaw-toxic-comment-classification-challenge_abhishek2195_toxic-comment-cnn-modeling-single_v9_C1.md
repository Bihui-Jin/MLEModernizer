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

0.963559276192663

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.50612) has done: 'I fix the import/runtime error in the Keras/TensorFlow stack by using `tf.keras` consistently (this avoids the protobuf-related `MessageFactory.GetPrototype` crash and missing-symbol NameErrors). I also correct the data-loading paths to use the actual competition CSVs available under `/kaggle/input/jigsaw-toxic-comment-classification-challenge/` (with a safe fallback to `/kaggle/input/`), and create the expected `cleaned` text field from `comment_text` so the existing tokenizer logic remains intact. Finally, I ensure the pipeline trains, runs inference on the test set, and writes a valid `submission.csv` with the required columns and order.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:10]:
        print(os.path.join(dirname, filename))



## === cell 1
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

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

from sklearn.metrics import roc_auc_score



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

vectorize_layer = tf.keras.layers.TextVectorization(
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
    input_1 = Input(shape=(MAXLEN,), dtype=tf.int32)
    embedding_1 = Embedding(vocabulary, 100)(input_1)
    conv_1 = Conv1D(filters=100, kernel_size=3, padding="same")(embedding_1)
    dropout_1 = Dropout(0.2)(conv_1)
    pool_1 = GlobalMaxPooling1D()(dropout_1)

    dense = Dense(128, activation="relu")(pool_1)
    output = Dense(6, activation="sigmoid")(dense)

    model = Model(inputs=[input_1], outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )

model.summary()



## === cell 14
es = EarlyStopping(
    monitor="val_loss",
    mode="min",
    verbose=1,
    patience=5,
    min_delta=1e-5,
    restore_best_weights=True,
)

mc = ModelCheckpoint(
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
    if labels is None:
        ds = tf.data.Dataset.from_tensor_slices(texts)
        ds = ds.batch(4096)
        ds = ds.map(vectorize_layer, num_parallel_calls=AUTO)
        ds = ds.batch(256)
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



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/303890230.py in <cell line: 0>()
      1 # Speed fix: stream test vectorization+prediction; avoids building x_test_seq tensor.
      2 test_ds_pred = _make_text_ds(texts_test, labels=None, training=False)
----> 3 final_pred = model.predict(test_ds_pred, verbose=1)
      4 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

InvalidArgumentError: {{function_node __wrapped__IteratorGetNext_output_types_1_device_/job:localhost/replica:0/task:0/device:CPU:0}} Cannot add tensor to the batch: number of elements does not match. Shapes are: [tensor]: [1612,100], [batch]: [4096,100] [Op:IteratorGetNext] name: 

## === cell 18
prob = pd.DataFrame({"id": df_test["id"].values})
for i, col in enumerate(label_cols):
    prob[col] = final_pred[:, i]
prob = prob[["id"] + label_cols]



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2888444897.py in <cell line: 0>()
      1 prob = pd.DataFrame({"id": df_test["id"].values})
      2 for i, col in enumerate(label_cols):
----> 3     prob[col] = final_pred[:, i]
      4 prob = prob[["id"] + label_cols]
      5 

NameError: name 'final_pred' is not defined

## === cell 19
prob.head()



## === cell 20
out_path = "/kaggle/working/submission.csv"
prob.to_csv(out_path, index=False)
print("Wrote submission to:", out_path)
print(prob.shape)
print(prob.columns.tolist())

## --- ERROR in outputing the csv:
Invalid submission: Submission is missing the following columns: {'threat', 'identity_hate', 'obscene', 'insult', 'toxic', 'severe_toxic'}
