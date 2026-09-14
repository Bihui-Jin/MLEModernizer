# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.config.optimizer.set_jit(True)

IMAGE_SIZE = (512, 512)
BATCH_SIZE = 8
EPOCHS = 2  # minimal training to produce a valid submission within time

BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
SAMPLE_SUB = os.path.join(BASE_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train_images")
TEST_IMG_DIR = os.path.join(BASE_DIR, "test_images")

TRAIN_TFREC_DIR = os.path.join(BASE_DIR, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(BASE_DIR, "test_tfrecords")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

AUTOTUNE = tf.data.AUTOTUNE

DATASET_OPTIONS = tf.data.Options()
DATASET_OPTIONS.experimental_deterministic = False
DATASET_OPTIONS.experimental_optimization.apply_default_optimizations = True
DATASET_OPTIONS.experimental_optimization.map_parallelization = True
DATASET_OPTIONS.experimental_optimization.parallel_batch = True
DATASET_OPTIONS.threading.private_threadpool_size = 0
DATASET_OPTIONS.threading.max_intra_op_parallelism = 0

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

train_df = pd.read_csv(TRAIN_CSV)
train_df["label"] = train_df["label"].astype(int)
assert train_df["label"].between(0, 4).all()

val_frac = 0.1
train_df = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
n_val_csv = int(len(train_df) * val_frac)
val_df = train_df.iloc[:n_val_csv].copy()
trn_df = train_df.iloc[n_val_csv:].copy()

TFREC_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}


@tf.function
def _decode_resize_normalize(img_bytes):
    img = tf.image.decode_jpeg(
        img_bytes, channels=3, dct_method="INTEGER_FAST", ratio=2
    )
    img = tf.image.resize(img, IMAGE_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) / 255.0
    return img


@tf.function
def _parse_train_tfrecord(example_proto):
    ex = tf.io.parse_single_example(example_proto, TFREC_FEATURES)
    img = _decode_resize_normalize(ex["image"])
    label = tf.cast(ex["target"], tf.int32)
    return img, tf.one_hot(label, depth=5)


@tf.function
def _parse_test_tfrecord(example_proto):
    ex = tf.io.parse_single_example(
        example_proto,
        {
            "image": tf.io.FixedLenFeature([], tf.string),
            "image_name": tf.io.FixedLenFeature([], tf.string),
        },
    )
    img = _decode_resize_normalize(ex["image"])
    return img, ex["image_name"]


@tf.function
def augment(img, label):
    img = tf.image.random_flip_left_right(img, seed=SEED)
    pad = 16
    img = tf.image.resize_with_crop_or_pad(
        img, IMAGE_SIZE[0] + pad, IMAGE_SIZE[1] + pad
    )
    img = tf.image.random_crop(img, size=[IMAGE_SIZE[0], IMAGE_SIZE[1], 3], seed=SEED)
    return img, label


train_tfrecord_files = tf.io.gfile.glob(os.path.join(TRAIN_TFREC_DIR, "*.tfrec"))
test_tfrecord_files = tf.io.gfile.glob(os.path.join(TEST_TFREC_DIR, "*.tfrec"))
train_tfrecord_files = sorted(train_tfrecord_files)
test_tfrecord_files = sorted(test_tfrecord_files)

if len(train_tfrecord_files) == 0 or len(test_tfrecord_files) == 0:
    raise FileNotFoundError(
        "TFRecord files not found; expected train_tfrecords/test_tfrecords under BASE_DIR."
    )

n_total = len(train_df)
n_val = int(n_total * val_frac)

EX_PER_FILE = 1338
n_val_files = max(1, int(np.ceil(n_val / EX_PER_FILE)))
n_val_files = min(n_val_files, len(train_tfrecord_files) - 1)

val_files = train_tfrecord_files[:n_val_files]
trn_files = train_tfrecord_files[n_val_files:]

n_val_effective = min(n_val, len(val_files) * EX_PER_FILE)
n_trn_effective = min(n_total - n_val_effective, len(trn_files) * EX_PER_FILE)


def _make_train_dataset(files):
    ds = tf.data.TFRecordDataset(
        files,
        num_parallel_reads=AUTOTUNE,
        buffer_size=16 * 1024 * 1024,
    )
    ds = ds.with_options(DATASET_OPTIONS)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = ds.map(_parse_train_tfrecord, num_parallel_calls=AUTOTUNE)

    ds = ds.shuffle(
        buffer_size=min(n_trn_effective, 8192), seed=SEED, reshuffle_each_iteration=True
    )
    ds = ds.map(augment, num_parallel_calls=AUTOTUNE)

    ds = ds.repeat()

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def _make_val_dataset(files):
    ds = tf.data.TFRecordDataset(
        files,
        num_parallel_reads=AUTOTUNE,
        buffer_size=16 * 1024 * 1024,
    )
    ds = ds.with_options(DATASET_OPTIONS)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = ds.map(_parse_train_tfrecord, num_parallel_calls=AUTOTUNE)
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = _make_train_dataset(trn_files)
val_ds = _make_val_dataset(val_files)

steps_per_epoch = int(np.ceil(n_trn_effective / BATCH_SIZE))
validation_steps = int(np.ceil(n_val_effective / BATCH_SIZE))

print(
    "Train samples (csv count):", len(trn_df), "Val samples (csv count):", len(val_df)
)
print(
    "Effective Train samples:",
    n_trn_effective,
    "Effective Val samples:",
    n_val_effective,
)
print("TFRecord train files:", len(trn_files), "TFRecord val files:", len(val_files))
print(
    "TFRecord total train files:",
    len(train_tfrecord_files),
    "TFRecord test files:",
    len(test_tfrecord_files),
)
print("steps_per_epoch:", steps_per_epoch, "validation_steps:", validation_steps)



## === cell 1
num_classes = 5

inputs = keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs, outputs)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()



## === cell 2
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    validation_steps=validation_steps,
    verbose=1,
)



## === cell 3
test_df = pd.read_csv(SAMPLE_SUB)

test_ds = tf.data.TFRecordDataset(
    test_tfrecord_files,
    num_parallel_reads=AUTOTUNE,
    buffer_size=16 * 1024 * 1024,
)
test_ds = test_ds.with_options(DATASET_OPTIONS)
test_ds = test_ds.apply(tf.data.experimental.ignore_errors())
test_ds = test_ds.map(_parse_test_tfrecord, num_parallel_calls=AUTOTUNE)

test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)


class _PredictWithNames(keras.Model):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    @tf.function
    def call(self, inputs, training=False):
        img, name = inputs
        probs = self.base_model(img, training=training)
        return probs, name


wrapped = _PredictWithNames(model)
probs, names = wrapped.predict(test_ds, verbose=0)

names = np.asarray(names)
if names.dtype.kind in ("S",):  # fixed-width bytes
    names = np.char.decode(names, "utf-8")
else:
    names = names.astype("U")

pred_labels = np.argmax(probs, axis=1).astype(np.int32)

name_to_label = dict(zip(names.tolist(), pred_labels.tolist()))
ordered_labels = test_df["image_id"].map(name_to_label).astype(int).to_numpy()
assert len(ordered_labels) == len(test_df)

submission = pd.DataFrame(
    {"image_id": test_df["image_id"].values, "label": ordered_labels}
)
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Submission columns:", submission.columns.tolist())
