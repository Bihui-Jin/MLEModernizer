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
import os, math, gc, random, re

if "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION" in os.environ:
    os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault(
    "TF_XLA_FLAGS",
    "--tf_xla_auto_jit=0 --tf_xla_cpu_global_jit=false",
)

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras

print("Tensorflow version " + tf.__version__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception as e:
    print("Determinism not enabled:", repr(e))

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception as e:
    print("Threading config not set:", repr(e))

pass




## === cell 1
DATA_DIR = "../input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 5
EPOCHS = 3  # keep identical core logic


@tf.function(reduce_retracing=True)
def _decode_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)  # same decoder/format
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.AREA, antialias=False
    )
    img = tf.cast(img, tf.float32) / 255.0
    img.set_shape([IMG_SIZE, IMG_SIZE, 3])
    return img


_DATASET_OPTIONS = tf.data.Options()
_DATASET_OPTIONS.experimental_deterministic = True
_DATASET_OPTIONS.experimental_optimization.apply_default_optimizations = True
_DATASET_OPTIONS.experimental_optimization.map_parallelization = True
_DATASET_OPTIONS.experimental_optimization.parallel_batch = True


def make_dataset_from_paths_labels(
    paths, labels, batch_size=BATCH_SIZE, training=True, cache=False, cache_name=None
):
    paths = tf.constant(paths)
    labels = tf.constant(labels, dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels)).with_options(
        _DATASET_OPTIONS
    )

    if training:
        ds = ds.shuffle(
            buffer_size=min(8192, int(paths.shape[0])),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    def _map_fn(p, y):
        return _decode_resize(p), y

    ds = ds.map(
        _map_fn,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    ).ignore_errors()

    if cache:
        if cache_name is None:
            ds = ds.cache()
        else:
            ds = ds.cache(cache_name)

    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds


def make_test_dataset_from_paths(
    paths, batch_size=BATCH_SIZE, cache=True, cache_name=None
):
    paths = tf.constant(paths)
    ds = tf.data.Dataset.from_tensor_slices(paths).with_options(_DATASET_OPTIONS)

    ds = ds.map(
        _decode_resize,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    ).ignore_errors()

    if cache:
        if cache_name is None:
            ds = ds.cache()
        else:
            ds = ds.cache(cache_name)

    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds




## === cell 2
submission = pd.read_csv(SAMPLE_SUB)
assert "image_id" in submission.columns and "label" in submission.columns
test_image_ids = submission["image_id"].astype(str).values
print("Test images:", len(test_image_ids))

train_df = pd.read_csv(TRAIN_CSV)
assert "image_id" in train_df.columns and "label" in train_df.columns
train_df["image_id"] = train_df["image_id"].astype(str)
train_df["label"] = train_df["label"].astype(int)
print(
    "Train rows:",
    len(train_df),
    "label counts:",
    train_df["label"].value_counts().to_dict(),
)

idx = np.arange(len(train_df))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
val_size = int(0.1 * len(train_df))
val_idx = idx[:val_size]
trn_idx = idx[val_size:]

trn_ids = train_df.iloc[trn_idx]["image_id"].values
trn_y = train_df.iloc[trn_idx]["label"].values
val_ids = train_df.iloc[val_idx]["image_id"].values
val_y = train_df.iloc[val_idx]["label"].values

trn_paths = (TRAIN_IMG_DIR + "/" + trn_ids).astype(str)
val_paths = (TRAIN_IMG_DIR + "/" + val_ids).astype(str)

ds_train = make_dataset_from_paths_labels(
    trn_paths, trn_y, batch_size=BATCH_SIZE, training=True, cache=False
)
ds_val = make_dataset_from_paths_labels(
    val_paths,
    val_y,
    batch_size=BATCH_SIZE,
    training=False,
    cache=True,
    cache_name="./_val_cache",
)




## === cell 3
def build_model(img_size=IMG_SIZE, num_classes=NUM_CLASSES):
    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2, seed=SEED)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


model = build_model()
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    steps_per_execution=32,
)

history = model.fit(ds_train, validation_data=ds_val, epochs=EPOCHS, verbose=1)
print(
    "Training done. Last val acc:",
    history.history.get(
        "val_acc", history.history.get("val_sparse_categorical_accuracy", [None])
    )[-1],
)




## === cell 4
test_paths = (TEST_IMG_DIR + "/" + test_image_ids).astype(str)

ds_test = make_test_dataset_from_paths(
    test_paths, batch_size=BATCH_SIZE, cache=True, cache_name="./_test_cache"
)

probs = model.predict(ds_test, verbose=1)
probs = np.asarray(probs)
if probs.shape[0] != len(test_image_ids):
    probs = probs[: len(test_image_ids)]
preds = np.argmax(probs, axis=-1).astype(int)

print("Preds shape:", preds.shape, "unique:", np.unique(preds, return_counts=True))




## === cell 5
submission["label"] = preds
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())




## === cell 6
submission
