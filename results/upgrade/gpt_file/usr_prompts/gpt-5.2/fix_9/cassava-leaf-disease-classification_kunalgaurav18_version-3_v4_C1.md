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
import time
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("TF version:", tf.__version__)

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
TRAIN_TFREC_DIR = os.path.join(DATA_DIR, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(DATA_DIR, "test_tfrecords")

IMG_SIZE = 512
NB_CHANNELS = 3
BATCH_SIZE = 32

SEED = 42
tf.keras.utils.set_random_seed(SEED)

tf.config.optimizer.set_jit(True)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.experimental.set_memory_growth(
        tf.config.list_physical_devices("GPU")[0], True
    )
except Exception:
    pass

tf.config.run_functions_eagerly(False)




## === cell 1
EfficientNetB0 = tf.keras.applications.EfficientNetB0

cnn_base = EfficientNetB0(
    include_top=False,
    weights=None,
    input_shape=(IMG_SIZE, IMG_SIZE, NB_CHANNELS),
)

cnn = tf.keras.Sequential(
    [
        cnn_base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(5, activation="softmax"),
    ]
)

cnn.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer="adam",
    metrics=["accuracy"],
)

cnn.summary()




## === cell 2
train_df = pd.read_csv(TRAIN_CSV, usecols=["image_id", "label"])
train_df["label"] = train_df["label"].astype(int)

val_frac = 0.1
train_df = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
n_val = int(len(train_df) * val_frac)
val_df = train_df.iloc[:n_val].copy()
trn_df = train_df.iloc[n_val:].copy()

TRAIN_TFRECS = tf.io.gfile.glob(os.path.join(TRAIN_TFREC_DIR, "*.tfrec"))
TEST_TFRECS = tf.io.gfile.glob(os.path.join(TEST_TFREC_DIR, "*.tfrec"))
TRAIN_TFRECS = sorted(TRAIN_TFRECS)
TEST_TFRECS = sorted(TEST_TFRECS)

_TRAIN_FEATURE_DESCRIPTION = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}
_TEST_FEATURE_DESCRIPTION = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


@tf.function(jit_compile=False)
def _parse_tfrec_train(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TRAIN_FEATURE_DESCRIPTION)
    img = tf.io.decode_jpeg(ex["image"], channels=NB_CHANNELS)
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    img = tf.ensure_shape(img, (IMG_SIZE, IMG_SIZE, NB_CHANNELS))
    label = tf.cast(ex["target"], tf.int32)
    return img, label


@tf.function(jit_compile=False)
def _parse_tfrec_test(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TEST_FEATURE_DESCRIPTION)
    img = tf.io.decode_jpeg(ex["image"], channels=NB_CHANNELS)
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    img = tf.ensure_shape(img, (IMG_SIZE, IMG_SIZE, NB_CHANNELS))
    image_id = tf.cast(ex["image_name"], tf.string)
    return img, image_id


def _ds_options(deterministic=True):
    opts = tf.data.Options()
    opts.experimental_deterministic = bool(deterministic)
    try:
        opts.threading.private_threadpool_size = 0
        opts.threading.max_intra_op_parallelism = 0
    except Exception:
        pass
    try:
        opts.experimental_optimization.apply_default_optimizations = True
        opts.experimental_optimization.map_fusion = True
        opts.experimental_optimization.noop_elimination = True
        opts.experimental_optimization.map_parallelization = True
        opts.experimental_optimization.filter_fusion = True
        opts.experimental_optimization.parallel_batch = True
        opts.experimental_optimization.autotune_buffers = True
        opts.experimental_optimization.autotune = True
    except Exception:
        pass
    return opts


def make_tfrec_train_val_datasets(train_tfrecs, val_frac=0.1):
    n_total = len(train_df)
    n_val_local = int(n_total * val_frac)

    ds_all = tf.data.TFRecordDataset(
        train_tfrecs,
        num_parallel_reads=tf.data.AUTOTUNE,
        buffer_size=16 * 1024 * 1024,
    )
    ds_all = ds_all.with_options(_ds_options(deterministic=True))

    ds_all = ds_all.map(_parse_tfrec_train, num_parallel_calls=tf.data.AUTOTUNE)

    ds_all = ds_all.cache()

    ds_all = ds_all.shuffle(2048, seed=SEED, reshuffle_each_iteration=False)

    val_ds = ds_all.take(n_val_local)
    trn_ds = ds_all.skip(n_val_local)

    trn_ds = trn_ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)

    trn_ds = trn_ds.batch(BATCH_SIZE, drop_remainder=False)
    val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=False)

    try:
        trn_ds = trn_ds.prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    except Exception:
        trn_ds = trn_ds.prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return trn_ds, val_ds


train_ds, val_ds = make_tfrec_train_val_datasets(TRAIN_TFRECS, val_frac=val_frac)

EPOCHS = 1
history = cnn.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)

cnn.save_weights("./final_weights1.h5")




## === cell 3
print("Weights file exists:", os.path.exists("./final_weights1.h5"))




## === cell 4
sample_sub = pd.read_csv(SAMPLE_SUB)
test_image_ids = sample_sub["image_id"].tolist()  # ensures correct ordering

test_ds = tf.data.TFRecordDataset(
    TEST_TFRECS,
    num_parallel_reads=tf.data.AUTOTUNE,
    buffer_size=16 * 1024 * 1024,
)
test_ds = test_ds.with_options(_ds_options(deterministic=False))
test_ds = test_ds.map(_parse_tfrec_test, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

probs = cnn.predict(
    test_ds.map(lambda x, _id: x, num_parallel_calls=tf.data.AUTOTUNE), verbose=0
)
preds = np.argmax(probs, axis=1).astype(int)

ids = []
for _, batch_ids in test_ds:
    ids.append(batch_ids.numpy())
ids = np.concatenate(ids).astype("U")

pred_map = dict(zip(ids.tolist(), preds.tolist()))
ordered_preds = np.fromiter(
    (pred_map[i] for i in test_image_ids), dtype=int, count=len(test_image_ids)
)

submission = pd.DataFrame({"image_id": test_image_ids, "label": ordered_preds})
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Saved at:", os.path.abspath("submission.csv"))
