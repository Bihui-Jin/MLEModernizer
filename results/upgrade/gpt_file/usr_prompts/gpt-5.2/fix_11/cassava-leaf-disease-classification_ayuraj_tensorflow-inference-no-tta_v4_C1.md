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

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.models import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("TF version:", tf.__version__)




## === cell 1
DATA_ROOT_CANDIDATES = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "../input/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
    "../data/cassava-leaf-disease-classification",
]
DATA_ROOT = None
for p in DATA_ROOT_CANDIDATES:
    if os.path.exists(p):
        DATA_ROOT = p
        break

if DATA_ROOT is None:
    raise FileNotFoundError(
        f"Could not find cassava dataset folder in: {DATA_ROOT_CANDIDATES}"
    )

TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_DIR = os.path.join(DATA_ROOT, "test_images")
TRAIN_TFREC_DIR = os.path.join(DATA_ROOT, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(DATA_ROOT, "test_tfrecords")

print("DATA_ROOT:", DATA_ROOT)
print("Train CSV exists:", os.path.exists(TRAIN_CSV))
print("Sample submission exists:", os.path.exists(SAMPLE_SUB))
print("Train images dir exists:", os.path.exists(TRAIN_DIR))
print("Test images dir exists:", os.path.exists(TEST_DIR))
print("Train TFRecords dir exists:", os.path.exists(TRAIN_TFREC_DIR))
print("Test TFRecords dir exists:", os.path.exists(TEST_TFREC_DIR))




## === cell 2
IMAGE_SIZE = 380
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE
CLASS_NUMS = 5
SEED = 42

tf.keras.utils.set_random_seed(SEED)

tf.config.optimizer.set_jit(True)

train_df = pd.read_csv(TRAIN_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB)

sample_sub["image_id"] = sample_sub["image_id"].astype(str)
test_images = sample_sub["image_id"].tolist()
image_paths_test = [os.path.join(TEST_DIR, img) for img in test_images]

assert len(test_images) == len(sample_sub), "Sample submission rows mismatch."
missing = [p for p in image_paths_test[:50] if not os.path.exists(p)]
print("Example missing test paths (should be empty):", missing)

train_df["image_id"] = train_df["image_id"].astype(str)
train_df["path"] = (TRAIN_DIR + "/" + train_df["image_id"]).astype(str)

_check_n = min(200, len(train_df))
_check_missing = sum(
    0 if os.path.exists(p) else 1 for p in train_df["path"].iloc[:_check_n].tolist()
)
if _check_missing > 0:
    raise FileNotFoundError(
        f"Found {_check_missing} missing train images in first {_check_n} paths."
    )




## === cell 3
def _parse_tfrecord(example_proto, labeled=True):
    if labeled:
        feature_description = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "target": tf.io.FixedLenFeature([], tf.int64),
        }
    else:
        feature_description = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "image_name": tf.io.FixedLenFeature([], tf.string),
        }
    ex = tf.io.parse_single_example(example_proto, feature_description)
    img = tf.image.decode_jpeg(ex["image"], channels=3)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    if labeled:
        return img, tf.cast(ex["target"], tf.int32)
    return img, ex["image_name"]


def augment(img, label):
    img = tf.image.random_flip_left_right(img)
    return img, label


train_tfrecord_files = tf.io.gfile.glob(os.path.join(TRAIN_TFREC_DIR, "*.tfrec"))
test_tfrecord_files = tf.io.gfile.glob(os.path.join(TEST_TFREC_DIR, "*.tfrec"))
if not train_tfrecord_files:
    raise FileNotFoundError(f"No train tfrecords found in {TRAIN_TFREC_DIR}")
if not test_tfrecord_files:
    raise FileNotFoundError(f"No test tfrecords found in {TEST_TFREC_DIR}")
train_tfrecord_files = sorted(train_tfrecord_files)
test_tfrecord_files = sorted(test_tfrecord_files)

data_opts = tf.data.Options()
data_opts.experimental_deterministic = True
data_opts.threading.private_threadpool_size = 0
data_opts.threading.max_intra_op_parallelism = 0


def _make_labeled_raw_ds(tfrecord_files):
    files_ds = tf.data.Dataset.from_tensor_slices(tfrecord_files)
    ds = files_ds.interleave(
        lambda x: tf.data.TFRecordDataset(x, num_parallel_reads=AUTOTUNE),
        cycle_length=min(16, len(tfrecord_files)),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
        block_length=16,
    )
    ds = ds.apply(tf.data.experimental.ignore_errors())
    return ds


def _make_test_raw_ds(tfrecord_files):
    files_ds = tf.data.Dataset.from_tensor_slices(tfrecord_files)
    ds = files_ds.interleave(
        lambda x: tf.data.TFRecordDataset(x, num_parallel_reads=AUTOTUNE),
        cycle_length=min(16, len(tfrecord_files)),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
        block_length=16,
    )
    ds = ds.apply(tf.data.experimental.ignore_errors())
    return ds


SHARD_SIZE = 1338
valid_size = int(0.1 * len(train_df))  # identical definition
valid_shards = int(np.ceil(valid_size / SHARD_SIZE))
valid_shards = max(1, min(valid_shards, len(train_tfrecord_files) - 1))

valid_tfrecord_files = train_tfrecord_files[:valid_shards]
train_tfrecord_files_split = train_tfrecord_files[valid_shards:]

expected_valid = valid_shards * SHARD_SIZE
expected_train = (len(train_tfrecord_files) - valid_shards) * SHARD_SIZE
print(
    "Using deterministic TFRecord shard-based split. Shards(valid/train):",
    valid_shards,
    len(train_tfrecord_files) - valid_shards,
    "Approx counts:",
    expected_train,
    expected_valid,
)

train_raw = (
    _make_labeled_raw_ds(train_tfrecord_files_split).with_options(data_opts).cache()
)
valid_raw = _make_labeled_raw_ds(valid_tfrecord_files).with_options(data_opts).cache()

train_ds = train_raw.map(
    lambda x: _parse_tfrecord(x, labeled=True), num_parallel_calls=AUTOTUNE
)
train_ds = train_ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
train_ds = train_ds.map(augment, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False)
train_ds = train_ds.prefetch(AUTOTUNE)

valid_ds = valid_raw.map(
    lambda x: _parse_tfrecord(x, labeled=True), num_parallel_calls=AUTOTUNE
)
valid_ds = valid_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

test_raw = _make_test_raw_ds(test_tfrecord_files).with_options(data_opts).cache()
test_named_ds = test_raw.map(
    lambda x: _parse_tfrecord(x, labeled=False), num_parallel_calls=AUTOTUNE
)
test_ds = test_named_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)




## === cell 4
base = tf.keras.applications.EfficientNetB4(
    include_top=False,
    weights="imagenet",
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
)
x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
x = tf.keras.layers.Dropout(0.3)(x)
out = tf.keras.layers.Dense(CLASS_NUMS, activation="softmax")(x)
model = tf.keras.Model(inputs=base.input, outputs=out)

base.trainable = False
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history1 = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=2,
    verbose=1,
)

base.trainable = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
history2 = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=1,
    verbose=1,
)

model.summary()




## === cell 5
if isinstance(test_ds.element_spec, tuple) and len(test_ds.element_spec) == 2:
    pred_names_list = []
    pred_labels_list = []

    for batch_imgs, batch_names in test_ds:
        batch_probs = model(batch_imgs, training=False).numpy()
        pred_labels_list.append(np.argmax(batch_probs, axis=1).astype(np.int32))
        pred_names_list.append(batch_names.numpy())

    pred_labels = np.concatenate(pred_labels_list, axis=0).astype(int)
    pred_names = np.concatenate(pred_names_list, axis=0)

    name_to_pred = {n: p for n, p in zip(pred_names.tolist(), pred_labels.tolist())}
    first_key = next(iter(name_to_pred.keys()))
    if isinstance(first_key, (bytes, bytearray)):
        pred_labels = np.array(
            [name_to_pred[img.encode("utf-8")] for img in test_images], dtype=int
        )
    else:
        pred_labels = np.array([name_to_pred[img] for img in test_images], dtype=int)
else:
    pred_probs = model.predict(test_ds, verbose=1)
    pred_labels = np.argmax(pred_probs, axis=1).astype(int)

print("Predictions:", pred_labels[:10], "count:", len(pred_labels))
print("Test images:", len(test_images))

if len(pred_labels) != len(sample_sub):
    raise ValueError(
        f"Prediction length {len(pred_labels)} != sample_submission length {len(sample_sub)}"
    )

sub = pd.DataFrame({"image_id": test_images, "label": pred_labels})
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with rows:", len(sub))
