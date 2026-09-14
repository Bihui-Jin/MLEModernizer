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

import glob, json, itertools, shutil, io
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from collections import Counter

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Activation,
    Dropout,
    Flatten,
    Dense,
)
from tensorflow.keras import backend as K

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

BASE_PATH = "../input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(BASE_PATH, "train_images")
TEST_IMG_DIR = os.path.join(BASE_PATH, "test_images")

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing {TEST_IMG_DIR}"


def _fast_count_jpg(dir_path):
    n = 0
    with os.scandir(dir_path) as it:
        for e in it:
            if e.is_file() and e.name.endswith(".jpg"):
                n += 1
    return n


print("TensorFlow:", tf.__version__)
print("Train images:", _fast_count_jpg(TRAIN_IMG_DIR))
print("Test images:", _fast_count_jpg(TEST_IMG_DIR))



## === cell 1
df = pd.read_csv(TRAIN_CSV)

df["path"] = (TRAIN_IMG_DIR.rstrip("/") + "/" + df["image_id"].astype(str)).astype(str)
df["label"] = df["label"].astype(str)

df_train, df_valid = train_test_split(
    df, test_size=0.15, random_state=SEED, stratify=df["label"]
)

print("Train/Valid sizes:", df_train.shape, df_valid.shape)
print("Label distribution (train):", Counter(df_train["label"]))

IMG_SIZE = 224
SIZE = (IMG_SIZE, IMG_SIZE)
BATCH_SIZE = 32

AUTOTUNE = tf.data.AUTOTUNE

class_names = sorted(df_train["label"].unique().tolist())
label_to_index = {name: i for i, name in enumerate(class_names)}
NUM_CLASSES = len(class_names)
print("NUM_CLASSES:", NUM_CLASSES)
print("class_indices:", label_to_index)

idx_to_label_str = {v: k for k, v in label_to_index.items()}
print("idx_to_label_str (first):", dict(list(idx_to_label_str.items())[:5]))

ds_options = tf.data.Options()
ds_options.experimental_deterministic = True
try:
    ds_options.experimental_optimization.apply_default_optimizations = True
    ds_options.experimental_optimization.map_parallelization = True
    ds_options.experimental_optimization.parallel_batch = True
    ds_options.experimental_optimization.autotune_buffers = True
    ds_options.experimental_optimization.map_fusion = True
    ds_options.experimental_slack = True
except Exception:
    pass

y_train_idx = df_train["label"].map(label_to_index).astype(np.int32).values
y_valid_idx = df_valid["label"].map(label_to_index).astype(np.int32).values
x_train = df_train["path"].values
x_valid = df_valid["path"].values


def _stable_u32_hash(s: str) -> int:
    h = 2166136261
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return h


train_seed0 = np.fromiter(
    (_stable_u32_hash(p) for p in x_train), dtype=np.uint32, count=len(x_train)
).astype(np.int32)
valid_seed0 = np.fromiter(
    (_stable_u32_hash(p) for p in x_valid), dtype=np.uint32, count=len(x_valid)
).astype(np.int32)

_PI_OVER_180 = tf.constant(np.pi / 180.0, dtype=tf.float32)
_SEED_CONST = tf.constant(SEED, tf.int32)
_IMG_SIZE_F = tf.constant(float(IMG_SIZE), tf.float32)
_CX = tf.constant((IMG_SIZE - 1) / 2.0, tf.float32)
_CY = tf.constant((IMG_SIZE - 1) / 2.0, tf.float32)
_OUT_SHAPE = tf.constant([IMG_SIZE, IMG_SIZE], dtype=tf.int32)


@tf.function
def _decode_resize_rescale(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) / 255.0
    return img


@tf.function
def _augment(img, seed):
    img = tf.image.stateless_random_flip_left_right(img, seed=seed)

    angle = tf.random.stateless_uniform(
        [],
        seed=seed + tf.constant([1, 0], tf.int32),
        minval=-15.0,
        maxval=15.0,
        dtype=tf.float32,
    )
    angle = angle * _PI_OVER_180

    zoom = tf.random.stateless_uniform(
        [],
        seed=seed + tf.constant([2, 0], tf.int32),
        minval=0.9,
        maxval=1.1,
        dtype=tf.float32,
    )

    tx = (
        tf.random.stateless_uniform(
            [],
            seed=seed + tf.constant([3, 0], tf.int32),
            minval=-0.08,
            maxval=0.08,
            dtype=tf.float32,
        )
        * _IMG_SIZE_F
    )
    ty = (
        tf.random.stateless_uniform(
            [],
            seed=seed + tf.constant([4, 0], tf.int32),
            minval=-0.08,
            maxval=0.08,
            dtype=tf.float32,
        )
        * _IMG_SIZE_F
    )

    cos_a = tf.cos(angle) / zoom
    sin_a = tf.sin(angle) / zoom

    a0 = cos_a
    a1 = -sin_a
    b0 = sin_a
    b1 = cos_a

    a2 = _CX - a0 * _CX - a1 * _CY - tx
    b2 = _CY - b0 * _CX - b1 * _CY - ty

    transform = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0], axis=0)

    img_b = tf.expand_dims(img, 0)
    transform_b = tf.expand_dims(transform, 0)

    img_b = tf.raw_ops.ImageProjectiveTransformV3(
        images=img_b,
        transforms=transform_b,
        output_shape=_OUT_SHAPE,
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )
    img = tf.squeeze(img_b, axis=0)
    return img


CACHE_DIR = "./tfds_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

y_train_int = y_train_idx
y_valid_int = y_valid_idx


def make_train_ds(paths, labels_int, seed0_arr, batch_size):
    paths = tf.convert_to_tensor(paths)
    labels_int = tf.convert_to_tensor(labels_int)
    seed0_arr = tf.convert_to_tensor(seed0_arr)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels_int, seed0_arr))
    ds = ds.with_options(ds_options)

    def _decode_map(path, y, seed0):
        img = _decode_resize_rescale(path)
        return img, y, seed0

    def _augment_map(img, y, seed0):
        seed = tf.stack([tf.cast(seed0, tf.int32), _SEED_CONST], axis=0)
        img = _augment(img, seed)
        return img, y

    buffer_size = int(min(int(paths.shape[0]), 8192))
    ds = ds.shuffle(buffer_size=buffer_size, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(_decode_map, num_parallel_calls=AUTOTUNE)
    ds = ds.cache()
    ds = ds.map(_augment_map, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_valid_ds(paths, labels_int, batch_size):
    paths = tf.convert_to_tensor(paths)
    labels_int = tf.convert_to_tensor(labels_int)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels_int))
    ds = ds.with_options(ds_options)

    def _map_fn(path, y):
        img = _decode_resize_rescale(path)
        return img, y

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)

    valid_cache = os.path.join(CACHE_DIR, "valid_cache")
    ds = ds.cache(valid_cache)

    ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


train_ds = make_train_ds(x_train, y_train_int, train_seed0, BATCH_SIZE)
valid_ds = make_valid_ds(x_valid, y_valid_int, BATCH_SIZE)



## === cell 2
model = Sequential(
    [
        Conv2D(32, (3, 3), input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        Activation("relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3)),
        Activation("relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3)),
        Activation("relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(256),
        Activation("relu"),
        Dropout(0.5),
        Dense(NUM_CLASSES),
        Activation("softmax"),
    ]
)

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
    steps_per_execution=128,
)
model.summary()

EPOCHS = 8

history = model.fit(train_ds, epochs=EPOCHS, validation_data=valid_ds, verbose=1)



## === cell 3
sample_sub = pd.read_csv(SAMPLE_SUB)
df_test = sample_sub.copy()

df_test["path"] = (
    TEST_IMG_DIR.rstrip("/") + "/" + df_test["image_id"].astype(str)
).astype(str)

x_test = df_test["path"].values


def make_test_ds(paths, batch_size):
    paths = tf.convert_to_tensor(paths)
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.with_options(ds_options)

    def _map_fn(path):
        img = _decode_resize_rescale(path)
        return img

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)

    test_cache = os.path.join(CACHE_DIR, "test_cache")
    ds = ds.cache(test_cache)

    ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


test_ds = make_test_ds(x_test, BATCH_SIZE)

pred_test = model.predict(test_ds, verbose=1)
pred_test_idx = np.argmax(pred_test, axis=-1).astype(int)

pred_test_labels = np.array(
    [int(idx_to_label_str[i]) for i in pred_test_idx], dtype=int
)

submission = pd.DataFrame(
    {"image_id": df_test["image_id"].values, "label": pred_test_labels}
)
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("submission.csv exists:", os.path.exists("submission.csv"))
print("Label value counts (submission):")
print(submission["label"].value_counts().sort_index())
