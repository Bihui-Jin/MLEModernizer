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

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", str(os.cpu_count() or 4))
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")

import glob
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras import layers

SEED = 42
DEBUG = False

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

BASE_PATH = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(BASE_PATH, "train_images")
TEST_IMG_DIR = os.path.join(BASE_PATH, "test_images")

print("TF version:", tf.__version__)
print("Base path exists:", os.path.exists(BASE_PATH))
print("Train CSV exists:", os.path.exists(TRAIN_CSV))
print("Sample submission exists:", os.path.exists(SAMPLE_SUB))
print("Train img dir exists:", os.path.exists(TRAIN_IMG_DIR))
print("Test img dir exists:", os.path.exists(TEST_IMG_DIR))

assert os.path.isdir(TRAIN_IMG_DIR), "TRAIN_IMG_DIR not found"
assert os.path.isdir(TEST_IMG_DIR), "TEST_IMG_DIR not found"



## === cell 1
IMG_SIZE = (300, 300)
NUM_CLASSES = 5
BATCH_SIZE = 32
EPOCHS = 2  # keep as-is per original core logic

df_train = pd.read_csv(TRAIN_CSV)
df_train["label"] = df_train["label"].astype(str)
df_train["path"] = TRAIN_IMG_DIR.rstrip("/") + "/" + df_train["image_id"].astype(str)
df_train = df_train.reset_index(drop=True)

idx = np.arange(len(df_train))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
split = int(0.9 * len(idx))
train_idx, val_idx = idx[:split], idx[split:]
train_df = df_train.iloc[train_idx].reset_index(drop=True)
val_df = df_train.iloc[val_idx].reset_index(drop=True)

print("Train/Val sizes:", len(train_df), len(val_df))
print("Train label sample:", train_df["label"].value_counts().sort_index().to_dict())

AUTOTUNE = tf.data.AUTOTUNE


def _decode_and_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(
        img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    return img


_ROT_MAX = 10.0 * np.pi / 180.0
_WSHIFT = 0.05
_HSHIFT = 0.05
_ZOOM = 0.1


def _fold_in(seed2, data):
    seed2 = tf.convert_to_tensor(seed2, dtype=tf.int32)
    data = tf.cast(data, tf.int32)
    mix = tf.random.stateless_uniform(
        shape=(2,), seed=seed2, minval=-(2**31), maxval=2**31 - 1, dtype=tf.int32
    )
    return mix ^ tf.stack([data, data + 1])


def _augment(img, seed):
    seed = tf.convert_to_tensor(seed, dtype=tf.int32)  # shape [2]

    s1 = _fold_in(seed, 1)
    s2 = _fold_in(seed, 2)
    s3 = _fold_in(seed, 3)
    s4 = _fold_in(seed, 4)
    s5 = _fold_in(seed, 5)

    do_flip = tf.random.stateless_uniform((), s1, 0.0, 1.0) < 0.5
    img = tf.cond(do_flip, lambda: tf.image.flip_left_right(img), lambda: img)

    h = tf.cast(tf.shape(img)[0], tf.float32)
    w = tf.cast(tf.shape(img)[1], tf.float32)
    dx = tf.random.stateless_uniform((), s2, -_WSHIFT, _WSHIFT) * w
    dy = tf.random.stateless_uniform((), s3, -_HSHIFT, _HSHIFT) * h
    z = tf.random.stateless_uniform((), s4, 1.0 - _ZOOM, 1.0 + _ZOOM)
    ang = tf.random.stateless_uniform((), s5, -_ROT_MAX, _ROT_MAX)

    cx = (w - 1.0) / 2.0
    cy = (h - 1.0) / 2.0

    cos_a = tf.cos(ang)
    sin_a = tf.sin(ang)
    inv_z = 1.0 / z
    a0 = cos_a * inv_z
    a1 = -sin_a * inv_z
    a3 = sin_a * inv_z
    a4 = cos_a * inv_z

    tx = cx - (a0 * cx + a1 * cy) - dx
    ty = cy - (a3 * cx + a4 * cy) - dy

    transform = tf.stack([a0, a1, tx, a3, a4, ty, 0.0, 0.0])
    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[None, ...],
        transforms=transform[None, ...],
        output_shape=tf.constant([IMG_SIZE[0], IMG_SIZE[1]], dtype=tf.int32),
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )[0]
    return img


def make_train_val_datasets(train_df, val_df, batch_size):
    train_paths = train_df["path"].to_numpy()
    train_labels = train_df["label"].astype(np.int32).to_numpy()
    val_paths = val_df["path"].to_numpy()
    val_labels = val_df["label"].astype(np.int32).to_numpy()

    ds_train = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    ds_train = ds_train.shuffle(
        len(train_paths), seed=SEED, reshuffle_each_iteration=True
    )

    ds_train = ds_train.enumerate()
    ds_train = ds_train.map(
        lambda i, xy: (
            xy[0],
            xy[1],
            tf.stack([tf.cast(SEED, tf.int32), tf.cast(i, tf.int32)], axis=0),
        ),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    ds_train = ds_train.map(
        lambda p, y, s: (_augment(_decode_and_resize(p), s), y),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    ds_train = ds_train.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)

    ds_val = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    ds_val = ds_val.map(
        lambda p, y: (_decode_and_resize(p), y),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    ds_val = ds_val.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)

    return ds_train, ds_val


train_ds, val_ds = make_train_val_datasets(train_df, val_df, BATCH_SIZE)

unique_labels = set(train_df["label"].unique().tolist())
assert unique_labels == set(
    map(str, range(NUM_CLASSES))
), "Unexpected number of classes parsed."

base = EfficientNetB3(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    pooling="avg",
)
base.trainable = False

inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base(inputs, training=False)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
my_model = Model(inputs, outputs)

my_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

my_model.summary()

steps_per_epoch = int(np.ceil(len(train_df) / BATCH_SIZE))
validation_steps = int(np.ceil(len(val_df) / BATCH_SIZE))

history = my_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
)



## === cell 2
df_sub = pd.read_csv(SAMPLE_SUB)
df_test = df_sub.copy()
df_test["path"] = TEST_IMG_DIR.rstrip("/") + "/" + df_test["image_id"].astype(str)

print("Test rows:", len(df_test))


def make_test_gen(batch_size=64):
    test_paths = df_test["path"].to_numpy()
    ds_test = tf.data.Dataset.from_tensor_slices(test_paths)
    ds_test = ds_test.map(
        _decode_and_resize, num_parallel_calls=AUTOTUNE, deterministic=True
    )
    ds_test = ds_test.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    return ds_test




## === cell 3
test_gen = make_test_gen(batch_size=128)

pred_test = my_model.predict(
    test_gen,
    verbose=1,
)

pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

final_csv = df_test[["image_id"]].copy()
final_csv["label"] = pred_test_labels

assert len(final_csv) == len(df_sub), "Submission row count mismatch."
assert list(final_csv.columns) == ["image_id", "label"], "Submission columns mismatch."
assert final_csv["label"].between(0, 4).all(), "Predicted labels out of range [0,4]."

final_csv.to_csv("submission.csv", index=False)
print(final_csv.head())
print("Wrote submission.csv with", len(final_csv), "rows")



## === cell 4
print(pd.read_csv("submission.csv").head())
print(
    "submission.csv exists:",
    os.path.exists("submission.csv"),
    "size:",
    os.path.getsize("submission.csv") if os.path.exists("submission.csv") else None,
)
