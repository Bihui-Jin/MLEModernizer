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

3.13

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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import re
from datetime import datetime

import numpy as np
import pandas as pd

import tensorflow as tf

print("TF version:", tf.__version__)
print("Eager:", tf.executing_eagerly())

tf.keras.utils.set_random_seed(42)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    tf.data.experimental.enable_debug_mode(False)
except Exception:
    pass




## === cell 1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

label_to_disease = pd.read_json(
    "/kaggle/input/cassava-leaf-disease-classification/label_num_to_disease_map.json",
    typ="series",
)

train_csv = pd.read_csv("/kaggle/input/cassava-leaf-disease-classification/train.csv")

train_csv["disease"] = train_csv["label"].map(label_to_disease)

train_csv["path"] = (
    "/kaggle/input/cassava-leaf-disease-classification/train_images/"
    + train_csv["image_id"].astype(str)
)

le = LabelEncoder()
train_csv["label_encoded"] = le.fit_transform(train_csv["disease"])

train_csv["disease"] = train_csv["disease"].astype(str)
train_csv["label"] = train_csv["label"].astype(str)

train, valid = train_test_split(
    train_csv, test_size=0.2, stratify=train_csv["label"], random_state=42
)

class_names = sorted(train["disease"].unique().tolist())
class_indices = {name: i for i, name in enumerate(class_names)}

NUM_CLASSES = len(class_names)
print("Num classes:", NUM_CLASSES)
print("Class indices (disease->idx):", class_indices)
print("Train samples:", len(train))
print("Valid samples:", len(valid))




## === cell 2
AUTOTUNE = tf.data.AUTOTUNE
IMG_SIZE = 224
BATCH_SIZE = 32  # keep identical
SEED = 42

_IMG_SIZE_F = tf.constant(float(IMG_SIZE), dtype=tf.float32)
_HALF_C = _IMG_SIZE_F / 2.0
_MAX_ANGLE = tf.constant(45.0 * np.pi / 180.0, dtype=tf.float32)

_disease_to_idx = {name: i for i, name in enumerate(class_names)}


@tf.function(jit_compile=False)
def _decode_and_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(
        img, (IMG_SIZE, IMG_SIZE), method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.image.convert_image_dtype(img, tf.float32) * 255.0  # match previous 0..255
    return img


@tf.function(jit_compile=False)
def _augment_like_datagen_batch(imgs, seed2s):
    seed2s = tf.cast(seed2s, tf.int32)
    batch_seed2 = tf.reduce_sum(seed2s)

    seed_lr = tf.stack([tf.constant(SEED, tf.int32), batch_seed2], axis=0)
    seed_ud = tf.stack([tf.constant(SEED + 1, tf.int32), batch_seed2], axis=0)

    imgs = tf.image.stateless_random_flip_left_right(imgs, seed=seed_lr)
    imgs = tf.image.stateless_random_flip_up_down(imgs, seed=seed_ud)

    b = tf.shape(imgs)[0]
    rnd = tf.random.stateless_uniform(
        [b, 6],
        seed=tf.stack([tf.constant(SEED + 2, tf.int32), batch_seed2]),
        minval=0.0,
        maxval=1.0,
        dtype=tf.float32,
    )

    angle = (rnd[:, 0] * 2.0 - 1.0) * _MAX_ANGLE
    tx = (rnd[:, 1] * 2.0 - 1.0) * 0.2 * _IMG_SIZE_F
    ty = (rnd[:, 2] * 2.0 - 1.0) * 0.2 * _IMG_SIZE_F
    zoom = 1.0 + (rnd[:, 3] * 2.0 - 1.0) * 0.2
    shear = (rnd[:, 4] * 2.0 - 1.0) * 0.2

    cos_a = tf.cos(angle) / zoom
    sin_a = tf.sin(angle) / zoom

    a0 = cos_a + shear * sin_a
    a1 = -sin_a
    a2 = (1.0 - a0) * _HALF_C - a1 * _HALF_C - tx

    b0 = sin_a + shear * cos_a
    b1 = cos_a
    b2 = (1.0 - b1) * _HALF_C - b0 * _HALF_C - ty

    transform = tf.stack(
        [a0, a1, a2, b0, b1, b2, tf.zeros_like(a0), tf.zeros_like(a0)], axis=1
    )

    imgs = tf.raw_ops.ImageProjectiveTransformV3(
        images=imgs,
        transforms=transform,
        output_shape=tf.constant([IMG_SIZE, IMG_SIZE], dtype=tf.int32),
        interpolation="BILINEAR",
        fill_mode="NEAREST",
        fill_value=0.0,
    )
    return imgs


@tf.function(jit_compile=False)
def _train_after_decode_batch(imgs, label_idx):
    label_i32 = tf.cast(label_idx, tf.int32)
    seed2s = tf.bitcast(label_i32 * tf.constant(2654435761, tf.int32), tf.int32)

    imgs = _augment_like_datagen_batch(imgs, seed2s)
    imgs = preprocess_input(imgs)
    labels = tf.one_hot(
        tf.cast(label_idx, tf.int32), depth=NUM_CLASSES, dtype=tf.float32
    )
    return imgs, labels


@tf.function(jit_compile=False)
def _valid_after_decode(img, label_idx):
    img = preprocess_input(img)
    label = tf.one_hot(
        tf.cast(label_idx, tf.int32), depth=NUM_CLASSES, dtype=tf.float32
    )
    return img, label


train_paths = train["path"].to_numpy(dtype=object)
valid_paths = valid["path"].to_numpy(dtype=object)

train_label_idx = train["disease"].map(_disease_to_idx).to_numpy(dtype=np.int64)
valid_label_idx = valid["disease"].map(_disease_to_idx).to_numpy(dtype=np.int64)

opts = tf.data.Options()
opts.deterministic = True
opts.experimental_optimization.map_parallelization = True
opts.experimental_optimization.parallel_batch = True
opts.experimental_optimization.apply_default_optimizations = True
opts.threading.private_threadpool_size = 0
opts.threading.max_intra_op_parallelism = 0

shuffle_buffer = min(len(train_paths), 4096)

train_cache_path = "/kaggle/working/train_decode_cache"
valid_cache_path = "/kaggle/working/valid_decode_cache"

train_ds = tf.data.Dataset.from_tensor_slices(
    (train_paths, train_label_idx)
).with_options(opts)
train_ds = train_ds.map(
    lambda p, y: (_decode_and_resize(p), y),
    num_parallel_calls=AUTOTUNE,
    deterministic=True,
).apply(tf.data.experimental.ignore_errors())
train_ds = train_ds.cache(train_cache_path)
train_ds = train_ds.shuffle(shuffle_buffer, seed=SEED, reshuffle_each_iteration=True)
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False)
train_ds = train_ds.map(
    _train_after_decode_batch, num_parallel_calls=AUTOTUNE, deterministic=True
)
train_ds = train_ds.prefetch(AUTOTUNE)

valid_ds = tf.data.Dataset.from_tensor_slices(
    (valid_paths, valid_label_idx)
).with_options(opts)
valid_ds = valid_ds.map(
    lambda p, y: (_decode_and_resize(p), y),
    num_parallel_calls=AUTOTUNE,
    deterministic=True,
).apply(tf.data.experimental.ignore_errors())
valid_ds = valid_ds.cache(valid_cache_path)
valid_ds = valid_ds.map(
    _valid_after_decode, num_parallel_calls=AUTOTUNE, deterministic=True
)
valid_ds = valid_ds.batch(BATCH_SIZE, drop_remainder=False)
valid_ds = valid_ds.prefetch(AUTOTUNE)




## === cell 3
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback


class EarlyStoppingCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if self.model.stop_training:
            print(f"Early stopping triggered at epoch {epoch + 1}.")


early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1,
)

learning_rate_reduction = ReduceLROnPlateau(
    monitor="val_loss",
    patience=2,
    factor=0.5,
    min_lr=1e-7,
    verbose=1,
)




## === cell 4
base_model = EfficientNetB0(
    include_top=False, weights="imagenet", input_shape=(224, 224, 3)
)
x = base_model.output
x = GlobalAveragePooling2D()(x)
outputs = Dense(NUM_CLASSES, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

try:
    tf.data.experimental.assert_cardinality(
        train_ds, int(np.ceil(len(train_paths) / BATCH_SIZE))
    )
except Exception:
    pass

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=10,
    callbacks=[early_stopping, learning_rate_reduction, EarlyStoppingCallback()],
    verbose=1,
)




## === cell 5
sample_sub = pd.read_csv(
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
test_image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"

sample_sub["path"] = (
    test_image_dir.rstrip("/") + "/" + sample_sub["image_id"].astype(str)
)
test_paths = sample_sub["path"].to_numpy(dtype=object)


@tf.function(jit_compile=False)
def make_test_example(path):
    img = _decode_and_resize(path)
    img = preprocess_input(img)
    return img


test_cache_path = "/kaggle/working/test_decode_cache"

test_ds = tf.data.Dataset.from_tensor_slices(test_paths).with_options(opts)
test_ds = test_ds.map(
    _decode_and_resize, num_parallel_calls=AUTOTUNE, deterministic=True
).apply(tf.data.experimental.ignore_errors())
test_ds = test_ds.cache(test_cache_path)
test_ds = test_ds.map(
    lambda img: preprocess_input(img), num_parallel_calls=AUTOTUNE, deterministic=True
)
test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False)
test_ds = test_ds.prefetch(AUTOTUNE)

probs = model.predict(test_ds, verbose=1)
pred_idx = np.argmax(probs, axis=1)

idx_to_disease = {v: k for k, v in class_indices.items()}

label_to_disease_dict = label_to_disease.to_dict()
disease_to_label = {str(v): int(k) for k, v in label_to_disease_dict.items()}

pred_disease = [idx_to_disease[int(i)] for i in pred_idx]
pred_label = [disease_to_label[str(d)] for d in pred_disease]

submission_df = pd.DataFrame(
    {"image_id": sample_sub["image_id"].values, "label": pred_label}
)
submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)

print("Submission file created:", submission_path)
print(submission_df.head())
print("Rows:", len(submission_df), "Cols:", submission_df.shape[1])
print("Unique labels:", sorted(submission_df["label"].unique().tolist()))
