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
    + train_csv["image_id"]
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
BATCH_SIZE = 32  # match the original generator batch size for training/validation
SEED = 42

_IMG_SIZE_F = tf.constant(float(IMG_SIZE), dtype=tf.float32)
_HALF_C = _IMG_SIZE_F / 2.0
_MAX_ANGLE = tf.constant(45.0 * np.pi / 180.0, dtype=tf.float32)

_class_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(class_names, dtype=tf.string),
        values=tf.constant(list(range(NUM_CLASSES)), dtype=tf.int64),
    ),
    default_value=tf.constant(-1, dtype=tf.int64),
)


@tf.function
def _decode_and_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(
        img, (IMG_SIZE, IMG_SIZE), method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32)
    return img


@tf.function
def _augment_like_datagen(img):
    img = tf.image.random_flip_left_right(img, seed=SEED)
    img = tf.image.random_flip_up_down(img, seed=SEED + 1)

    rnd = tf.random.stateless_uniform([6], seed=[SEED, 123], minval=0.0, maxval=1.0)

    angle = (rnd[0] * 2.0 - 1.0) * _MAX_ANGLE
    tx = (rnd[1] * 2.0 - 1.0) * 0.2 * _IMG_SIZE_F
    ty = (rnd[2] * 2.0 - 1.0) * 0.2 * _IMG_SIZE_F
    zoom = 1.0 + (rnd[3] * 2.0 - 1.0) * 0.2
    shear = (rnd[4] * 2.0 - 1.0) * 0.2

    cos_a = tf.cos(angle) / zoom
    sin_a = tf.sin(angle) / zoom

    a0 = cos_a + shear * sin_a
    a1 = -sin_a
    a2 = (1.0 - a0) * _HALF_C - a1 * _HALF_C - tx

    b0 = sin_a + shear * cos_a
    b1 = cos_a
    b2 = (1.0 - b1) * _HALF_C - b0 * _HALF_C - ty

    transform = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])[tf.newaxis, :]

    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[tf.newaxis, ...],
        transforms=transform,
        output_shape=tf.constant([IMG_SIZE, IMG_SIZE], dtype=tf.int32),
        interpolation="BILINEAR",
        fill_mode="NEAREST",  # match fill_mode="nearest"
        fill_value=0.0,
    )[0]
    return img


@tf.function
def make_train_example_from_decoded(img, disease_str):
    img = _augment_like_datagen(img)
    img = preprocess_input(img)  # match original EfficientNet preprocess
    idx = _class_table.lookup(tf.cast(disease_str, tf.string))
    label = tf.one_hot(tf.cast(idx, tf.int32), depth=NUM_CLASSES, dtype=tf.float32)
    return img, label


@tf.function
def make_valid_example_from_decoded(img, disease_str):
    img = preprocess_input(img)
    idx = _class_table.lookup(tf.cast(disease_str, tf.string))
    label = tf.one_hot(tf.cast(idx, tf.int32), depth=NUM_CLASSES, dtype=tf.float32)
    return img, label


train_paths = train["path"].values
train_disease = train["disease"].values
valid_paths = valid["path"].values
valid_disease = valid["disease"].values

opts = tf.data.Options()
opts.deterministic = True

train_cache_path = "/kaggle/working/tf_cache_train_decoded"
valid_cache_path = "/kaggle/working/tf_cache_valid_decoded"

train_base_ds = tf.data.Dataset.from_tensor_slices(
    (train_paths, train_disease)
).with_options(opts)
train_base_ds = train_base_ds.shuffle(
    len(train_paths), seed=SEED, reshuffle_each_iteration=True
)

train_base_ds = train_base_ds.map(
    lambda p, d: (_decode_and_resize(p), d),
    num_parallel_calls=AUTOTUNE,
    deterministic=True,
).cache(train_cache_path)

train_ds = (
    train_base_ds.map(
        make_train_example_from_decoded,
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

valid_base_ds = tf.data.Dataset.from_tensor_slices(
    (valid_paths, valid_disease)
).with_options(opts)
valid_base_ds = valid_base_ds.map(
    lambda p, d: (_decode_and_resize(p), d),
    num_parallel_calls=AUTOTUNE,
    deterministic=True,
).cache(valid_cache_path)

valid_ds = (
    valid_base_ds.map(
        make_valid_example_from_decoded,
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)



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
sample_sub["path"] = sample_sub["image_id"].apply(
    lambda x: os.path.join(test_image_dir, x)
)

test_paths = sample_sub["path"].values

test_cache_path = "/kaggle/working/tf_cache_test_decoded"


@tf.function
def make_test_example_from_decoded(img):
    img = preprocess_input(img)
    return img


test_base_ds = tf.data.Dataset.from_tensor_slices(test_paths).with_options(opts)
test_base_ds = test_base_ds.map(
    _decode_and_resize,
    num_parallel_calls=AUTOTUNE,
    deterministic=True,
).cache(test_cache_path)

test_ds = (
    test_base_ds.map(
        make_test_example_from_decoded,
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

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
