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

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import re
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass



## === cell 1
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.preprocessing import LabelEncoder

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

train_csv["label_encoded"] = LabelEncoder().fit_transform(train_csv["disease"])

train_csv["disease"] = train_csv["disease"].astype(str)
train_csv["label"] = train_csv["label"].astype(str)

train, valid = train_test_split(
    train_csv, test_size=0.2, stratify=train_csv["label"], random_state=SEED
)



## === cell 2
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 32
IMG_SIZE = 224
NUM_CLASSES = train["label_encoded"].nunique()

_ds_options = tf.data.Options()
_ds_options.experimental_deterministic = True

_AUG_SEED = tf.constant([SEED, 0], dtype=tf.int32)

_CACHE_TRAIN_PATH = "/kaggle/working/cache_train_decode_resize"
_CACHE_VALID_PATH = "/kaggle/working/cache_valid_decode_resize"


@tf.function
def _read_decode_resize(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE], antialias=True)
    return image


@tf.function
def _preprocess(image):
    image = tf.cast(image, tf.float32)
    return preprocess_input(image)


@tf.function
def _augment(image, seed):
    seeds = tf.random.experimental.stateless_split(seed, 6)

    image = tf.image.stateless_random_flip_left_right(image, seed=seeds[0])
    image = tf.image.stateless_random_flip_up_down(image, seed=seeds[1])

    zoom = tf.random.stateless_uniform([], seed=seeds[2], minval=0.8, maxval=1.2)
    new_size = tf.cast(tf.round(zoom * IMG_SIZE), tf.int32)

    image = tf.image.resize(image, [new_size, new_size], antialias=True)
    image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE, IMG_SIZE)

    max_shift = int(0.2 * IMG_SIZE)
    dx = tf.random.stateless_uniform(
        [], seed=seeds[3], minval=-max_shift, maxval=max_shift + 1, dtype=tf.int32
    )
    dy = tf.random.stateless_uniform(
        [], seed=seeds[4], minval=-max_shift, maxval=max_shift + 1, dtype=tf.int32
    )
    image = tf.roll(image, shift=[dy, dx], axis=[0, 1])

    shear = tf.random.stateless_uniform([], seed=seeds[5], minval=-0.2, maxval=0.2)
    one = tf.constant(1.0, tf.float32)
    zero = tf.constant(0.0, tf.float32)
    transform = tf.stack([one, shear, zero, zero, one, zero, zero, zero])[None, :]
    image = tf.raw_ops.ImageProjectiveTransformV3(
        images=image[None, ...],
        transforms=transform,
        output_shape=tf.constant([IMG_SIZE, IMG_SIZE], dtype=tf.int32),
        fill_mode="NEAREST",
        interpolation="BILINEAR",
        fill_value=0.0,
    )[0]
    return image


def _make_train_ds(paths, labels):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels)).with_options(_ds_options)
    ds = ds.shuffle(buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True)

    def _decode_resize_keep_path(path, label):
        image = _read_decode_resize(path)
        return path, image, label

    ds = ds.map(_decode_resize_keep_path, num_parallel_calls=AUTOTUNE)
    ds = ds.cache()  # faster than on-disk cache in most Kaggle CPU environments

    @tf.function
    def _augment_preprocess_onehot(path, image, label):
        h = tf.strings.to_hash_bucket_fast(path, 2**31 - 1)
        seed = tf.stack([_AUG_SEED[0], tf.cast(h, tf.int32)])
        image = _augment(image, seed)
        image = _preprocess(image)
        label_oh = tf.one_hot(tf.cast(label, tf.int32), depth=NUM_CLASSES)
        return image, label_oh

    ds = ds.map(_augment_preprocess_onehot, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def _make_valid_ds(paths, labels):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels)).with_options(_ds_options)

    def _decode_resize(path, label):
        image = _read_decode_resize(path)
        return image, label

    ds = ds.map(_decode_resize, num_parallel_calls=AUTOTUNE)

    ds = ds.cache()  # memory cache for speed

    @tf.function
    def _preprocess_onehot(image, label):
        image = _preprocess(image)
        label_oh = tf.one_hot(tf.cast(label, tf.int32), depth=NUM_CLASSES)
        return image, label_oh

    ds = ds.map(_preprocess_onehot, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_paths = train["path"].values
train_labels = train["label_encoded"].values
valid_paths = valid["path"].values
valid_labels = valid["label_encoded"].values

train_ds = _make_train_ds(train_paths, train_labels)
valid_ds = _make_valid_ds(valid_paths, valid_labels)



## === cell 3
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

early_stopping = EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)

learning_rate_reduction = ReduceLROnPlateau(
    monitor="val_loss", patience=2, factor=0.5, min_lr=1e-6, verbose=1
)

base = EfficientNetB0(
    include_top=False, weights="imagenet", input_tensor=Input(shape=(224, 224, 3))
)
x = GlobalAveragePooling2D()(base.output)
x = Dropout(0.2)(x)
out = Dense(NUM_CLASSES, activation="softmax")(x)
model = Model(inputs=base.input, outputs=out)

base.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

EPOCHS = 8

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    callbacks=[early_stopping, learning_rate_reduction],
    verbose=1,
)

disease_by_index = (
    train_csv[["label_encoded", "disease"]]
    .drop_duplicates("label_encoded")
    .sort_values("label_encoded")["disease"]
    .tolist()
)
disease_to_labelnum = {str(v): int(k) for k, v in label_to_disease.items()}
index_to_labelnum = {
    idx: disease_to_labelnum[disease] for idx, disease in enumerate(disease_by_index)
}



## === cell 4
import numpy as np
import pandas as pd
import math

sample_sub = pd.read_csv(
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)

test_df = sample_sub.copy()
test_df["path"] = (
    "/kaggle/input/cassava-leaf-disease-classification/test_images/"
    + test_df["image_id"].astype(str)
)

test_paths = test_df["path"].values

_CACHE_TEST_PATH = "/kaggle/working/cache_test_decode_resize"


def _make_test_ds(paths):
    ds = tf.data.Dataset.from_tensor_slices(paths).with_options(_ds_options)

    def _decode_resize(path):
        image = _read_decode_resize(path)
        return image

    ds = ds.map(_decode_resize, num_parallel_calls=AUTOTUNE)

    ds = ds.cache()

    @tf.function
    def _preprocess_only(image):
        return _preprocess(image)

    ds = ds.map(_preprocess_only, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


test_ds = _make_test_ds(test_paths)

probs = model.predict(test_ds, verbose=1)
pred_indices = np.argmax(probs, axis=1).astype(int)

map_arr = np.fromiter(
    (index_to_labelnum[i] for i in range(NUM_CLASSES)),
    dtype=np.int32,
    count=NUM_CLASSES,
)
pred_labels = map_arr[pred_indices].astype(int)

submission_df = pd.DataFrame(
    {"image_id": test_df["image_id"].to_numpy(), "label": pred_labels}
)

submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"Submission file created: {submission_path}")
print(submission_df.head())
print("Unique predicted labels:", sorted(submission_df["label"].unique().tolist()))
print("Submission shape:", submission_df.shape)
assert submission_df.shape[0] == sample_sub.shape[0]
assert list(submission_df.columns) == ["image_id", "label"]
