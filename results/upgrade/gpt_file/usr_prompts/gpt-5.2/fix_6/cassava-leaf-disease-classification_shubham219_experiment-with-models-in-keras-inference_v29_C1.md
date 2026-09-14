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

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("KERAS_HOME", "/kaggle/working/.keras")

import glob
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import EfficientNetB3

SEED = 42
DEBUG = False

tf.random.set_seed(SEED)
np.random.seed(SEED)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE

if os.path.exists("/kaggle/input/cassava-leaf-disease-classification"):
    BASE_INPUT = "/kaggle/input/cassava-leaf-disease-classification"
elif os.path.exists("../input/cassava-leaf-disease-classification"):
    BASE_INPUT = "../input/cassava-leaf-disease-classification"
else:
    BASE_INPUT = "/kaggle/data/cassava-leaf-disease-classification"

TRAIN_CSV = os.path.join(BASE_INPUT, "train.csv")
SAMPLE_SUB = os.path.join(BASE_INPUT, "sample_submission.csv")
TEST_IMG_DIR = os.path.join(BASE_INPUT, "test_images")
TRAIN_IMG_DIR = os.path.join(BASE_INPUT, "train_images")

print("BASE_INPUT:", BASE_INPUT)
print("TensorFlow:", tf.__version__)



## === cell 1
train_df = pd.read_csv(TRAIN_CSV)
num_classes = int(train_df["label"].nunique())

train_df = train_df.copy()
train_df["label"] = train_df["label"].astype(int)

IMG_SIZE = (300, 300)
BATCH_SIZE = 32
EPOCHS = 2  # keep runtime within budget while still yielding non-random predictions
VAL_SPLIT = 0.10


paths = (TRAIN_IMG_DIR.rstrip("/") + "/" + train_df["image_id"].astype(str)).values
labels = train_df["label"].values.astype(np.int32)

n = len(train_df)
val_n = int(round(n * VAL_SPLIT))
train_n = n - val_n

train_paths = paths[:train_n]
train_labels = labels[:train_n]
valid_paths = paths[train_n:]
valid_labels = labels[train_n:]


def _decode_and_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) / 255.0
    return img


def _augment(img, seed_pair):
    seed0 = tf.stack([seed_pair[0], seed_pair[1]])
    seed1 = tf.stack([seed_pair[0] + 1, seed_pair[1] + 1])
    seed2 = tf.stack([seed_pair[0] + 2, seed_pair[1] + 2])
    seed3 = tf.stack([seed_pair[0] + 3, seed_pair[1] + 3])
    seed4 = tf.stack([seed_pair[0] + 4, seed_pair[1] + 4])

    img = tf.image.stateless_random_flip_left_right(img, seed=seed0)

    max_dx = tf.cast(tf.round(IMG_SIZE[1] * 0.05), tf.int32)
    max_dy = tf.cast(tf.round(IMG_SIZE[0] * 0.05), tf.int32)
    dx = tf.random.stateless_uniform(
        [], seed=seed1, minval=-max_dx, maxval=max_dx + 1, dtype=tf.int32
    )
    dy = tf.random.stateless_uniform(
        [], seed=seed2, minval=-max_dy, maxval=max_dy + 1, dtype=tf.int32
    )
    pad_x = max_dx
    pad_y = max_dy
    img_padded = tf.image.pad_to_bounding_box(
        img, pad_y, pad_x, IMG_SIZE[0] + 2 * pad_y, IMG_SIZE[1] + 2 * pad_x
    )
    img = tf.image.crop_to_bounding_box(
        img_padded, pad_y + dy, pad_x + dx, IMG_SIZE[0], IMG_SIZE[1]
    )

    scale = tf.random.stateless_uniform(
        [], seed=seed3, minval=0.9, maxval=1.1, dtype=tf.float32
    )
    new_h = tf.cast(tf.round(tf.cast(IMG_SIZE[0], tf.float32) * scale), tf.int32)
    new_w = tf.cast(tf.round(tf.cast(IMG_SIZE[1], tf.float32) * scale), tf.int32)
    img2 = tf.image.resize(img, (new_h, new_w), method=tf.image.ResizeMethod.BILINEAR)
    img2 = tf.image.resize_with_crop_or_pad(img2, IMG_SIZE[0], IMG_SIZE[1])
    img = img2

    angle = tf.random.stateless_uniform(
        [], seed=seed4, minval=-10.0, maxval=10.0, dtype=tf.float32
    ) * (np.pi / 180.0)

    c = tf.cast(IMG_SIZE[1], tf.float32) / 2.0
    r = tf.cast(IMG_SIZE[0], tf.float32) / 2.0
    cos_a = tf.math.cos(angle)
    sin_a = tf.math.sin(angle)

    a0 = cos_a
    a1 = -sin_a
    b0 = sin_a
    b1 = cos_a
    a2 = c - a0 * c - a1 * r
    b2 = r - b0 * c - b1 * r
    transform = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])[tf.newaxis, :]

    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[tf.newaxis, ...],
        transforms=transform,
        output_shape=tf.constant([IMG_SIZE[0], IMG_SIZE[1]], dtype=tf.int32),
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )[0]
    return img


def _train_map_fn(path, label, idx):
    img = _decode_and_resize(path)
    seed_pair = (tf.constant(SEED, tf.int32), tf.cast(idx, tf.int32))
    img = _augment(img, seed_pair)
    return img, label


def _valid_map_fn(path, label):
    img = _decode_and_resize(path)
    return img, label


train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_ds = train_ds.shuffle(
    buffer_size=min(8192, train_n), seed=SEED, reshuffle_each_iteration=True
)
train_ds = train_ds.enumerate()
train_ds = train_ds.map(
    lambda idx, data: _train_map_fn(data[0], data[1], idx), num_parallel_calls=AUTOTUNE
)
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

valid_ds = tf.data.Dataset.from_tensor_slices((valid_paths, valid_labels))
valid_ds = valid_ds.map(_valid_map_fn, num_parallel_calls=AUTOTUNE)
valid_ds = valid_ds.cache().batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

inputs = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base = EfficientNetB3(include_top=False, weights="imagenet", input_tensor=inputs)
x = GlobalAveragePooling2D()(base.output)
outputs = Dense(num_classes, activation="softmax")(x)
my_model = Model(inputs=inputs, outputs=outputs)

base.trainable = False
my_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

_ = my_model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    verbose=1,
)

base.trainable = True
my_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
_ = my_model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=1,
    verbose=1,
)



## === cell 2
sample_df = pd.read_csv(SAMPLE_SUB)

sample_df = sample_df.copy()
sample_df["path"] = TEST_IMG_DIR.rstrip("/") + "/" + sample_df["image_id"].astype(str)

if not os.path.exists(sample_df["path"].iloc[0]):
    alt_dir = os.path.join(TEST_IMG_DIR, "test_images")
    sample_df["path"] = alt_dir.rstrip("/") + "/" + sample_df["image_id"].astype(str)

df_test = sample_df[["path", "image_id"]].copy()

test_paths = df_test["path"].values


def _test_map_fn(path):
    img = _decode_and_resize(path)
    return img


test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
test_ds = test_ds.map(_test_map_fn, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.batch(128, drop_remainder=False).prefetch(AUTOTUNE)



## === cell 3
pred_test = my_model.predict(
    test_ds,
    verbose=1,
)
pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

final_csv = pd.DataFrame(
    {
        "image_id": df_test["image_id"].values,
        "label": pred_test_labels,
    }
)

assert (
    final_csv.shape[0] == sample_df.shape[0]
), "Submission row count does not match sample_submission."
assert list(final_csv.columns) == [
    "image_id",
    "label",
], "Submission columns are incorrect."

final_csv.to_csv("submission.csv", index=False)
print(final_csv.head())
print("Wrote submission.csv with", len(final_csv), "rows")



## === cell 4
final_csv.head(10)
