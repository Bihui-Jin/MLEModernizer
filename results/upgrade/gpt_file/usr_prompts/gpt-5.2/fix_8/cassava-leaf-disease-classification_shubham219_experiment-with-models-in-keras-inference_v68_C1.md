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

import glob
import math
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB3

SEED = 42
DEBUG = False

np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE

BASE_PATH = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
TRAIN_IMG_DIR = os.path.join(BASE_PATH, "train_images")
TEST_IMG_DIR = os.path.join(BASE_PATH, "test_images")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")
TRAIN_TFREC_DIR = os.path.join(BASE_PATH, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(BASE_PATH, "test_tfrecords")

assert os.path.exists(TRAIN_CSV), f"Missing train.csv at {TRAIN_CSV}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing train_images dir at {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing test_images dir at {TEST_IMG_DIR}"
assert os.path.exists(SAMPLE_SUB), f"Missing sample_submission.csv at {SAMPLE_SUB}"
assert os.path.isdir(
    TRAIN_TFREC_DIR
), f"Missing train_tfrecords dir at {TRAIN_TFREC_DIR}"
assert os.path.isdir(TEST_TFREC_DIR), f"Missing test_tfrecords dir at {TEST_TFREC_DIR}"

print("TensorFlow:", tf.__version__)




## === cell 1
df = pd.read_csv(TRAIN_CSV)
df["path"] = TRAIN_IMG_DIR + "/" + df["image_id"].astype(str)

num_classes = df["label"].nunique()
assert num_classes == 5, f"Expected 5 classes, got {num_classes}"

IMG_SIZE = (512, 512)
BATCH_SIZE = 16  # keep identical batch size

FEATURE_DESC_TRAIN = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}
FEATURE_DESC_TEST = {
    "image": tf.io.FixedLenFeature([], tf.string),
}


@tf.function
def _decode_image(image_bytes):
    img = tf.io.decode_jpeg(image_bytes, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.ensure_shape(img, [IMG_SIZE[0], IMG_SIZE[1], 3])
    return img


@tf.function
def _rotate_translate_projective(img, angle_rad, tx, ty):
    h = tf.cast(tf.shape(img)[0], tf.float32)
    w = tf.cast(tf.shape(img)[1], tf.float32)
    cx = (w - 1.0) / 2.0
    cy = (h - 1.0) / 2.0

    cos_a = tf.math.cos(angle_rad)
    sin_a = tf.math.sin(angle_rad)

    a0 = cos_a
    a1 = -sin_a
    a3 = sin_a
    a4 = cos_a
    a2 = cx - a0 * cx - a1 * cy
    a5 = cy - a3 * cx - a4 * cy

    a2 = a2 - tx
    a5 = a5 - ty

    transform = tf.stack([a0, a1, a2, a3, a4, a5, 0.0, 0.0])
    transform = tf.reshape(transform, [1, 8])

    out = tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.expand_dims(img, 0),
        transforms=transform,
        output_shape=tf.constant([IMG_SIZE[0], IMG_SIZE[1]]),
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )
    return tf.squeeze(out, 0)


@tf.function
def _augment(img, seed):
    img = tf.image.stateless_random_flip_left_right(img, seed=seed)

    angle = tf.random.stateless_uniform(
        [],
        seed=seed + tf.constant([1, 0], tf.int32),
        minval=-10.0,
        maxval=10.0,
        dtype=tf.float32,
    )
    angle = angle * (np.pi / 180.0)

    dx = tf.random.stateless_uniform(
        [],
        seed=seed + tf.constant([2, 0], tf.int32),
        minval=-0.05,
        maxval=0.05,
        dtype=tf.float32,
    )
    dy = tf.random.stateless_uniform(
        [],
        seed=seed + tf.constant([3, 0], tf.int32),
        minval=-0.05,
        maxval=0.05,
        dtype=tf.float32,
    )

    z = tf.random.stateless_uniform(
        [],
        seed=seed + tf.constant([4, 0], tf.int32),
        minval=0.9,
        maxval=1.1,
        dtype=tf.float32,
    )

    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    new_h = tf.cast(tf.cast(h, tf.float32) / z, tf.int32)
    new_w = tf.cast(tf.cast(w, tf.float32) / z, tf.int32)
    new_h = tf.clip_by_value(new_h, 1, h)
    new_w = tf.clip_by_value(new_w, 1, w)

    img = tf.image.resize_with_crop_or_pad(img, new_h, new_w)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)

    tx = dx * tf.cast(w, tf.float32)
    ty = dy * tf.cast(h, tf.float32)

    img = _rotate_translate_projective(img, angle, tx, ty)
    img = tf.ensure_shape(img, [IMG_SIZE[0], IMG_SIZE[1], 3])
    return img


@tf.function
def _parse_train_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, FEATURE_DESC_TRAIN)
    img = _decode_image(ex["image"])
    label = tf.cast(ex["target"], tf.int32)
    return img, label


@tf.function
def _parse_test_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, FEATURE_DESC_TEST)
    img = _decode_image(ex["image"])
    return img


train_tfrecs = sorted(glob.glob(os.path.join(TRAIN_TFREC_DIR, "*.tfrec")))
assert len(train_tfrecs) > 0, "No training TFRecords found."

rng = np.random.RandomState(SEED)
perm = rng.permutation(len(train_tfrecs))
val_n = max(1, int(round(0.15 * len(train_tfrecs))))
val_idx = set(perm[:val_n].tolist())
train_files = [f for i, f in enumerate(train_tfrecs) if i not in val_idx]
val_files = [f for i, f in enumerate(train_tfrecs) if i in val_idx]


def _with_performance_options(ds):
    options = tf.data.Options()
    options.deterministic = True  # preserve determinism guarantees
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    options.experimental_optimization.map_and_batch_fusion = True
    options.threading.max_intra_op_parallelism = 0
    options.threading.private_threadpool_size = 0
    return ds.with_options(options)


total_count = int(df.shape[0])
val_count = int(round(total_count * (len(val_files) / len(train_tfrecs))))
train_count = total_count - val_count
steps_per_epoch = math.ceil(train_count / BATCH_SIZE)
val_steps = math.ceil(val_count / BATCH_SIZE)

print("Train examples:", train_count, "Val examples:", val_count)
print("Steps/epoch:", steps_per_epoch, "Val steps:", val_steps)


@tf.function
def _parse_decode_augment(example_proto, i):
    ex = tf.io.parse_single_example(example_proto, FEATURE_DESC_TRAIN)
    img = _decode_image(ex["image"])
    label = tf.cast(ex["target"], tf.int32)
    seed = tf.stack([tf.cast(SEED, tf.int32), tf.cast(i, tf.int32)])
    img = _augment(img, seed)
    return img, label


@tf.function
def _parse_decode_only(example_proto):
    ex = tf.io.parse_single_example(example_proto, FEATURE_DESC_TRAIN)
    img = _decode_image(ex["image"])
    label = tf.cast(ex["target"], tf.int32)
    return img, label


TFRECORD_READ_BUFFER = 8 * 1024 * 1024


def make_train_ds(files, batch_size):
    ds = tf.data.TFRecordDataset(
        files, num_parallel_reads=AUTOTUNE, buffer_size=TFRECORD_READ_BUFFER
    )
    ds = _with_performance_options(ds)
    ds = ds.shuffle(8192, seed=SEED, reshuffle_each_iteration=True)

    seed_ctr = tf.data.Dataset.counter()
    ds = tf.data.Dataset.zip((ds, seed_ctr)).map(
        lambda ex, i: _parse_decode_augment(ex, i),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_val_ds(files, batch_size):
    ds = tf.data.TFRecordDataset(
        files, num_parallel_reads=AUTOTUNE, buffer_size=TFRECORD_READ_BUFFER
    )
    ds = _with_performance_options(ds)
    ds = ds.map(_parse_decode_only, num_parallel_calls=AUTOTUNE, deterministic=True)

    ds = ds.cache()

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_train_ds(train_files, BATCH_SIZE)
val_ds = make_val_ds(val_files, BATCH_SIZE)




## === cell 2
try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

base = EfficientNetB3(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
)
base.trainable = False

my_model = Sequential(
    [
        base,
        GlobalAveragePooling2D(),
        Dropout(0.2),
        Dense(num_classes, activation="softmax"),
    ]
)

my_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

EPOCHS = 3

history = my_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    verbose=1,
)




## === cell 3
sample = pd.read_csv(SAMPLE_SUB)

test_tfrecs = sorted(glob.glob(os.path.join(TEST_TFREC_DIR, "*.tfrec")))
assert len(test_tfrecs) > 0, "No test TFRecords found."


def make_test_ds(files, batch_size=32):
    ds = tf.data.TFRecordDataset(
        files, num_parallel_reads=AUTOTUNE, buffer_size=TFRECORD_READ_BUFFER
    )
    ds = _with_performance_options(ds)
    ds = ds.map(_parse_test_example, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


test_ds = make_test_ds(test_tfrecs, batch_size=32)
pred_test = my_model.predict(test_ds, verbose=1)
pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

final_csv = sample[["image_id"]].copy()
final_csv["label"] = pred_test_labels
final_csv["label"] = final_csv["label"].astype(int)
final_csv.to_csv("submission.csv", index=False)

print(final_csv.head())
print("Wrote submission.csv with shape:", final_csv.shape)




## === cell 4
sub = pd.read_csv("submission.csv")
assert list(sub.columns) == ["image_id", "label"]
assert sub.shape[0] == pd.read_csv(SAMPLE_SUB).shape[0]
assert sub["label"].between(0, 4).all()
sub.head()
