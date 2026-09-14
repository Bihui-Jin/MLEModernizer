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

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import json
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import warnings

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
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

try:
    from tensorflow.keras import mixed_precision

    mixed_precision.set_global_policy("mixed_float16")
except Exception as e:
    print("Mixed precision not enabled (continuing in float32). Reason:", repr(e))

try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

print("TF:", tf.__version__)
print("Keras module:", keras.__name__)




## === cell 1
BASE_PATH = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")
TRAIN_DIR = os.path.join(BASE_PATH, "train_images")
TEST_DIR = os.path.join(BASE_PATH, "test_images")
TRAIN_TFREC_DIR = os.path.join(BASE_PATH, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(BASE_PATH, "test_tfrecords")

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing: {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_DIR), f"Missing dir: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing dir: {TEST_DIR}"
assert os.path.isdir(TRAIN_TFREC_DIR), f"Missing dir: {TRAIN_TFREC_DIR}"
assert os.path.isdir(TEST_TFREC_DIR), f"Missing dir: {TEST_TFREC_DIR}"

train_df = pd.read_csv(TRAIN_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB)

assert {"image_id", "label"}.issubset(train_df.columns)
assert {"image_id", "label"}.issubset(sample_sub.columns)

train_df["filepath"] = (TRAIN_DIR + "/" + train_df["image_id"].astype(str)).astype(str)
sample_sub["filepath"] = (TEST_DIR + "/" + sample_sub["image_id"].astype(str)).astype(
    str
)

num_classes = train_df["label"].nunique()
print(
    "Train rows:", len(train_df), "Test rows:", len(sample_sub), "Classes:", num_classes
)




## === cell 2
from sklearn.model_selection import train_test_split

image_size = 512
batch_size = 8  # unchanged
epochs = 3  # unchanged

train_df["label"] = train_df["label"].astype("int32")

trn_df, val_df = train_test_split(
    train_df, test_size=0.15, random_state=SEED, stratify=train_df["label"]
)

from tensorflow.keras.applications.efficientnet import preprocess_input

AUTOTUNE = tf.data.AUTOTUNE

TRAIN_TFRECS = sorted(
    [
        os.path.join(TRAIN_TFREC_DIR, f)
        for f in os.listdir(TRAIN_TFREC_DIR)
        if f.endswith(".tfrec")
    ]
)
TEST_TFRECS = sorted(
    [
        os.path.join(TEST_TFREC_DIR, f)
        for f in os.listdir(TEST_TFREC_DIR)
        if f.endswith(".tfrec")
    ]
)
assert len(TRAIN_TFRECS) > 0, "No train tfrecords found"
assert len(TEST_TFRECS) > 0, "No test tfrecords found"

_TFREC_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}


def _decode_center_crop_resize_from_jpegbytes(img_bytes):
    shape = tf.io.extract_jpeg_shape(img_bytes)  # [height, width, channels]
    h = tf.cast(shape[0], tf.int32)
    w = tf.cast(shape[1], tf.int32)

    side = tf.minimum(h, w)
    offset_y = (h - side) // 2
    offset_x = (w - side) // 2

    crop_window = tf.stack([offset_y, offset_x, side, side])
    img = tf.io.decode_and_crop_jpeg(img_bytes, crop_window=crop_window, channels=3)

    img = tf.image.resize(
        img, [image_size, image_size], method=tf.image.ResizeMethod.BILINEAR
    )
    return img


def _decode_and_resize(path):
    img_bytes = tf.io.read_file(path)
    return _decode_center_crop_resize_from_jpegbytes(img_bytes)


def _make_rng(idx):
    idx = tf.cast(idx, tf.int32)
    return tf.stack([tf.constant(SEED, tf.int32), idx])


def _augment(img, idx):
    img = tf.cast(img, tf.float32)

    seed = _make_rng(idx)

    img = tf.image.stateless_random_flip_left_right(img, seed=seed)

    angle = tf.random.stateless_uniform(
        [],
        seed=seed + tf.constant([1, 0], tf.int32),
        minval=-20.0,
        maxval=20.0,
        dtype=tf.float32,
    )
    angle = angle * (np.pi / 180.0)

    img = tf.image.rotate(img, angles=angle, interpolation="BILINEAR")

    dx = tf.random.stateless_uniform(
        [],
        seed=seed + tf.constant([2, 0], tf.int32),
        minval=-0.1,
        maxval=0.1,
        dtype=tf.float32,
    )
    dy = tf.random.stateless_uniform(
        [],
        seed=seed + tf.constant([3, 0], tf.int32),
        minval=-0.1,
        maxval=0.1,
        dtype=tf.float32,
    )
    tx = tf.cast(dx * tf.cast(image_size, tf.float32), tf.int32)
    ty = tf.cast(dy * tf.cast(image_size, tf.float32), tf.int32)

    pad_x = tf.abs(tx)
    pad_y = tf.abs(ty)
    img_pad = tf.pad(img, [[pad_y, pad_y], [pad_x, pad_x], [0, 0]], mode="SYMMETRIC")
    start_y = pad_y - ty
    start_x = pad_x - tx
    img = tf.image.crop_to_bounding_box(
        img_pad, start_y, start_x, image_size, image_size
    )

    z = tf.random.stateless_uniform(
        [],
        seed=seed + tf.constant([4, 0], tf.int32),
        minval=0.85,
        maxval=1.15,
        dtype=tf.float32,
    )
    new_size = tf.cast(tf.round(z * tf.cast(image_size, tf.float32)), tf.int32)
    new_size = tf.maximum(1, new_size)

    img_z = tf.image.resize(img, [new_size, new_size], method="bilinear")
    if z >= 1.0:
        offset = (new_size - image_size) // 2
        img = tf.image.crop_to_bounding_box(
            img_z, offset, offset, image_size, image_size
        )
    else:
        pad_total = image_size - new_size
        pad_b = pad_total // 2
        pad_a = pad_total - pad_b
        img_p = tf.pad(
            img_z, [[pad_b, pad_a], [pad_b, pad_a], [0, 0]], mode="SYMMETRIC"
        )
        img = tf.image.crop_to_bounding_box(img_p, 0, 0, image_size, image_size)

    img = preprocess_input(img)
    return img


def _prepare_train_from_decoded(img, label, idx):
    img = _augment(img, idx)
    return img, tf.cast(label, tf.int32)


def _prepare_eval_from_decoded(img, label=None):
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    if label is None:
        return img
    return img, tf.cast(label, tf.int32)


def _parse_tfrecord_train(ex):
    x = tf.io.parse_single_example(ex, _TFREC_FEATURES)
    img = _decode_center_crop_resize_from_jpegbytes(x["image"])
    label = tf.cast(x["target"], tf.int32)
    image_id = x["image_name"]
    return img, label, image_id


def _parse_tfrecord_test(ex):
    x = tf.io.parse_single_example(ex, _TFREC_FEATURES)
    img = _decode_center_crop_resize_from_jpegbytes(x["image"])
    image_id = x["image_name"]
    return img, image_id


SHUFFLE_BUFFER = min(len(trn_df), 4096)

options = tf.data.Options()
options.experimental_deterministic = True
options.threading.private_threadpool_size = max(4, (os.cpu_count() or 8))
options.autotune.enabled = True

trn_ids = tf.constant(trn_df["image_id"].values.astype("S"))
val_ids = tf.constant(val_df["image_id"].values.astype("S"))
trn_id_set = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(trn_ids, tf.ones_like(trn_ids, dtype=tf.int32)),
    default_value=0,
)
val_id_set = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(val_ids, tf.ones_like(val_ids, dtype=tf.int32)),
    default_value=0,
)

train_files_ds = tf.data.Dataset.from_tensor_slices(TRAIN_TFRECS)
train_files_ds = train_files_ds.shuffle(
    len(TRAIN_TFRECS), seed=SEED, reshuffle_each_iteration=True
)
train_raw = train_files_ds.interleave(
    lambda fn: tf.data.TFRecordDataset(fn, num_parallel_reads=AUTOTUNE),
    cycle_length=min(8, len(TRAIN_TFRECS)),
    num_parallel_calls=AUTOTUNE,
    deterministic=True,
)
train_raw = train_raw.map(
    _parse_tfrecord_train, num_parallel_calls=AUTOTUNE, deterministic=True
)
train_raw = train_raw.filter(lambda img, y, image_id: trn_id_set.lookup(image_id) > 0)
train_raw = train_raw.map(
    lambda img, y, image_id: (img, y), num_parallel_calls=AUTOTUNE, deterministic=True
)
train_raw = train_raw.apply(tf.data.experimental.ignore_errors())

train_ds = train_raw.shuffle(SHUFFLE_BUFFER, seed=SEED, reshuffle_each_iteration=True)
train_ds = train_ds.enumerate()
train_ds = train_ds.map(
    lambda idx, xy: _prepare_train_from_decoded(xy[0], xy[1], idx),
    num_parallel_calls=AUTOTUNE,
    deterministic=True,
)
train_ds = train_ds.batch(batch_size, drop_remainder=False)
train_ds = train_ds.with_options(options)
train_ds = train_ds.prefetch(AUTOTUNE)

val_files_ds = tf.data.Dataset.from_tensor_slices(TRAIN_TFRECS)
val_raw = val_files_ds.interleave(
    lambda fn: tf.data.TFRecordDataset(fn, num_parallel_reads=AUTOTUNE),
    cycle_length=min(8, len(TRAIN_TFRECS)),
    num_parallel_calls=AUTOTUNE,
    deterministic=True,
)
val_raw = val_raw.map(
    _parse_tfrecord_train, num_parallel_calls=AUTOTUNE, deterministic=True
)
val_raw = val_raw.filter(lambda img, y, image_id: val_id_set.lookup(image_id) > 0)
val_raw = val_raw.map(
    lambda img, y, image_id: (img, y), num_parallel_calls=AUTOTUNE, deterministic=True
)
val_raw = val_raw.apply(tf.data.experimental.ignore_errors())

val_ds = val_raw.map(
    lambda img, y: _prepare_eval_from_decoded(img, y),
    num_parallel_calls=AUTOTUNE,
    deterministic=True,
)
val_ds = (
    val_ds.batch(batch_size, drop_remainder=False)
    .with_options(options)
    .prefetch(AUTOTUNE)
)

steps_per_epoch = int(np.ceil(len(trn_df) / batch_size))
validation_steps = int(np.ceil(len(val_df) / batch_size))

WORKERS = max(2, (os.cpu_count() or 4) - 1)
print("WORKERS (unused with tf.data):", WORKERS)
print(
    f"Using TFRecords for faster I/O + fused center-crop+resize; shuffle_buffer={SHUFFLE_BUFFER}."
)




## === cell 3
from tensorflow.keras.applications import EfficientNetB0

inputs = keras.Input(shape=(image_size, image_size, 3))
base = EfficientNetB0(include_top=False, weights="imagenet", input_tensor=inputs)
x = layers.GlobalAveragePooling2D()(base.output)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
model = keras.Model(inputs, outputs)

base.trainable = True

opt = keras.optimizers.Adam(learning_rate=1e-4)
model.compile(
    optimizer=opt,
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    jit_compile=True,
)

callbacks = [
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_accuracy", factor=0.5, patience=1, verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        "best_model.keras", monitor="val_accuracy", save_best_only=True, verbose=1
    ),
]

model.summary()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=callbacks,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
)

model = keras.models.load_model("best_model.keras")




## === cell 4
test_files_ds = tf.data.Dataset.from_tensor_slices(TEST_TFRECS)
test_raw = test_files_ds.interleave(
    lambda fn: tf.data.TFRecordDataset(fn, num_parallel_reads=AUTOTUNE),
    cycle_length=min(8, len(TEST_TFRECS)),
    num_parallel_calls=AUTOTUNE,
    deterministic=True,
)
test_raw = test_raw.map(
    _parse_tfrecord_test, num_parallel_calls=AUTOTUNE, deterministic=True
)
test_raw = test_raw.apply(tf.data.experimental.ignore_errors())

test_img_ds = test_raw.map(
    lambda img, image_id: _prepare_eval_from_decoded(img, None),
    num_parallel_calls=AUTOTUNE,
    deterministic=True,
)
test_img_ds = test_img_ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)

test_steps = int(np.ceil(len(sample_sub) / batch_size))

probs = model.predict(
    test_img_ds,
    steps=test_steps,
    verbose=1,
)

probs = probs[: len(sample_sub)]
preds = np.argmax(probs, axis=1).astype(int)

test_id_ds = test_raw.map(
    lambda img, image_id: image_id, num_parallel_calls=AUTOTUNE, deterministic=True
)
test_ids = []
for b in test_id_ds.batch(4096):
    test_ids.extend([x.decode("utf-8") for x in b.numpy().tolist()])
test_ids = test_ids[: len(sample_sub)]

if len(test_ids) != len(preds):
    raise RuntimeError(
        f"Test id/pred length mismatch: ids={len(test_ids)} preds={len(preds)}"
    )

pred_map = dict(zip(test_ids, preds))

ordered_preds = sample_sub["image_id"].map(pred_map).values
if np.any(pd.isna(ordered_preds)):
    missing = sample_sub.loc[pd.isna(ordered_preds), "image_id"].head(5).tolist()
    raise RuntimeError(f"Missing predictions for some test ids (example): {missing}")

ordered_preds = ordered_preds.astype(int)

submission = pd.DataFrame(
    {"image_id": sample_sub["image_id"].values, "label": ordered_preds}
)
submission = submission[["image_id", "label"]]
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with rows:", len(submission))
print(
    "submission.csv exists:",
    os.path.exists("submission.csv"),
    "size:",
    os.path.getsize("submission.csv"),
)
