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

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input

SEED = 42
DEBUG = False

tf.random.set_seed(SEED)
np.random.seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

tf.config.optimizer.set_jit(False)  # keep off to avoid long XLA compile overhead

CANDIDATE_ROOTS = [
    "../input/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
]
DATA_ROOT = next((p for p in CANDIDATE_ROOTS if os.path.exists(p)), None)
if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not find cassava-leaf-disease-classification dataset directory."
    )

TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")
TRAIN_TFREC_DIR = os.path.join(DATA_ROOT, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(DATA_ROOT, "test_tfrecords")

for p in [TRAIN_CSV, SAMPLE_SUB, TRAIN_IMG_DIR, TEST_IMG_DIR]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing expected path: {p}")

print("Using DATA_ROOT:", DATA_ROOT)
print("TensorFlow:", tf.__version__)




## === cell 1
df_train = pd.read_csv(TRAIN_CSV)
if DEBUG:
    df_train = df_train.sample(2000, random_state=SEED).reset_index(drop=True)

df_train["path"] = TRAIN_IMG_DIR.rstrip("/") + "/" + df_train["image_id"].astype(str)
num_classes = int(df_train["label"].nunique())
assert num_classes == 5, f"Expected 5 classes, got {num_classes}"

val_frac = 0.1
val_mask = (
    pd.util.hash_pandas_object(df_train["image_id"], index=False).values % 100
) < int(val_frac * 100)
df_val = df_train.loc[val_mask].reset_index(drop=True).copy()
df_tr = df_train.loc[~val_mask].reset_index(drop=True).copy()

IMG_SIZE = (300, 300)
BATCH_SIZE = 32 if not DEBUG else 16
EPOCHS = 2 if not DEBUG else 1

AUTO = tf.data.AUTOTUNE

DS_OPTIONS = tf.data.Options()
DS_OPTIONS.experimental_deterministic = True
DS_OPTIONS.experimental_optimization.apply_default_optimizations = True
DS_OPTIONS.experimental_optimization.map_parallelization = True
DS_OPTIONS.experimental_optimization.map_and_batch_fusion = True
DS_OPTIONS.experimental_optimization.parallel_batch = True
DS_OPTIONS.threading.private_threadpool_size = 0
DS_OPTIONS.threading.max_intra_op_parallelism = 0


def _list_tfrecord_files(tfrecord_dir, prefix):
    if not os.path.isdir(tfrecord_dir):
        return []
    files = []
    for fn in tf.io.gfile.listdir(tfrecord_dir):
        if fn.startswith(prefix) and (
            fn.endswith(".tfrec") or fn.endswith(".tfrecord")
        ):
            files.append(os.path.join(tfrecord_dir, fn))
    return sorted(files)


TRAIN_TFRECS = _list_tfrecord_files(TRAIN_TFREC_DIR, "ld_train")
TEST_TFRECS = _list_tfrecord_files(TEST_TFREC_DIR, "ld_test")

USE_TFRECORDS_TRAIN = len(TRAIN_TFRECS) > 0
USE_TFRECORDS_TEST = len(TEST_TFRECS) > 0

print(
    "Found train tfrecords:",
    len(TRAIN_TFRECS),
    "->",
    "using" if USE_TFRECORDS_TRAIN else "not using",
)
print(
    "Found test tfrecords:",
    len(TEST_TFRECS),
    "->",
    "using" if USE_TFRECORDS_TEST else "not using",
)


IMG_H = IMG_SIZE[0]
IMG_W = IMG_SIZE[1]
IMG_SIZE_TF = tf.constant([IMG_H, IMG_W], tf.int32)


@tf.function(reduce_retracing=True)
def _decode_resize_float32_from_jpeg_bytes(img_bytes):
    img = tf.io.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img.set_shape([None, None, 3])
    img = tf.image.resize(
        img, [IMG_H, IMG_W], method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    img.set_shape([IMG_H, IMG_W, 3])
    return img


@tf.function(reduce_retracing=True)
def _decode_resize_float32(path):
    img_bytes = tf.io.read_file(path)
    return _decode_resize_float32_from_jpeg_bytes(img_bytes)


_FEATURE_DESC = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}


@tf.function(reduce_retracing=True)
def _parse_train_example(serialized):
    ex = tf.io.parse_single_example(serialized, _FEATURE_DESC)
    img = _decode_resize_float32_from_jpeg_bytes(ex["image"])
    label = tf.cast(ex["target"], tf.int32)
    image_id = ex["image_name"]
    return img, label, image_id


@tf.function(reduce_retracing=True)
def _parse_test_example(serialized):
    ex = tf.io.parse_single_example(serialized, _FEATURE_DESC)
    img = _decode_resize_float32_from_jpeg_bytes(ex["image"])
    image_id = ex["image_name"]
    return img, image_id


@tf.function(reduce_retracing=True)
def _augment_stateless(img, label, image_id):
    sid = tf.strings.to_hash_bucket_fast(image_id, 2**31 - 1)
    seed = tf.stack([tf.cast(SEED, tf.int32), tf.cast(sid, tf.int32)], axis=0)

    img = tf.image.stateless_random_flip_left_right(img, seed=seed)

    h = tf.cast(IMG_H, tf.float32)
    w = tf.cast(IMG_W, tf.float32)
    t_seed = seed + tf.constant([1, 0], tf.int32)

    dx = tf.random.stateless_uniform([], t_seed, minval=-0.05, maxval=0.05) * w
    dy = (
        tf.random.stateless_uniform(
            [], t_seed + tf.constant([0, 1], tf.int32), minval=-0.05, maxval=0.05
        )
        * h
    )
    img = tf.roll(
        img,
        shift=[tf.cast(tf.round(dy), tf.int32), tf.cast(tf.round(dx), tf.int32)],
        axis=[0, 1],
    )

    z_seed = seed + tf.constant([2, 0], tf.int32)
    zh = tf.random.stateless_uniform([], z_seed, minval=-0.1, maxval=0.1)
    zw = tf.random.stateless_uniform(
        [], z_seed + tf.constant([0, 1], tf.int32), minval=-0.1, maxval=0.1
    )
    scale_h = 1.0 + zh
    scale_w = 1.0 + zw

    def _zoom_in(img_):
        crop_h = tf.cast(tf.round(h * scale_h), tf.int32)
        crop_w = tf.cast(tf.round(w * scale_w), tf.int32)
        crop_h = tf.clip_by_value(crop_h, 1, IMG_H)
        crop_w = tf.clip_by_value(crop_w, 1, IMG_W)
        img_ = tf.image.resize_with_crop_or_pad(img_, crop_h, crop_w)
        img_ = tf.image.resize(
            img_, [IMG_H, IMG_W], method=tf.image.ResizeMethod.BILINEAR, antialias=False
        )
        return img_

    def _zoom_out(img_):
        pad_h = tf.cast(tf.round(h * scale_h), tf.int32)
        pad_w = tf.cast(tf.round(w * scale_w), tf.int32)
        pad_h = tf.maximum(pad_h, IMG_H)
        pad_w = tf.maximum(pad_w, IMG_W)
        img_ = tf.image.resize_with_crop_or_pad(img_, pad_h, pad_w)
        img_ = tf.image.resize(
            img_, [IMG_H, IMG_W], method=tf.image.ResizeMethod.BILINEAR, antialias=False
        )
        return img_

    img = tf.cond(
        tf.logical_and(scale_h <= 1.0, scale_w <= 1.0),
        lambda: _zoom_in(img),
        lambda: _zoom_out(img),
    )

    r_seed = seed + tf.constant([3, 0], tf.int32)
    angle = tf.random.stateless_uniform([], r_seed, minval=-15.0, maxval=15.0) * (
        np.pi / 180.0
    )

    cos_a = tf.math.cos(angle)
    sin_a = tf.math.sin(angle)
    cx = (w - 1.0) / 2.0
    cy = (h - 1.0) / 2.0

    a0 = cos_a
    a1 = -sin_a
    a2 = cx - cos_a * cx + sin_a * cy
    a3 = sin_a
    a4 = cos_a
    a5 = cy - sin_a * cx - cos_a * cy
    transform = tf.stack([a0, a1, a2, a3, a4, a5, 0.0, 0.0])[tf.newaxis, :]

    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[tf.newaxis, ...],
        transforms=transform,
        output_shape=IMG_SIZE_TF,
        interpolation="BILINEAR",
        fill_mode="NEAREST",
        fill_value=0.0,
    )[0]
    img.set_shape([IMG_H, IMG_W, 3])
    return img, label


@tf.function(reduce_retracing=True)
def _decode_resize_augment(path, label):
    img = _decode_resize_float32(path)
    image_id = tf.strings.split(path, os.sep)[-1]
    img, label = _augment_stateless(img, label, image_id)
    return img, label


@tf.function(reduce_retracing=True)
def _decode_and_resize(path):
    return _decode_resize_float32(path)


def make_train_ds_from_files(df, batch_size):
    paths = df["path"].values
    labels = df["label"].values.astype(np.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.with_options(DS_OPTIONS)
    ds = ds.shuffle(
        buffer_size=min(len(df), 8192), seed=SEED, reshuffle_each_iteration=True
    )
    ds = ds.map(_decode_resize_augment, num_parallel_calls=AUTO, deterministic=False)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(AUTO)
    return ds


def make_val_ds_from_files(df, batch_size):
    paths = df["path"].values
    labels = df["label"].values.astype(np.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.with_options(DS_OPTIONS)
    ds = ds.map(
        lambda p, y: (_decode_and_resize(p), y),
        num_parallel_calls=AUTO,
        deterministic=False,
    )
    ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTO)
    return ds


def make_train_val_ds_from_tfrecords(train_tfrecs, df_val_ids_set, batch_size):
    keys = tf.constant(df_val_ids_set, dtype=tf.string)
    vals = tf.ones([tf.shape(keys)[0]], dtype=tf.int32)
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys, vals), default_value=0
    )

    def _is_val(image_id):
        return table.lookup(image_id) > 0

    ds = tf.data.TFRecordDataset(train_tfrecs, num_parallel_reads=AUTO).with_options(
        DS_OPTIONS
    )
    ds = ds.map(_parse_train_example, num_parallel_calls=AUTO, deterministic=False)

    val_ds = ds.filter(lambda img, y, image_id: _is_val(image_id)).map(
        lambda img, y, image_id: (img, y), num_parallel_calls=AUTO, deterministic=False
    )
    tr_ds = ds.filter(lambda img, y, image_id: tf.logical_not(_is_val(image_id))).map(
        lambda img, y, image_id: _augment_stateless(img, y, image_id),
        num_parallel_calls=AUTO,
        deterministic=False,
    )

    tr_ds = tr_ds.shuffle(
        buffer_size=min(int(len(df_train) * (1 - val_frac)), 8192),
        seed=SEED,
        reshuffle_each_iteration=True,
    )
    tr_ds = tr_ds.batch(batch_size, drop_remainder=True).prefetch(AUTO)

    val_ds = val_ds.cache().batch(batch_size, drop_remainder=False).prefetch(AUTO)
    return tr_ds, val_ds


if USE_TFRECORDS_TRAIN:
    val_ids = df_val["image_id"].astype(str).values.tolist()
    train_ds, val_ds = make_train_val_ds_from_tfrecords(
        TRAIN_TFRECS, val_ids, BATCH_SIZE
    )
else:
    train_ds = make_train_ds_from_files(df_tr, BATCH_SIZE)
    val_ds = make_val_ds_from_files(df_val, BATCH_SIZE)

inputs = Input(shape=(IMG_H, IMG_W, 3))
base = EfficientNetB3(include_top=False, weights="imagenet", input_tensor=inputs)

base.trainable = True

x = GlobalAveragePooling2D()(base.output)
x = Dropout(0.2)(x)
outputs = Dense(num_classes, activation="softmax")(x)
my_model = Model(inputs=inputs, outputs=outputs)

my_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history = my_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1,
)




## === cell 2
sample = pd.read_csv(SAMPLE_SUB)
df_test = sample.copy()
df_test["path"] = TEST_IMG_DIR.rstrip("/") + "/" + df_test["image_id"].astype(str)


def make_test_ds_from_files(df, batch_size=128):
    paths = df["path"].values
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.with_options(DS_OPTIONS)
    ds = ds.map(_decode_and_resize, num_parallel_calls=AUTO, deterministic=False)
    ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTO)
    return ds


def make_test_ds_from_tfrecords(test_tfrecs, batch_size=128):
    ds = tf.data.TFRecordDataset(test_tfrecs, num_parallel_reads=AUTO).with_options(
        DS_OPTIONS
    )
    ds = ds.map(_parse_test_example, num_parallel_calls=AUTO, deterministic=False)
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTO)
    return ds


if USE_TFRECORDS_TEST:
    test_ds = make_test_ds_from_tfrecords(TEST_TFRECS, batch_size=128)
else:
    test_ds = make_test_ds_from_files(df_test, batch_size=128)




## === cell 3
if USE_TFRECORDS_TEST:
    sample_ids = tf.constant(sample["image_id"].astype(str).values, dtype=tf.string)
    sample_pos = tf.range(tf.shape(sample_ids)[0], dtype=tf.int32)
    pos_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(sample_ids, sample_pos),
        default_value=-1,
    )

    pred_buf = np.empty((len(sample), num_classes), dtype=np.float32)

    for batch_imgs, batch_ids in test_ds:
        preds = my_model(batch_imgs, training=False).numpy()
        pos = pos_table.lookup(batch_ids)
        pos_np = pos.numpy()
        valid = pos_np >= 0
        pred_buf[pos_np[valid]] = preds[valid]

    pred_test = pred_buf
else:
    pred_test = my_model.predict(test_ds, verbose=1)

pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

final_csv = pd.DataFrame(
    {
        "image_id": sample["image_id"].values,
        "label": pred_test_labels[: len(sample)],
    }
)

assert list(final_csv.columns) == ["image_id", "label"]
assert len(final_csv) == len(sample), (len(final_csv), len(sample))
assert final_csv["label"].isna().sum() == 0

final_csv["label"] = final_csv["label"].astype(int)
final_csv.to_csv("submission.csv", index=False)
print(final_csv.head())
print("Wrote submission.csv with shape:", final_csv.shape)
print("submission.csv columns:", list(final_csv.columns))
