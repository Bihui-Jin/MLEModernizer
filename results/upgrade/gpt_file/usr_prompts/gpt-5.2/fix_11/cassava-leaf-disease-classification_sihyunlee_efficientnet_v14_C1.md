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

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import json
import math
from copy import deepcopy
import matplotlib.pyplot as plt

tf.random.set_seed(42)
np.random.seed(42)

try:
    tf.config.experimental.enable_op_determinism()
except Exception as e:
    print("Warning: could not enable op determinism:", repr(e))

AUTOTUNE = tf.data.AUTOTUNE

tf.config.threading.set_intra_op_parallelism_threads(0)
tf.config.threading.set_inter_op_parallelism_threads(0)

print("TF version:", tf.__version__)
print("Keras version:", keras.__version__)

_HAS_GPU = len(tf.config.list_physical_devices("GPU")) > 0
_DEVICE_PREFETCH = "/GPU:0" if _HAS_GPU else None




## === cell 1
def draw_grpah(history):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, "bo", label="Training loss", markersize=1)
    plt.plot(epochs, val_loss, "b", label="Validation loss", markersize=1)
    plt.legend()
    plt.show()




## === cell 2
def stratifying_data(data):
    data.head()
    orderby_label = []
    sample_rate = [2.0, 1.5, 1.5, 0.3, 1.5]
    for i in range(5):
        orderby_label.append(
            data[data["label"] == i].sample(
                frac=sample_rate[i], replace=True, random_state=42
            )
        )

    stratified_data = pd.concat(orderby_label)
    return stratified_data




## === cell 3
def get_count_by_class(data):
    vc = data["label"].value_counts().reindex(range(5), fill_value=0).astype(int)
    return [(i, (int(vc.loc[i]), data.shape[1])) for i in range(5)]




## === cell 4
PATH = "../input/cassava-leaf-disease-classification/"

if not os.path.exists(os.path.join(PATH, "train.csv")):
    alt = "../input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/"
    if os.path.exists(os.path.join(alt, "train.csv")):
        PATH = alt

data = pd.read_csv(PATH + "train.csv")
with open(PATH + "label_num_to_disease_map.json", "r") as f:
    real_labels = json.load(f)
real_labels = {int(k): v for k, v in real_labels.items()}
data["class_name"] = data.label.map(real_labels)

stratified_data = stratifying_data(data)
print(get_count_by_class(stratified_data))

train, val = train_test_split(
    stratified_data,
    test_size=0.05,
    stratify=stratified_data["class_name"],
    random_state=42,
)

print("train rows:", len(train), "val rows:", len(val))




## === cell 5
IMG_SIZE = (512, 512)
BATCH_SIZE = 6

CLASS_NAMES = [real_labels[i] for i in range(5)]
CLASS_NAME_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

ROTATION_RANGE = 45.0  # degrees
WIDTH_SHIFT_RANGE = 0.2
HEIGHT_SHIFT_RANGE = 0.2
SHEAR_RANGE = 0.2
ZOOM_RANGE = 0.2
HFLIP = True
VFLIP = True

BASE_SEED = tf.constant([42, 123], dtype=tf.int32)

_PI_OVER_180 = tf.constant(math.pi / 180.0, dtype=tf.float32)
_OUT_SHAPE = tf.constant(list(IMG_SIZE), dtype=tf.int32)

_seed_add_1 = tf.constant([1, 0], tf.int32)
_seed_add_2 = tf.constant([2, 0], tf.int32)
_seed_add_3 = tf.constant([3, 0], tf.int32)
_seed_add_4 = tf.constant([4, 0], tf.int32)
_seed_add_5 = tf.constant([5, 0], tf.int32)
_seed_add_7 = tf.constant([7, 0], tf.int32)


def _list_tfrec(prefix):
    tfrec_dir = os.path.join(PATH, prefix + "_tfrecords")
    if not tf.io.gfile.exists(tfrec_dir):
        return []
    files = tf.io.gfile.glob(os.path.join(tfrec_dir, "*.tfrec"))
    return sorted(files)


TRAIN_TFRECS = _list_tfrec("train")
TEST_TFRECS = _list_tfrec("test")
print("Found train tfrecs:", len(TRAIN_TFRECS), "test tfrecs:", len(TEST_TFRECS))


@tf.function
def _read_jpeg(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method="nearest")
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    return img


def _one_hot_from_class_name(class_name):
    return class_name


keys_tensor = tf.constant(CLASS_NAMES, dtype=tf.string)
vals_tensor = tf.constant(list(range(5)), dtype=tf.int64)
table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor),
    default_value=-1,
)


@tf.function
def _apply_affine(img, seed):
    h = tf.cast(tf.shape(img)[0], tf.float32)
    w = tf.cast(tf.shape(img)[1], tf.float32)

    s0 = seed
    s1 = seed + _seed_add_1
    s2 = seed + _seed_add_2
    s3 = seed + _seed_add_3
    s4 = seed + _seed_add_4
    s5 = seed + _seed_add_5

    angle = (
        tf.random.stateless_uniform(
            [], s0, minval=-ROTATION_RANGE, maxval=ROTATION_RANGE
        )
        * _PI_OVER_180
    )
    tx = (
        tf.random.stateless_uniform(
            [], s1, minval=-HEIGHT_SHIFT_RANGE, maxval=HEIGHT_SHIFT_RANGE
        )
        * h
    )
    ty = (
        tf.random.stateless_uniform(
            [], s2, minval=-WIDTH_SHIFT_RANGE, maxval=WIDTH_SHIFT_RANGE
        )
        * w
    )
    shear = tf.random.stateless_uniform([], s3, minval=-SHEAR_RANGE, maxval=SHEAR_RANGE)
    zoom = tf.random.stateless_uniform(
        [], s4, minval=1.0 - ZOOM_RANGE, maxval=1.0 + ZOOM_RANGE
    )

    cx = (w - 1.0) / 2.0
    cy = (h - 1.0) / 2.0

    cos_a = tf.cos(angle)
    sin_a = tf.sin(angle)

    a0 = zoom * cos_a
    a1 = -zoom * sin_a
    b0 = zoom * sin_a
    b1 = zoom * cos_a

    sh = tf.tan(shear)
    m00 = a0 + sh * b0
    m01 = a1 + sh * b1
    m10 = b0
    m11 = b1

    t0 = (cx + ty) - m00 * cx - m01 * cy
    t1 = (cy + tx) - m10 * cx - m11 * cy

    transform = tf.stack([m00, m01, t0, m10, m11, t1, 0.0, 0.0], axis=0)[tf.newaxis, :]

    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[tf.newaxis, ...],
        transforms=transform,
        output_shape=_OUT_SHAPE,
        interpolation="NEAREST",
        fill_mode="REFLECT",
        fill_value=0.0,
    )[0]

    if HFLIP:
        do = tf.random.stateless_uniform([], s5, minval=0.0, maxval=1.0)
        img = tf.cond(do < 0.5, lambda: tf.image.flip_left_right(img), lambda: img)
    if VFLIP:
        do2 = tf.random.stateless_uniform([], s5 + _seed_add_7, minval=0.0, maxval=1.0)
        img = tf.cond(do2 < 0.5, lambda: tf.image.flip_up_down(img), lambda: img)

    return img


_TFREC_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
    "image_name": tf.io.FixedLenFeature([], tf.string, default_value=""),
}


@tf.function
def _decode_tfrec_common(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TFREC_FEATURES)
    img = tf.io.decode_jpeg(ex["image"], channels=3)
    img = tf.image.resize(img, IMG_SIZE, method="nearest")
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    return img, ex["target"], ex["image_name"]


@tf.function
def _decode_tfrec_train(idx, example_proto):
    img, target, _name = _decode_tfrec_common(example_proto)
    seed = BASE_SEED + tf.stack([tf.cast(idx, tf.int32), 0])
    img = _apply_affine(img, seed)
    y = tf.one_hot(tf.cast(target, tf.int32), 5, dtype=tf.float32)
    return img, y


@tf.function
def _decode_tfrec_eval(example_proto):
    img, target, _name = _decode_tfrec_common(example_proto)
    y = tf.one_hot(tf.cast(target, tf.int32), 5, dtype=tf.float32)
    return img, y


@tf.function
def _decode_tfrec_test(example_proto):
    img, _target, _name = _decode_tfrec_common(example_proto)
    return img


def _apply_ds_options(ds):
    options = tf.data.Options()
    options.deterministic = True
    options.autotune.enabled = True
    options.threading.private_threadpool_size = 0
    options.threading.max_intra_op_parallelism = 0
    return ds.with_options(options)


def _infer_tfrec_from_image_ids(image_ids):
    return None


def make_train_val_ds_from_tfrecords(train_df, val_df, shuffle=True):
    val_names = tf.constant(val_df["image_id"].values.astype(str))
    val_lookup = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            val_names, tf.ones_like(val_names, dtype=tf.int32)
        ),
        default_value=0,
    )

    ds = tf.data.TFRecordDataset(TRAIN_TFRECS, num_parallel_reads=AUTOTUNE)
    ds = _apply_ds_options(ds)
    ds = ds.apply(tf.data.experimental.ignore_errors())

    @tf.function
    def _parse_and_flag(example_proto):
        ex = tf.io.parse_single_example(example_proto, _TFREC_FEATURES)
        name = ex["image_name"]
        is_val = tf.not_equal(name, "") & tf.equal(val_lookup.lookup(name), 1)
        return example_proto, is_val

    ds_flagged = ds.map(
        _parse_and_flag, num_parallel_calls=AUTOTUNE, deterministic=True
    )

    val_ds = ds_flagged.filter(lambda ex, is_val: is_val).map(
        lambda ex, is_val: ex, num_parallel_calls=AUTOTUNE, deterministic=True
    )
    train_ds = ds_flagged.filter(lambda ex, is_val: tf.logical_not(is_val)).map(
        lambda ex, is_val: ex, num_parallel_calls=AUTOTUNE, deterministic=True
    )

    if shuffle:
        buf = int(min(len(train_df), 4096))
        train_ds = train_ds.shuffle(
            buffer_size=buf, seed=42, reshuffle_each_iteration=True
        )

    train_ds = train_ds.enumerate()
    train_ds = train_ds.map(
        _decode_tfrec_train, num_parallel_calls=AUTOTUNE, deterministic=True
    )

    val_ds = val_ds.map(
        _decode_tfrec_eval, num_parallel_calls=AUTOTUNE, deterministic=True
    )


    train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False)
    val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=False)

    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)

    if _DEVICE_PREFETCH is not None:
        train_ds = train_ds.apply(
            tf.data.experimental.prefetch_to_device(_DEVICE_PREFETCH)
        )
        val_ds = val_ds.apply(tf.data.experimental.prefetch_to_device(_DEVICE_PREFETCH))

    return train_ds, val_ds


def make_train_ds(df, shuffle=True):
    if len(TRAIN_TFRECS) > 0:
        ds = tf.data.TFRecordDataset(TRAIN_TFRECS, num_parallel_reads=AUTOTUNE)
        ds = _apply_ds_options(ds)
        ds = ds.apply(tf.data.experimental.ignore_errors())
        if shuffle:
            buf = int(min(len(df), 4096))
            ds = ds.shuffle(buffer_size=buf, seed=42, reshuffle_each_iteration=True)
        ds = ds.enumerate()
        ds = ds.map(
            _decode_tfrec_train, num_parallel_calls=AUTOTUNE, deterministic=True
        )
    else:
        file_paths = (PATH + "train_images/" + df["image_id"]).values
        class_names = df["class_name"].values
        ds = tf.data.Dataset.from_tensor_slices((file_paths, class_names))
        if shuffle:
            buf = int(min(len(df), 4096))
            ds = ds.shuffle(buffer_size=buf, seed=42, reshuffle_each_iteration=True)
        ds = ds.enumerate()

        @tf.function
        def _map_fn(idx, x):
            path, cname = x
            img = _read_jpeg(path)
            seed = BASE_SEED + tf.stack([tf.cast(idx, tf.int32), 0])
            img = _apply_affine(img, seed)
            y = tf.one_hot(tf.cast(table.lookup(cname), tf.int32), 5, dtype=tf.float32)
            return img, y

        ds = _apply_ds_options(ds)
        ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE, deterministic=True)

    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    if _DEVICE_PREFETCH is not None:
        ds = ds.apply(tf.data.experimental.prefetch_to_device(_DEVICE_PREFETCH))
    return ds


def make_val_ds(df):
    if len(TRAIN_TFRECS) > 0:
        ds = tf.data.TFRecordDataset(TRAIN_TFRECS, num_parallel_reads=AUTOTUNE)
        ds = _apply_ds_options(ds)
        ds = ds.apply(tf.data.experimental.ignore_errors())

        val_set_names = tf.constant(df["image_id"].values.astype(str))
        val_lookup = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                val_set_names, tf.ones_like(val_set_names, dtype=tf.int32)
            ),
            default_value=0,
        )

        @tf.function
        def _parse_and_filter(example_proto):
            ex = tf.io.parse_single_example(example_proto, _TFREC_FEATURES)
            name = ex["image_name"]
            return tf.not_equal(name, "") & tf.equal(val_lookup.lookup(name), 1)

        ds = ds.filter(_parse_and_filter)
        ds = ds.map(_decode_tfrec_eval, num_parallel_calls=AUTOTUNE, deterministic=True)

    else:
        file_paths = (PATH + "train_images/" + df["image_id"]).values
        class_names = df["class_name"].values
        ds = tf.data.Dataset.from_tensor_slices((file_paths, class_names))

        @tf.function
        def _map_fn(path, cname):
            img = _read_jpeg(path)
            y = tf.one_hot(tf.cast(table.lookup(cname), tf.int32), 5, dtype=tf.float32)
            return img, y

        ds = _apply_ds_options(ds)
        ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE, deterministic=True)
        ds = ds.cache()

    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    if _DEVICE_PREFETCH is not None:
        ds = ds.apply(tf.data.experimental.prefetch_to_device(_DEVICE_PREFETCH))
    return ds


if len(TRAIN_TFRECS) > 0:
    train_set, val_set = make_train_val_ds_from_tfrecords(train, val, shuffle=True)
else:
    train_set = make_train_ds(train, shuffle=True)
    val_set = make_val_ds(val)

train_steps = int(math.ceil(len(train) / BATCH_SIZE))
val_steps = int(math.ceil(len(val) / BATCH_SIZE))
print("train steps:", train_steps)
print("val steps:", val_steps)




## === cell 6
class SEUnit(keras.layers.Layer):
    def __init__(self, feature_map_len, se_ratio, **kwargs):
        super().__init__(**kwargs)
        self.feature_map_len = int(feature_map_len)
        self.se_ratio = float(se_ratio)
        self.global_avg_pool = keras.layers.GlobalAvgPool2D()
        self.reshape = keras.layers.Reshape((1, 1, self.feature_map_len))

        squeeze_filters = int(max(1, round(self.feature_map_len * self.se_ratio)))
        self.squeeze = keras.layers.Conv2D(
            squeeze_filters, kernel_size=1, activation="relu"
        )
        self.excitation = keras.layers.Conv2D(
            self.feature_map_len, kernel_size=1, activation="sigmoid"
        )

    def call(self, inputs):
        Z = inputs
        Z = self.global_avg_pool(Z)
        Z = self.reshape(Z)
        Z = self.squeeze(Z)
        excitation_vector = self.excitation(Z)
        excitation_vector = tf.reshape(
            excitation_vector, [-1, 1, 1, self.feature_map_len]
        )
        return inputs * excitation_vector

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "feature_map_len": self.feature_map_len,
            "se_ratio": self.se_ratio,
        }




## === cell 7
def SiLU(x):
    return x * tf.keras.backend.sigmoid(x)




## === cell 8
class MBConv_v2(keras.layers.Layer):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        multiplier,
        strides,
        name,
        dropout,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.in_channel = int(in_channel)
        self.out_channel = int(out_channel)
        self.kernel_size = int(kernel_size)
        self.multiplier = int(multiplier)
        self.strides = int(strides)
        self.dropout = float(dropout)
        self.Name = name

        self.expansion_layers = []
        bn_axis = 3
        if multiplier != 1:
            self.expansion_layers = [
                keras.layers.Conv2D(
                    filters=self.in_channel * self.multiplier,
                    kernel_size=1,
                    padding="same",
                    use_bias=False,
                ),
                keras.layers.BatchNormalization(axis=bn_axis),
                keras.layers.Activation(SiLU),
            ]

        self.depthwise_layers = [
            keras.layers.DepthwiseConv2D(
                kernel_size=self.kernel_size,
                strides=self.strides,
                padding="same",
                use_bias=False,
            ),
            keras.layers.BatchNormalization(axis=bn_axis),
            keras.layers.Activation(SiLU),
        ]
        se_ratio = 0.25 / self.multiplier
        self.se_unit = SEUnit(self.in_channel * self.multiplier, se_ratio)

        self.reduction_layers = [
            keras.layers.Conv2D(
                filters=self.out_channel, kernel_size=1, padding="same", use_bias=False
            ),
            keras.layers.BatchNormalization(axis=bn_axis),
        ]
        if self.dropout > 0:
            self.reduction_layers.append(
                keras.layers.Dropout(self.dropout, noise_shape=(None, 1, 1, 1))
            )

    def call(self, inputs):
        Z = inputs
        for layer in self.expansion_layers:
            Z = layer(Z)
        for layer in self.depthwise_layers:
            Z = layer(Z)
        Z = self.se_unit(Z)
        for layer in self.reduction_layers:
            Z = layer(Z)

        if self.strides == 1 and self.in_channel == self.out_channel:
            Z = Z + inputs
        return Z

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "in_channel": self.in_channel,
            "out_channel": self.out_channel,
            "kernel_size": self.kernel_size,
            "multiplier": self.multiplier,
            "strides": self.strides,
            "dropout": self.dropout,
            "name": self.Name,
        }




## === cell 9
default_efficient = [
    {
        "in_channel": 32,
        "out_channel": 16,
        "kernel_size": 3,
        "multiplier": 1,
        "strides": 1,
        "n_repeat": 1,
    },
    {
        "in_channel": 16,
        "out_channel": 24,
        "kernel_size": 3,
        "multiplier": 6,
        "strides": 2,
        "n_repeat": 2,
    },
    {
        "in_channel": 24,
        "out_channel": 40,
        "kernel_size": 5,
        "multiplier": 6,
        "strides": 2,
        "n_repeat": 2,
    },
    {
        "in_channel": 40,
        "out_channel": 80,
        "kernel_size": 3,
        "multiplier": 6,
        "strides": 2,
        "n_repeat": 3,
    },
    {
        "in_channel": 80,
        "out_channel": 112,
        "kernel_size": 5,
        "multiplier": 6,
        "strides": 1,
        "n_repeat": 3,
    },
    {
        "in_channel": 112,
        "out_channel": 192,
        "kernel_size": 5,
        "multiplier": 6,
        "strides": 2,
        "n_repeat": 4,
    },
    {
        "in_channel": 192,
        "out_channel": 320,
        "kernel_size": 3,
        "multiplier": 6,
        "strides": 1,
        "n_repeat": 1,
    },
]




## === cell 10
def rf(filters, width_coef):
    depth_divisor = 8
    filters *= width_coef
    new_filters = int(filters + depth_divisor / 2) // depth_divisor * depth_divisor
    new_filters = max(depth_divisor, new_filters)
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)


def rd(repeat, depth_coef):
    return int(math.ceil(repeat * depth_coef))




## === cell 11
def condition_tensor(y, target_class):
    return tf.argmax(y, axis=-1) == target_class




## === cell 12
class condition_penalty_loss(tf.keras.losses.Loss):
    def __init__(self, condition_penalty, **kwargs):
        self.condition_penalty = condition_penalty
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)

        class4 = condition_tensor(y_true, 4)
        predict0 = condition_tensor(y_pred, 0)
        predict4_0 = tf.cast(class4 & predict0, dtype=tf.float32)
        loss4_0 = loss * predict4_0 * 5

        class0123 = class4 == False
        predict4 = condition_tensor(y_pred, 4)
        predict0123_4 = tf.cast(class0123 & predict4, dtype=tf.float32)
        loss0123_4 = loss * predict0123_4 * 2

        class0 = condition_tensor(y_true, 0)
        predict0_4 = tf.cast(class0 & predict4, dtype=tf.float32)
        loss0_4 = loss * predict0_4 * 10

        return loss + loss0123_4 + loss4_0 + loss0_4

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "condition_penalty": self.condition_penalty}




## === cell 13
class EfficientNet(keras.models.Model):
    def __init__(
        self,
        default_efficient,
        width_coef,
        depth_coef,
        resolution,
        dropout,
        dropout_connect=0.2,
        **kwargs
    ):
        super().__init__(**kwargs)
        default_efficient = deepcopy(default_efficient)
        self.default_efficient = default_efficient
        self.width_coef = width_coef
        self.depth_coef = depth_coef
        self.resolution = resolution
        self.dropout = dropout
        self.dropout_connect = dropout_connect
        bn_axis = 3

        self.first_conv = [
            keras.layers.Conv2D(
                rf(32, width_coef),
                kernel_size=3,
                strides=2,
                padding="same",
                use_bias=False,
            ),
            keras.layers.BatchNormalization(axis=bn_axis),
            keras.layers.Activation(SiLU),
        ]

        self.MB_layers = []
        total_n_repeat = sum([block["n_repeat"] for block in default_efficient])
        block_num = 0
        name = "a"
        for block in default_efficient:
            block["in_channel"] = rf(block["in_channel"], width_coef)
            block["out_channel"] = rf(block["out_channel"], width_coef)
            block["n_repeat"] = rd(block["n_repeat"], depth_coef)
            block_args = dict(block)
            block_args.pop("n_repeat")

            drop_rate = dropout_connect * float(block_num) / total_n_repeat
            block_num += 1

            self.MB_layers.append(MBConv_v2(**block_args, dropout=drop_rate, name=name))
            name = chr(ord(name) + 1)
            if block["n_repeat"] > 1:
                block_args["in_channel"] = block_args["out_channel"]
                block_args["strides"] = 1
                for _ in range(block["n_repeat"] - 1):
                    drop_rate = dropout_connect * float(block_num) / total_n_repeat
                    self.MB_layers.append(
                        MBConv_v2(**block_args, dropout=drop_rate, name=name)
                    )
                    block_num += 1
                    name = chr(ord(name) + 1)

        self.last_conv = [
            keras.layers.Conv2D(
                rf(1280, width_coef),
                kernel_size=1,
                strides=1,
                padding="same",
                use_bias=False,
            ),
            keras.layers.BatchNormalization(axis=bn_axis),
            keras.layers.Activation(SiLU),
        ]

        self.top = [
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(5, activation="softmax"),
        ]

    def call(self, inputs):
        Z = inputs
        for layer in self.first_conv:
            Z = layer(Z)
        for layer in self.MB_layers:
            Z = layer(Z)
        for layer in self.last_conv:
            Z = layer(Z)
        for layer in self.top:
            Z = layer(Z)
        return Z

    def get_config(self):
        basic_config = super().get_config()
        return {
            **basic_config,
            "default_efficient": self.default_efficient,
            "width_coef": self.width_coef,
            "depth_coef": self.depth_coef,
            "resolution": self.resolution,
            "dropout": self.dropout,
            "dropout_connect": self.dropout_connect,
        }

    def model(self):
        x = tf.keras.layers.Input(shape=(self.resolution, self.resolution, 3))
        model = keras.models.Model(inputs=x, outputs=self.call(x))

        model.compile(
            optimizer=keras.optimizers.RMSprop(),
            loss=condition_penalty_loss(1),
            metrics=[keras.metrics.CategoricalAccuracy()],
            jit_compile=True,
        )
        return model




## === cell 14
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_categorical_accuracy", mode="max", patience=4, verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        "EfficientNet_best.h5", save_best_only=True, monitor="val_loss", mode="min"
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.1, patience=2, min_lr=1e-6, verbose=1
    ),
]




## === cell 15
count_by_class = get_count_by_class(train)
print(count_by_class)

counts = np.array([count_by_class[i][1][0] for i in range(5)], dtype=np.float64)
class_weights = {i: (1.0 / counts[i]) * (len(train) / 2.0) for i in range(5)}
mi = min(class_weights.values())
print(mi)
for k in list(class_weights.keys()):
    class_weights[k] = (class_weights[k] / mi) * 1.5
class_weights[4] *= 4
print(class_weights)




## === cell 16
MODEL_PATH = "../input/efficientnet-day7/EfficientNet_day7.h5"
efficientNet_B3 = None

custom_objects = {
    "MBConv_v2": MBConv_v2,
    "SEUnit": SEUnit,
    "SiLU": SiLU,
    "condition_penalty_loss": condition_penalty_loss,
}

if os.path.isfile(MODEL_PATH):
    efficientNet_B3 = keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
    print("Using previous Model:", MODEL_PATH)
else:
    efficientNet_B3 = EfficientNet(default_efficient, 1.2, 1.4, 512, 0.3).model()
    print("Using New Model (training because pretrained .h5 not found)")

    history = efficientNet_B3.fit(
        train_set,
        validation_data=val_set,
        epochs=5,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
    )

    if os.path.isfile("EfficientNet_best.h5"):
        efficientNet_B3 = keras.models.load_model(
            "EfficientNet_best.h5", custom_objects=custom_objects
        )
        print("Loaded checkpoint: EfficientNet_best.h5")

assert efficientNet_B3 is not None, "Model was not created/loaded."




## === cell 17
test_csv = pd.read_csv(PATH + "sample_submission.csv")
test_paths = (PATH + "test_images/" + test_csv["image_id"]).values

if len(TEST_TFRECS) > 0:
    test_ds = tf.data.TFRecordDataset(TEST_TFRECS, num_parallel_reads=AUTOTUNE)
    test_ds = _apply_ds_options(test_ds)
    test_ds = test_ds.apply(tf.data.experimental.ignore_errors())
    test_ds = test_ds.map(
        _decode_tfrec_test, num_parallel_calls=AUTOTUNE, deterministic=True
    )

else:
    test_ds = tf.data.Dataset.from_tensor_slices(test_paths)

    @tf.function
    def _map_test(path):
        return _read_jpeg(path)

    test_ds = _apply_ds_options(test_ds)
    test_ds = test_ds.map(_map_test, num_parallel_calls=AUTOTUNE, deterministic=True)
    test_ds = test_ds.cache()

test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
if _DEVICE_PREFETCH is not None:
    test_ds = test_ds.apply(tf.data.experimental.prefetch_to_device(_DEVICE_PREFETCH))

probs = efficientNet_B3.predict(test_ds, verbose=1)
preds = np.argmax(probs, axis=1).astype(int)

submission = test_csv.copy()
submission["label"] = preds
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Saved to:", os.path.abspath("submission.csv"))
