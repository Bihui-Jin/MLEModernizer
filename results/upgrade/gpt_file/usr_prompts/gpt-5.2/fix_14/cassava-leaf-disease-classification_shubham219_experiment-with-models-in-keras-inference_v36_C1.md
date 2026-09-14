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

import glob
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, Input

SEED = 42
DEBUG = False

tf.keras.utils.set_random_seed(SEED)
tf.config.optimizer.set_jit(False)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE



## === cell 1
BASE_PATH = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
TRAIN_IMG_DIR = os.path.join(BASE_PATH, "train_images")
TEST_IMG_DIR = os.path.join(BASE_PATH, "test_images")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(TRAIN_IMG_DIR), f"Missing {TRAIN_IMG_DIR}"
assert os.path.exists(TEST_IMG_DIR), f"Missing {TEST_IMG_DIR}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"

train_df = pd.read_csv(TRAIN_CSV)
sample_df = pd.read_csv(SAMPLE_SUB)

train_df["path"] = TRAIN_IMG_DIR.rstrip("/") + "/" + train_df["image_id"].astype(str)
sample_df["path"] = TEST_IMG_DIR.rstrip("/") + "/" + sample_df["image_id"].astype(str)

if DEBUG:
    train_df = train_df[train_df["path"].apply(os.path.exists)].reset_index(drop=True)
    sample_df = sample_df[sample_df["path"].apply(os.path.exists)].reset_index(
        drop=True
    )
else:
    assert os.path.exists(train_df["path"].iloc[0])
    assert os.path.exists(sample_df["path"].iloc[0])

num_classes = int(train_df["label"].nunique())
assert num_classes == 5, f"Expected 5 classes, got {num_classes}"

if DEBUG:
    train_df = train_df.sample(2000, random_state=SEED).reset_index(drop=True)

train_df.head()



## === cell 2
IMG_SIZE = (300, 300)
BATCH_SIZE = 32

from sklearn.model_selection import train_test_split

tr_df, va_df = train_test_split(
    train_df,
    test_size=0.1,
    random_state=SEED,
    stratify=train_df["label"],
)

tr_paths = tr_df["path"].to_numpy()
tr_labels = tr_df["label"].to_numpy().astype(np.int32)
va_paths = va_df["path"].to_numpy()
va_labels = va_df["label"].to_numpy().astype(np.int32)

test_paths = sample_df["path"].to_numpy()

preprocess_fn = tf.keras.applications.efficientnet.preprocess_input

_have_tfa = False
try:
    import tensorflow_addons as tfa  # only used if present

    _have_tfa = True
except Exception:
    _have_tfa = False

TRAIN_TFREC_DIR = os.path.join(BASE_PATH, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(BASE_PATH, "test_tfrecords")
_train_tfrecs = sorted(glob.glob(os.path.join(TRAIN_TFREC_DIR, "*.tfrec")))
_test_tfrecs = sorted(glob.glob(os.path.join(TEST_TFREC_DIR, "*.tfrec")))

USE_TFRECORDS = False


@tf.function
def _decode_resize_preprocess_from_jpeg_bytes(jpeg_bytes):
    img = tf.image.decode_jpeg(jpeg_bytes, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    img = preprocess_fn(img)
    return img


@tf.function
def _decode_resize_uint8_from_path(path):
    jpeg_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(jpeg_bytes, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0.0, 255.0), tf.uint8)
    return img


@tf.function
def _decode_resize_from_path(path):
    jpeg_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(jpeg_bytes, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    return img


@tf.function
def _augment(img, seed2):
    img = tf.image.stateless_random_flip_left_right(img, seed=seed2)

    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    max_dx = tf.cast(tf.round(0.05 * tf.cast(w, tf.float32)), tf.int32)
    max_dy = tf.cast(tf.round(0.05 * tf.cast(h, tf.float32)), tf.int32)
    dx = tf.random.stateless_uniform(
        [],
        seed=seed2 + tf.constant([1, 0], tf.int32),
        minval=-max_dx,
        maxval=max_dx + 1,
        dtype=tf.int32,
    )
    dy = tf.random.stateless_uniform(
        [],
        seed=seed2 + tf.constant([0, 1], tf.int32),
        minval=-max_dy,
        maxval=max_dy + 1,
        dtype=tf.int32,
    )
    img = tf.roll(img, shift=[dy, dx], axis=[0, 1])

    scale = tf.random.stateless_uniform(
        [],
        seed=seed2 + tf.constant([2, 2], tf.int32),
        minval=0.9,
        maxval=1.1,
        dtype=tf.float32,
    )
    new_h = tf.cast(
        tf.round(tf.cast(h, tf.float32) / tf.maximum(scale, 1e-6)), tf.int32
    )
    new_w = tf.cast(
        tf.round(tf.cast(w, tf.float32) / tf.maximum(scale, 1e-6)), tf.int32
    )
    new_h = tf.clip_by_value(new_h, 1, h)
    new_w = tf.clip_by_value(new_w, 1, w)
    img = tf.image.stateless_random_crop(
        img, size=[new_h, new_w, 3], seed=seed2 + tf.constant([3, 3], tf.int32)
    )
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)

    if _have_tfa:
        angle = tf.random.stateless_uniform(
            [],
            seed=seed2 + tf.constant([4, 4], tf.int32),
            minval=-10.0 * np.pi / 180.0,
            maxval=10.0 * np.pi / 180.0,
            dtype=tf.float32,
        )
        img = tfa.image.rotate(
            img, angles=angle, interpolation="BILINEAR", fill_mode="reflect"
        )
    return img


@tf.function
def _prep_train_from_decoded(img, label, idx):
    seed2 = tf.stack([tf.cast(SEED, tf.int32), tf.cast(idx, tf.int32)])
    img = _augment(img, seed2)
    img = preprocess_fn(img)
    return img, label


@tf.function
def _prep_train_from_path(path, label, idx):
    img = _decode_resize_from_path(path)
    seed2 = tf.stack([tf.cast(SEED, tf.int32), tf.cast(idx, tf.int32)])
    img = _augment(img, seed2)
    img = preprocess_fn(img)
    return img, label


@tf.function
def _prep_val_from_path(path, label):
    img = _decode_resize_from_path(path)
    img = preprocess_fn(img)
    return img, label


@tf.function
def _prep_test_from_path(path):
    img = _decode_resize_from_path(path)
    img = preprocess_fn(img)
    return img


_TFREC_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}
_TFREC_FEATURES_TEST = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


@tf.function
def _parse_train_tfrec(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TFREC_FEATURES)
    jpeg = ex["image"]
    label = tf.cast(ex["target"], tf.int32)
    name = ex["image_name"]
    return jpeg, label, name


@tf.function
def _parse_test_tfrec(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TFREC_FEATURES_TEST)
    jpeg = ex["image"]
    return jpeg, ex["image_name"]


@tf.function
def _prep_train_from_jpeg(jpeg, label, idx):
    img = tf.image.decode_jpeg(jpeg, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    seed2 = tf.stack([tf.cast(SEED, tf.int32), tf.cast(idx, tf.int32)])
    img = _augment(img, seed2)
    img = preprocess_fn(img)
    return img, label


@tf.function
def _prep_val_from_jpeg(jpeg, label):
    img = _decode_resize_preprocess_from_jpeg_bytes(jpeg)
    return img, label


@tf.function
def _prep_test_from_jpeg(jpeg):
    img = _decode_resize_preprocess_from_jpeg_bytes(jpeg)
    return img


def _with_deterministic_options(ds):
    opts = tf.data.Options()
    opts.experimental_deterministic = True
    opts.experimental_optimization.apply_default_optimizations = True
    opts.experimental_optimization.map_and_batch_fusion = True
    opts.experimental_optimization.parallel_batch = True
    return ds.with_options(opts)


tr_count = len(tr_df)
va_count = len(va_df)
train_steps = int(np.ceil(tr_count / BATCH_SIZE))
val_steps = int(np.ceil(va_count / BATCH_SIZE))

_split_table = None


def make_train_ds(paths, labels, batch_size):
    img_label_ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    img_label_ds = img_label_ds.map(
        lambda p, y: (_decode_resize_uint8_from_path(p), y),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    cache_path = (
        None if DEBUG else os.path.join("/kaggle/working", "cache_train_b3_uint8")
    )
    img_label_ds = img_label_ds.cache(cache_path)

    img_label_ds = img_label_ds.shuffle(
        buffer_size=8192, seed=SEED, reshuffle_each_iteration=True
    )

    idx_ds = tf.data.Dataset.range(len(paths))
    ds = tf.data.Dataset.zip((img_label_ds, idx_ds))

    def _to_train(img_label, idx):
        img_u8, y = img_label
        img = tf.cast(img_u8, tf.float32)
        return _prep_train_from_decoded(img, y, idx)

    ds = ds.map(_to_train, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = _with_deterministic_options(ds)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_val_ds(paths, labels, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(
        lambda p, y: (_decode_resize_uint8_from_path(p), y),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    cache_path = (
        None if DEBUG else os.path.join("/kaggle/working", "cache_val_b3_uint8")
    )
    ds = ds.cache(cache_path)

    def _to_val(img_u8, y):
        img = tf.cast(img_u8, tf.float32)
        img = preprocess_fn(img)
        return img, y

    ds = ds.map(_to_val, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = _with_deterministic_options(ds)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_test_ds(paths, batch_size):
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(_prep_test_from_path, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = _with_deterministic_options(ds)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_train_ds(tr_paths, tr_labels, BATCH_SIZE)
val_ds = make_val_ds(va_paths, va_labels, BATCH_SIZE)
test_ds = make_test_ds(test_paths, 256)

idx_to_label_int = {i: i for i in range(num_classes)}



## === cell 3
inputs = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base = EfficientNetB3(include_top=False, weights="imagenet", input_tensor=inputs)
x = GlobalAveragePooling2D()(base.output)
x = Dropout(0.2)(x)
outputs = Dense(num_classes, activation="softmax")(x)
my_model = Model(inputs=inputs, outputs=outputs)

base.trainable = False
my_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

EPOCHS_HEAD = 2 if not DEBUG else 1

fit_kwargs = {}
my_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_HEAD,
    verbose=1,
    **fit_kwargs,
)

base.trainable = True
for layer in base.layers[:-40]:
    layer.trainable = False

my_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

EPOCHS_FT = 2 if not DEBUG else 1
my_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FT,
    verbose=1,
    **fit_kwargs,
)



## === cell 4
test_ds_for_pred = test_ds

pred_test = my_model.predict(test_ds_for_pred, verbose=1)
pred_test_idx = np.argmax(pred_test, axis=-1).astype(int)

pred_test_labels = pred_test_idx.astype(int)

submission = sample_df[["image_id"]].copy()
submission["label"] = pred_test_labels

sample_check = pd.read_csv(SAMPLE_SUB)
assert len(submission) == len(
    sample_check
), "Submission row count mismatch vs sample_submission.csv"
assert submission["image_id"].isna().sum() == 0
assert set(submission.columns) == {"image_id", "label"}

submission.to_csv("submission.csv", index=False)
submission.head()
