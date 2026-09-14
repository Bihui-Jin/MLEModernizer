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
import random
import numpy as np
import pandas as pd

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import tensorflow as tf
from tensorflow.keras import layers, models

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
try:
    tf.keras.utils.set_random_seed(SEED)
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("TF version:", tf.__version__)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass



## === cell 1
IMAGE_SIZE = (512, 512)
BATCH_SIZE = 16
EPOCHS = 1  # minimal training to ensure end-to-end run and a valid submission

BASE_DIR = "../input/cassava-leaf-disease-classification"
TRAIN_CSV = f"{BASE_DIR}/train.csv"
SAMPLE_SUB = f"{BASE_DIR}/sample_submission.csv"
TRAIN_IMG_DIR = f"{BASE_DIR}/train_images/"
TEST_IMG_DIR = f"{BASE_DIR}/test_images/"

TRAIN_TFREC_DIR = f"{BASE_DIR}/train_tfrecords"
TEST_TFREC_DIR = f"{BASE_DIR}/test_tfrecords"

MODEL_DIR = "../input/cassava-leaf-disease-classification-model/model.h5"



## === cell 2
train_df = pd.read_csv(TRAIN_CSV)
sample_df = pd.read_csv(SAMPLE_SUB)

train_df["label"] = train_df["label"].astype(str)

idx = np.arange(len(train_df))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
split = int(0.9 * len(idx))
train_idx, val_idx = idx[:split], idx[split:]

tr_df = train_df.iloc[train_idx].reset_index(drop=True)
va_df = train_df.iloc[val_idx].reset_index(drop=True)

AUTOTUNE = tf.data.AUTOTUNE

class_names = [str(i) for i in sorted(train_df["label"].astype(int).unique())]
NUM_CLASSES = len(class_names)
class_to_idx = {name: i for i, name in enumerate(class_names)}


def _build_paths_and_labels(df, img_dir, has_label=True):
    paths = (img_dir + df["image_id"].astype(str)).values
    if has_label:
        labels = df["label"].map(class_to_idx).astype(np.int32).values
        return paths, labels
    return paths


tr_paths, tr_labels = _build_paths_and_labels(tr_df, TRAIN_IMG_DIR, has_label=True)
va_paths, va_labels = _build_paths_and_labels(va_df, TRAIN_IMG_DIR, has_label=True)
te_paths = _build_paths_and_labels(sample_df, TEST_IMG_DIR, has_label=False)

train_tfrec_files = sorted(tf.io.gfile.glob(os.path.join(TRAIN_TFREC_DIR, "*.tfrec")))
test_tfrec_files = sorted(tf.io.gfile.glob(os.path.join(TEST_TFREC_DIR, "*.tfrec")))
assert len(train_tfrec_files) > 0, "No train tfrecords found"
assert len(test_tfrec_files) > 0, "No test tfrecords found"


@tf.function
def _decode_and_resize_from_jpeg_bytes(jpeg_bytes):
    img = tf.image.decode_jpeg(jpeg_bytes, channels=3)  # dataset is jpg
    img = tf.image.resize(
        img, IMAGE_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img = tf.cast(img, tf.float32) / 255.0  # rescale=1/255
    return img


@tf.function
def _decode_and_resize_from_path(path):
    img = tf.io.read_file(path)
    return _decode_and_resize_from_jpeg_bytes(img)


_SHIFT_FRAC = 0.05
_ZOOM_FRAC = 0.05


@tf.function
def _augment(img, seed_pair):
    img = tf.image.stateless_random_flip_left_right(img, seed=seed_pair)

    k = tf.random.stateless_uniform(
        [],
        seed=seed_pair + tf.constant([1, 0], tf.int32),
        minval=0,
        maxval=4,
        dtype=tf.int32,
    )
    img = tf.image.rot90(img, k=k)

    h = tf.cast(tf.shape(img)[0], tf.float32)
    w = tf.cast(tf.shape(img)[1], tf.float32)
    dx = (
        tf.random.stateless_uniform(
            [],
            seed=seed_pair + tf.constant([2, 0], tf.int32),
            minval=-_SHIFT_FRAC,
            maxval=_SHIFT_FRAC,
        )
        * w
    )
    dy = (
        tf.random.stateless_uniform(
            [],
            seed=seed_pair + tf.constant([3, 0], tf.int32),
            minval=-_SHIFT_FRAC,
            maxval=_SHIFT_FRAC,
        )
        * h
    )

    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.expand_dims(img, 0),
        transforms=tf.expand_dims(
            tf.stack([1.0, 0.0, -dx, 0.0, 1.0, -dy, 0.0, 0.0]), 0
        ),
        output_shape=tf.shape(img)[:2],
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )[0]

    zoom = tf.random.stateless_uniform(
        [],
        seed=seed_pair + tf.constant([4, 0], tf.int32),
        minval=1.0 - _ZOOM_FRAC,
        maxval=1.0 + _ZOOM_FRAC,
    )
    new_h = tf.cast(tf.round(h / zoom), tf.int32)
    new_w = tf.cast(tf.round(w / zoom), tf.int32)
    new_h = tf.clip_by_value(new_h, 1, tf.cast(h, tf.int32))
    new_w = tf.clip_by_value(new_w, 1, tf.cast(w, tf.int32))

    img = tf.image.stateless_random_crop(
        img,
        size=tf.stack([new_h, new_w, 3]),
        seed=seed_pair + tf.constant([5, 0], tf.int32),
    )
    img = tf.image.resize(
        img, IMAGE_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    return img


_DATASET_OPTIONS = tf.data.Options()
_DATASET_OPTIONS.experimental_deterministic = True
try:
    _DATASET_OPTIONS.experimental_optimization.map_vectorization.enabled = True
except Exception:
    pass
try:
    _DATASET_OPTIONS.experimental_optimization.apply_default_optimizations = True
except Exception:
    pass

_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
}


@tf.function
def _parse_train_example(serialized):
    ex = tf.io.parse_single_example(serialized, _FEATURES)
    img = _decode_and_resize_from_jpeg_bytes(ex["image"])
    label = tf.cast(ex["target"], tf.int32)
    return img, label


@tf.function
def _parse_test_example(serialized):
    ex = tf.io.parse_single_example(serialized, _FEATURES)
    img = _decode_and_resize_from_jpeg_bytes(ex["image"])
    return img


def _count_tfrecord_examples(tfrec_files):
    total = 0
    for fn in tfrec_files:
        for _ in tf.compat.v1.io.tf_record_iterator(fn):
            total += 1
    return total


va_image_id_set = set(va_df["image_id"].astype(str).tolist())
va_image_id_const = tf.constant(sorted(list(va_image_id_set)), dtype=tf.string)
va_lookup = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=va_image_id_const,
        values=tf.ones_like(va_image_id_const, dtype=tf.int32),
    ),
    default_value=tf.constant(0, dtype=tf.int32),
)


@tf.function
def _parse_train_with_name(serialized):
    ex = tf.io.parse_single_example(serialized, _FEATURES)
    img = _decode_and_resize_from_jpeg_bytes(ex["image"])
    label = tf.cast(ex["target"], tf.int32)
    name = ex["image_name"]
    return img, label, name


@tf.function
def _is_val(img, label, name):
    return tf.equal(va_lookup.lookup(name), 1)


def make_train_ds_from_tfrecords(tfrec_files):
    ds = tf.data.TFRecordDataset(tfrec_files, num_parallel_reads=AUTOTUNE)
    ds = ds.with_options(_DATASET_OPTIONS)
    ds = ds.map(_parse_train_example, num_parallel_calls=AUTOTUNE, deterministic=True)

    ds = ds.shuffle(buffer_size=2048, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.enumerate()

    @tf.function
    def _map_aug(i, il):
        img, label = il
        seed_pair = tf.stack([tf.cast(SEED, tf.int32), tf.cast(i, tf.int32)])
        img = _augment(img, seed_pair)
        return img, label

    ds = ds.map(_map_aug, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_val_ds_from_tfrecords(tfrec_files, cache_path=None):
    ds = tf.data.TFRecordDataset(tfrec_files, num_parallel_reads=AUTOTUNE)
    ds = ds.with_options(_DATASET_OPTIONS)
    ds = ds.map(_parse_train_with_name, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.filter(_is_val)
    ds = ds.map(
        lambda img, label, name: (img, label),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )

    if cache_path is None:
        ds = ds.cache()
    else:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        ds = ds.cache(cache_path)

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_test_ds_from_tfrecords(tfrec_files, cache_path=None):
    ds = tf.data.TFRecordDataset(tfrec_files, num_parallel_reads=AUTOTUNE)
    ds = ds.with_options(_DATASET_OPTIONS)
    ds = ds.map(_parse_test_example, num_parallel_calls=AUTOTUNE, deterministic=True)

    if cache_path is None:
        ds = ds.cache()
    else:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        ds = ds.cache(cache_path)

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


n_train_total = _count_tfrecord_examples(train_tfrec_files)
n_test_total = _count_tfrecord_examples(test_tfrec_files)
n_val = len(va_df)
n_train = n_train_total - n_val
assert n_train > 0 and n_val > 0, (n_train, n_val, n_train_total)

steps_per_epoch = max(1, int(np.ceil(n_train / BATCH_SIZE)))
validation_steps = max(1, int(np.ceil(n_val / BATCH_SIZE)))
test_steps = max(1, int(np.ceil(n_test_total / BATCH_SIZE)))

train_ds = (
    make_train_ds_from_tfrecords(train_tfrec_files).repeat().take(steps_per_epoch)
)
val_cache_path = "../working/tfdata_cache/val_cache"
test_cache_path = "../working/tfdata_cache/test_cache"
val_ds = make_val_ds_from_tfrecords(train_tfrec_files, cache_path=val_cache_path)
test_ds = make_test_ds_from_tfrecords(test_tfrec_files, cache_path=test_cache_path)

print("NUM_CLASSES:", NUM_CLASSES)
print("train/val/test samples (from TFRecords):", n_train, n_val, n_test_total)
print(
    "steps_per_epoch/validation_steps/test_steps:",
    steps_per_epoch,
    validation_steps,
    test_steps,
)



## === cell 3
inputs = layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(
    train_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    validation_steps=validation_steps,
    verbose=1,
)



## === cell 4
predictions = model.predict(
    test_ds,
    steps=test_steps,
    verbose=1,
)

pred_labels = np.argmax(predictions, axis=1).astype(int)

pred_labels = pred_labels[: len(sample_df)]
assert len(pred_labels) == len(sample_df), (len(pred_labels), len(sample_df))

print("Pred label distribution:", pd.Series(pred_labels).value_counts().to_dict())
print("First 10 preds:", pred_labels[:10])



## === cell 5
submission = pd.DataFrame(
    {"image_id": sample_df["image_id"].values, "label": pred_labels}
)
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with", len(submission), "rows")
assert submission.columns.tolist() == ["image_id", "label"]
assert submission["image_id"].nunique() == len(submission)
assert os.path.exists("submission.csv")
