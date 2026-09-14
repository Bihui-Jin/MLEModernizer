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
import json
import random
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")




## === cell 1
def _import_tf():
    import tensorflow as tf  # noqa: F401

    return tf


try:
    import tensorflow as tf
except AttributeError:
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"
    os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
    import sys

    for m in list(sys.modules.keys()):
        if m.startswith("tensorflow") or m.startswith("google.protobuf"):
            sys.modules.pop(m, None)
    tf = _import_tf()

from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

print("TF version:", tf.__version__)



## === cell 2
BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification/"
TRAIN_DIR = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
TEST_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_images/"

train_csv_path = os.path.join(BASE_DIR, "train.csv")
sample_sub_path = os.path.join(BASE_DIR, "sample_submission.csv")

assert os.path.exists(train_csv_path), f"Missing: {train_csv_path}"
assert os.path.exists(sample_sub_path), f"Missing: {sample_sub_path}"
assert os.path.isdir(TRAIN_DIR), f"Missing dir: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing dir: {TEST_DIR}"



## === cell 3
with open(os.path.join(BASE_DIR, "label_num_to_disease_map.json")) as file:
    map_classes = json.loads(file.read())

print(json.dumps(map_classes, indent=2))



## === cell 4
label_list = [int(key) for key in map_classes.keys()]
print("Labels:", label_list)



## === cell 5
train_df_tmp = pd.read_csv(train_csv_path, usecols=["image_id", "label"])
print(f"Number of train images: {train_df_tmp.shape[0]}")
del train_df_tmp



## === cell 6
IMG_HEIGHT = 400
IMG_WIDTH = 400
batch_size = 32

PRE_TRAINED_MODEL = "../input/xceptionv6/Cassava_Best_Xception_Model_V05.hdf5"
print(
    "Pretrained model exists?",
    os.path.exists(PRE_TRAINED_MODEL),
    "| Path:",
    PRE_TRAINED_MODEL,
)



## === cell 7
AUTOTUNE = tf.data.AUTOTUNE

TFREC_TRAIN_DIR = os.path.join(BASE_DIR, "train_tfrecords")
TFREC_TEST_DIR = os.path.join(BASE_DIR, "test_tfrecords")
USE_TFRECORDS = os.path.isdir(TFREC_TRAIN_DIR) and os.path.isdir(TFREC_TEST_DIR)
print(
    "Using TFRecords?",
    USE_TFRECORDS,
    "| train:",
    TFREC_TRAIN_DIR,
    "| test:",
    TFREC_TEST_DIR,
)

_RESIZE_METHOD = tf.image.ResizeMethod.BILINEAR


@tf.function
def _decode_resize_normalize_from_bytes(img_bytes):
    img = tf.io.decode_jpeg(img_bytes, channels=3)  # uint8 [H,W,3]
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH], method=_RESIZE_METHOD)
    img = tf.cast(img, tf.float32) / 255.0
    return img


@tf.function
def _read_decode_resize(image_path):
    img_bytes = tf.io.read_file(image_path)
    return _decode_resize_normalize_from_bytes(img_bytes)


_TRAIN_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64),
}
_TEST_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


@tf.function
def _basename(image_name):
    parts = tf.strings.split(image_name, sep="/")
    return parts[-1]


def _parse_train_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TRAIN_FEATURES)
    img = _decode_resize_normalize_from_bytes(ex["image"])
    y = tf.cast(ex["label"], tf.int32)
    image_name = _basename(ex["image_name"])
    return img, y, image_name


def _parse_test_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TEST_FEATURES)
    img = _decode_resize_normalize_from_bytes(ex["image"])
    image_name = _basename(ex["image_name"])
    return img, image_name


def _aug_hflip(img, p=0.5):
    do = tf.random.uniform(()) < p
    return tf.cond(do, lambda: tf.image.flip_left_right(img), lambda: img)


def _aug_rand_brightness(img, limit=0.2, p=0.5):
    do = tf.random.uniform(()) < p

    def _apply():
        delta = tf.random.uniform((), -limit, limit, dtype=tf.float32)
        out = img + delta
        return tf.clip_by_value(out, 0.0, 1.0)

    return tf.cond(do, _apply, lambda: img)


def _aug_rand_contrast(img, limit=0.2, p=0.5):
    do = tf.random.uniform(()) < p

    def _apply():
        factor = 1.0 + tf.random.uniform((), -limit, limit, dtype=tf.float32)
        mean = tf.reduce_mean(img, axis=[0, 1], keepdims=True)
        out = (img - mean) * factor + mean
        return tf.clip_by_value(out, 0.0, 1.0)

    return tf.cond(do, _apply, lambda: img)


def _augment_train_like_original(img):
    do_full = tf.random.uniform(()) > 0.5

    def _full():
        x = img
        x = _aug_hflip(x, p=0.5)
        x = _aug_rand_contrast(x, limit=0.2, p=0.5)
        x = _aug_rand_brightness(x, limit=0.2, p=0.5)
        return x

    return tf.cond(do_full, _full, lambda: img)


def _with_tfdata_options(ds, deterministic: bool):
    options = tf.data.Options()
    options.experimental_deterministic = deterministic
    try:
        options.experimental_optimization.map_parallelization = True
        options.experimental_optimization.parallel_batch = True
        options.experimental_optimization.autotune_buffers = True
        options.experimental_slack = True
    except Exception:
        pass
    return ds.with_options(options)


_TRAIN_DECODE_CACHE = "/kaggle/working/_cassava_train_decode_cache"
_USE_TRAIN_DECODE_CACHE = True


@tf.function
def _map_train_path_label_decode_only(path, y):
    img = _read_decode_resize(path)
    return img, y


@tf.function
def _map_train_apply_aug(img, y):
    img = _augment_train_like_original(img)
    return img, y


@tf.function
def _map_val_path_label(path, y):
    img = _read_decode_resize(path)
    return img, y


@tf.function
def _map_test_path(path):
    img = _read_decode_resize(path)
    return img


def _make_dataset(image_ids, labels, data_dir, mode):
    image_paths = tf.strings.join([data_dir, image_ids])
    if labels is None:
        ds = tf.data.Dataset.from_tensor_slices(image_paths)
    else:
        ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    if mode == "TRAIN":
        buffer_size = (
            int(min(int(image_ids.shape[0]), 8192))
            if image_ids.shape[0] is not None
            else 8192
        )
        ds = ds.shuffle(
            buffer_size=buffer_size, seed=SEED, reshuffle_each_iteration=True
        )

    if mode == "TEST":
        ds = ds.map(_map_test_path, num_parallel_calls=AUTOTUNE, deterministic=True)
        ds = ds.cache("/kaggle/working/_cassava_test_cache")
        ds = ds.batch(batch_size, drop_remainder=False)
        ds = ds.prefetch(AUTOTUNE)
        ds = _with_tfdata_options(ds, deterministic=True)
        return ds

    if mode == "TRAIN":
        ds = ds.map(
            _map_train_path_label_decode_only,
            num_parallel_calls=AUTOTUNE,
            deterministic=False,
        )
        if _USE_TRAIN_DECODE_CACHE:
            ds = ds.cache(_TRAIN_DECODE_CACHE)
        ds = ds.map(
            _map_train_apply_aug, num_parallel_calls=AUTOTUNE, deterministic=False
        )
        ds = ds.batch(batch_size, drop_remainder=False)
        ds = ds.prefetch(AUTOTUNE)
        ds = _with_tfdata_options(ds, deterministic=False)
        return ds

    ds = ds.map(_map_val_path_label, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.cache("/kaggle/working/_cassava_val_cache")
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    ds = _with_tfdata_options(ds, deterministic=True)
    return ds


def _list_tfrec_files(tfrec_dir):
    files = tf.io.gfile.glob(os.path.join(tfrec_dir, "*.tfrec"))
    return sorted(files)


def _tfrec_train_has_label():
    try:
        files = _list_tfrec_files(TFREC_TRAIN_DIR)
        if not files:
            return False
        ds = tf.data.TFRecordDataset(files[:1]).take(1)
        raw = next(iter(ds))
        ex = tf.train.Example.FromString(raw.numpy())
        return "label" in ex.features.feature
    except Exception:
        return False


_TFREC_CACHE_DIR = "/kaggle/working/_cassava_tfrec_cache"
os.makedirs(_TFREC_CACHE_DIR, exist_ok=True)


def _make_train_val_from_tfrecords(train_image_ids, val_image_ids):
    train_image_ids = np.asarray(train_image_ids).astype(str)
    val_image_ids = np.asarray(val_image_ids).astype(str)

    keys = tf.constant(
        np.concatenate([train_image_ids, val_image_ids]), dtype=tf.string
    )
    vals = tf.constant(
        np.concatenate(
            [
                np.zeros(len(train_image_ids), dtype=np.int32),
                np.ones(len(val_image_ids), dtype=np.int32),
            ]
        ),
        dtype=tf.int32,
    )
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys, vals),
        default_value=tf.constant(-1, dtype=tf.int32),
    )

    files = _list_tfrec_files(TFREC_TRAIN_DIR)

    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTOTUNE)
    ds = _with_tfdata_options(ds, deterministic=False)
    ds = ds.map(_parse_train_example, num_parallel_calls=AUTOTUNE, deterministic=False)

    def _split_id(img, y, image_name):
        split = table.lookup(image_name)  # -1 unknown, 0 train, 1 val
        return img, y, split

    ds = ds.map(_split_id, num_parallel_calls=AUTOTUNE, deterministic=False)
    ds = ds.filter(lambda img, y, split: split >= 0)

    train_raw = ds.filter(lambda img, y, split: tf.equal(split, 0))
    val_raw = ds.filter(lambda img, y, split: tf.equal(split, 1))

    def _drop_split(img, y, split):
        return img, y

    train_ds = train_raw.map(
        _drop_split, num_parallel_calls=AUTOTUNE, deterministic=False
    )
    train_ds = train_ds.shuffle(
        buffer_size=8192, seed=SEED, reshuffle_each_iteration=True
    )
    train_ds = train_ds.map(
        lambda img, y: (_augment_train_like_original(img), y),
        num_parallel_calls=AUTOTUNE,
        deterministic=False,
    )
    train_ds = train_ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    train_ds = _with_tfdata_options(train_ds, deterministic=False)

    val_ds = val_raw.map(_drop_split, num_parallel_calls=AUTOTUNE, deterministic=True)
    val_ds = val_ds.cache(os.path.join(_TFREC_CACHE_DIR, "val_cache"))
    val_ds = val_ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    val_ds = _with_tfdata_options(val_ds, deterministic=True)

    return train_ds, val_ds


def _make_test_from_tfrecords():
    files = _list_tfrec_files(TFREC_TEST_DIR)
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTOTUNE)
    ds = _with_tfdata_options(ds, deterministic=True)
    ds = ds.map(_parse_test_example, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.map(
        lambda img, image_name: img, num_parallel_calls=AUTOTUNE, deterministic=True
    )
    ds = (
        ds.cache("/kaggle/working/_cassava_test_tfrec_cache")
        .batch(batch_size, drop_remainder=False)
        .prefetch(AUTOTUNE)
    )
    ds = _with_tfdata_options(ds, deterministic=True)
    return ds




## === cell 8
train_df = pd.read_csv(train_csv_path)
print(train_df.head())
print("Train shape:", train_df.shape)
num_classes = train_df["label"].nunique()
print("Num classes:", num_classes)

sample_sub = pd.read_csv(sample_sub_path)
test_df = sample_sub[["image_id"]].copy()
test_samples = test_df.shape[0]
print("Test samples:", test_samples)



## === cell 9
from sklearn.model_selection import train_test_split

train_ids, val_ids, train_y, val_y = train_test_split(
    train_df["image_id"].values,
    train_df["label"].values,
    test_size=0.1,
    random_state=SEED,
    stratify=train_df["label"].values,
)

USE_TFRECORDS_FOR_TRAIN = bool(USE_TFRECORDS and _tfrec_train_has_label())
print("Using TFRecords for training?", USE_TFRECORDS_FOR_TRAIN)

if USE_TFRECORDS_FOR_TRAIN:
    train_ds, val_ds = _make_train_val_from_tfrecords(train_ids, val_ids)
else:
    train_ds = _make_dataset(
        image_ids=tf.constant(train_ids),
        labels=tf.constant(train_y, dtype=tf.int32),
        data_dir=tf.constant(TRAIN_DIR),
        mode="TRAIN",
    )
    val_ds = _make_dataset(
        image_ids=tf.constant(val_ids),
        labels=tf.constant(val_y, dtype=tf.int32),
        data_dir=tf.constant(TRAIN_DIR),
        mode="VALIDATE",
    )

if USE_TFRECORDS:
    test_ds = _make_test_from_tfrecords()
else:
    test_ds = _make_dataset(
        image_ids=tf.constant(test_df["image_id"].values),
        labels=None,
        data_dir=tf.constant(TEST_DIR),
        mode="TEST",
    )

train_steps = int(np.ceil(len(train_ids) / batch_size))
val_steps = int(np.ceil(len(val_ids) / batch_size))
test_steps = int(np.ceil(len(test_df) / batch_size))
print("Batches - train/val/test:", train_steps, val_steps, test_steps)





## === cell 10
def build_model(img_height=IMG_HEIGHT, img_width=IMG_WIDTH, n_classes=5):
    inputs = keras.Input(shape=(img_height, img_width, 3))
    x = keras.applications.xception.preprocess_input(inputs * 255.0)
    base = keras.applications.Xception(
        include_top=False, weights="imagenet", input_tensor=x
    )
    base.trainable = False  # keep fast and stable

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    return model


model = build_model(n_classes=5)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    jit_compile=False,
)
model.summary()



## === cell 11
EPOCHS = 2
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)



## === cell 12
pred_probs = model.predict(test_ds, verbose=1)
pred_labels = np.argmax(pred_probs, axis=1).astype(int)

print("Pred shape:", pred_probs.shape, "Labels shape:", pred_labels.shape)
assert len(pred_labels) == len(test_df), "Prediction length mismatch with test_df"
assert np.all(
    (pred_labels >= 0) & (pred_labels <= 4)
), "Predicted labels out of range 0..4"



## === cell 13
submission = sample_sub[["image_id"]].copy()
submission["label"] = pred_labels

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print("Wrote:", submission_path)
print(submission.head(10))

check = pd.read_csv(submission_path)
print(check.shape)
print(check.head(3))
print("Columns:", list(check.columns))
assert list(check.columns) == ["image_id", "label"]
assert check["image_id"].nunique() == len(check), "Duplicate image_id in submission"
assert check["label"].between(0, 4).all(), "Labels out of expected range 0..4"
assert os.path.exists(submission_path) and submission_path.endswith(".csv")
