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

2.7

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
from __future__ import print_function
import os

if "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION" in os.environ:
    os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(0)
try:
    tf.random.set_seed(0)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

print("TensorFlow:", getattr(tf, "__version__", "unknown"))
try:
    print("Eager:", tf.executing_eagerly())
except Exception:
    print("Eager: unknown")




## === cell 1
def _find_existing_dir(candidates):
    for p in candidates:
        if p and os.path.isdir(p):
            return p
    return None


def _find_existing_file(candidates):
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return None


DATA_DIR = _find_existing_dir(
    [
        "../input/cassava-leaf-disease-classification",
        "/kaggle/input/cassava-leaf-disease-classification",
        "../input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
        "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    ]
)
if DATA_DIR is None:
    raise OSError(
        "Could not locate cassava-leaf-disease-classification input directory."
    )

TRAIN_DIR = _find_existing_dir(
    [
        os.path.join(DATA_DIR, "train_images"),
        "../input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/train_images",
        "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/train_images",
    ]
)
TEST_DIR = _find_existing_dir(
    [
        os.path.join(DATA_DIR, "test_images"),
        "../input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/test_images",
        "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/test_images",
    ]
)
train_csv_path = _find_existing_file(
    [
        os.path.join(DATA_DIR, "train.csv"),
        "../input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/train.csv",
        "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/train.csv",
    ]
)
sample_sub_path = _find_existing_file(
    [
        os.path.join(DATA_DIR, "sample_submission.csv"),
        "../input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/sample_submission.csv",
        "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/sample_submission.csv",
    ]
)

if TRAIN_DIR is None or TEST_DIR is None:
    raise OSError(
        "train_images/test_images directories not found in expected locations."
    )
if train_csv_path is None or sample_sub_path is None:
    raise OSError("train.csv/sample_submission.csv not found in expected locations.")

print("DATA_DIR:", DATA_DIR)
print("TRAIN_DIR:", TRAIN_DIR)
print("TEST_DIR:", TEST_DIR)
print("train.csv:", train_csv_path)
print("sample_submission.csv:", sample_sub_path)

train_df = pd.read_csv(train_csv_path)
sample_sub = pd.read_csv(sample_sub_path)
test_df = sample_sub[["image_id"]].copy()

assert "image_id" in train_df.columns and "label" in train_df.columns
assert "image_id" in test_df.columns
print("Train rows:", len(train_df), "Test rows:", len(test_df))
print("Train label distribution:\n", train_df["label"].value_counts().sort_index())




## === cell 2
def random_crop(img, random_crop_size):
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y : (y + dy), x : (x + dx), :]


def crop_generator(batches, crop_length):
    """Take as input a Keras ImageGen (Iterator) and generate random
    crops from the image batches generated by the original iterator.
    """
    while True:
        batch_x = next(batches)
        batch_crops = np.zeros(
            (batch_x.shape[0], crop_length, crop_length, 3), dtype=batch_x.dtype
        )
        for i in range(batch_x.shape[0]):
            batch_crops[i] = random_crop(batch_x[i], (crop_length, crop_length))
        yield batch_crops




## === cell 3
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5
NUM_CLASSES = 5

from sklearn.model_selection import train_test_split

train_df["label"] = train_df["label"].astype(str)

tr_df, va_df = train_test_split(
    train_df,
    test_size=0.15,
    random_state=0,
    stratify=train_df["label"].values,
)

AUTOTUNE = getattr(tf.data, "AUTOTUNE", None)
if AUTOTUNE is None:
    AUTOTUNE = tf.data.experimental.AUTOTUNE

_classes = sorted(tr_df["label"].unique().tolist())
class_indices = dict((c, i) for i, c in enumerate(_classes))
print("class_indices:", class_indices)

tr_image_ids = tr_df["image_id"].values
tr_labels_str = tr_df["label"].values
va_image_ids = va_df["image_id"].values
va_labels_str = va_df["label"].values
te_image_ids = test_df["image_id"].values

tr_paths = (TRAIN_DIR + os.sep + tr_image_ids).astype(str)
va_paths = (TRAIN_DIR + os.sep + va_image_ids).astype(str)
te_paths = (TEST_DIR + os.sep + te_image_ids).astype(str)

tr_labels_int = pd.Categorical(
    tr_labels_str, categories=_classes, ordered=True
).codes.astype(np.int32)
va_labels_int = pd.Categorical(
    va_labels_str, categories=_classes, ordered=True
).codes.astype(np.int32)


@tf.function
def _decode_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) / 255.0
    return img


BASE_SEED = tf.constant([0, 0], dtype=tf.int32)
_ROT_FACTOR = 15.0 / 360.0

_aug_layers = tf.keras.Sequential(
    [
        tf.keras.layers.RandomRotation(factor=_ROT_FACTOR, fill_mode="reflect", seed=0),
        tf.keras.layers.RandomTranslation(
            height_factor=0.10, width_factor=0.10, fill_mode="reflect", seed=0
        ),
        tf.keras.layers.RandomZoom(
            height_factor=(-0.10, 0.10),
            width_factor=(-0.10, 0.10),
            fill_mode="reflect",
            seed=0,
        ),
    ],
    name="aug",
)


@tf.function
def _augment(img, seed):
    img = _aug_layers(img, training=True)
    img = tf.image.stateless_random_flip_left_right(
        img, seed=seed + tf.constant([2, 2], tf.int32)
    )
    return img


_ds_options = tf.data.Options()
try:
    _ds_options.experimental_deterministic = True
except Exception:
    pass
try:
    _ds_options.experimental_optimization.apply_default_optimizations = True
except Exception:
    pass
try:
    _ds_options.experimental_optimization.map_and_batch_fusion = True
except Exception:
    pass
try:
    _ds_options.experimental_optimization.parallel_batch = True
except Exception:
    pass

SHUFFLE_BUFFER = int(min(len(tr_paths), 4096))


def _make_train_ds(paths, labels_int):
    paths = np.asarray(paths)
    labels_int = np.asarray(labels_int, dtype=np.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels_int))
    ds = ds.with_options(_ds_options)

    def _load_noaug(path, y):
        img = _decode_resize(path)
        return img, tf.cast(y, tf.int32)

    ds = ds.map(_load_noaug, num_parallel_calls=AUTOTUNE)
    ds = ds.cache()

    ds = ds.shuffle(buffer_size=SHUFFLE_BUFFER, seed=0, reshuffle_each_iteration=True)
    ds = ds.enumerate()

    def _apply_aug_and_encode(idx, img_y):
        img, y = img_y
        seed = BASE_SEED + tf.cast(tf.stack([idx, idx]), tf.int32)
        img = _augment(img, seed)
        y = tf.one_hot(tf.cast(y, tf.int32), depth=NUM_CLASSES, dtype=tf.float32)
        return img, y

    ds = ds.map(_apply_aug_and_encode, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)

    try:
        ds = ds.prefetch(AUTOTUNE)
    except Exception:
        ds = ds.prefetch(1)
    try:
        ds = ds.apply(tf.data.experimental.prefetch_to_device("/GPU:0", buffer_size=1))
    except Exception:
        pass
    return ds


def _make_val_ds(paths, labels_int):
    paths = np.asarray(paths)
    labels_int = np.asarray(labels_int, dtype=np.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels_int))
    ds = ds.with_options(_ds_options)

    def _load(path, y):
        img = _decode_resize(path)
        y = tf.one_hot(tf.cast(y, tf.int32), depth=NUM_CLASSES, dtype=tf.float32)
        return img, y

    ds = ds.map(_load, num_parallel_calls=AUTOTUNE)
    ds = ds.cache()
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    try:
        ds = ds.prefetch(AUTOTUNE)
    except Exception:
        ds = ds.prefetch(1)
    try:
        ds = ds.apply(tf.data.experimental.prefetch_to_device("/GPU:0", buffer_size=1))
    except Exception:
        pass
    return ds


def _make_test_ds(paths):
    paths = np.asarray(paths)

    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.with_options(_ds_options)

    def _load(path):
        img = _decode_resize(path)
        return img

    ds = ds.map(_load, num_parallel_calls=AUTOTUNE)
    ds = ds.cache()
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    try:
        ds = ds.prefetch(AUTOTUNE)
    except Exception:
        ds = ds.prefetch(1)
    try:
        ds = ds.apply(tf.data.experimental.prefetch_to_device("/GPU:0", buffer_size=1))
    except Exception:
        pass
    return ds


train_dataset = _make_train_ds(tr_paths, tr_labels_int)
val_dataset = _make_val_ds(va_paths, va_labels_int)
test_dataset = _make_test_ds(te_paths)

inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.3)(x)
outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

steps_per_epoch = int(np.ceil(float(len(tr_df)) / float(BATCH_SIZE)))
val_steps = int(np.ceil(float(len(va_df)) / float(BATCH_SIZE)))

history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_dataset,
    validation_steps=val_steps,
    verbose=1,
)




## === cell 4
test_steps = int(np.ceil(float(len(test_df)) / float(BATCH_SIZE)))
pred = model.predict(test_dataset, verbose=1, steps=test_steps)
pred = np.asarray(pred)

if pred.ndim != 2 or pred.shape[1] != NUM_CLASSES:
    raise ValueError("Unexpected prediction shape: {}".format(pred.shape))

predicted_class_indices = np.argmax(pred, axis=1).astype(int)
predicted_class_indices = predicted_class_indices[: len(test_df)]

submission = pd.DataFrame(
    {"image_id": test_df["image_id"].values, "label": predicted_class_indices}
)
submission["label"] = submission["label"].astype(int)
submission = submission[["image_id", "label"]]

out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(submission.head())
print("Rows:", len(submission), "Cols:", submission.columns.tolist())
