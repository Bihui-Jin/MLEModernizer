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
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import sys
import subprocess
import glob
import numpy as np
import pandas as pd

import tensorflow as tf

SEED = 42
DEBUG = False

np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

print("TF version:", tf.__version__)



## === cell 1
DATA_CANDIDATES = [
    "../input/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
]

DATA_ROOT = None
for p in DATA_CANDIDATES:
    if os.path.exists(p):
        DATA_ROOT = p
        break

if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not find cassava-leaf-disease-classification dataset directory in expected locations: "
        + ", ".join(DATA_CANDIDATES)
    )

TRAIN_CSV_PATH = os.path.join(DATA_ROOT, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")

for req in [TRAIN_CSV_PATH, SAMPLE_SUB_PATH, TRAIN_IMG_DIR, TEST_IMG_DIR]:
    if not os.path.exists(req):
        raise FileNotFoundError(f"Missing required path: {req}")

print("DATA_ROOT:", DATA_ROOT)

try:
    print("Train images:", len(os.listdir(TRAIN_IMG_DIR)))
    print("Test images:", len(os.listdir(TEST_IMG_DIR)))
except Exception:
    print("Train images:", len(glob.glob(os.path.join(TRAIN_IMG_DIR, "*.jpg"))))
    print("Test images:", len(glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg"))))



## === cell 2
train_df = pd.read_csv(TRAIN_CSV_PATH)

train_image_ids = train_df["image_id"].astype(str).values
train_paths = [os.path.join(TRAIN_IMG_DIR, x) for x in train_image_ids]

if len(train_paths) > 0:
    for p in (train_paths[0], train_paths[-1]):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required train image: {p}")

train_df["path"] = train_paths

y = train_df["label"].astype(int).values
num_classes = int(train_df["label"].nunique())
if num_classes != 5:
    print("Warning: expected 5 classes, got:", num_classes)

rng = np.random.RandomState(SEED)
val_mask = np.zeros(len(train_df), dtype=bool)
val_frac = 0.15

for c in np.unique(y):
    idx = np.where(y == c)[0]
    rng.shuffle(idx)
    n_val = max(1, int(round(len(idx) * val_frac)))
    val_mask[idx[:n_val]] = True

df_val = train_df[val_mask].copy().reset_index(drop=True)
df_train = train_df[~val_mask].copy().reset_index(drop=True)

print("Train size:", len(df_train), "Val size:", len(df_val))
print("Train label counts:\n", df_train["label"].value_counts().sort_index())
print("Val label counts:\n", df_val["label"].value_counts().sort_index())



## === cell 3
IMG_SIZE = (512, 512)
BATCH_SIZE = 16  # keep identical

AUTOTUNE = tf.data.AUTOTUNE


def _read_decode_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    return img


def _one_hot(label):
    return tf.one_hot(tf.cast(label, tf.int32), depth=5)


_ROT = 15.0 * np.pi / 180.0
_SHIFT = 0.08
_ZOOM = 0.1

_random_rotation = tf.keras.layers.RandomRotation(
    factor=15.0 / 360.0, fill_mode="nearest", interpolation="bilinear", seed=SEED
)
_random_translation = tf.keras.layers.RandomTranslation(
    height_factor=_SHIFT,
    width_factor=_SHIFT,
    fill_mode="nearest",
    interpolation="bilinear",
    seed=SEED,
)
_random_zoom = tf.keras.layers.RandomZoom(
    height_factor=(-_ZOOM, _ZOOM),
    width_factor=(-_ZOOM, _ZOOM),
    fill_mode="nearest",
    interpolation="bilinear",
    seed=SEED,
)


def _augment(img, seed):
    seed1 = tf.stack([seed, 0])

    img = tf.image.stateless_random_flip_left_right(img, seed=seed1)

    img = _random_rotation(tf.expand_dims(img, 0), training=True)[0]
    img = _random_translation(tf.expand_dims(img, 0), training=True)[0]
    img = _random_zoom(tf.expand_dims(img, 0), training=True)[0]

    return img


def make_train_ds(df, batch_size):
    paths = df["path"].astype(str).values
    labels = df["label"].astype(np.int32).values

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.shuffle(buffer_size=len(df), seed=SEED, reshuffle_each_iteration=True)

    ds = ds.enumerate()

    def _map(i, pl):
        path, label = pl
        img = _read_decode_resize(path)
        img = _augment(img, tf.cast(i, tf.int32) + SEED)
        return img, _one_hot(label)

    ds = ds.map(_map, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_val_ds(df, batch_size):
    paths = df["path"].astype(str).values
    labels = df["label"].astype(np.int32).values
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _map(path, label):
        img = _read_decode_resize(path)
        return img, _one_hot(label)

    ds = ds.map(_map, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_train_ds(df_train, BATCH_SIZE)
val_ds = make_val_ds(df_val, BATCH_SIZE)

steps_per_epoch = int(np.ceil(len(df_train) / BATCH_SIZE))
validation_steps = int(np.ceil(len(df_val) / BATCH_SIZE))

print("steps_per_epoch:", steps_per_epoch, "validation_steps:", validation_steps)



## === cell 4
inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.25)(x)
outputs = tf.keras.layers.Dense(5, activation="softmax")(x)

my_model = tf.keras.Model(inputs=inputs, outputs=outputs)
my_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

my_model.summary()



## === cell 5
EPOCHS = 3 if not DEBUG else 1

history = my_model.fit(
    train_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    validation_steps=validation_steps,
    verbose=1,
)

val_metrics = my_model.evaluate(
    val_ds,
    steps=validation_steps,
    verbose=0,
)
print("Validation metrics:", dict(zip(my_model.metrics_names, val_metrics)))



## === cell 6
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

test_image_ids = sample_sub["image_id"].astype(str).values
test_paths = [os.path.join(TEST_IMG_DIR, x) for x in test_image_ids]

if len(test_paths) > 0:
    for p in (test_paths[0], test_paths[-1]):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required test image: {p}")

sample_sub["path"] = test_paths
df_test = sample_sub[["path"]].copy()


def make_test_ds(df, batch_size=64):
    paths = df["path"].astype(str).values
    ds = tf.data.Dataset.from_tensor_slices(paths)

    def _map(path):
        return _read_decode_resize(path)

    ds = ds.map(_map, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


test_ds = make_test_ds(df_test, batch_size=64)
test_steps = int(np.ceil(len(df_test) / 64))



## === cell 7
pred_test = my_model.predict(
    test_ds,
    steps=test_steps,
    verbose=1,
)
pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

final_csv = sample_sub[["image_id"]].copy()
final_csv["label"] = pred_test_labels.astype(int)

if len(final_csv) != len(sample_sub):
    raise RuntimeError("Submission length mismatch with sample_submission.csv")

final_csv.to_csv("submission.csv", index=False)

print(final_csv.head())
print("Wrote submission.csv with shape:", final_csv.shape)



## === cell 8
_check = pd.read_csv("submission.csv")
print(_check.head())
print(_check.dtypes)
assert list(_check.columns) == ["image_id", "label"]
assert _check["label"].dtype in [np.int64, np.int32, int]
assert len(_check) == len(pd.read_csv(SAMPLE_SUB_PATH))
print("submission.csv OK")
