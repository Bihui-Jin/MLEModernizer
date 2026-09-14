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

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import glob, json, shutil, io  # keep original imports available
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Activation,
    Dropout,
    Flatten,
    Dense,
)
from tensorflow.keras import backend as K

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

BASE_PATH = "../input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
TRAIN_IMG_DIR = os.path.join(BASE_PATH, "train_images")
TEST_IMG_DIR = os.path.join(BASE_PATH, "test_images")
TEST_IMG_GLOB = os.path.join(TEST_IMG_DIR, "*.jpg")

sample_sub_path = os.path.join(BASE_PATH, "sample_submission.csv")
if os.path.exists(sample_sub_path):
    _test_image_ids = pd.read_csv(sample_sub_path, usecols=["image_id"])[
        "image_id"
    ].tolist()
    _test_image_paths = [os.path.join(TEST_IMG_DIR, fn) for fn in _test_image_ids]
else:
    _test_image_paths = glob.glob(TEST_IMG_GLOB)

print("TensorFlow:", tf.__version__)
print("Train CSV exists:", os.path.exists(TRAIN_CSV))
print("Train image dir exists:", os.path.isdir(TRAIN_IMG_DIR))
print("Test images found:", len(_test_image_paths))

try:
    tf.config.optimizer.set_jit(False)
except Exception as e:
    print("Could not set XLA JIT flag:", repr(e))

try:
    tf.config.threading.set_inter_op_parallelism_threads(0)
    tf.config.threading.set_intra_op_parallelism_threads(0)
except Exception as e:
    print("Could not set threading options:", repr(e))

CPU_COUNT = os.cpu_count() or 2
GEN_WORKERS = max(2, min(8, CPU_COUNT))
GEN_MAX_QUEUE = 32

AUTOTUNE = tf.data.AUTOTUNE
_DS_OPTIONS = tf.data.Options()
_DS_OPTIONS.experimental_deterministic = True
try:
    _DS_OPTIONS.experimental_optimization.apply_default_optimizations = True
    _DS_OPTIONS.experimental_optimization.autotune = True
    _DS_OPTIONS.experimental_optimization.map_parallelization = True
    _DS_OPTIONS.experimental_optimization.map_and_batch_fusion = True
except Exception:
    pass



## === cell 1
df_train = pd.read_csv(TRAIN_CSV)

df_train["path"] = (TRAIN_IMG_DIR.rstrip("/") + "/") + df_train["image_id"].astype(
    "string"
)
df_train["label"] = df_train["label"].astype("int32")

test_images = list(_test_image_paths)
df_test = pd.DataFrame({"path": pd.Series(test_images, dtype="string")})

IMG_SIZE = 224
SIZE = (IMG_SIZE, IMG_SIZE)
BATCH_SIZE = 32

paths_all = df_train["path"].astype(str).to_numpy()
labels_all = df_train["label"].to_numpy(np.int32)

num_samples = len(paths_all)
val_size = int(round(num_samples * 0.15))
train_size = num_samples - val_size

train_paths = paths_all[:train_size]
train_labels = labels_all[:train_size]
valid_paths = paths_all[train_size:]
valid_labels = labels_all[train_size:]

unique_labels = np.unique(labels_all)
NUM_CLASSES = int(unique_labels.max() + 1)
class_indices = {str(i): i for i in range(NUM_CLASSES)}

print("Detected classes:", NUM_CLASSES, class_indices)

_SEED_T = tf.constant(SEED, tf.int32)
_PI_OVER_180 = tf.constant(np.pi / 180.0, tf.float32)
_AUG_OFFSETS = tf.constant([[1, 0], [2, 0], [3, 0], [4, 0]], tf.int32)


@tf.function
def _decode_and_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) / 255.0
    img.set_shape((IMG_SIZE, IMG_SIZE, 3))
    return img


@tf.function
def _one_hot(label):
    y = tf.one_hot(label, depth=NUM_CLASSES, dtype=tf.float32)
    y.set_shape((NUM_CLASSES,))
    return y


@tf.function
def _augment_image(img, seed2):
    img = tf.image.stateless_random_flip_left_right(img, seed=seed2)

    angle = (
        tf.random.stateless_uniform(
            [],
            seed=seed2 + _AUG_OFFSETS[0],
            minval=-10.0,
            maxval=10.0,
            dtype=tf.float32,
        )
        * _PI_OVER_180
    )
    try:
        img = tf.image.rotate(img, angles=angle, interpolation="BILINEAR")
    except Exception:
        img = img

    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    max_dx = tf.cast(tf.round(0.05 * tf.cast(w, tf.float32)), tf.int32)
    max_dy = tf.cast(tf.round(0.05 * tf.cast(h, tf.float32)), tf.int32)

    dx = tf.random.stateless_uniform(
        [],
        seed=seed2 + _AUG_OFFSETS[1],
        minval=-max_dx,
        maxval=max_dx + 1,
        dtype=tf.int32,
    )
    dy = tf.random.stateless_uniform(
        [],
        seed=seed2 + _AUG_OFFSETS[2],
        minval=-max_dy,
        maxval=max_dy + 1,
        dtype=tf.int32,
    )

    img_pad = tf.pad(img, [[max_dy, max_dy], [max_dx, max_dx], [0, 0]], mode="REFLECT")
    img = tf.image.crop_to_bounding_box(img_pad, max_dy + dy, max_dx + dx, h, w)

    zoom = tf.random.stateless_uniform(
        [],
        seed=seed2 + _AUG_OFFSETS[3],
        minval=0.95,
        maxval=1.05,
        dtype=tf.float32,
    )
    new_h = tf.cast(tf.round(tf.cast(h, tf.float32) * zoom), tf.int32)
    new_w = tf.cast(tf.round(tf.cast(w, tf.float32) * zoom), tf.int32)
    img_zoom = tf.image.resize(
        img, [new_h, new_w], method=tf.image.ResizeMethod.BILINEAR
    )
    img_zoom = tf.image.resize_with_crop_or_pad(img_zoom, h, w)
    img_zoom.set_shape((IMG_SIZE, IMG_SIZE, 3))
    return img_zoom


def make_train_ds(paths, labels, batch_size):
    paths = tf.convert_to_tensor(paths, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.with_options(_DS_OPTIONS)

    def _decode_map(path, label):
        img = _decode_and_resize(path)
        return img, label

    ds = ds.map(_decode_map, num_parallel_calls=AUTOTUNE, deterministic=True)

    ds = ds.cache()

    ds = ds.shuffle(
        buffer_size=int(paths.shape[0]), seed=SEED, reshuffle_each_iteration=True
    )

    ds = ds.enumerate()

    def _aug_map(i, il):
        img, label = il
        seed2 = tf.stack([_SEED_T, tf.cast(i, tf.int32)])
        img = _augment_image(img, seed2)
        y = _one_hot(label)
        return img, y

    ds = ds.map(_aug_map, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=False)

    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_eval_ds(paths, labels_or_none, batch_size, cache=False):
    paths = tf.convert_to_tensor(paths, dtype=tf.string)
    if labels_or_none is None:
        ds = tf.data.Dataset.from_tensor_slices(paths)
        ds = ds.with_options(_DS_OPTIONS)

        def _map_fn(path):
            return _decode_and_resize(path)

        ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE, deterministic=True)
    else:
        labels = tf.convert_to_tensor(labels_or_none, dtype=tf.int32)
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.with_options(_DS_OPTIONS)

        def _map_fn(path, label):
            img = _decode_and_resize(path)
            y = _one_hot(label)
            return img, y

        ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE, deterministic=True)

    if cache:
        ds = ds.cache()

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_train_ds(train_paths, train_labels, BATCH_SIZE)
valid_ds = make_eval_ds(valid_paths, valid_labels, BATCH_SIZE, cache=True)

test_paths = df_test["path"].astype(str).to_numpy()
test_ds = make_eval_ds(test_paths, None, BATCH_SIZE, cache=True)

steps_per_epoch = int(np.ceil(len(train_paths) / BATCH_SIZE))
validation_steps = int(np.ceil(len(valid_paths) / BATCH_SIZE))
test_steps = int(np.ceil(len(test_paths) / BATCH_SIZE))

print("Train samples:", len(train_paths), "steps/epoch:", steps_per_epoch)
print("Valid samples:", len(valid_paths), "validation_steps:", validation_steps)
print("Test samples:", len(test_paths), "test_steps:", test_steps)



## === cell 2
model = Sequential(
    [
        Conv2D(32, (3, 3), input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        Activation("relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3)),
        Activation("relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3)),
        Activation("relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(256),
        Activation("relu"),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation="softmax"),
    ]
)

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()



## === cell 3
EPOCHS = 5  # keep core approach/loop unchanged

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    verbose=1,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
)

pred_test = model.predict(
    test_ds,
    verbose=1,
    steps=test_steps,
)

pred_test_labels = np.argmax(pred_test, axis=-1)

final_submission = df_test.copy()
final_submission["image_id"] = final_submission["path"].str.rsplit("/", n=1).str[-1]
final_submission["label"] = pred_test_labels.astype(int)

final_csv = final_submission[["image_id", "label"]]

sample_sub_path = os.path.join(BASE_PATH, "sample_submission.csv")
if os.path.exists(sample_sub_path):
    sample_sub = pd.read_csv(sample_sub_path)
    final_csv = sample_sub[["image_id"]].merge(final_csv, on="image_id", how="left")
    final_csv["label"] = final_csv["label"].fillna(0).astype(int)

print(final_csv.head())
print("Submission rows:", len(final_csv))

final_csv.to_csv("submission.csv", index=False)
print("Wrote submission.csv")
