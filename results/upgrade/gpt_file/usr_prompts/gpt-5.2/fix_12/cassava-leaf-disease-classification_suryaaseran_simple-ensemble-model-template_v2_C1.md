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

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("TensorFlow:", tf.__version__)
print("Skipping /kaggle/input file walk (sanity check) to save time.")




## === cell 1
BASE_CANDIDATES = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
]
DATA_BASE = None
for c in BASE_CANDIDATES:
    if os.path.exists(c):
        DATA_BASE = c
        break

if DATA_BASE is None:
    raise FileNotFoundError(
        "Could not locate cassava dataset directory under /kaggle/input. "
        f"Tried: {BASE_CANDIDATES}"
    )

SAMPLE_SUB_PATH = os.path.join(DATA_BASE, "sample_submission.csv")
TRAIN_CSV_PATH = os.path.join(DATA_BASE, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_BASE, "train_images")
TEST_IMG_DIR = os.path.join(DATA_BASE, "test_images")

print("Using DATA_BASE:", DATA_BASE)
print("Train CSV:", TRAIN_CSV_PATH, "exists:", os.path.exists(TRAIN_CSV_PATH))
print("Train image dir:", TRAIN_IMG_DIR, "isdir:", os.path.isdir(TRAIN_IMG_DIR))
print(
    "Sample submission path:",
    SAMPLE_SUB_PATH,
    "exists:",
    os.path.exists(SAMPLE_SUB_PATH),
)
print("Test image dir:", TEST_IMG_DIR, "isdir:", os.path.isdir(TEST_IMG_DIR))

sample = pd.read_csv(SAMPLE_SUB_PATH)
train_df = pd.read_csv(TRAIN_CSV_PATH)

train_df["image_id"] = train_df["image_id"].astype(str)
train_df["label"] = train_df["label"].astype(np.int32)

print("sample.head():\n", sample.head())
print("train_df.head():\n", train_df.head())
print("Rows(sample):", len(sample), "Rows(train):", len(train_df))




## === cell 2
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
    print("XLA JIT enabled.")
except Exception as e:
    print("Could not enable XLA JIT:", repr(e))

try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

try:
    tf.config.set_soft_device_placement(True)
except Exception:
    pass

TARGET_SIZE = 450  # preserve original core setting
BATCH_SIZE = 16  # preserve original
EPOCHS = 2  # preserve original
NUM_CLASSES = 5

train_df_shuffled = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
val_frac = 0.1
val_n = int(len(train_df_shuffled) * val_frac)
val_df = train_df_shuffled.iloc[:val_n].copy()
trn_df = train_df_shuffled.iloc[val_n:].copy()

AUTOTUNE = tf.data.AUTOTUNE


def _read_decode_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)  # cassava images are jpg
    img = tf.image.resize(
        img,
        [TARGET_SIZE, TARGET_SIZE],
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=True,
    )
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)  # rescale
    return img


rot_layer = layers.RandomRotation(factor=10.0 / 180.0, fill_mode="reflect", seed=SEED)


def _augment(img, seed):
    img = rot_layer(img, training=True)

    tx = tf.random.stateless_uniform(
        [], seed=seed + tf.constant([1, 0], tf.int32), minval=-0.05, maxval=0.05
    )
    ty = tf.random.stateless_uniform(
        [], seed=seed + tf.constant([0, 1], tf.int32), minval=-0.05, maxval=0.05
    )
    dx = tf.cast(tf.round(tx * TARGET_SIZE), tf.int32)
    dy = tf.cast(tf.round(ty * TARGET_SIZE), tf.int32)
    img = tf.roll(img, shift=[dy, dx], axis=[0, 1])

    zoom = tf.random.stateless_uniform(
        [], seed=seed + tf.constant([2, 2], tf.int32), minval=0.9, maxval=1.0
    )
    crop_sz = tf.cast(tf.round(zoom * TARGET_SIZE), tf.int32)
    crop_sz = tf.maximum(1, tf.minimum(TARGET_SIZE, crop_sz))
    img = tf.image.stateless_random_crop(
        img, size=[crop_sz, crop_sz, 3], seed=seed + tf.constant([3, 3], tf.int32)
    )
    img = tf.image.resize(
        img,
        [TARGET_SIZE, TARGET_SIZE],
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=True,
    )

    img = tf.image.stateless_random_flip_left_right(
        img, seed=seed + tf.constant([4, 4], tf.int32)
    )
    return img


def make_train_ds(df, shuffle, augment):
    paths = tf.constant(
        (TRAIN_IMG_DIR + "/" + df["image_id"].astype(str)).values, dtype=tf.string
    )
    labels_int = tf.constant(df["label"].values, dtype=tf.int32)
    labels_oh = tf.one_hot(labels_int, depth=NUM_CLASSES, dtype=tf.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels_oh))
    if shuffle:
        ds = ds.shuffle(
            buffer_size=min(len(df), 4096), seed=SEED, reshuffle_each_iteration=True
        )

    def _decode_map(path, y):
        img = _read_decode_resize(path)
        return path, img, y

    ds = ds.map(_decode_map, num_parallel_calls=AUTOTUNE, deterministic=True)

    ds = ds.cache()

    def _final_map(path, img, y):
        if augment:
            h = tf.strings.to_hash_bucket_fast(path, 2**31 - 1)
            s0 = tf.cast(h, tf.int32) ^ tf.constant(SEED, tf.int32)
            s1 = tf.cast(h, tf.int32) + tf.constant(SEED, tf.int32)
            s = tf.stack([s0, s1])
            img = _augment(img, s)
        return img, y

    ds = ds.map(_final_map, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)

    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_test_ds(df):
    paths = tf.constant(
        (TEST_IMG_DIR + "/" + df["image_id"].astype(str)).values, dtype=tf.string
    )
    ds = tf.data.Dataset.from_tensor_slices(paths)

    def _map_fn(path):
        img = _read_decode_resize(path)
        return img

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.cache()
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_train_ds(trn_df, shuffle=True, augment=True)
val_ds = make_train_ds(val_df, shuffle=False, augment=False)

inputs = keras.Input(shape=(TARGET_SIZE, TARGET_SIZE, 3))
x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = keras.Model(inputs, outputs)

compile_kwargs = dict(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
try:
    model.compile(**compile_kwargs, jit_compile=True)
    print("model.compile(..., jit_compile=True)")
except TypeError:
    model.compile(**compile_kwargs)
    print("model.compile(...), jit_compile not supported in this TF version")

print(model.summary())

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1,
)




## === cell 3
test_ds = make_test_ds(sample)

pred = model.predict(
    test_ds,
    verbose=1,
)
print("Pred shape:", pred.shape)

if pred.ndim == 1:
    preds = (pred > 0.5).astype(int)
else:
    preds = np.argmax(pred, axis=1).astype(int)

if len(preds) != len(sample):
    raise ValueError(
        f"Prediction length mismatch: got {len(preds)}, expected {len(sample)}"
    )

submission = pd.DataFrame(
    {"image_id": sample["image_id"].astype(str).values, "label": preds}
)
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with rows:", len(submission))
print("submission.csv exists:", os.path.exists("submission.csv"))
