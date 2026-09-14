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

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import json
import random
import numpy as np
import pandas as pd

from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

print("TF:", tf.__version__)
print("Keras:", keras.__version__)




## === cell 1
BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification/"
TRAIN_DIR = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
TEST_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_images/"

TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
SAMPLE_SUB = os.path.join(BASE_DIR, "sample_submission.csv")
MAP_JSON = os.path.join(BASE_DIR, "label_num_to_disease_map.json")

TRAIN_TFREC_DIR = os.path.join(BASE_DIR, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(BASE_DIR, "test_tfrecords")

assert os.path.exists(TRAIN_CSV), TRAIN_CSV
assert os.path.exists(SAMPLE_SUB), SAMPLE_SUB
assert os.path.exists(TRAIN_DIR), TRAIN_DIR
assert os.path.exists(TEST_DIR), TEST_DIR
assert os.path.exists(TRAIN_TFREC_DIR), TRAIN_TFREC_DIR
assert os.path.exists(TEST_TFREC_DIR), TEST_TFREC_DIR




## === cell 2
with open(MAP_JSON, "r") as f:
    map_classes = json.load(f)
print(json.dumps(map_classes, indent=2))
label_list = sorted([int(k) for k in map_classes.keys()])
NUM_CLASSES = len(label_list)
print("NUM_CLASSES:", NUM_CLASSES, "labels:", label_list)




## === cell 3
IMG_HEIGHT = 500
IMG_WIDTH = 500
batch_size = 16

PRE_TRAINED_MODEL = "../input/unitedmodelv01/Cassava_Best_UnitedModel_V01.hdf5"
print("Pretrained exists?", os.path.exists(PRE_TRAINED_MODEL), PRE_TRAINED_MODEL)




## === cell 4
train_df = pd.read_csv(TRAIN_CSV)
sample_sub_df = pd.read_csv(SAMPLE_SUB)

assert set(train_df.columns) == {"image_id", "label"}
assert set(sample_sub_df.columns) == {"image_id", "label"}

print("Train rows:", len(train_df), "Test rows:", len(sample_sub_df))

missing = 0
for fn in train_df["image_id"].head(50):
    if not os.path.exists(os.path.join(TRAIN_DIR, fn)):
        missing += 1
print("Missing in first 50 train images:", missing)




## === cell 5
_RESAMPLE = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS


def load_single_image(data_type, image_id):
    if data_type == "TEST_DATA":
        image_path = os.path.join(TEST_DIR, image_id)
    else:
        image_path = os.path.join(TRAIN_DIR, image_id)
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT), _RESAMPLE)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # float32 0..1
    return arr




## === cell 6
AUTOTUNE = tf.data.AUTOTUNE


def _tf_load_image_from_bytes(img_bytes):
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img,
        [IMG_HEIGHT, IMG_WIDTH],
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False,
    )
    img = tf.cast(img, tf.float32) / 255.0
    img.set_shape([IMG_HEIGHT, IMG_WIDTH, 3])
    return img


_TRAIN_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}
_TEST_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


def _parse_train_tfrecord(ex):
    x = tf.io.parse_single_example(ex, _TRAIN_FEATURES)
    img = _tf_load_image_from_bytes(x["image"])
    y = tf.cast(x["target"], tf.int32)
    y.set_shape([])
    return img, y


def _parse_test_tfrecord(ex):
    x = tf.io.parse_single_example(ex, _TEST_FEATURES)
    img = _tf_load_image_from_bytes(x["image"])
    image_name = x["image_name"]  # bytes
    return img, image_name


@tf.function
def _augment(x, y):
    x = tf.image.random_flip_left_right(x, seed=SEED)
    x = tf.image.random_flip_up_down(x, seed=SEED)
    x = tf.image.random_brightness(x, max_delta=0.15, seed=SEED)
    x = tf.clip_by_value(x, 0.0, 1.0)
    return x, y


train_tfrecs = sorted(
    [
        os.path.join(TRAIN_TFREC_DIR, f)
        for f in tf.io.gfile.listdir(TRAIN_TFREC_DIR)
        if f.endswith(".tfrec")
    ]
)
test_tfrecs = sorted(
    [
        os.path.join(TEST_TFREC_DIR, f)
        for f in tf.io.gfile.listdir(TEST_TFREC_DIR)
        if f.endswith(".tfrec")
    ]
)
print("Train tfrecs:", len(train_tfrecs), "Test tfrecs:", len(test_tfrecs))

rng = np.random.RandomState(SEED)
perm = rng.permutation(len(train_tfrecs))
train_tfrecs_shuf = [train_tfrecs[i] for i in perm.tolist()]

val_frac = 0.1
val_files = max(1, int(round(len(train_tfrecs_shuf) * val_frac)))
val_tfrecs = train_tfrecs_shuf[:val_files]
trn_tfrecs = train_tfrecs_shuf[val_files:]

print(
    "TFRecord file split -> train files:",
    len(trn_tfrecs),
    "val files:",
    len(val_tfrecs),
)

opts = tf.data.Options()
opts.experimental_deterministic = True  # stable results
opts.threading.private_threadpool_size = max(4, (os.cpu_count() or 8) // 2)

_TFREC_BUFFER = 64 * 1024 * 1024  # larger sequential read buffer reduces IO overhead

trn_raw = tf.data.TFRecordDataset(
    trn_tfrecs, num_parallel_reads=AUTOTUNE, buffer_size=_TFREC_BUFFER
).with_options(opts)
trn_ds = (
    trn_raw.map(_parse_train_tfrecord, num_parallel_calls=AUTOTUNE)
    .shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    .map(_augment, num_parallel_calls=AUTOTUNE)
    .batch(batch_size, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

val_raw = tf.data.TFRecordDataset(
    val_tfrecs, num_parallel_reads=AUTOTUNE, buffer_size=_TFREC_BUFFER
).with_options(opts)

val_ds = (
    val_raw.map(_parse_train_tfrecord, num_parallel_calls=AUTOTUNE)
    .cache()
    .batch(batch_size, drop_remainder=False)
    .prefetch(AUTOTUNE)
)


def _check_no_missing(ds, name):
    for _, y in ds.take(1):
        if tf.reduce_any(y < 0):
            raise ValueError(
                f"{name} has missing labels (-1). Unexpected for train TFRecords."
            )


_check_no_missing(trn_ds, "trn_ds")
_check_no_missing(val_ds, "val_ds")




## === cell 7
inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2, seed=SEED)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = keras.Model(inputs, outputs)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()




## === cell 8
EPOCHS = 3
history = model.fit(trn_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)




## === cell 9
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
)  # kept to preserve core logic imports

test_ids = sample_sub_df["image_id"].astype(str).values
n_aug = 5
tta_total = 1 + n_aug

test_opts = tf.data.Options()
test_opts.experimental_deterministic = True
test_opts.threading.private_threadpool_size = max(4, (os.cpu_count() or 8) // 2)

raw_test = tf.data.TFRecordDataset(
    test_tfrecs, num_parallel_reads=AUTOTUNE, buffer_size=_TFREC_BUFFER
).with_options(test_opts)

test_ds = (
    raw_test.map(_parse_test_tfrecord, num_parallel_calls=AUTOTUNE)
    .cache()
    .batch(64, drop_remainder=False)
    .prefetch(AUTOTUNE)
)


def _build_tta_model(seed):
    return keras.Sequential(
        [
            layers.RandomFlip(mode="horizontal", seed=seed),
            layers.RandomFlip(mode="vertical", seed=seed),
            layers.RandomRotation(factor=45.0 / 360.0, fill_mode="nearest", seed=seed),
            layers.RandomZoom(
                height_factor=(-0.3, 0.3),
                width_factor=(-0.3, 0.3),
                fill_mode="nearest",
                seed=seed,
            ),
            layers.RandomTranslation(
                height_factor=0.1,
                width_factor=0.1,
                fill_mode="nearest",
                seed=seed,
            ),
            layers.RandomShear(
                x_factor=0.1,
                y_factor=0.1,
                fill_mode="nearest",
                seed=seed,
            ),
        ],
        name=f"tta_{seed}",
    )


tta_layers = [_build_tta_model(SEED + k) for k in range(n_aug)]


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None, IMG_HEIGHT, IMG_WIDTH, 3], dtype=tf.float32)
    ]
)
def _predict_with_tta(batch_x):
    aug_list = [batch_x]
    for layer in tta_layers:
        ax = layer(batch_x, training=True)
        ax = tf.clip_by_value(ax, 0.0, 1.0)
        aug_list.append(ax)
    all_x = tf.stack(aug_list, axis=1)  # [B, tta_total, H, W, C]
    shape = tf.shape(all_x)
    b = shape[0]
    t = shape[1]
    flat_x = tf.reshape(all_x, [b * t, IMG_HEIGHT, IMG_WIDTH, 3])

    flat_probs = model(flat_x, training=False)  # [B*T, num_classes]
    probs = tf.reshape(flat_probs, [b, t, NUM_CLASSES])
    probs_sum = tf.reduce_sum(probs, axis=1)  # [B, num_classes]
    return probs_sum


preds_out = np.empty((len(test_ids),), dtype=np.int64)

offset = 0
for batch_x, batch_name in test_ds:
    batch_probs_sum = _predict_with_tta(batch_x).numpy().astype(np.float32)
    batch_probs_mean = batch_probs_sum / float(tta_total)
    batch_pred = batch_probs_mean.argmax(axis=1).astype(np.int64)

    bsz = batch_pred.shape[0]
    preds_out[offset : offset + bsz] = batch_pred
    offset += bsz

assert offset == len(test_ids), (offset, len(test_ids))

test_results = preds_out.astype(int).tolist()
submission = pd.DataFrame({"image_id": test_ids, "label": test_results})
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())




## === cell 10
sub_check = pd.read_csv("submission.csv")
assert list(sub_check.columns) == ["image_id", "label"]
assert len(sub_check) == len(sample_sub_df)
assert sub_check["label"].between(0, NUM_CLASSES - 1).all()
print("submission.csv OK")
