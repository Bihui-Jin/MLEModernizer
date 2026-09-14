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

import pandas as pd
import numpy as np
import tensorflow as tf
import glob
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB3
from sklearn.model_selection import train_test_split

SEED = 42
DEBUG = False

tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

BASE_INPUT = "/kaggle/input/cassava-leaf-disease-classification"
if not os.path.exists(BASE_INPUT):
    BASE_INPUT = "/kaggle/data/cassava-leaf-disease-classification"

TRAIN_CSV = os.path.join(BASE_INPUT, "train.csv")
TRAIN_IMG_DIR = os.path.join(BASE_INPUT, "train_images")
TEST_IMG_DIR = os.path.join(BASE_INPUT, "test_images")
SAMPLE_SUB = os.path.join(BASE_INPUT, "sample_submission.csv")

assert os.path.exists(TRAIN_CSV), f"Missing train.csv at {TRAIN_CSV}"
assert os.path.exists(TRAIN_IMG_DIR), f"Missing train_images at {TRAIN_IMG_DIR}"
assert os.path.exists(TEST_IMG_DIR), f"Missing test_images at {TEST_IMG_DIR}"
assert os.path.exists(SAMPLE_SUB), f"Missing sample_submission.csv at {SAMPLE_SUB}"

print("Using BASE_INPUT:", BASE_INPUT)
print("TensorFlow:", tf.__version__)

try:
    tf.config.threading.set_inter_op_parallelism_threads(0)
    tf.config.threading.set_intra_op_parallelism_threads(0)
except Exception:
    pass



## === cell 1
train_df = pd.read_csv(TRAIN_CSV)

train_df["path"] = (TRAIN_IMG_DIR + "/" + train_df["image_id"].astype(str)).astype(str)
train_df["label"] = train_df["label"].astype(str)

tr_df, va_df = train_test_split(
    train_df, test_size=0.10, random_state=SEED, stratify=train_df["label"]
)

IMG_SIZE = (300, 300)
BATCH_SIZE = 32 if not DEBUG else 8

AUTOTUNE = tf.data.AUTOTUNE
IMG_H, IMG_W = IMG_SIZE
NUM_CLASSES = train_df["label"].nunique()
print("NUM_CLASSES:", NUM_CLASSES)

class_names = sorted(train_df["label"].unique().tolist())
class_to_idx = {c: i for i, c in enumerate(class_names)}
print("class_indices:", class_to_idx)

tr_paths = tr_df["path"].to_numpy()
va_paths = va_df["path"].to_numpy()
tr_labels = tr_df["label"].map(class_to_idx).to_numpy(dtype=np.int32)
va_labels = va_df["label"].map(class_to_idx).to_numpy(dtype=np.int32)

tr_steps = int(np.ceil(len(tr_paths) / BATCH_SIZE))
va_steps = int(np.ceil(len(va_paths) / BATCH_SIZE))


@tf.function
def _decode_and_resize_u8(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)  # uint8
    img = tf.image.resize(
        img, [IMG_H, IMG_W], method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img = tf.clip_by_value(tf.round(img), 0.0, 255.0)
    img = tf.cast(img, tf.uint8)
    return img


@tf.function
def _to_float01(img_u8):
    img = tf.cast(img_u8, tf.float32) / 255.0
    return img


@tf.function
def _augment(img, seed):
    img = tf.image.stateless_random_flip_left_right(img, seed=seed)

    angle = tf.random.stateless_uniform(
        [], seed=seed + tf.constant([1, 0], tf.int32), minval=-10.0, maxval=10.0
    ) * (np.pi / 180.0)
    try:
        img = tf.image.rotate(img, angles=angle, interpolation="BILINEAR")
    except Exception:
        img = tf.raw_ops.ImageProjectiveTransformV3(
            images=tf.expand_dims(img, 0),
            transforms=tf.expand_dims(
                tf.stack(
                    [
                        tf.cos(angle),
                        -tf.sin(angle),
                        0.0,
                        tf.sin(angle),
                        tf.cos(angle),
                        0.0,
                        0.0,
                        0.0,
                    ]
                ),
                0,
            ),
            output_shape=[IMG_H, IMG_W],
            interpolation="BILINEAR",
            fill_mode="REFLECT",
            fill_value=0.0,
        )[0]

    max_dx = 0.05 * tf.cast(IMG_W, tf.float32)
    max_dy = 0.05 * tf.cast(IMG_H, tf.float32)
    dx = tf.random.stateless_uniform(
        [], seed=seed + tf.constant([2, 0], tf.int32), minval=-max_dx, maxval=max_dx
    )
    dy = tf.random.stateless_uniform(
        [], seed=seed + tf.constant([3, 0], tf.int32), minval=-max_dy, maxval=max_dy
    )

    scale = tf.random.stateless_uniform(
        [], seed=seed + tf.constant([4, 0], tf.int32), minval=0.9, maxval=1.1
    )

    cx = (tf.cast(IMG_W, tf.float32) - 1.0) / 2.0
    cy = (tf.cast(IMG_H, tf.float32) - 1.0) / 2.0
    a0 = 1.0 / scale
    a1 = 0.0
    a2 = (dx - cx * (scale - 1.0)) / scale
    b0 = 0.0
    b1 = 1.0 / scale
    b2 = (dy - cy * (scale - 1.0)) / scale

    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.expand_dims(img, 0),
        transforms=tf.expand_dims(tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0]), 0),
        output_shape=[IMG_H, IMG_W],
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )[0]
    return img


_DATA_OPTIONS = tf.data.Options()
_DATA_OPTIONS.deterministic = True
try:
    _DATA_OPTIONS.experimental_optimization.apply_default_optimizations = True
    _DATA_OPTIONS.experimental_optimization.map_parallelization = True
    _DATA_OPTIONS.experimental_optimization.parallel_batch = True
except Exception:
    pass

_SHUFFLE_BUFFER = min(len(tr_paths), 4096)

_EPOCH_COUNTER = tf.Variable(0, dtype=tf.int64, trainable=False)


class _EpochCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        _EPOCH_COUNTER.assign(epoch)


def _make_train_ds(paths, labels, batch_size):
    n = len(paths)
    base = tf.data.Dataset.from_tensor_slices((paths, labels))
    base = base.shuffle(
        buffer_size=_SHUFFLE_BUFFER, seed=SEED, reshuffle_each_iteration=True
    )

    idx_ds = tf.data.Dataset.range(n)
    base = tf.data.Dataset.zip((idx_ds, base))  # (idx, (path,label))

    def _decode_map(idx, pl):
        path, label = pl
        img_u8 = _decode_and_resize_u8(path)
        return idx, img_u8, label

    base = base.map(_decode_map, num_parallel_calls=AUTOTUNE, deterministic=True)

    base = base.cache()

    def _aug_map(idx, img_u8, label):
        img = _to_float01(img_u8)
        s0 = tf.cast(SEED, tf.int32)
        s = tf.stack([s0, tf.cast(_EPOCH_COUNTER, tf.int32)], axis=0)
        s = tf.random.experimental.stateless_fold_in(s, tf.cast(idx, tf.int32))
        img = _augment(img, s)
        y = tf.one_hot(label, NUM_CLASSES, dtype=tf.float32)
        return img, y

    ds = base.map(_aug_map, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds.with_options(_DATA_OPTIONS)


def _make_valid_ds(paths, labels, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _map_fn(path, label):
        img_u8 = _decode_and_resize_u8(path)
        img = _to_float01(img_u8)
        y = tf.one_hot(label, NUM_CLASSES, dtype=tf.float32)
        return img, y

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds.with_options(_DATA_OPTIONS)


train_ds = _make_train_ds(tr_paths, tr_labels, BATCH_SIZE)
valid_ds = _make_valid_ds(va_paths, va_labels, BATCH_SIZE)

inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
backbone = EfficientNetB3(include_top=False, weights="imagenet", input_tensor=inputs)
x = tf.keras.layers.GlobalAveragePooling2D()(backbone.output)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
my_model = Model(inputs, outputs)

backbone.trainable = False
my_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

EPOCHS_1 = 1 if DEBUG else 2

my_model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS_1,
    verbose=1,
    callbacks=[_EpochCallback()],
)

backbone.trainable = True
my_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

EPOCHS_2 = 1 if DEBUG else 1
my_model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS_1 + EPOCHS_2,
    initial_epoch=EPOCHS_1,
    verbose=1,
    callbacks=[_EpochCallback()],
)



## === cell 2
test_images = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg")))
assert len(test_images) > 0, f"No test images found in {TEST_IMG_DIR}"


def make_test_ds(batch_size=64):
    ds = tf.data.Dataset.from_tensor_slices(test_images)

    def _map_fn(path):
        img_u8 = _decode_and_resize_u8(path)
        img = _to_float01(img_u8)
        return img

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds.with_options(_DATA_OPTIONS)




## === cell 3
test_batch = 128 if not DEBUG else 16
test_ds = make_test_ds(batch_size=test_batch)

pred_test = my_model.predict(test_ds, verbose=1)
pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

image_ids = np.fromiter(
    (os.path.basename(p) for p in test_images), dtype=object, count=len(test_images)
)
final_submission = pd.DataFrame({"image_id": image_ids, "label": pred_test_labels})

sample_sub = pd.read_csv(SAMPLE_SUB)
final_csv = sample_sub[["image_id"]].merge(
    final_submission,
    on="image_id",
    how="left",
    validate="one_to_one",
)

if final_csv["label"].isna().any():
    final_csv["label"] = final_csv["label"].fillna(0).astype(int)

final_csv["label"] = final_csv["label"].astype(int)
final_csv.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", final_csv.shape)
print(final_csv.head())



## === cell 4
final_csv.head()
