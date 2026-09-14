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
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import glob
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.applications import EfficientNetB3
from sklearn.model_selection import train_test_split

SEED = 42
DEBUG = False

tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(
        False
    )  # keep deterministic-ish behavior; no XLA surprises
except Exception:
    pass

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing {TEST_IMG_DIR}"

print("TF version:", tf.__version__)
print("Train images:", len(glob.glob(os.path.join(TRAIN_IMG_DIR, "*.jpg"))))
print("Test images:", len(glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg"))))

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass




## === cell 1
df = pd.read_csv(TRAIN_CSV)
assert {"image_id", "label"}.issubset(df.columns)

df["path"] = TRAIN_IMG_DIR.rstrip("/") + "/" + df["image_id"].astype(str)

assert (
    len(glob.glob(os.path.join(TRAIN_IMG_DIR, "*.jpg"))) >= len(df) * 0.99
), "Many train images missing."

df["label_str"] = df["label"].astype(str)

train_df, val_df = train_test_split(
    df[["path", "label", "label_str", "image_id"]],
    test_size=0.15,
    random_state=SEED,
    stratify=df["label"],
)

IMG_SIZE = (300, 300)
BATCH_SIZE = 32
NUM_CLASSES = int(df["label"].nunique())
classes = [str(i) for i in sorted(df["label"].unique())]

print("NUM_CLASSES:", NUM_CLASSES)

AUTOTUNE = tf.data.AUTOTUNE

train_paths = train_df["path"].to_numpy()
train_labels = train_df["label"].to_numpy(dtype=np.int32)
val_paths = val_df["path"].to_numpy()
val_labels = val_df["label"].to_numpy(dtype=np.int32)

IMG_H, IMG_W = IMG_SIZE

DATA_OPTS = tf.data.Options()
DATA_OPTS.experimental_deterministic = True  # keep deterministic order where applicable


@tf.function
def _decode_resize_rescale(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, [IMG_H, IMG_W], method=tf.image.ResizeMethod.BILINEAR, antialias=True
    )
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    return img


@tf.function
def _rotate_projective(image, angle_rad):
    c = tf.math.cos(angle_rad)
    s = tf.math.sin(angle_rad)

    cx = (tf.cast(IMG_W, tf.float32) - 1.0) / 2.0
    cy = (tf.cast(IMG_H, tf.float32) - 1.0) / 2.0

    a0 = c
    a1 = -s
    a2 = cx - c * cx + s * cy
    b0 = s
    b1 = c
    b2 = cy - s * cx - c * cy

    transform = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])[None, :]
    rotated = tf.raw_ops.ImageProjectiveTransformV3(
        images=image[None, ...],
        transforms=transform,
        output_shape=tf.constant([IMG_H, IMG_W], tf.int32),
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )[0]
    return rotated


@tf.function
def _augment(image, seed_pair):
    image = tf.image.stateless_random_flip_left_right(image, seed=seed_pair)

    angle = tf.random.stateless_uniform(
        [], seed=seed_pair + tf.constant([1, 0], tf.int32), minval=-10.0, maxval=10.0
    ) * (np.pi / 180.0)
    image = _rotate_projective(image, tf.cast(angle, tf.float32))

    dx = tf.random.stateless_uniform(
        [], seed=seed_pair + tf.constant([2, 0], tf.int32), minval=-0.05, maxval=0.05
    ) * tf.cast(IMG_W, tf.float32)
    dy = tf.random.stateless_uniform(
        [], seed=seed_pair + tf.constant([3, 0], tf.int32), minval=-0.05, maxval=0.05
    ) * tf.cast(IMG_H, tf.float32)
    transform = tf.stack([1.0, 0.0, -dx, 0.0, 1.0, -dy, 0.0, 0.0])[None, :]
    image = tf.raw_ops.ImageProjectiveTransformV3(
        images=image[None, ...],
        transforms=transform,
        output_shape=tf.constant([IMG_H, IMG_W], tf.int32),
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )[0]

    z = tf.random.stateless_uniform(
        [],
        seed=seed_pair + tf.constant([4, 0], tf.int32),
        minval=1.0 - 0.05,
        maxval=1.0 + 0.05,
    )

    def _zoom_in():
        crop_h = tf.cast(tf.round(tf.cast(IMG_H, tf.float32) / z), tf.int32)
        crop_w = tf.cast(tf.round(tf.cast(IMG_W, tf.float32) / z), tf.int32)
        crop_h = tf.clip_by_value(crop_h, 1, IMG_H)
        crop_w = tf.clip_by_value(crop_w, 1, IMG_W)
        crop = tf.image.stateless_random_crop(
            image,
            size=[crop_h, crop_w, 3],
            seed=seed_pair + tf.constant([5, 0], tf.int32),
        )
        return tf.image.resize(
            crop,
            [IMG_H, IMG_W],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
            antialias=False,
        )

    def _zoom_out():
        pad_h = tf.cast(
            tf.round((tf.cast(IMG_H, tf.float32) * (z - 1.0)) / 2.0), tf.int32
        )
        pad_w = tf.cast(
            tf.round((tf.cast(IMG_W, tf.float32) * (z - 1.0)) / 2.0), tf.int32
        )
        pad_h = tf.maximum(pad_h, 0)
        pad_w = tf.maximum(pad_w, 0)
        padded = tf.pad(image, [[pad_h, pad_h], [pad_w, pad_w], [0, 0]], mode="REFLECT")
        return tf.image.resize(
            padded,
            [IMG_H, IMG_W],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
            antialias=False,
        )

    image = tf.cond(z >= 1.0, _zoom_out, _zoom_in)
    return image


def make_train_ds(paths, labels, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.with_options(DATA_OPTS)
    ds = ds.shuffle(buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True)

    def _map_all(path, label):
        image = _decode_resize_rescale(path)
        h = tf.strings.to_hash_bucket_fast(path, 2**31 - 1)
        seed_pair = tf.stack([tf.cast(h, tf.int32), tf.constant(SEED, tf.int32)])
        image = _augment(image, seed_pair)
        return image, tf.cast(label, tf.float32)

    ds = ds.map(_map_all, num_parallel_calls=AUTOTUNE, deterministic=True)

    cache_path = "/kaggle/working/train_cache.tf-data"
    ds = ds.cache(cache_path)

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_val_ds(paths, labels, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.with_options(DATA_OPTS)

    def _map_fn(path, label):
        image = _decode_resize_rescale(path)
        return image, tf.cast(label, tf.float32)

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE, deterministic=True)

    cache_path = "/kaggle/working/val_cache.tf-data"
    ds = ds.cache(cache_path)

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_train_ds(train_paths, train_labels, BATCH_SIZE)
val_ds = make_val_ds(val_paths, val_labels, BATCH_SIZE)

train_steps = int(np.ceil(len(train_paths) / BATCH_SIZE))
val_steps = int(np.ceil(len(val_paths) / BATCH_SIZE))

print("Train steps:", train_steps, "Val steps:", val_steps)




## === cell 2
inputs = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base = EfficientNetB3(include_top=False, weights="imagenet", input_tensor=inputs)
x = base.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
outputs = Dense(NUM_CLASSES, activation="softmax")(x)
my_model = Model(inputs=inputs, outputs=outputs)

base.trainable = False
my_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

EPOCHS_HEAD = 2 if not DEBUG else 1
history1 = my_model.fit(
    train_ds,
    validation_data=val_ds,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
    epochs=EPOCHS_HEAD,
    verbose=1,
)

base.trainable = True
my_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

EPOCHS_FT = 1 if not DEBUG else 1
history2 = my_model.fit(
    train_ds,
    validation_data=val_ds,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
    epochs=EPOCHS_FT,
    verbose=1,
)




## === cell 3
sample_sub = pd.read_csv(SAMPLE_SUB)
assert {"image_id", "label"}.issubset(sample_sub.columns)

sample_sub["path"] = TEST_IMG_DIR.rstrip("/") + "/" + sample_sub["image_id"].astype(str)

assert os.path.isdir(TEST_IMG_DIR), "Missing test image directory."

test_paths = sample_sub["path"].to_numpy()


def make_test_ds(paths, batch_size):
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.with_options(DATA_OPTS)

    def _map_fn(path):
        return _decode_resize_rescale(path)

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE, deterministic=True)

    cache_path = "/kaggle/working/test_cache.tf-data"
    ds = ds.cache(cache_path)

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


test_batch = 128
test_ds = make_test_ds(test_paths, batch_size=test_batch)
test_steps = int(np.ceil(len(test_paths) / test_batch))

pred_test = my_model.predict(test_ds, steps=test_steps, verbose=1)
pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

submission = sample_sub[["image_id"]].copy()
submission["label"] = pred_test_labels
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with rows:", len(submission))
assert os.path.exists("submission.csv")
assert submission.shape[0] == sample_sub.shape[0]
assert list(submission.columns) == ["image_id", "label"]
