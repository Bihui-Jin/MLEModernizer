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
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

SEED = 42
DEBUG = False

np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

print("TF version:", tf.__version__)




## === cell 1
NUM_CLASSES = 5
IMG_SIZE = (512, 512)


def build_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=NUM_CLASSES):
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        pooling="avg",
    )
    base.trainable = False

    inputs = keras.Input(shape=input_shape)
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.Dropout(0.2, seed=SEED)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        jit_compile=True,
    )
    return model


my_model = build_model()
my_model.summary()




## === cell 2
DATA_ROOT = "../input/cassava-leaf-disease-classification"
TRAIN_CSV_PATH = os.path.join(DATA_ROOT, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")

for p in [TRAIN_CSV_PATH, TRAIN_IMG_DIR, TEST_IMG_DIR, SAMPLE_SUB_PATH]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Required path not found: {p}")

df_train = pd.read_csv(TRAIN_CSV_PATH)
if not {"image_id", "label"}.issubset(df_train.columns):
    raise ValueError("train.csv must contain columns: image_id, label")
df_train["path"] = (TRAIN_IMG_DIR + "/" + df_train["image_id"].astype(str)).astype(str)
df_train["label"] = df_train["label"].astype(int)

df_sample = pd.read_csv(SAMPLE_SUB_PATH)
if "image_id" not in df_sample.columns:
    raise ValueError("sample_submission.csv missing required column: image_id")

df_test = df_sample.copy()
df_test["path"] = (TEST_IMG_DIR + "/" + df_test["image_id"].astype(str)).astype(str)

print("Train rows:", len(df_train), " Test rows:", len(df_test))
print("Train label distribution:\n", df_train["label"].value_counts().sort_index())




## === cell 3
idx = np.arange(len(df_train))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)

val_frac = 0.1
val_size = int(len(df_train) * val_frac)
val_idx = idx[:val_size]
trn_idx = idx[val_size:]

df_trn = df_train.iloc[trn_idx].reset_index(drop=True)
df_val = df_train.iloc[val_idx].reset_index(drop=True)

BATCH_SIZE = 16 if DEBUG else 32
EPOCHS = 1 if DEBUG else 3

AUTOTUNE = tf.data.AUTOTUNE


@tf.function
def _decode_jpeg(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(
        img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img = tf.cast(img, tf.float32)  # keep float32 like Keras generators produce
    return img


_ROT = tf.constant(10.0 * np.pi / 180.0, dtype=tf.float32)
_W_SHIFT = tf.constant(0.05, dtype=tf.float32)
_H_SHIFT = tf.constant(0.05, dtype=tf.float32)
_ZOOM = tf.constant(0.1, dtype=tf.float32)


@tf.function
def _zoom_batch_projective(imgs, z):
    b = tf.shape(imgs)[0]
    h = tf.cast(tf.shape(imgs)[1], tf.float32)
    w = tf.cast(tf.shape(imgs)[2], tf.float32)

    cx = (w - 1.0) / 2.0
    cy = (h - 1.0) / 2.0
    invz = 1.0 / z

    a0 = invz
    a1 = tf.zeros([b], tf.float32)
    a2 = cx - cx * invz

    a3 = tf.zeros([b], tf.float32)
    a4 = invz
    a5 = cy - cy * invz

    a6 = tf.zeros([b], tf.float32)
    a7 = tf.zeros([b], tf.float32)

    transforms = tf.stack([a0, a1, a2, a3, a4, a5, a6, a7], axis=1)  # [B,8]

    return tf.raw_ops.ImageProjectiveTransformV3(
        images=imgs,
        transforms=transforms,
        output_shape=tf.shape(imgs)[1:3],
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )


@tf.function
def _augment_batch(imgs):
    imgs = tf.image.random_flip_left_right(imgs, seed=SEED)

    b = tf.shape(imgs)[0]
    h = tf.cast(tf.shape(imgs)[1], tf.float32)
    w = tf.cast(tf.shape(imgs)[2], tf.float32)

    angles = tf.random.uniform(
        [b], minval=-_ROT, maxval=_ROT, seed=SEED, dtype=tf.float32
    )
    imgs = tf.image.rotate(imgs, angles=angles, fill_mode="reflect")

    dx = tf.random.uniform([b], -_W_SHIFT, _W_SHIFT, seed=SEED, dtype=tf.float32) * w
    dy = tf.random.uniform([b], -_H_SHIFT, _H_SHIFT, seed=SEED, dtype=tf.float32) * h

    transforms = tf.stack(
        [
            tf.ones([b], tf.float32),
            tf.zeros([b], tf.float32),
            -dx,
            tf.zeros([b], tf.float32),
            tf.ones([b], tf.float32),
            -dy,
            tf.zeros([b], tf.float32),
            tf.zeros([b], tf.float32),
        ],
        axis=1,
    )  # [B,8]

    imgs = tf.raw_ops.ImageProjectiveTransformV3(
        images=imgs,
        transforms=transforms,
        output_shape=tf.shape(imgs)[1:3],
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )

    z = tf.random.uniform([b], 1.0 - _ZOOM, 1.0 + _ZOOM, seed=SEED, dtype=tf.float32)
    imgs = _zoom_batch_projective(imgs, z)
    return imgs


def make_ds(paths, labels=None, training=False, batch_size=BATCH_SIZE):
    paths = np.asarray(paths)
    if labels is not None:
        labels = np.asarray(labels)

    if labels is None:
        ds = tf.data.Dataset.from_tensor_slices((paths,))
    else:
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    options = tf.data.Options()
    options.deterministic = True
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    ds = ds.with_options(options)

    if training:
        buf = int(min(len(paths), 2048))
        ds = ds.shuffle(buffer_size=buf, seed=SEED, reshuffle_each_iteration=True)

    if labels is None:
        ds = ds.map(
            lambda p: _decode_jpeg(p), num_parallel_calls=AUTOTUNE, deterministic=True
        )
    else:
        ds = ds.map(
            lambda p, y: (_decode_jpeg(p), tf.cast(y, tf.int32)),
            num_parallel_calls=AUTOTUNE,
            deterministic=True,
        )

    ds = ds.cache()

    ds = ds.batch(batch_size, drop_remainder=False)

    if training:
        if labels is None:
            ds = ds.map(
                lambda x: _augment_batch(x),
                num_parallel_calls=AUTOTUNE,
                deterministic=True,
            )
        else:
            ds = ds.map(
                lambda x, y: (_augment_batch(x), y),
                num_parallel_calls=AUTOTUNE,
                deterministic=True,
            )

    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_ds(df_trn["path"].values, df_trn["label"].values, training=True)
val_ds = make_ds(df_val["path"].values, df_val["label"].values, training=False)

history = my_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1,
)

base_model = None
for layer in my_model.layers:
    if isinstance(layer, tf.keras.Model) and layer.name.startswith("efficientnet"):
        base_model = layer
        break

if base_model is not None:
    base_model.trainable = True
    for l in base_model.layers[:-20]:
        l.trainable = False

    my_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        jit_compile=True,
    )
    ft_epochs = 1 if DEBUG else 2
    my_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=ft_epochs,
        verbose=1,
    )




## === cell 4
test_ds = make_ds(df_test["path"].values, labels=None, training=False)

pred_test = my_model.predict(test_ds, verbose=1)
pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

final_csv = df_test[["image_id"]].copy()
final_csv["label"] = pred_test_labels

final_csv["image_id"] = final_csv["image_id"].astype(str)
final_csv["label"] = final_csv["label"].astype(int)

final_csv.to_csv("submission.csv", index=False)

print(final_csv.head())
print("Wrote submission.csv with shape:", final_csv.shape)




## === cell 5
sub = pd.read_csv("submission.csv")
assert list(sub.columns) == ["image_id", "label"]
assert len(sub) == len(df_sample)
assert sub["label"].between(0, 4).all()
print(sub.head())
