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
import glob
import numpy as np
import pandas as pd

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB3

SEED = 42
DEBUG = False

tf.random.set_seed(SEED)
np.random.seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

tf.config.optimizer.set_jit(False)
tf.config.optimizer.set_experimental_options(
    {
        "map_parallelization": True,
        "map_and_batch_fusion": True,
        "noop_elimination": True,
        "shuffle_and_repeat_fusion": True,
    }
)


def resolve_path(*candidates):
    """Return the first existing path among candidates."""
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


DATA_ROOT = resolve_path(
    "/kaggle/input/cassava-leaf-disease-classification",
    "../input/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
)

if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not locate cassava-leaf-disease-classification dataset directory."
    )

TRAIN_CSV = resolve_path(
    os.path.join(DATA_ROOT, "train.csv"),
    "/kaggle/input/train.csv",
    "../input/train.csv",
)
SAMPLE_SUB = resolve_path(
    os.path.join(DATA_ROOT, "sample_submission.csv"),
    "/kaggle/input/sample_submission.csv",
    "../input/sample_submission.csv",
)
TRAIN_IMG_DIR = resolve_path(
    os.path.join(DATA_ROOT, "train_images"),
    "/kaggle/input/cassava-leaf-disease-classification/train_images",
    "../input/cassava-leaf-disease-classification/train_images",
)
TEST_IMG_DIR = resolve_path(
    os.path.join(DATA_ROOT, "test_images"),
    "/kaggle/input/cassava-leaf-disease-classification/test_images",
    "../input/cassava-leaf-disease-classification/test_images",
)

for pth, name in [
    (TRAIN_CSV, "train.csv"),
    (SAMPLE_SUB, "sample_submission.csv"),
    (TRAIN_IMG_DIR, "train_images/"),
    (TEST_IMG_DIR, "test_images/"),
]:
    if pth is None:
        raise FileNotFoundError(f"Missing required dataset artifact: {name}")

print("Using DATA_ROOT:", DATA_ROOT)
print("TRAIN_CSV:", TRAIN_CSV)
print("SAMPLE_SUB:", SAMPLE_SUB)
print("TRAIN_IMG_DIR:", TRAIN_IMG_DIR)
print("TEST_IMG_DIR:", TEST_IMG_DIR)



## === cell 1
IMG_SIZE = (300, 300)
BATCH_SIZE = 32
NUM_CLASSES = 5
EPOCHS = 2  # keep small to fit runtime; still provides a non-random signal

train_df = pd.read_csv(TRAIN_CSV)

train_df["path"] = TRAIN_IMG_DIR.rstrip("/") + "/" + train_df["image_id"].astype(str)
train_df["label"] = train_df["label"].astype(np.int32)

if DEBUG:
    train_df = train_df.sample(n=2000, random_state=SEED).reset_index(drop=True)

from sklearn.model_selection import train_test_split

tr_df, va_df = train_test_split(
    train_df, test_size=0.15, random_state=SEED, stratify=train_df["label"]
)


@tf.function
def _decode_resize_rescale(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) / 255.0
    return img


@tf.function
def _one_hot(label):
    return tf.one_hot(tf.cast(label, tf.int32), depth=NUM_CLASSES, dtype=tf.float32)


aug_rotate = layers.RandomRotation(factor=15.0 / 180.0, fill_mode="nearest", seed=SEED)
aug_translate = layers.RandomTranslation(
    height_factor=0.05, width_factor=0.05, fill_mode="nearest", seed=SEED
)
aug_zoom = layers.RandomZoom(
    height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1), fill_mode="nearest", seed=SEED
)
aug_flip = layers.RandomFlip(mode="horizontal", seed=SEED)


@tf.function
def _apply_aug(img):
    x = aug_flip(img, training=True)
    x = aug_rotate(x, training=True)
    x = aug_translate(x, training=True)
    x = aug_zoom(x, training=True)
    return x


def make_train_val_ds(df, training, batch_size):
    paths = tf.constant(df["path"].values)
    labels = tf.constant(df["label"].values)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    options = tf.data.Options()
    options.experimental_deterministic = True
    options.experimental_optimization.apply_default_optimizations = True
    ds = ds.with_options(options)

    if training:
        shuffle_buf = int(min(len(df), 4096))
        ds = ds.shuffle(
            buffer_size=shuffle_buf, seed=SEED, reshuffle_each_iteration=True
        )

    def _load_base(path, label):
        img = _decode_resize_rescale(path)
        return img, label

    ds = ds.map(_load_base, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        ds = ds.cache()

        def _aug_and_encode(img, label):
            img = _apply_aug(img)
            return img, _one_hot(label)

        ds = ds.map(_aug_and_encode, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        def _encode(img, label):
            return img, _one_hot(label)

        ds = ds.map(_encode, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.cache()

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


train_ds = make_train_val_ds(tr_df, training=True, batch_size=BATCH_SIZE)
valid_ds = make_train_val_ds(va_df, training=False, batch_size=BATCH_SIZE)

backbone = EfficientNetB3(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    pooling=None,
)
backbone.trainable = False  # keep core logic stable and runtime bounded

inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = backbone(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2, seed=SEED)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
my_model = Model(inputs, outputs)

my_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

my_model.summary()

history = my_model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    verbose=1,
)



## === cell 2
test_images = sorted(tf.io.gfile.glob(os.path.join(TEST_IMG_DIR, "*.jpg")))
if len(test_images) == 0:
    raise FileNotFoundError(f"No test images found in {TEST_IMG_DIR}")

df_test = pd.DataFrame({"path": test_images})


def make_test_ds(batch_size=64):
    paths = tf.constant(df_test["path"].values)

    ds = tf.data.Dataset.from_tensor_slices(paths)
    options = tf.data.Options()
    options.experimental_deterministic = True
    options.experimental_optimization.apply_default_optimizations = True
    ds = ds.with_options(options)

    ds = ds.map(_decode_resize_rescale, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.cache()
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds




## === cell 3
test_ds = make_test_ds(batch_size=256)
pred_test = my_model.predict(test_ds, verbose=1)
pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

final_submission = df_test.copy()
final_submission["image_id"] = pd.Series(final_submission["path"]).map(os.path.basename)
final_submission["label"] = pred_test_labels

sample_sub = pd.read_csv(SAMPLE_SUB)
sub = sample_sub[["image_id"]].merge(
    final_submission[["image_id", "label"]], on="image_id", how="left"
)

if sub["label"].isna().any():
    fill_label = int(pd.Series(pred_test_labels).mode().iloc[0])
    sub["label"] = sub["label"].fillna(fill_label).astype(int)
else:
    sub["label"] = sub["label"].astype(int)

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)



## === cell 4
assert os.path.exists("submission.csv"), "submission.csv was not created."
check = pd.read_csv("submission.csv")
assert list(check.columns) == [
    "image_id",
    "label",
], "Submission columns must be image_id,label"
assert len(check) == len(
    pd.read_csv(SAMPLE_SUB)
), "Submission row count must match sample_submission."
check.head()
