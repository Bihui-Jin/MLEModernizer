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
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import glob
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB3

SEED = 42
DEBUG = False

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE

BASE_INPUT = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(BASE_INPUT, "train.csv")
SAMPLE_SUB = os.path.join(BASE_INPUT, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(BASE_INPUT, "train_images")
TEST_IMG_DIR = os.path.join(BASE_INPUT, "test_images")

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"
assert os.path.exists(TRAIN_IMG_DIR), f"Missing {TRAIN_IMG_DIR}"
assert os.path.exists(TEST_IMG_DIR), f"Missing {TEST_IMG_DIR}"

print("TensorFlow:", tf.__version__)




## === cell 1
df = pd.read_csv(TRAIN_CSV)

df["path"] = TRAIN_IMG_DIR.rstrip("/") + "/" + df["image_id"].astype(str)

exists_mask = df["path"].map(os.path.exists)
df = df[exists_mask].reset_index(drop=True)

if DEBUG:
    df = df.sample(1024, random_state=SEED).reset_index(drop=True)

val_frac = 0.1
train_parts = []
val_parts = []
for lbl, g in df.groupby("label", sort=False):
    g = g.sample(frac=1.0, random_state=SEED).reset_index(drop=True)  # shuffle
    n_val = max(1, int(len(g) * val_frac))
    val_parts.append(g.iloc[:n_val])
    train_parts.append(g.iloc[n_val:])

df_train = (
    pd.concat(train_parts, axis=0)
    .sample(frac=1.0, random_state=SEED)
    .reset_index(drop=True)
)
df_val = (
    pd.concat(val_parts, axis=0)
    .sample(frac=1.0, random_state=SEED)
    .reset_index(drop=True)
)

num_classes = int(df["label"].nunique())
assert num_classes == 5, f"Expected 5 classes, got {num_classes}"

print("Train size:", len(df_train), "Val size:", len(df_val), "Classes:", num_classes)




## === cell 2
IMG_SIZE = (300, 300)
BATCH_SIZE = 32 if not DEBUG else 16


def _decode_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) / 255.0
    return img


def _augment(img):
    img = tf.image.random_flip_left_right(img, seed=SEED)

    return img


rand_rot = tf.keras.layers.RandomRotation(
    factor=15.0 / 180.0, fill_mode="nearest", seed=SEED
)
rand_zoom = tf.keras.layers.RandomZoom(
    height_factor=(-0.10, 0.10),
    width_factor=(-0.10, 0.10),
    fill_mode="nearest",
    seed=SEED,
)
rand_trans = tf.keras.layers.RandomTranslation(
    height_factor=0.08, width_factor=0.08, fill_mode="nearest", seed=SEED
)


@tf.function
def _train_map(path, label):
    img = _decode_resize(path)
    img = rand_rot(img, training=True)
    img = rand_trans(img, training=True)
    img = rand_zoom(img, training=True)
    img = tf.image.random_flip_left_right(img, seed=SEED)
    return img, label


@tf.function
def _val_map(path, label):
    img = _decode_resize(path)
    return img, label


def make_train_val_ds(df_train_in, df_val_in, batch_size):
    train_paths = df_train_in["path"].values
    train_labels = df_train_in["label"].values.astype(np.int32)
    val_paths = df_val_in["path"].values
    val_labels = df_val_in["label"].values.astype(np.int32)

    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    train_ds = train_ds.shuffle(
        buffer_size=len(train_paths), seed=SEED, reshuffle_each_iteration=True
    )
    train_ds = train_ds.map(_train_map, num_parallel_calls=AUTOTUNE, deterministic=True)
    train_ds = train_ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    val_ds = val_ds.map(_val_map, num_parallel_calls=AUTOTUNE, deterministic=True)
    val_ds = val_ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    return train_ds, val_ds


train_ds, val_ds = make_train_val_ds(df_train, df_val, BATCH_SIZE)

steps_per_epoch = int(np.ceil(len(df_train) / BATCH_SIZE))
validation_steps = int(np.ceil(len(df_val) / BATCH_SIZE))

print("steps_per_epoch:", steps_per_epoch, "validation_steps:", validation_steps)




## === cell 3
base = EfficientNetB3(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
)

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
out = Dense(num_classes, activation="softmax")(x)
my_model = Model(inputs=base.input, outputs=out)

base.trainable = False

my_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

ckpt_path = "best_effnetb3.keras"
callbacks = [
    ModelCheckpoint(
        ckpt_path, monitor="val_accuracy", mode="max", save_best_only=True, verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_accuracy",
        mode="max",
        factor=0.5,
        patience=2,
        verbose=1,
        min_lr=1e-6,
    ),
]




## === cell 4
EPOCHS_HEAD = 5 if not DEBUG else 2
history1 = my_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_HEAD,
    callbacks=callbacks,
    verbose=1,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
)

base.trainable = True
my_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

EPOCHS_FT = 3 if not DEBUG else 1
history2 = my_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_HEAD + EPOCHS_FT,
    initial_epoch=EPOCHS_HEAD,
    callbacks=callbacks,
    verbose=1,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
)

my_model = tf.keras.models.load_model(ckpt_path)




## === cell 5
sub = pd.read_csv(SAMPLE_SUB)

sub["path"] = TEST_IMG_DIR.rstrip("/") + "/" + sub["image_id"].astype(str)

missing = (~sub["path"].map(os.path.exists)).sum()
if missing:
    print(
        f"Warning: {missing} test images listed in sample_submission are missing on disk."
    )


def make_test_ds(df_test, batch_size=128):
    test_paths = df_test["path"].values

    def _test_map(path):
        img = _decode_resize(path)
        return img

    ds = tf.data.Dataset.from_tensor_slices(test_paths)
    ds = ds.map(_test_map, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


test_ds = make_test_ds(sub, batch_size=128)
test_steps = int(np.ceil(len(sub) / 128))

pred_test = my_model.predict(
    test_ds,
    verbose=1,
    steps=test_steps,
)
pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

sub["label"] = pred_test_labels
final_csv = sub[["image_id", "label"]]

assert final_csv.shape[0] == len(
    sub
), "Submission row count mismatch vs sample_submission"
final_csv.to_csv("submission.csv", index=False)
print(final_csv.head())
print("Wrote submission.csv with", len(final_csv), "rows")
