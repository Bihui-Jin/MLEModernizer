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
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_enable_xla_devices=false")

import pandas as pd
import numpy as np
import tensorflow as tf
import glob
import math

from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB3
from sklearn.model_selection import train_test_split

SEED = 42
DEBUG = False

tf.keras.utils.set_random_seed(SEED)

try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE

if DEBUG:
    tf.data.experimental.enable_debug_mode()
try:
    tf.config.run_functions_eagerly(False)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

print("TF version:", tf.__version__)
print("Num GPUs:", len(tf.config.list_physical_devices("GPU")))
print("CPU count:", os.cpu_count())




## === cell 1
BASE_PATH = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = f"{BASE_PATH}/train.csv"
TRAIN_IMG_DIR = f"{BASE_PATH}/train_images"
TEST_IMG_DIR = f"{BASE_PATH}/test_images"
SAMPLE_SUB = f"{BASE_PATH}/sample_submission.csv"

df_train = pd.read_csv(TRAIN_CSV)
df_train["path"] = TRAIN_IMG_DIR + "/" + df_train["image_id"].astype(str)

num_classes = df_train["label"].nunique()
assert num_classes == 5, f"Expected 5 classes, got {num_classes}"

train_df, val_df = train_test_split(
    df_train,
    test_size=0.15,
    random_state=SEED,
    stratify=df_train["label"],
)

train_df = train_df.copy()
val_df = val_df.copy()

print("Train size:", len(train_df), "Val size:", len(val_df))
train_df.head()




## === cell 2
IMG_SIZE = (300, 300)
BATCH_SIZE = 32

IMG_H, IMG_W = IMG_SIZE
NUM_CLASSES = 5


def _decode_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(
        img, (IMG_H, IMG_W), method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img = tf.cast(img, tf.float32) / 255.0  # same rescale=1/255
    return img


def _one_hot(label):
    return tf.one_hot(tf.cast(label, tf.int32), NUM_CLASSES)


data_augment = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal", seed=SEED),
        tf.keras.layers.RandomRotation(
            factor=15.0 / 180.0, fill_mode="reflect", seed=SEED
        ),
        tf.keras.layers.RandomTranslation(
            height_factor=0.08, width_factor=0.08, fill_mode="reflect", seed=SEED
        ),
        tf.keras.layers.RandomZoom(
            height_factor=(-0.10, 0.10),
            width_factor=(-0.10, 0.10),
            fill_mode="reflect",
            seed=SEED,
        ),
    ],
    name="data_augment",
)

_DATA_OPTIONS = tf.data.Options()
_DATA_OPTIONS.experimental_deterministic = False
try:
    _DATA_OPTIONS.experimental_optimization.map_parallelization = True
    _DATA_OPTIONS.experimental_optimization.parallel_batch = True
    _DATA_OPTIONS.experimental_optimization.autotune_buffers = True
    _DATA_OPTIONS.experimental_slack = True
except Exception:
    pass


def make_train_ds(paths, labels, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.with_options(_DATA_OPTIONS)

    shuffle_buf = min(len(train_df), 8192)
    ds = ds.shuffle(buffer_size=shuffle_buf, seed=SEED, reshuffle_each_iteration=True)

    def _decode_only(path, label):
        img = _decode_resize(path)
        return img, label

    ds = ds.map(_decode_only, num_parallel_calls=AUTOTUNE)

    def _augment_and_label(img, label):
        img = data_augment(img, training=True)
        return img, _one_hot(label)

    ds = ds.map(_augment_and_label, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_val_ds(paths, labels, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.with_options(_DATA_OPTIONS)

    def _map_fn(path, label):
        img = _decode_resize(path)
        return img, _one_hot(label)

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)

    ds = ds.cache()

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_train_ds(train_df["path"].values, train_df["label"].values, BATCH_SIZE)
val_ds = make_val_ds(val_df["path"].values, val_df["label"].values, BATCH_SIZE)

train_steps = math.ceil(len(train_df) / BATCH_SIZE)
val_steps = math.ceil(len(val_df) / BATCH_SIZE)

print("train_steps:", train_steps, "val_steps:", val_steps)




## === cell 3
base = EfficientNetB3(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    pooling="avg",
)

base.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base(inputs, training=False)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(5, activation="softmax")(x)
my_model = Model(inputs, outputs)

my_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    steps_per_execution=10,
)

my_model.summary()




## === cell 4
EPOCHS_WARMUP = 2

history1 = my_model.fit(
    train_ds,
    validation_data=val_ds,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
    epochs=EPOCHS_WARMUP,
    verbose=1,
)

base.trainable = True
for layer in base.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

my_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    steps_per_execution=10,
)

EPOCHS_FINETUNE = 1
history2 = my_model.fit(
    train_ds,
    validation_data=val_ds,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
    epochs=EPOCHS_WARMUP + EPOCHS_FINETUNE,
    initial_epoch=EPOCHS_WARMUP,
    verbose=1,
)




## === cell 5
test_images = sorted(glob.glob(f"{TEST_IMG_DIR}/*.jpg"))
df_test = pd.DataFrame({"path": test_images})


def make_test_ds(paths, batch_size):
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.with_options(_DATA_OPTIONS)

    def _map_fn(path):
        return _decode_resize(path)

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)

    ds = ds.cache()

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


test_batch = 128
test_ds = make_test_ds(df_test["path"].values, test_batch)
print("Test images:", len(df_test))




## === cell 6
pred_test = my_model.predict(
    test_ds,
    verbose=1,
)
pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

final_submission = df_test.copy()
final_submission["image_id"] = final_submission["path"].str.rsplit("/", n=1).str[-1]
final_submission["label"] = pred_test_labels

final_csv = final_submission[["image_id", "label"]].copy()

sample = pd.read_csv(SAMPLE_SUB)
final_csv = sample[["image_id"]].merge(final_csv, on="image_id", how="left")
assert final_csv["label"].notna().all(), "Some test images did not receive predictions."

final_csv["label"] = final_csv["label"].astype(int)
final_csv.to_csv("submission.csv", index=False)

print("Saved submission.csv with shape:", final_csv.shape)
print(final_csv.dtypes)
print(final_csv["label"].value_counts().sort_index())
final_csv.head()
