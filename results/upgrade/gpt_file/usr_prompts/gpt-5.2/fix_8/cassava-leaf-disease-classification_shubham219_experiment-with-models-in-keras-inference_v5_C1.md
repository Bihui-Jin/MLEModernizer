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
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

SEED = 42
DEBUG = False

tf.keras.utils.set_random_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.threading.set_inter_op_parallelism_threads(0)
    tf.config.threading.set_intra_op_parallelism_threads(0)
except Exception:
    pass

print("TensorFlow:", tf.__version__)



## === cell 1
CANDIDATE_BASES = [
    "../input/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
]

BASE = None
for b in CANDIDATE_BASES:
    if os.path.exists(b):
        BASE = b
        break

if BASE is None:
    raise FileNotFoundError(f"Could not find dataset folder. Tried: {CANDIDATE_BASES}")

TRAIN_CSV = os.path.join(BASE, "train.csv")
SAMPLE_SUB = os.path.join(BASE, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(BASE, "train_images")
TEST_IMG_DIR = os.path.join(BASE, "test_images")
TRAIN_TFREC_DIR = os.path.join(BASE, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(BASE, "test_tfrecords")

print("Using BASE:", BASE)
print("Train CSV exists:", os.path.exists(TRAIN_CSV))
print("Train images dir exists:", os.path.exists(TRAIN_IMG_DIR))
print("Test images dir exists:", os.path.exists(TEST_IMG_DIR))
print("Train TFRecords dir exists:", os.path.exists(TRAIN_TFREC_DIR))
print("Test TFRecords dir exists:", os.path.exists(TEST_TFREC_DIR))

train_df = pd.read_csv(TRAIN_CSV)
sample_df = pd.read_csv(SAMPLE_SUB)

train_df["path"] = TRAIN_IMG_DIR.rstrip("/") + "/" + train_df["image_id"].astype(str)

assert set(sample_df.columns) == {"image_id", "label"}
assert set(train_df.columns) >= {"image_id", "label", "path"}
assert train_df["label"].nunique() == 5, "Expected 5 classes (0..4)."

if DEBUG:
    train_df = train_df.sample(2000, random_state=SEED).reset_index(drop=True)

train_df.head()



## === cell 2
IMG_SIZE = (300, 300)
BATCH_SIZE = 32
NUM_CLASSES = 5

trn_df, val_df = train_test_split(
    train_df,
    test_size=0.15,
    random_state=SEED,
    stratify=train_df["label"],
)

AUTOTUNE = tf.data.AUTOTUNE

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal", seed=SEED),
        layers.RandomRotation(factor=10.0 / 360.0, fill_mode="reflect", seed=SEED),
        layers.RandomTranslation(
            height_factor=0.05, width_factor=0.05, fill_mode="reflect", seed=SEED
        ),
        layers.RandomZoom(
            height_factor=(-0.1, 0.1),
            width_factor=(-0.1, 0.1),
            fill_mode="reflect",
            seed=SEED,
        ),
    ],
    name="data_augmentation",
)


@tf.function
def _decode_resize_rescale_from_bytes(jpeg_bytes):
    img = tf.image.decode_jpeg(jpeg_bytes, channels=3)
    img = tf.image.resize(
        img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=True
    )
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    return img


@tf.function
def _decode_resize_rescale_from_path(path):
    img = tf.io.read_file(path)
    return _decode_resize_rescale_from_bytes(img)


@tf.function
def _apply_aug(x, y):
    x = data_augmentation(x, training=True)
    return x, y


def _base_options():
    options = tf.data.Options()
    options.deterministic = True  # preserve deterministic ordering/behavior
    return options


_TFREC_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


@tf.function
def _parse_train_example(serialized):
    ex = tf.io.parse_single_example(serialized, _TFREC_FEATURES)
    x = _decode_resize_rescale_from_bytes(ex["image"])
    y = tf.cast(ex["target"], tf.int32)
    return x, y


@tf.function
def _parse_test_example(serialized):
    ex = tf.io.parse_single_example(serialized, _TFREC_FEATURES)
    x = _decode_resize_rescale_from_bytes(ex["image"])
    return x, ex["image_name"]


def _list_tfrecords(directory, pattern="*.tfrec"):
    if not os.path.exists(directory):
        return []
    files = sorted(glob.glob(os.path.join(directory, pattern)))
    return files


def make_train_ds(df, batch_size):
    tfrec_files = _list_tfrecords(TRAIN_TFREC_DIR, "*.tfrec")
    if len(tfrec_files) > 0:
        ds = tf.data.TFRecordDataset(tfrec_files, num_parallel_reads=AUTOTUNE)
        shuffle_buf = min(len(df), 8192)
        ds = ds.shuffle(
            buffer_size=shuffle_buf, seed=SEED, reshuffle_each_iteration=True
        )
        ds = ds.map(
            _parse_train_example, num_parallel_calls=AUTOTUNE, deterministic=True
        )
        ds = ds.map(_apply_aug, num_parallel_calls=AUTOTUNE, deterministic=True)
        ds = ds.batch(batch_size, drop_remainder=False)
        ds = ds.prefetch(AUTOTUNE)
        ds = ds.with_options(_base_options())
        return ds

    paths = df["path"].to_numpy(dtype=object)
    labels = df["label"].to_numpy(dtype=np.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    shuffle_buf = min(len(df), 4096)
    ds = ds.shuffle(buffer_size=shuffle_buf, seed=SEED, reshuffle_each_iteration=True)

    @tf.function
    def _load(path, y):
        x = _decode_resize_rescale_from_path(path)
        return x, y

    ds = ds.map(_load, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.map(_apply_aug, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    ds = ds.with_options(_base_options())
    return ds


def make_val_ds(df, batch_size):
    tfrec_files = _list_tfrecords(TRAIN_TFREC_DIR, "*.tfrec")
    if len(tfrec_files) > 0:
        ds = tf.data.TFRecordDataset(tfrec_files, num_parallel_reads=AUTOTUNE)
        pass

    paths = df["path"].to_numpy(dtype=object)
    labels = df["label"].to_numpy(dtype=np.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    @tf.function
    def _load(path, y):
        x = _decode_resize_rescale_from_path(path)
        return x, y

    ds = ds.map(_load, num_parallel_calls=AUTOTUNE, deterministic=True)

    ds = ds.cache()

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    ds = ds.with_options(_base_options())
    return ds


train_ds = make_train_ds(trn_df, BATCH_SIZE)
val_ds = make_val_ds(val_df, BATCH_SIZE)

steps_per_epoch = int(np.ceil(len(trn_df) / BATCH_SIZE))
validation_steps = int(np.ceil(len(val_df) / BATCH_SIZE))

backbone = EfficientNetB3(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    pooling="avg",
)
backbone.trainable = False  # keep core logic simple/stable and fast

inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = backbone(inputs, training=False)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

my_model = Model(inputs, outputs)

my_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    steps_per_execution=16,
)

my_model.summary()



## === cell 3
EPOCHS = 2 if not DEBUG else 1

history = my_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
)

backbone.trainable = True
for layer in backbone.layers[:-30]:
    layer.trainable = False

my_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    steps_per_execution=16,
)

history_ft = my_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=1 if not DEBUG else 1,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
)



## === cell 4
df_test = pd.DataFrame({"image_id": sample_df["image_id"].astype(str)})
df_test["path"] = TEST_IMG_DIR.rstrip("/") + "/" + df_test["image_id"]

if not os.path.exists(TEST_IMG_DIR):
    test_images = sorted(
        glob.glob(os.path.join(BASE, "**", "test_images", "*.jpg"), recursive=True)
    )
    if len(test_images) == 0:
        raise FileNotFoundError("No test images found under expected directories.")
    df_test = pd.DataFrame(test_images, columns=["path"])
    df_test["image_id"] = df_test["path"].str.split("/").str[-1]


def make_test_ds(df, batch_size=128):
    tfrec_files = _list_tfrecords(TEST_TFREC_DIR, "*.tfrec")
    if len(tfrec_files) > 0:
        ds = tf.data.TFRecordDataset(tfrec_files, num_parallel_reads=AUTOTUNE)
        ds = ds.map(
            _parse_test_example, num_parallel_calls=AUTOTUNE, deterministic=True
        )
        ds_img = ds.map(
            lambda x, name: x, num_parallel_calls=AUTOTUNE, deterministic=True
        )
        ds_img = ds_img.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
        ds_img = ds_img.with_options(_base_options())
        return ds_img

    paths = df["path"].to_numpy(dtype=object)
    ds = tf.data.Dataset.from_tensor_slices(paths)

    @tf.function
    def _load(path):
        x = _decode_resize_rescale_from_path(path)
        return x

    ds = ds.map(_load, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    ds = ds.with_options(_base_options())
    return ds


test_ds = make_test_ds(df_test, batch_size=128)

pred_test = my_model.predict(
    test_ds,
    verbose=1,
)

pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

final_submission = df_test.copy()
final_submission["label"] = pred_test_labels

final_csv = final_submission[["image_id", "label"]]

final_csv = sample_df[["image_id"]].merge(final_csv, on="image_id", how="left")
if final_csv["label"].isna().any():
    fill_label = int(pd.Series(pred_test_labels).mode().iloc[0])
    final_csv["label"] = final_csv["label"].fillna(fill_label).astype(int)
else:
    final_csv["label"] = final_csv["label"].astype(int)

final_csv.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", final_csv.shape)
final_csv.head()
