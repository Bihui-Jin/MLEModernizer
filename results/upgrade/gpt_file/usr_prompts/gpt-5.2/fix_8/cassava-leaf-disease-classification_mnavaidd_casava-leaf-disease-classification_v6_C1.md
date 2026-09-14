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
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization,
    GlobalAveragePooling2D,
)

SEED = 42
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE



## === cell 1
data_path = "/kaggle/input/cassava-leaf-disease-classification/"
if not os.path.exists(data_path):
    data_path = "../input/cassava-leaf-disease-classification/"

train_csv_data_path = os.path.join(data_path, "train.csv")
label_json_data_path = os.path.join(data_path, "label_num_to_disease_map.json")
train_images_dir_data_path = os.path.join(data_path, "train_images")
test_images_dir_data_path = os.path.join(data_path, "test_images")
sample_sub_path = os.path.join(data_path, "sample_submission.csv")

train_tfrecords_dir_data_path = os.path.join(data_path, "train_tfrecords")
test_tfrecords_dir_data_path = os.path.join(data_path, "test_tfrecords")

assert os.path.exists(train_csv_data_path), f"Missing {train_csv_data_path}"
assert os.path.exists(
    train_images_dir_data_path
), f"Missing {train_images_dir_data_path}"
assert os.path.exists(test_images_dir_data_path), f"Missing {test_images_dir_data_path}"
assert os.path.exists(sample_sub_path), f"Missing {sample_sub_path}"

assert os.path.exists(
    train_tfrecords_dir_data_path
), f"Missing {train_tfrecords_dir_data_path}"
assert os.path.exists(
    test_tfrecords_dir_data_path
), f"Missing {test_tfrecords_dir_data_path}"



## === cell 2
train_csv = pd.read_csv(train_csv_data_path)
train_csv["label"] = train_csv["label"].astype(int)

label_class = pd.read_json(label_json_data_path, orient="index")
label_class = label_class.values.flatten().tolist()

train_csv["filepath"] = (
    train_images_dir_data_path + "/" + train_csv["image_id"].astype(str)
)

assert len(train_csv) > 0 and train_csv["filepath"].iloc[0].endswith(".jpg")

NUM_CLASSES = train_csv["label"].nunique()
assert NUM_CLASSES == 5, f"Expected 5 classes, got {NUM_CLASSES}"



## === cell 3
print("Label names :")
for i, label in enumerate(label_class):
    print(f" {i}. {label}")



## === cell 4
train_csv.head()



## === cell 5
BATCH_SIZE = 18
IMG_SIZE = 224
EPOCHS = 2  # keep identical

from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(
    train_csv,
    test_size=0.1,
    random_state=SEED,
    stratify=train_csv["label"],
)



## === cell 6
base_model = applications.ResNet152(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)

base_model.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.25)(x)
outputs = Dense(NUM_CLASSES, activation="softmax")(x)
model_model = tf.keras.Model(inputs, outputs)

model_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    jit_compile=True,
)



## === cell 7
model_model.summary()



## === cell 8
ss = pd.read_csv(sample_sub_path)
test_img_path = os.path.join(test_images_dir_data_path, ss["image_id"].iloc[0])

img = cv2.imread(test_img_path)
assert img is not None, f"Failed to read image: {test_img_path}"

resized_img = (
    cv2.resize(img, (IMG_SIZE, IMG_SIZE)).reshape(-1, IMG_SIZE, IMG_SIZE, 3) / 255.0
)

plt.figure(figsize=(8, 4))
plt.title("TEST IMAGE")
plt.imshow(resized_img[0][:, :, ::-1])  # BGR->RGB for cv2
plt.axis("off")
plt.show()



## === cell 9
data_augment = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal", seed=SEED),
        tf.keras.layers.RandomRotation(15 / 360.0, fill_mode="reflect", seed=SEED),
        tf.keras.layers.RandomTranslation(
            height_factor=0.05, width_factor=0.05, fill_mode="reflect", seed=SEED
        ),
        tf.keras.layers.RandomZoom(
            height_factor=(-0.1, 0.1),
            width_factor=(-0.1, 0.1),
            fill_mode="reflect",
            seed=SEED,
        ),
    ],
    name="data_augment",
)

_FEATURES_TRAIN = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}
_FEATURES_TEST = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


@tf.function
def _decode_and_resize_from_bytes(img_bytes):
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) / 255.0
    return img


@tf.function
def _parse_train_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURES_TRAIN)
    img = _decode_and_resize_from_bytes(ex["image"])
    y = tf.cast(ex["target"], tf.int32)
    image_id = ex["image_name"]
    return img, y, image_id


@tf.function
def _parse_test_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURES_TEST)
    img = _decode_and_resize_from_bytes(ex["image"])
    image_id = ex["image_name"]
    return img, image_id


train_tfrec_files = tf.io.gfile.glob(
    os.path.join(train_tfrecords_dir_data_path, "*.tfrec")
)
test_tfrec_files = tf.io.gfile.glob(
    os.path.join(test_tfrecords_dir_data_path, "*.tfrec")
)
train_tfrec_files = sorted(train_tfrec_files)
test_tfrec_files = sorted(test_tfrec_files)
assert len(train_tfrec_files) > 0, "No train tfrecords found"
assert len(test_tfrec_files) > 0, "No test tfrecords found"


def _num_steps(n, batch_size):
    return (int(n) + int(batch_size) - 1) // int(batch_size)


train_count = len(train_df)
val_count = len(val_df)

train_steps = _num_steps(train_count, BATCH_SIZE)
val_steps = _num_steps(val_count, BATCH_SIZE)
test_steps = _num_steps(len(ss), BATCH_SIZE)

options = tf.data.Options()
options.experimental_deterministic = True
options.experimental_optimization.apply_default_optimizations = True
options.experimental_optimization.map_parallelization = True
options.experimental_optimization.parallel_batch = True


def _shard_index_from_image_id_series(
    image_id_series: pd.Series, num_shards: int = 16
) -> np.ndarray:
    s = image_id_series.astype(str).str.replace(".jpg", "", regex=False)
    ids_int = s.astype(np.int64).values
    return (ids_int % num_shards).astype(np.int64)


NUM_TRAIN_SHARDS = (
    16  # fixed for this dataset's TFRecord naming convention (ld_train00..15)
)

train_shards = np.unique(
    _shard_index_from_image_id_series(train_df["image_id"], NUM_TRAIN_SHARDS)
)
val_shards = np.unique(
    _shard_index_from_image_id_series(val_df["image_id"], NUM_TRAIN_SHARDS)
)


def _select_train_tfrec_files_by_shards(shards: np.ndarray):
    selected = []
    shard_set = set(int(x) for x in shards.tolist())
    for fp in train_tfrec_files:
        base = os.path.basename(fp)
        try:
            xx = int(base.split("ld_train")[1][:2])
        except Exception:
            continue
        if xx in shard_set:
            selected.append(fp)
    selected = sorted(selected)
    assert len(selected) > 0, "Shard selection resulted in empty TFRecord file list"
    return selected


train_tfrec_files_split = _select_train_tfrec_files_by_shards(train_shards)
val_tfrec_files_split = _select_train_tfrec_files_by_shards(val_shards)

tfrec_read_opts = tf.io.TFRecordOptions(
    compression_type=""
)  # explicit, same as default


def _make_base_train_ds(files):
    ds = tf.data.TFRecordDataset(
        files,
        num_parallel_reads=AUTOTUNE,
        compression_type=tfrec_read_opts.compression_type,
    )
    ds = ds.with_options(options)
    ds = ds.map(_parse_train_example, num_parallel_calls=AUTOTUNE)
    return ds


SHUFFLE_BUFFER = min(4096, train_count)


def _make_train_ds_from_tfrecords():
    ds = _make_base_train_ds(train_tfrec_files_split)

    ds = ds.take(train_count)

    ds = ds.map(lambda x, y, image_id: (x, y), num_parallel_calls=AUTOTUNE)
    ds = ds.shuffle(
        buffer_size=SHUFFLE_BUFFER, seed=SEED, reshuffle_each_iteration=True
    )
    ds = ds.map(
        lambda x, y: (data_augment(x, training=True), y), num_parallel_calls=AUTOTUNE
    )
    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


def _make_val_ds_from_tfrecords():
    ds = _make_base_train_ds(val_tfrec_files_split)

    ds = ds.take(val_count)

    ds = ds.map(lambda x, y, image_id: (x, y), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


def _make_test_ds_from_tfrecords():
    ds = tf.data.TFRecordDataset(
        test_tfrec_files,
        num_parallel_reads=AUTOTUNE,
        compression_type=tfrec_read_opts.compression_type,
    )
    ds = ds.with_options(options)
    ds = ds.map(_parse_test_example, num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda x, image_id: x, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


train_ds = _make_train_ds_from_tfrecords()
val_ds = _make_val_ds_from_tfrecords()

history = model_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
)

test_ds = _make_test_ds_from_tfrecords()

probs = model_model.predict(test_ds, verbose=1, steps=test_steps)
preds = np.argmax(probs, axis=1).astype(int)

my_submission = pd.DataFrame({"image_id": ss["image_id"].values, "label": preds})
my_submission.to_csv("submission.csv", index=False)



## === cell 10
print("Submission File: \n---------------\n")
print(my_submission.head())
print("\nSaved to: submission.csv")
print("Rows:", len(my_submission), "Columns:", list(my_submission.columns))
