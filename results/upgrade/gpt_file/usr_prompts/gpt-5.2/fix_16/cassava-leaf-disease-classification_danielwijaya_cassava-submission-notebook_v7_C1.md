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

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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
import numpy as np
import pandas as pd

DATA_ROOT = "/kaggle/input"
print("Listing input root:", DATA_ROOT)
try:
    print("Top-level entries:", sorted(os.listdir(DATA_ROOT))[:50])
except Exception as e:
    print("Could not list input root:", repr(e))




## === cell 1
import sys
import subprocess


def _ensure_tf_compatible_protobuf(target_version="4.25.3"):
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver

        print(f"protobuf=={pb_ver} (no runtime pinning to avoid timeout risk).")
    except Exception as e:
        print("WARNING: Could not check protobuf due to:", repr(e))


_ensure_tf_compatible_protobuf()




## === cell 2
import re
import random
import math
from functools import partial

import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

SEED = 42

try:
    tf.keras.utils.set_random_seed(SEED)
except Exception:
    tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("TensorFlow:", tf.__version__)

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass




## === cell 3
DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_TFRECORDS_GLOB = os.path.join(DATA_DIR, "train_tfrecords", "ld_train*.tfrec")
TEST_TFRECORDS_GLOB = os.path.join(DATA_DIR, "test_tfrecords", "ld_test*.tfrec")

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(SAMPLE_SUB)
print("train_df:", train_df.shape, "test_df:", test_df.shape)

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 64
IMAGE_SIZE = (224, 224)
NUM_CLASSES = 5


_size_re = re.compile(r"-([0-9]*)\.")


def dataset_sizes(filenames):
    n = [int(_size_re.search(filename).group(1)) for filename in filenames]
    return int(np.sum(n))


TRAIN_FILENAMES = sorted(tf.io.gfile.glob(TRAIN_TFRECORDS_GLOB))
TEST_FILENAMES = sorted(tf.io.gfile.glob(TEST_TFRECORDS_GLOB))

NUM_TRAIN_IMAGES = dataset_sizes(TRAIN_FILENAMES)
NUM_TEST_IMAGES = dataset_sizes(TEST_FILENAMES)

print(
    "TFRecords:",
    len(TRAIN_FILENAMES),
    "train files;",
    len(TEST_FILENAMES),
    "test files",
)
print("NUM_TRAIN_IMAGES:", NUM_TRAIN_IMAGES, "NUM_TEST_IMAGES:", NUM_TEST_IMAGES)




## === cell 4
@tf.function
def decode_img(img_bytes):
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE, method="bilinear")
    img = tf.cast(img, tf.float32)  # keep [0,255] for preprocess_input
    return img


@tf.function
def _read_tfrecord_labeled(example):
    tfrec_format = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, tfrec_format)
    img = decode_img(example["image"])
    label = tf.cast(example["target"], tf.int32)
    label = tf.one_hot(label, depth=NUM_CLASSES, dtype=tf.float32)
    return img, label


@tf.function
def _read_tfrecord_unlabeled(example):
    tfrec_format = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, tfrec_format)
    img = decode_img(example["image"])
    image_id = example["image_name"]
    return img, image_id


_TFDATA_OPTS_FAST = tf.data.Options()
try:
    _TFDATA_OPTS_FAST.experimental_optimization.map_and_batch_fusion = True
except Exception:
    pass
try:
    _TFDATA_OPTS_FAST.experimental_optimization.parallel_batch = True
except Exception:
    pass

try:
    _TFDATA_OPTS_FAST.threading.private_threadpool_size = max(8, os.cpu_count() or 8)
    _TFDATA_OPTS_FAST.threading.max_intra_op_parallelism = 1
except Exception:
    pass


def load_dataset(filenames, labeled=True, ordered=False):
    options = tf.data.Options()
    options.experimental_deterministic = bool(ordered)

    ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
    ds = ds.with_options(options)
    ds = ds.with_options(_TFDATA_OPTS_FAST)

    if labeled:
        ds = ds.map(_read_tfrecord_labeled, num_parallel_calls=AUTOTUNE)
    else:
        ds = ds.map(_read_tfrecord_unlabeled, num_parallel_calls=AUTOTUNE)

    ds = ds.apply(tf.data.experimental.ignore_errors())
    return ds


def split_filenames(filenames, val_ratio=0.1, seed=SEED):
    filenames = list(filenames)
    filenames = sorted(filenames)  # deterministic
    tr, va = train_test_split(filenames, test_size=val_ratio, random_state=seed)
    return tr, va


TRAIN_FILES_SPLIT, VAL_FILES_SPLIT = split_filenames(
    TRAIN_FILENAMES, val_ratio=0.1, seed=SEED
)
NUM_TRAIN_IMAGES_SPLIT = dataset_sizes(TRAIN_FILES_SPLIT)
NUM_VAL_IMAGES_SPLIT = dataset_sizes(VAL_FILES_SPLIT)

print(
    "Split TFRecords -> train files:",
    len(TRAIN_FILES_SPLIT),
    "val files:",
    len(VAL_FILES_SPLIT),
)
print(
    "Split sizes -> train images:",
    NUM_TRAIN_IMAGES_SPLIT,
    "val images:",
    NUM_VAL_IMAGES_SPLIT,
)

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal", seed=SEED),
        tf.keras.layers.RandomRotation(0.05, seed=SEED),
        tf.keras.layers.RandomZoom(0.10, seed=SEED),
        tf.keras.layers.RandomContrast(0.10, seed=SEED),
    ],
    name="data_augmentation",
)


@tf.function
def _augment_train(image, label):
    image = data_augmentation(image, training=True)
    return image, label


def _maybe_prefetch_to_device(ds):
    try:
        if tf.config.list_physical_devices("GPU"):
            ds = ds.apply(tf.data.experimental.prefetch_to_device("/GPU:0"))
    except Exception:
        pass
    return ds


_TRAIN_CACHE_PATH = os.path.join("/kaggle/working", "train_decoded_cache")
_VAL_CACHE_PATH = os.path.join("/kaggle/working", "val_decoded_cache")


def get_train_data():
    ds = load_dataset(TRAIN_FILES_SPLIT, labeled=True, ordered=False)

    ds = ds.cache(_TRAIN_CACHE_PATH)

    ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(_augment_train, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.repeat()

    ds = ds.prefetch(AUTOTUNE)
    ds = _maybe_prefetch_to_device(ds)
    return ds


def get_val_data():
    ds = load_dataset(VAL_FILES_SPLIT, labeled=True, ordered=True)

    ds = ds.cache(_VAL_CACHE_PATH)

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.repeat()  # safe with explicit validation_steps
    ds = ds.prefetch(AUTOTUNE)
    ds = _maybe_prefetch_to_device(ds)
    return ds


def get_test_data(ordered=False, cache=False):
    ds = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
    if cache:
        ds = ds.cache()
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    ds = _maybe_prefetch_to_device(ds)
    return ds




## === cell 5
inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
x = tf.keras.layers.Lambda(tf.keras.applications.efficientnet.preprocess_input)(inputs)
backbone = tf.keras.applications.EfficientNetB0(
    include_top=False, weights="imagenet", input_shape=(*IMAGE_SIZE, 3)
)
backbone.trainable = False
x = backbone(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
model_19 = tf.keras.Model(inputs, outputs)

model_19.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=["accuracy"],
)

EPOCHS_STAGE1 = 6
steps_per_epoch = math.ceil(NUM_TRAIN_IMAGES_SPLIT / BATCH_SIZE)
val_steps = math.ceil(NUM_VAL_IMAGES_SPLIT / BATCH_SIZE)

train_ds = get_train_data()
val_ds = get_val_data()

history = model_19.fit(
    train_ds,
    epochs=EPOCHS_STAGE1,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    validation_steps=val_steps,
    verbose=1,
)

backbone.trainable = True
for layer in backbone.layers[:-30]:
    layer.trainable = False

model_19.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=["accuracy"],
)

EPOCHS_STAGE2 = 2
history_ft = model_19.fit(
    train_ds,
    epochs=EPOCHS_STAGE2,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    validation_steps=val_steps,
    verbose=1,
)




## === cell 6
test_df = pd.read_csv(SAMPLE_SUB)
print(test_df.head())

GCS_PATH = DATA_DIR
CLASSES = ["0", "1", "2", "3", "4"]




## === cell 7
test_ds = get_test_data(ordered=True, cache=True)

id_ds = test_ds.map(lambda img, img_id: img_id, num_parallel_calls=AUTOTUNE)
test_ids = np.concatenate([x.numpy() for x in id_ds.unbatch().batch(4096)], axis=0)
test_ids = np.array([x.decode("utf-8") for x in test_ids.tolist()], dtype=object)

probs = model_19.predict(test_ds, verbose=1)
predictions = np.argmax(probs, axis=-1).astype(np.int32)

if len(test_ids) != NUM_TEST_IMAGES or predictions.shape[0] != NUM_TEST_IMAGES:
    raise ValueError(
        f"Collected ids={len(test_ids)} preds={predictions.shape[0]} expected={NUM_TEST_IMAGES}."
    )

print("predictions shape:", predictions.shape, "unique:", np.unique(predictions))

print("Generating submission.csv file...")
sub = pd.DataFrame({"image_id": test_ids, "label": predictions})
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)

with open("submission.csv", "r") as f:
    for _ in range(5):
        print(f.readline().strip())
