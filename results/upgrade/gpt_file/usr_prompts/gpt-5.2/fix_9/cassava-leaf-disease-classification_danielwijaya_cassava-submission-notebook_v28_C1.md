# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


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

# 5. Target score

0.885766092475068

# 6. Current score

0.61099

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.61099) has done: 'I fix the runtime crash caused by an incompatibility between TensorFlow 2.18 and protobuf 6 by pinning protobuf to a compatible 4.x version at runtime before importing TensorFlow. Then I fix inference so we correctly collect `image_id`s from the test dataset (Keras `predict` only returns model outputs, not the extra dataset element), ensuring `test_ids` aligns with the prediction order. Finally, I keep the model/training logic unchanged and write a valid `submission.csv` with exactly the required columns and row count.'

# 9. Code solution

## === cell 0
import os
import re
import random
import sys
import subprocess
import numpy as np
import pandas as pd

try:
    import google.protobuf  # noqa: F401
    import pkgutil, importlib  # noqa: F401
    import importlib.metadata as importlib_metadata

    pb_ver = importlib_metadata.version("protobuf")
    if pb_ver.startswith("6."):
        print(
            "Detected protobuf",
            pb_ver,
            "-> installing protobuf==4.25.3 for TF compatibility...",
        )
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
        )
except Exception as e:
    print("protobuf version check/install skipped:", repr(e))

import tensorflow as tf
import matplotlib.pyplot as plt

from functools import partial

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

print("TF version:", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Enabled GPU memory growth.")
    except Exception as e:
        print("Could not set memory growth:", repr(e))

try:
    tf.config.experimental.enable_op_determinism()
except Exception as e:
    print("Determinism enable skipped:", repr(e))

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception as e:
    print("Threading config skipped:", repr(e))

CACHE_DIR = "/kaggle/working/tfdata_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

try:
    tf.config.optimizer.set_jit(True)
    print("XLA JIT enabled.")
except Exception as e:
    print("XLA enable skipped:", repr(e))




## === cell 1
AUTOTUNE = tf.data.AUTOTUNE

GCS_PATH = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_FILENAMES = tf.io.gfile.glob(GCS_PATH + "/train_tfrecords/ld_train*.tfrec")
TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + "/test_tfrecords/ld_test*.tfrec")


def dataset_sizes(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(fn).group(1)) for fn in filenames]
    return int(np.sum(n))


NUM_TRAIN_IMAGES = dataset_sizes(TRAIN_FILENAMES)
NUM_TEST_IMAGES = dataset_sizes(TEST_FILENAMES)

BATCH_SIZE = 8
IMAGE_SIZE = [512, 512]
NUM_CLASSES = 5

print("Train TFRecords:", len(TRAIN_FILENAMES), "images:", NUM_TRAIN_IMAGES)
print("Test TFRecords :", len(TEST_FILENAMES), "images:", NUM_TEST_IMAGES)




## === cell 2
sample_sub = pd.read_csv(GCS_PATH + "/sample_submission.csv")
print(sample_sub.head(), "\nrows:", len(sample_sub))




## === cell 3
@tf.function
def decode_img(img_bytes):
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE, method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    return img


@tf.function
def read_tfrecord_labeled(example):
    tfrec_format = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, tfrec_format)
    img = decode_img(example["image"])
    label = tf.cast(example["target"], tf.int32)
    return img, label


@tf.function
def read_tfrecord_unlabeled(example):
    tfrec_format = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, tfrec_format)
    img = decode_img(example["image"])
    image_id = example["image_name"]
    return img, image_id


def load_dataset(filenames, labeled=True, ordered=False):
    options = tf.data.Options()
    if not ordered:
        options.experimental_deterministic = False
    try:
        options.experimental_optimization.map_parallelization = True
        options.experimental_optimization.parallel_batch = True
        options.experimental_optimization.autotune_buffers = True
    except Exception:
        pass

    ds = tf.data.TFRecordDataset(
        filenames,
        num_parallel_reads=AUTOTUNE,
        compression_type=None,
        buffer_size=16 * 1024 * 1024,
    )
    ds = ds.with_options(options)

    if labeled:
        ds = ds.map(
            read_tfrecord_labeled, num_parallel_calls=AUTOTUNE, deterministic=ordered
        )
    else:
        ds = ds.map(
            read_tfrecord_unlabeled, num_parallel_calls=AUTOTUNE, deterministic=ordered
        )
    return ds


def _cache_path(name: str) -> str:
    return os.path.join(CACHE_DIR, name)


def get_training_data(filenames, ordered=False):
    ds = load_dataset(filenames, labeled=True, ordered=ordered)
    ds = ds.cache()
    ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE, drop_remainder=True)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def get_val_data(filenames, ordered=True):
    ds = load_dataset(filenames, labeled=True, ordered=ordered)
    ds = ds.cache()
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def get_test_data(ordered=False):
    ds = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(AUTOTUNE)
    return ds




## === cell 4
rng = np.random.RandomState(SEED)
shuffled = TRAIN_FILENAMES.copy()
rng.shuffle(shuffled)

val_count = max(1, int(0.1 * len(shuffled)))
val_files = shuffled[:val_count]
train_files = shuffled[val_count:]

train_ds = get_training_data(train_files, ordered=False)
val_ds = get_val_data(val_files, ordered=True)

NUM_TRAIN = dataset_sizes(train_files)
NUM_VAL = dataset_sizes(val_files)
STEPS_PER_EPOCH = NUM_TRAIN // BATCH_SIZE
VALIDATION_STEPS = int(np.ceil(NUM_VAL / BATCH_SIZE))

print("Train files:", len(train_files), "Val files:", len(val_files))
print("Train images:", NUM_TRAIN, "Val images:", NUM_VAL)
print("STEPS_PER_EPOCH:", STEPS_PER_EPOCH, "VALIDATION_STEPS:", VALIDATION_STEPS)




## === cell 5
base = tf.keras.applications.EfficientNetB5(
    include_top=False,
    weights="imagenet",
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    pooling="avg",
)
base.trainable = False

inputs = tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
x = base(inputs, training=False)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
model_28_1 = tf.keras.Model(inputs, outputs)

model_28_1.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

model_28_1.summary()




## === cell 6
EPOCHS = 3

history = model_28_1.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=VALIDATION_STEPS,
    verbose=1,
)




## === cell 7
test_ds = get_test_data(ordered=True)

print("Collecting test image_ids...")
test_ids_list = []
for _, batch_ids in test_ds:
    test_ids_list.append(batch_ids.numpy())
test_ids = np.concatenate(test_ids_list, axis=0).astype("U")

print("Computing predictions...")
test_images_ds = test_ds.map(lambda img, img_id: img, num_parallel_calls=AUTOTUNE)
probs = model_28_1.predict(test_images_ds, verbose=1)

predictions = np.argmax(probs, axis=-1).astype(np.int64)

print("Pred shape:", predictions.shape, "unique:", np.unique(predictions))
print("IDs shape:", test_ids.shape)

n = min(len(test_ids), len(predictions))
test_ids = test_ids[:n]
predictions = predictions[:n]




## === cell 8
pred_df = pd.DataFrame({"image_id": test_ids, "label": predictions})
sub = sample_sub[["image_id"]].merge(pred_df, on="image_id", how="left")

sub["label"] = sub["label"].fillna(0).astype(int)

assert len(sub) == len(sample_sub), "Submission row count mismatch"
assert list(sub.columns) == ["image_id", "label"], "Bad submission columns"

out_path = "submission.csv"
sub.to_csv(out_path, index=False)
print("Wrote", out_path, "rows:", len(sub))
print(sub.head())
