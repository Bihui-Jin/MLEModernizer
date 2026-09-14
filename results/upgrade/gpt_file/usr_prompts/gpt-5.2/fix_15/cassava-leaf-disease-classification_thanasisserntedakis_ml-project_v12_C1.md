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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import glob
import re
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Activation,
    Dropout,
    Flatten,
    Dense,
)

from sklearn.model_selection import train_test_split

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception as e:
    print("Determinism not enabled (non-fatal):", repr(e))

try:
    tf.config.optimizer.set_jit(True)
except Exception as e:
    print("XLA JIT not enabled (non-fatal):", repr(e))

try:
    tf.config.optimizer.set_experimental_options(
        {
            "layout_optimizer": True,
            "constant_folding": True,
            "shape_optimization": True,
            "remapping": True,
            "arithmetic_optimization": True,
            "dependency_optimization": True,
            "loop_optimization": True,
            "function_optimization": True,
        }
    )
except Exception as e:
    print("Experimental optimizer options not set (non-fatal):", repr(e))

try:
    tf.config.threading.set_inter_op_parallelism_threads(0)
    tf.config.threading.set_intra_op_parallelism_threads(0)
except Exception as e:
    print("Thread settings not applied (non-fatal):", repr(e))

BASE_PATH = "../input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
TRAIN_IMG_DIR = os.path.join(BASE_PATH, "train_images")
TEST_IMG_GLOB = os.path.join(BASE_PATH, "test_images", "*.jpg")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")

TRAIN_TFREC_GLOB = os.path.join(BASE_PATH, "train_tfrecords", "*.tfrec")
TEST_TFREC_GLOB = os.path.join(BASE_PATH, "test_tfrecords", "*.tfrec")

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing {TRAIN_IMG_DIR}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"

print("TF version:", tf.__version__)
print("Keras version:", tf.keras.__version__)

train_tfrec_files = sorted(glob.glob(TRAIN_TFREC_GLOB))
test_tfrec_files = sorted(glob.glob(TEST_TFREC_GLOB))
print(
    "Found train tfrecords:",
    len(train_tfrec_files),
    "test tfrecords:",
    len(test_tfrec_files),
)

CACHE_DIR = "./tfdata_cache"
os.makedirs(CACHE_DIR, exist_ok=True)




## === cell 1
df_train = pd.read_csv(TRAIN_CSV)

df_train["path"] = (TRAIN_IMG_DIR + "/" + df_train["image_id"]).astype(str)
df_train["label"] = df_train["label"].astype(str)

train_df, valid_df = train_test_split(
    df_train, test_size=0.15, random_state=SEED, stratify=df_train["label"]
)

print("Train size:", len(train_df), "Valid size:", len(valid_df))
print("Label distribution (train):", train_df["label"].value_counts().to_dict())




## === cell 2
IMG_SIZE = 224
SIZE = (IMG_SIZE, IMG_SIZE)
BATCH_SIZE = 32

AUTOTUNE = tf.data.AUTOTUNE

classes = sorted(df_train["label"].unique().tolist())
NUM_CLASSES = len(classes)
class_to_idx = {c: i for i, c in enumerate(classes)}
print("NUM_CLASSES:", NUM_CLASSES)
print("Class indices:", class_to_idx)


@tf.function
def _decode_resize_rescale(image_bytes):
    img = tf.io.decode_jpeg(image_bytes, channels=3)
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    return img


@tf.function
def _load_train(path, label_idx):
    image_bytes = tf.io.read_file(path)
    img = _decode_resize_rescale(image_bytes)
    y = tf.one_hot(label_idx, NUM_CLASSES, dtype=tf.float32)
    return img, y


@tf.function
def _load_test(path):
    image_bytes = tf.io.read_file(path)
    img = _decode_resize_rescale(image_bytes)
    return img


def make_train_valid_ds_from_jpegs(train_df, valid_df, batch_size):
    train_paths = train_df["path"].to_numpy()
    valid_paths = valid_df["path"].to_numpy()
    train_y = train_df["label"].map(class_to_idx).to_numpy(np.int32)
    valid_y = valid_df["label"].map(class_to_idx).to_numpy(np.int32)

    options = tf.data.Options()
    options.experimental_deterministic = (
        False  # throughput; label correctness preserved
    )
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True

    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_y)).with_options(
        options
    )
    train_ds = train_ds.shuffle(
        buffer_size=len(train_paths), seed=SEED, reshuffle_each_iteration=True
    )
    train_ds = train_ds.map(_load_train, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.cache(os.path.join(CACHE_DIR, "jpeg_train.cache"))
    train_ds = train_ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)

    valid_ds = tf.data.Dataset.from_tensor_slices((valid_paths, valid_y)).with_options(
        options
    )
    valid_ds = valid_ds.map(_load_train, num_parallel_calls=AUTOTUNE)
    valid_ds = valid_ds.cache(os.path.join(CACHE_DIR, "jpeg_valid.cache"))
    valid_ds = valid_ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    return train_ds, valid_ds


_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


@tf.function
def _parse_train_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURES)
    img = _decode_resize_rescale(ex["image"])
    y = tf.one_hot(tf.cast(ex["target"], tf.int32), NUM_CLASSES, dtype=tf.float32)
    return img, y


@tf.function
def _parse_test_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURES)
    img = _decode_resize_rescale(ex["image"])
    return img, ex["image_name"]


def make_train_valid_ds_from_tfrecords(train_files, valid_files, batch_size):
    options = tf.data.Options()
    options.experimental_deterministic = (
        False  # throughput; label correctness preserved
    )
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True

    train_ds = tf.data.TFRecordDataset(
        train_files, num_parallel_reads=AUTOTUNE
    ).with_options(options)
    train_ds = train_ds.shuffle(
        buffer_size=8192, seed=SEED, reshuffle_each_iteration=True
    )
    train_ds = train_ds.map(_parse_train_example, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.cache(os.path.join(CACHE_DIR, "tfrec_train.cache"))
    train_ds = train_ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)

    valid_ds = tf.data.TFRecordDataset(
        valid_files, num_parallel_reads=AUTOTUNE
    ).with_options(options)
    valid_ds = valid_ds.map(_parse_train_example, num_parallel_calls=AUTOTUNE)
    valid_ds = valid_ds.cache(os.path.join(CACHE_DIR, "tfrec_valid.cache"))
    valid_ds = valid_ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    return train_ds, valid_ds


def _extract_train_shard_id_from_path(p: str) -> int:
    m = re.search(r"ld_train(\d+)-", os.path.basename(p))
    if not m:
        raise ValueError(f"Unrecognized train tfrecord name: {p}")
    return int(m.group(1))


def _compute_shard_ids(image_ids: np.ndarray) -> np.ndarray:
    stems = np.char.partition(image_ids.astype(str), ".")[:, 0]
    ids_int = stems.astype(np.int64, copy=False)
    return (ids_int % 14).astype(np.int32, copy=False)


def _tfrecord_files_for_split(split_df: pd.DataFrame, shard_to_path: dict) -> list:
    shard_ids = np.unique(_compute_shard_ids(split_df["image_id"].to_numpy()))
    files = []
    missing = []
    for sid in shard_ids.tolist():
        if int(sid) in shard_to_path:
            files.append(shard_to_path[int(sid)])
        else:
            missing.append(int(sid))
    if missing:
        raise FileNotFoundError(
            f"Missing TFRecord shard ids {missing} from available files: {sorted(shard_to_path.keys())}"
        )
    return sorted(files)


if len(train_tfrec_files) > 0:
    shard_to_path = {_extract_train_shard_id_from_path(p): p for p in train_tfrec_files}
    train_files = _tfrecord_files_for_split(train_df, shard_to_path)
    valid_files = _tfrecord_files_for_split(valid_df, shard_to_path)
    print(
        "Using TFRecords for training. Train shards:",
        len(train_files),
        "Valid shards:",
        len(valid_files),
    )
    train_ds, valid_ds = make_train_valid_ds_from_tfrecords(
        train_files, valid_files, BATCH_SIZE
    )
else:
    print("TFRecords not found; falling back to JPEG loading (slower).")
    train_ds, valid_ds = make_train_valid_ds_from_jpegs(train_df, valid_df, BATCH_SIZE)


def make_test_ds_from_tfrecords(test_files, batch_size):
    options = tf.data.Options()
    options.experimental_deterministic = True  # fixed order for name alignment
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True

    ds = tf.data.TFRecordDataset(test_files, num_parallel_reads=AUTOTUNE).with_options(
        options
    )
    ds = ds.map(_parse_test_example, num_parallel_calls=AUTOTUNE)
    ds = ds.cache(os.path.join(CACHE_DIR, "tfrec_test.cache"))
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


if len(test_tfrec_files) > 0:
    test_ds = make_test_ds_from_tfrecords(test_tfrec_files, BATCH_SIZE)
    df_test = None
else:
    test_images = sorted(glob.glob(TEST_IMG_GLOB))
    df_test = pd.DataFrame(
        {
            "path": test_images,
            "image_id": [os.path.basename(p) for p in test_images],
        }
    )

    def make_test_ds(test_paths, batch_size):
        options = tf.data.Options()
        options.experimental_deterministic = True
        options.experimental_optimization.apply_default_optimizations = True
        options.experimental_optimization.map_parallelization = True
        options.experimental_optimization.parallel_batch = True

        ds = tf.data.Dataset.from_tensor_slices(test_paths).with_options(options)
        ds = ds.map(_load_test, num_parallel_calls=AUTOTUNE)
        ds = ds.cache(os.path.join(CACHE_DIR, "jpeg_test.cache"))
        ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
        return ds

    test_ds = make_test_ds(df_test["path"].to_numpy(), BATCH_SIZE)




## === cell 3
model = Sequential(
    [
        Conv2D(32, (3, 3), input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        Activation("relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3)),
        Activation("relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3)),
        Activation("relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(256),
        Activation("relu"),
        Dropout(0.5),
        Dense(NUM_CLASSES),
        Activation("softmax"),
    ]
)

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
    jit_compile=True,
    steps_per_execution=32,
)
model.summary()




## === cell 4
EPOCHS = 3
history = model.fit(train_ds, validation_data=valid_ds, epochs=EPOCHS, verbose=1)




## === cell 5
if len(test_tfrec_files) > 0:
    test_names = []
    for _, name_batch in test_ds:
        test_names.extend([n.decode("utf-8") for n in name_batch.numpy()])

    test_img_ds = test_ds.map(lambda img, name: img, num_parallel_calls=AUTOTUNE)
    pred_test = model.predict(test_img_ds, verbose=1)
    df_test = pd.DataFrame({"image_id": np.asarray(test_names, dtype=object)})
else:
    pred_test = model.predict(test_ds, verbose=1)

pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

final_submission = pd.DataFrame(
    {
        "image_id": df_test["image_id"].values,
        "label": pred_test_labels,
    }
)

sample = pd.read_csv(SAMPLE_SUB)
assert len(final_submission) == len(
    sample
), f"Pred rows {len(final_submission)} != sample {len(sample)}"

final_submission = sample[["image_id"]].merge(
    final_submission, on="image_id", how="left"
)
missing = final_submission["label"].isna().sum()
assert missing == 0, f"Some test image_ids missing predictions: {missing}"

final_submission["label"] = final_submission["label"].astype(int)
final_submission.to_csv("submission.csv", index=False)
print(final_submission.head())
print("Wrote submission.csv with shape:", final_submission.shape)
