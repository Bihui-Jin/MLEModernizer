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
os.environ.pop("OMP_NUM_THREADS", None)
os.environ.pop("TF_NUM_INTRAOP_THREADS", None)
os.environ.pop("TF_NUM_INTEROP_THREADS", None)

import numpy as np
import pandas as pd
import tensorflow as tf

SEED = 42
DEBUG = False

tf.random.set_seed(SEED)
np.random.seed(SEED)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

print("TensorFlow:", tf.__version__)




## === cell 1
def _find_first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


DATASET_ROOT = _find_first_existing(
    [
        "/kaggle/input/cassava-leaf-disease-classification",
        "../input/cassava-leaf-disease-classification",
        "/kaggle/data/cassava-leaf-disease-classification",
        "../data/cassava-leaf-disease-classification",
    ]
)

if DATASET_ROOT is None:
    raise FileNotFoundError(
        "Could not locate cassava-leaf-disease-classification dataset folder in known paths."
    )

TRAIN_CSV = os.path.join(DATASET_ROOT, "train.csv")
SAMPLE_SUB_CSV = os.path.join(DATASET_ROOT, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATASET_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATASET_ROOT, "test_images")
TRAIN_TFREC_DIR = os.path.join(DATASET_ROOT, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(DATASET_ROOT, "test_tfrecords")

for p in [TRAIN_CSV, SAMPLE_SUB_CSV, TRAIN_IMG_DIR, TEST_IMG_DIR]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing expected path: {p}")

print("DATASET_ROOT:", DATASET_ROOT)
print("TRAIN_TFREC_DIR exists:", os.path.exists(TRAIN_TFREC_DIR))
print("TEST_TFREC_DIR exists:", os.path.exists(TEST_TFREC_DIR))




## === cell 2
df_train = pd.read_csv(TRAIN_CSV)
df_sample = pd.read_csv(SAMPLE_SUB_CSV)

train_prefix = TRAIN_IMG_DIR.rstrip(os.sep) + os.sep
test_prefix = TEST_IMG_DIR.rstrip(os.sep) + os.sep

df_train = df_train.copy()
df_train["path"] = train_prefix + df_train["image_id"].astype(str)

df_test = df_sample.copy()
df_test["path"] = test_prefix + df_test["image_id"].astype(str)

NUM_CLASSES = df_train["label"].nunique()
assert NUM_CLASSES == 5, f"Expected 5 classes, got {NUM_CLASSES}"

print("Train rows:", len(df_train), "Test rows:", len(df_test))




## === cell 3
IMG_SIZE = (512, 512)
BATCH_SIZE = 16  # keep moderate for memory/time
EPOCHS = 1  # keep as-is

from sklearn.model_selection import train_test_split

df_tr, df_va = train_test_split(
    df_train,
    test_size=0.1,
    stratify=df_train["label"],
    random_state=SEED,
)

df_tr = df_tr.copy()
df_va = df_va.copy()

AUTOTUNE = tf.data.AUTOTUNE

train_paths = df_tr["path"].astype(str).to_numpy()
train_labels = df_tr["label"].to_numpy(np.int32)

valid_paths = df_va["path"].astype(str).to_numpy()
valid_labels = df_va["label"].to_numpy(np.int32)

test_paths = df_test["path"].astype(str).to_numpy()


def _glob_tfrecs(folder):
    if folder and os.path.exists(folder):
        files = sorted(
            [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.endswith(".tfrec")
                or f.endswith(".tfrecord")
                or f.endswith(".tfrec")
            ]
        )
        return files
    return []


train_tfrec_files = _glob_tfrecs(TRAIN_TFREC_DIR)
test_tfrec_files = _glob_tfrecs(TEST_TFREC_DIR)

print("Found train tfrecords:", len(train_tfrec_files))
print("Found test tfrecords:", len(test_tfrec_files))

options = tf.data.Options()
options.experimental_deterministic = False
try:
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_fusion = True
    options.experimental_optimization.map_parallelization = True
except Exception:
    pass

SHUFFLE_BUFFER = min(len(train_paths), 4096)


def _decode_resize_bytes(img_bytes):
    img = tf.io.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    return img


def _decode_resize_path(path):
    img_bytes = tf.io.read_file(path)
    return _decode_resize_bytes(img_bytes)


def _augment_stateless(img, key):
    seed0 = tf.cast(SEED, tf.int32)
    seed1 = tf.cast(key, tf.int32)
    img = tf.image.stateless_random_flip_left_right(img, seed=[seed0, seed1])
    img = tf.image.stateless_random_flip_up_down(img, seed=[seed0 + 1, seed1 + 1])
    k = tf.random.stateless_uniform(
        [], seed=[seed0 + 2, seed1 + 2], minval=0, maxval=4, dtype=tf.int32
    )
    img = tf.image.rot90(img, k)
    factor = tf.random.stateless_uniform(
        [], seed=[seed0 + 3, seed1 + 3], minval=0.1, maxval=0.3, dtype=tf.float32
    )
    img = tf.clip_by_value(img * factor, 0.0, 1.0)
    return img


_TFREC_FEATURES_TRAIN = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}
_TFREC_FEATURES_TEST = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


def _parse_train_example(example):
    ex = tf.io.parse_single_example(example, _TFREC_FEATURES_TRAIN)
    img = _decode_resize_bytes(ex["image"])
    label = tf.cast(ex["target"], tf.int32)
    return img, label


def _parse_test_example(example):
    ex = tf.io.parse_single_example(example, _TFREC_FEATURES_TEST)
    img = _decode_resize_bytes(ex["image"])
    name = ex["image_name"]
    return img, name


def _train_map_from_path(path, label):
    img = _decode_resize_path(path)
    key = tf.strings.to_hash_bucket_fast(path, 2**31 - 1)
    img = _augment_stateless(img, key)
    return img, label


def _valid_map_from_path(path, label):
    img = _decode_resize_path(path)
    return img, label


def _test_map_from_path(path):
    img = _decode_resize_path(path)
    return img


if len(train_tfrec_files) > 0 and len(test_tfrec_files) > 0:
    train_files_ds = tf.data.Dataset.from_tensor_slices(train_tfrec_files).shuffle(
        buffer_size=len(train_tfrec_files), seed=SEED, reshuffle_each_iteration=True
    )
    train_raw = train_files_ds.interleave(
        lambda fn: tf.data.TFRecordDataset(fn, num_parallel_reads=AUTOTUNE),
        cycle_length=AUTOTUNE,
        num_parallel_calls=AUTOTUNE,
        deterministic=False,
    )
    train_raw = train_raw.with_options(options)

    train_ds = train_raw.map(
        _parse_train_example, num_parallel_calls=AUTOTUNE, deterministic=False
    )
    train_ds = train_ds.shuffle(
        buffer_size=SHUFFLE_BUFFER, seed=SEED, reshuffle_each_iteration=True
    )

    aug_counter = tf.data.experimental.Counter()
    train_ds = tf.data.Dataset.zip((train_ds, aug_counter))
    train_ds = train_ds.map(
        lambda xy, c: (_augment_stateless(xy[0], tf.cast(c, tf.int32)), xy[1]),
        num_parallel_calls=AUTOTUNE,
        deterministic=False,
    )
    train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=True).prefetch(AUTOTUNE)

    valid_ds = tf.data.Dataset.from_tensor_slices((valid_paths, valid_labels))
    valid_ds = valid_ds.with_options(options)
    valid_ds = valid_ds.map(
        _valid_map_from_path, num_parallel_calls=AUTOTUNE, deterministic=False
    )
    valid_ds = valid_ds.cache()
    valid_ds = valid_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

    test_files_ds = tf.data.Dataset.from_tensor_slices(test_tfrec_files)
    test_raw = test_files_ds.interleave(
        lambda fn: tf.data.TFRecordDataset(fn, num_parallel_reads=AUTOTUNE),
        cycle_length=AUTOTUNE,
        num_parallel_calls=AUTOTUNE,
        deterministic=False,
    ).with_options(options)

    test_parsed = test_raw.map(
        _parse_test_example, num_parallel_calls=AUTOTUNE, deterministic=False
    )
    test_parsed = test_parsed.cache()

    test_ds = test_parsed.map(
        lambda img, name: img, num_parallel_calls=AUTOTUNE, deterministic=False
    )
    test_ds = test_ds.batch(64, drop_remainder=False).prefetch(AUTOTUNE)

    test_name_ds = test_parsed.map(
        lambda img, name: name, num_parallel_calls=AUTOTUNE, deterministic=False
    )
    test_name_ds = test_name_ds.batch(2048, drop_remainder=False).prefetch(AUTOTUNE)

else:
    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    train_ds = train_ds.with_options(options)
    train_ds = train_ds.shuffle(
        buffer_size=SHUFFLE_BUFFER, seed=SEED, reshuffle_each_iteration=True
    )
    train_ds = train_ds.map(
        _train_map_from_path, num_parallel_calls=AUTOTUNE, deterministic=False
    )
    train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=True).prefetch(AUTOTUNE)

    valid_ds = tf.data.Dataset.from_tensor_slices((valid_paths, valid_labels))
    valid_ds = valid_ds.with_options(options)
    valid_ds = valid_ds.map(
        _valid_map_from_path, num_parallel_calls=AUTOTUNE, deterministic=False
    )
    valid_ds = valid_ds.cache()
    valid_ds = valid_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
    test_ds = test_ds.with_options(options)
    test_ds = test_ds.map(
        _test_map_from_path, num_parallel_calls=AUTOTUNE, deterministic=False
    )
    test_ds = test_ds.cache()
    test_ds = test_ds.batch(64, drop_remainder=False).prefetch(AUTOTUNE)
    test_name_ds = None

index_to_label_int = np.arange(NUM_CLASSES, dtype=int)

steps_per_epoch = max(
    1, len(train_paths) // BATCH_SIZE
)  # keep exact original semantics
validation_steps = max(1, int(np.ceil(len(valid_paths) / BATCH_SIZE)))

print("steps_per_epoch:", steps_per_epoch, "validation_steps:", validation_steps)




## === cell 4
from tensorflow.keras import layers, models

base = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
)
base.trainable = False  # unchanged

inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2, seed=SEED)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
my_model = models.Model(inputs, outputs)

my_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

my_model.summary()




## === cell 5
history = my_model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=valid_ds,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
)

pred_test = my_model.predict(test_ds, verbose=1)
pred_test_idx = np.argmax(pred_test, axis=-1).astype(int)
pred_test_labels = index_to_label_int[pred_test_idx].astype(int)

final_csv = df_sample.copy()

if test_name_ds is not None:
    test_names = []
    for batch in test_name_ds:
        test_names.extend([n.decode("utf-8") for n in batch.numpy().tolist()])
    if len(test_names) != len(final_csv):
        raise RuntimeError(
            f"Mismatch test names ({len(test_names)}) vs sample_submission ({len(final_csv)})"
        )
    name_to_pred = dict(zip(test_names, pred_test_labels.tolist()))
    final_csv["label"] = final_csv["image_id"].map(name_to_pred).astype(int)
else:
    final_csv["label"] = pred_test_labels

final_csv.to_csv("submission.csv", index=False)

print(final_csv.head())
print("Submission shape:", final_csv.shape)
print("Saved to: submission.csv")
assert list(final_csv.columns) == ["image_id", "label"]
assert final_csv["image_id"].nunique() == len(final_csv)
assert final_csv["label"].between(0, 4).all()
assert os.path.exists("submission.csv")
assert os.path.getsize("submission.csv") > 0
