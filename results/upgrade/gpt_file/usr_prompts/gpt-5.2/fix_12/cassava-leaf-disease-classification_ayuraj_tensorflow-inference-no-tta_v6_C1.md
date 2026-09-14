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

import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.models import *

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

print("TF version:", tf.__version__)

try:
    tf.config.experimental.enable_op_determinism()
except Exception as e:
    print("Determinism enable failed (non-fatal):", e)

try:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except Exception as e:
    print("GPU mem growth set failed (non-fatal):", e)

try:
    tf.config.optimizer.set_jit(True)
except Exception as e:
    print("XLA JIT enable failed (non-fatal):", e)

try:
    tf.data.experimental.enable_debug_mode(False)
except Exception:
    pass



## === cell 1
DATA_DIR = "../input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")

TRAIN_TFREC_DIR = os.path.join(DATA_DIR, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(DATA_DIR, "test_tfrecords")

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(TRAIN_IMG_DIR), f"Missing {TRAIN_IMG_DIR}"
assert os.path.exists(TEST_IMG_DIR), f"Missing {TEST_IMG_DIR}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_TFREC_DIR), f"Missing {TRAIN_TFREC_DIR}"
assert os.path.isdir(TEST_TFREC_DIR), f"Missing {TEST_TFREC_DIR}"

IMAGE_SIZE = 380
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE
CLASS_NUMS = 5

train_df = pd.read_csv(TRAIN_CSV)
print("Train rows:", len(train_df), "columns:", train_df.columns.tolist())
print("Label distribution:\n", train_df["label"].value_counts().sort_index())



## === cell 2
tf.keras.utils.set_random_seed(42)

train_image_ids = train_df["image_id"].astype(str).to_numpy()
train_labels = train_df["label"].astype("int32").to_numpy()

idx = np.arange(len(train_image_ids))
rng = np.random.default_rng(42)
rng.shuffle(idx)
split = int(0.9 * len(idx))
tr_idx, va_idx = idx[:split], idx[split:]

tr_image_ids, va_image_ids = train_image_ids[tr_idx], train_image_ids[va_idx]
tr_labels, va_labels = train_labels[tr_idx], train_labels[va_idx]

tr_id_set = set(tr_image_ids.tolist())
va_id_set = set(va_image_ids.tolist())

label_map = dict(zip(train_image_ids.tolist(), train_labels.tolist()))
label_keys = tf.constant(list(label_map.keys()), dtype=tf.string)
label_vals = tf.constant(list(label_map.values()), dtype=tf.int32)
label_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(label_keys, label_vals),
    default_value=tf.constant(-1, dtype=tf.int32),
)

split_keys = tf.constant(list(tr_id_set) + list(va_id_set), dtype=tf.string)
split_vals = tf.concat(
    [
        tf.ones([len(tr_id_set)], dtype=tf.int32),  # 1 = train
        tf.fill([len(va_id_set)], tf.constant(2, tf.int32)),  # 2 = val
    ],
    axis=0,
)
split_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(split_keys, split_vals),
    default_value=tf.constant(0, dtype=tf.int32),  # 0 = neither
)

print("Train/Val sizes:", len(tr_image_ids), len(va_image_ids))



## === cell 3
TFREC_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
}


@tf.function
def _parse_tfrecord_with_meta(example_proto):
    ex = tf.io.parse_single_example(example_proto, TFREC_FEATURES)
    image_id = ex["image_name"]
    img = tf.io.decode_jpeg(ex["image"], channels=3, dct_method="INTEGER_FAST")
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img = tf.image.resize(
        img, (IMAGE_SIZE, IMAGE_SIZE), method=tf.image.ResizeMethod.BILINEAR
    )
    y = label_table.lookup(image_id)
    split_id = split_table.lookup(image_id)  # 1=train, 2=val, 0=other
    return img, tf.cast(y, tf.int32), image_id, tf.cast(split_id, tf.int32)


@tf.function
def _parse_tfrecord_test(example_proto):
    ex = tf.io.parse_single_example(example_proto, TFREC_FEATURES)
    img = tf.io.decode_jpeg(ex["image"], channels=3, dct_method="INTEGER_FAST")
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img = tf.image.resize(
        img, (IMAGE_SIZE, IMAGE_SIZE), method=tf.image.ResizeMethod.BILINEAR
    )
    image_id = ex["image_name"]
    return img, image_id


@tf.function
def augment_if_training(img, label=None, training=False):
    if training:
        img = tf.image.random_flip_left_right(img)
    if label is None:
        return img
    return img, label


def _list_tfrecs(tfrecs_dir):
    files = tf.io.gfile.glob(os.path.join(tfrecs_dir, "*.tfrec"))
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No .tfrec files found in {tfrecs_dir}")
    return files


def make_train_val_ds_from_tfrecs(training=True):
    options = tf.data.Options()
    options.experimental_deterministic = True
    try:
        options.experimental_optimization.map_parallelization = True
        options.experimental_optimization.parallel_batch = True
        if hasattr(options.experimental_optimization, "autotune_buffers"):
            options.experimental_optimization.autotune_buffers = True
        if hasattr(options.experimental_optimization, "apply_default_optimizations"):
            options.experimental_optimization.apply_default_optimizations = True
    except Exception:
        pass

    files = _list_tfrecs(TRAIN_TFREC_DIR)
    ds = tf.data.Dataset.from_tensor_slices(files)
    ds = ds.interleave(
        lambda f: tf.data.TFRecordDataset(f, num_parallel_reads=AUTOTUNE),
        cycle_length=AUTOTUNE,
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )

    ds = ds.map(
        _parse_tfrecord_with_meta, num_parallel_calls=AUTOTUNE, deterministic=True
    )
    ds = ds.apply(tf.data.experimental.ignore_errors())

    want_split = tf.constant(1 if training else 2, dtype=tf.int32)
    ds = ds.filter(
        lambda img, y, image_id, split_id: tf.logical_and(
            tf.equal(split_id, want_split), y >= 0
        )
    )

    ds = ds.map(
        lambda img, y, image_id, split_id: (img, y),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    ds = ds.cache()

    if training:
        ds = ds.shuffle(2048, seed=42, reshuffle_each_iteration=True)
        ds = ds.map(
            lambda img, y: augment_if_training(img, y, training=True),
            num_parallel_calls=AUTOTUNE,
            deterministic=True,
        )

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    ds = ds.with_options(options)
    return ds


def make_test_ds_from_tfrecs():
    options = tf.data.Options()
    options.experimental_deterministic = True
    try:
        options.experimental_optimization.map_parallelization = True
        options.experimental_optimization.parallel_batch = True
        if hasattr(options.experimental_optimization, "autotune_buffers"):
            options.experimental_optimization.autotune_buffers = True
        if hasattr(options.experimental_optimization, "apply_default_optimizations"):
            options.experimental_optimization.apply_default_optimizations = True
    except Exception:
        pass

    files = _list_tfrecs(TEST_TFREC_DIR)
    ds = tf.data.Dataset.from_tensor_slices(files)
    ds = ds.interleave(
        lambda f: tf.data.TFRecordDataset(f, num_parallel_reads=AUTOTUNE),
        cycle_length=AUTOTUNE,
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    ds = ds.map(_parse_tfrecord_test, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.apply(tf.data.experimental.ignore_errors())

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    ds = ds.with_options(options)
    return ds


train_ds = make_train_val_ds_from_tfrecs(training=True)
val_ds = make_train_val_ds_from_tfrecs(training=False)

steps_per_epoch = int(np.ceil(len(tr_image_ids) / BATCH_SIZE))
validation_steps = int(np.ceil(len(va_image_ids) / BATCH_SIZE))

print("steps_per_epoch:", steps_per_epoch, "validation_steps:", validation_steps)



## === cell 4
base = tf.keras.applications.EfficientNetB4(
    include_top=False,
    weights="imagenet",
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    pooling="avg",
)
base.trainable = False  # minimal training time and stable behavior

inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
x = inputs
x = tf.keras.applications.efficientnet.preprocess_input(x)
x = base(x, training=False)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(CLASS_NUMS, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()



## === cell 5
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=2,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
)

base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
history_ft = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=1,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
)



## === cell 6
sample_sub = pd.read_csv(SAMPLE_SUB)
test_images = sample_sub["image_id"].tolist()  # canonical ordering

assert os.path.isdir(TEST_IMG_DIR), f"Missing test image directory: {TEST_IMG_DIR}"

test_ds_with_ids = make_test_ds_from_tfrecs()

test_ids_tf = tf.constant(test_images, dtype=tf.string)
test_pos_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        test_ids_tf, tf.range(tf.shape(test_ids_tf)[0], dtype=tf.int32)
    ),
    default_value=tf.constant(-1, dtype=tf.int32),
)

all_probs = []
all_pos = []
for imgs, ids in test_ds_with_ids:
    probs = model(imgs, training=False)  # [B,5]
    pos = test_pos_table.lookup(ids)  # [B]
    all_probs.append(probs)
    all_pos.append(pos)

probs_all = tf.concat(all_probs, axis=0)  # [N,5]
pos_all = tf.concat(all_pos, axis=0)  # [N]

valid = pos_all >= 0
pos_all_valid = tf.boolean_mask(pos_all, valid)
probs_all_valid = tf.boolean_mask(probs_all, valid)

n_test = len(test_images)
seen = int(pos_all_valid.shape[0])
assert seen == n_test, f"Failed to align all test ids: matched {seen}/{n_test}"

reordered_probs = tf.tensor_scatter_nd_update(
    tf.zeros([n_test, CLASS_NUMS], dtype=probs_all_valid.dtype),
    tf.expand_dims(pos_all_valid, axis=1),
    probs_all_valid,
)

predictions = (
    tf.argmax(reordered_probs, axis=1, output_type=tf.int32)
    .numpy()
    .astype(int)
    .tolist()
)

print("Num test images:", len(test_images))
print("Num predictions:", len(predictions))
assert len(test_images) == len(
    predictions
), "Prediction count must match test image count"



## === cell 7
predictions[:20], test_images[:5]



## === cell 8
sub = pd.DataFrame({"image_id": test_images, "label": predictions})
print(sub.head())
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print("Saved to:", os.path.abspath("submission.csv"))
