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

if os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "").lower() == "python":
    del os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]

import random
import numpy as np
import pandas as pd
import tensorflow as tf

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)  # keep as-is (can speed up graphs)
except Exception:
    pass

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
except Exception:
    pass

print("TF version:", tf.__version__)

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")
TRAIN_TFREC_DIR = os.path.join(DATA_ROOT, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(DATA_ROOT, "test_tfrecords")

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing {TEST_IMG_DIR}"
assert os.path.isdir(TRAIN_TFREC_DIR), f"Missing {TRAIN_TFREC_DIR}"
assert os.path.isdir(TEST_TFREC_DIR), f"Missing {TEST_TFREC_DIR}"

train_df = pd.read_csv(
    TRAIN_CSV,
    usecols=["image_id", "label"],
    dtype={"image_id": "string", "label": "int64"},
)
sample_sub = pd.read_csv(
    SAMPLE_SUB,
    usecols=["image_id", "label"],
    dtype={"image_id": "string", "label": "int64"},
)

print("train_df:", train_df.shape, train_df.columns.tolist())
print("sample_sub:", sample_sub.shape, sample_sub.columns.tolist())



## === cell 1
IMG_SIZE = 512
BATCH_SIZE = 8  # keep memory safe at 512x512
EPOCHS = 6  # keep same core training schedule

AUTOTUNE = tf.data.AUTOTUNE

tfdata_opts = tf.data.Options()
tfdata_opts.experimental_deterministic = True
try:
    tfdata_opts.experimental_optimization.apply_default_optimizations = True
    tfdata_opts.experimental_optimization.autotune_buffers = True
    tfdata_opts.experimental_optimization.autotune_cpu_budget = 0  # let TF pick
    tfdata_opts.experimental_optimization.autotune_ram_budget = 0  # let TF pick
    cpu = os.cpu_count() or 8
    tfdata_opts.threading.private_threadpool_size = min(32, max(8, cpu))
    tfdata_opts.threading.max_intra_op_parallelism = 0
except Exception:
    pass

TRAIN_TFRECS = sorted(
    [
        os.path.join(TRAIN_TFREC_DIR, f)
        for f in os.listdir(TRAIN_TFREC_DIR)
        if f.endswith(".tfrec")
    ]
)
TEST_TFRECS = sorted(
    [
        os.path.join(TEST_TFREC_DIR, f)
        for f in os.listdir(TEST_TFREC_DIR)
        if f.endswith(".tfrec")
    ]
)
assert len(TRAIN_TFRECS) > 0 and len(TEST_TFRECS) > 0, "TFRecord files not found."

_FEATURE_DESC_TRAIN = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}
_FEATURE_DESC_TEST = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


def _decode_and_resize(jpeg_bytes):
    try:
        return tf.image.decode_and_resize(jpeg_bytes, [IMG_SIZE, IMG_SIZE], channels=3)
    except Exception:
        img = tf.image.decode_jpeg(jpeg_bytes, channels=3)
        img = tf.image.resize(
            img,
            [IMG_SIZE, IMG_SIZE],
            method=tf.image.ResizeMethod.BILINEAR,
            antialias=False,
        )
        return img


@tf.function
def _train_parse_decode_aug(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURE_DESC_TRAIN)
    img = _decode_and_resize(ex["image"])
    img = tf.cast(img, tf.float32)  # keep [0,255]

    h = tf.strings.to_hash_bucket_fast(ex["image"], 2**31 - 1)
    seed = tf.stack([tf.cast(SEED, tf.int64), tf.cast(h, tf.int64)], axis=0)

    img = tf.image.stateless_random_flip_left_right(img, seed=seed)
    img = tf.image.stateless_random_brightness(
        img, max_delta=0.08, seed=seed + tf.constant([0, 1], tf.int64)
    )
    img = tf.clip_by_value(img, 0.0, 255.0)
    img = tf.ensure_shape(img, [IMG_SIZE, IMG_SIZE, 3])
    return img, tf.cast(ex["target"], tf.int32)


@tf.function
def _val_parse_decode(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURE_DESC_TRAIN)
    img = _decode_and_resize(ex["image"])
    img = tf.cast(img, tf.float32)  # keep [0,255]
    img = tf.ensure_shape(img, [IMG_SIZE, IMG_SIZE, 3])
    return img, tf.cast(ex["target"], tf.int32)


@tf.function
def _test_parse_decode(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURE_DESC_TEST)
    img = _decode_and_resize(ex["image"])
    img = tf.cast(img, tf.float32)  # keep [0,255]
    img = tf.ensure_shape(img, [IMG_SIZE, IMG_SIZE, 3])
    return img, ex["image_name"]


def make_ds_from_tfrecords(tfrecs, training, cache_path=None):
    ds = tf.data.TFRecordDataset(tfrecs, num_parallel_reads=AUTOTUNE)
    ds = ds.with_options(tfdata_opts)
    ds = ds.apply(tf.data.experimental.ignore_errors())

    if training:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False
        ds = ds.with_options(ignore_order)
        ds = ds.map(
            _train_parse_decode_aug, num_parallel_calls=AUTOTUNE, deterministic=False
        )
        ds = ds.batch(BATCH_SIZE, drop_remainder=True)
        ds = ds.repeat()
    else:
        ds = ds.map(_val_parse_decode, num_parallel_calls=AUTOTUNE, deterministic=True)
        if cache_path is not None:
            ds = ds.cache(cache_path)
        ds = ds.batch(BATCH_SIZE, drop_remainder=False)

    ds = ds.prefetch(AUTOTUNE)
    return ds


rng = np.random.RandomState(SEED)
perm = rng.permutation(len(TRAIN_TFRECS))
TRAIN_TFRECS_SHUF = [TRAIN_TFRECS[i] for i in perm]
n_val_files = max(1, int(round(0.1 * len(TRAIN_TFRECS_SHUF))))
VAL_TFRECS = TRAIN_TFRECS_SHUF[:n_val_files]
TRN_TFRECS = TRAIN_TFRECS_SHUF[n_val_files:]

val_cache = os.path.join("/kaggle/working", f"val_cache_{IMG_SIZE}.cache")
train_ds = make_ds_from_tfrecords(TRN_TFRECS, training=True, cache_path=None)
val_ds = make_ds_from_tfrecords(VAL_TFRECS, training=False, cache_path=val_cache)

total_examples = int(train_df.shape[0])  # exact dataset size

train_frac = len(TRN_TFRECS) / len(TRAIN_TFRECS)
n_train_examples = int(round(total_examples * train_frac))
n_val_examples = total_examples - n_train_examples

steps_per_epoch = n_train_examples // BATCH_SIZE  # drop_remainder=True so use floor
validation_steps = int(np.ceil(n_val_examples / BATCH_SIZE))

print(
    "Example counts (from train.csv + TFRecord file split) - train examples:",
    n_train_examples,
    "val examples:",
    n_val_examples,
)
print(
    "Computed steps_per_epoch:", steps_per_epoch, "validation_steps:", validation_steps
)



## === cell 2
from tensorflow.keras import layers, models

base = tf.keras.applications.EfficientNetB4(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)
base.trainable = False  # unchanged

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = inputs
x = tf.keras.applications.efficientnet.preprocess_input(x)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(5, activation="softmax")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_accuracy", factor=0.5, patience=1, verbose=1
    ),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=2,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
)

base.trainable = True
for layer in base.layers[:-40]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history_ft = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=2,
    callbacks=callbacks,
    verbose=2,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
)




## === cell 3
def make_test_ds_from_tfrecords(tfrecs, cache_path=None):
    ds = tf.data.TFRecordDataset(tfrecs, num_parallel_reads=AUTOTUNE)
    ds = ds.with_options(tfdata_opts)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = ds.map(_test_parse_decode, num_parallel_calls=AUTOTUNE, deterministic=True)
    if cache_path is not None:
        ds = ds.cache(cache_path)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


test_cache = os.path.join("/kaggle/working", f"test_cache_{IMG_SIZE}.cache")
test_ds = make_test_ds_from_tfrecords(TEST_TFRECS, cache_path=test_cache)

names = []
pred_chunks = []
for xb, name_batch in test_ds:
    pb = model(xb, training=False).numpy()
    pred_chunks.append(pb)
    names.extend(name_batch.numpy().astype(str).tolist())

probs = np.concatenate(pred_chunks, axis=0)
pred_labels = np.argmax(probs, axis=1).astype(int)

assert len(names) == len(pred_labels), "Mismatch between test names and predictions."

pred_df = pd.DataFrame({"image_id": names, "label": pred_labels})
sub = sample_sub[["image_id"]].merge(pred_df, on="image_id", how="left")
assert sub["label"].isna().sum() == 0, "Some test images missing predictions."
sub["label"] = sub["label"].astype(int)

sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)



## === cell 4
predictions = sub["label"].tolist()
predictions[:10]
