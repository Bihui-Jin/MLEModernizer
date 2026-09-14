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

import json
import numpy as np
import pandas as pd
import tensorflow as tf

SEED = 42
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("TF version:", tf.__version__)

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
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
assert os.path.exists(TRAIN_TFREC_DIR), f"Missing {TRAIN_TFREC_DIR}"
assert os.path.exists(TEST_TFREC_DIR), f"Missing {TEST_TFREC_DIR}"

train_df = pd.read_csv(TRAIN_CSV)
sample_df = pd.read_csv(SAMPLE_SUB)

print("train_df:", train_df.shape, train_df.columns.tolist())
print("sample_df:", sample_df.shape, sample_df.columns.tolist())

NUM_CLASSES = int(train_df["label"].nunique())
print("NUM_CLASSES:", NUM_CLASSES)



## === cell 1
IMG_SIZE = 380
BATCH_SIZE = 16

preprocess = tf.keras.applications.efficientnet.preprocess_input
_AUTO = tf.data.AUTOTUNE

_DS_OPTIONS = tf.data.Options()
_DS_OPTIONS.experimental_deterministic = True
try:
    _DS_OPTIONS.experimental_optimization.map_parallelization = True
except Exception:
    pass
try:
    _DS_OPTIONS.experimental_optimization.map_and_batch_fusion = True
except Exception:
    pass
try:
    _DS_OPTIONS.experimental_optimization.apply_default_optimizations = True
except Exception:
    pass

_TRAIN_FEATURE_DESC = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}
_TEST_FEATURE_DESC = {
    "image": tf.io.FixedLenFeature([], tf.string),
}

_VAL_MOD = 10  # matches test_size=0.1
_VAL_BUCKET = 0  # fixed bucket for validation


@tf.function
def _decode_and_preprocess_jpeg_bytes(img_bytes):
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method="bilinear", antialias=True)
    img = tf.cast(img, tf.float32)
    img = preprocess(img)
    return img


@tf.function
def _parse_train_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TRAIN_FEATURE_DESC)
    x = _decode_and_preprocess_jpeg_bytes(ex["image"])
    y = tf.cast(ex["target"], tf.int32)
    return x, y


@tf.function
def _parse_test_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TEST_FEATURE_DESC)
    x = _decode_and_preprocess_jpeg_bytes(ex["image"])
    return x


def _list_tfrecs(tfrecs_dir):
    files = tf.io.gfile.glob(os.path.join(tfrecs_dir, "*.tfrec"))
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No .tfrec files found under: {tfrecs_dir}")
    return files


def _stable_bucket_for_file(path, mod):
    base = os.path.basename(path).encode("utf-8")
    h = 2166136261
    for b in base:
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h % mod)


def _split_tfrecs_by_bucket(tfrecs_files, val_mod=_VAL_MOD, val_bucket=_VAL_BUCKET):
    train_files, val_files = [], []
    for f in tfrecs_files:
        b = _stable_bucket_for_file(f, val_mod)
        (val_files if b == val_bucket else train_files).append(f)
    if len(val_files) == 0:
        val_files = [tfrecs_files[0]]
        train_files = [f for f in tfrecs_files if f != val_files[0]]
    return train_files, val_files


def _make_tfrecord_ds(
    tfrecs_files, parse_fn, batch_size, shuffle=False, cache=False, drop_remainder=False
):
    ds = tf.data.TFRecordDataset(
        tfrecs_files,
        num_parallel_reads=_AUTO,
        compression_type=None,
    )
    if shuffle:
        ds = ds.shuffle(8192, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(parse_fn, num_parallel_calls=_AUTO, deterministic=True)

    if cache:
        ds = ds.cache()

    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.prefetch(_AUTO)
    ds = ds.with_options(_DS_OPTIONS)
    return ds


train_tfrecs = _list_tfrecs(TRAIN_TFREC_DIR)
test_tfrecs = _list_tfrecs(TEST_TFREC_DIR)

train_tfrecs_files, val_tfrecs_files = _split_tfrecs_by_bucket(train_tfrecs)

train_ds = _make_tfrecord_ds(
    train_tfrecs_files,
    parse_fn=_parse_train_example,
    batch_size=BATCH_SIZE,
    shuffle=True,
    cache=False,
    drop_remainder=True,
)
val_ds = _make_tfrecord_ds(
    val_tfrecs_files,
    parse_fn=_parse_train_example,
    batch_size=BATCH_SIZE,
    shuffle=False,
    cache=False,
    drop_remainder=False,
)

n_total = int(train_df.shape[0])
ratio_train = len(train_tfrecs_files) / max(
    1, (len(train_tfrecs_files) + len(val_tfrecs_files))
)
n_train_est = int(round(n_total * ratio_train))
n_val_est = n_total - n_train_est

steps_per_epoch = int(np.ceil(n_train_est / BATCH_SIZE))
validation_steps = int(np.ceil(n_val_est / BATCH_SIZE))

print("Estimated split -> n_train:", n_train_est, "n_val:", n_val_est)
print("steps_per_epoch:", steps_per_epoch, "validation_steps:", validation_steps)

base = tf.keras.applications.EfficientNetB4(
    include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    jit_compile=True,
)

model.summary()



## === cell 2
EPOCHS_HEAD = 2
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_HEAD,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
)

base.trainable = True
fine_tune_at = int(len(base.layers) * 0.85)
for layer in base.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    jit_compile=True,
)

EPOCHS_FT = 1
history_ft = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FT,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
)



## === cell 3
test_image_ids = sample_df["image_id"].tolist()

test_ds = _make_tfrecord_ds(
    test_tfrecs,
    parse_fn=_parse_test_example,
    batch_size=BATCH_SIZE,
    shuffle=False,
    cache=False,
    drop_remainder=False,
)

test_steps = int(np.ceil(sample_df.shape[0] / BATCH_SIZE))

probs = model.predict(test_ds, verbose=1, steps=test_steps)
pred_labels = probs.argmax(axis=1).astype(int)

pred_labels = pred_labels[: sample_df.shape[0]]
assert len(pred_labels) == sample_df.shape[0], (len(pred_labels), sample_df.shape[0])

sub = pd.DataFrame({"image_id": test_image_ids, "label": pred_labels})
assert sub.shape[0] == sample_df.shape[0]
assert list(sub.columns) == ["image_id", "label"]
assert sub["label"].between(0, NUM_CLASSES - 1).all()

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())



## === cell 4
predictions = pred_labels.tolist()
predictions[:20]
