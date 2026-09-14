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
import random
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)



## === cell 1
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

tf.random.set_seed(SEED)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

tf.config.run_functions_eagerly(False)
keras.backend.clear_session()

print("TF version:", tf.__version__)



## === cell 2
DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
TRAIN_TFREC_DIR = os.path.join(DATA_DIR, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(DATA_DIR, "test_tfrecords")

train_df = pd.read_csv(TRAIN_CSV)
sample_df = pd.read_csv(SAMPLE_SUB)

assert set(["image_id", "label"]).issubset(train_df.columns)
assert set(["image_id", "label"]).issubset(sample_df.columns)

num_classes = int(train_df["label"].nunique())
print("Train rows:", len(train_df), "Num classes:", num_classes)
print("Sample submission rows:", len(sample_df))



## === cell 3
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 3  # keep runtime within limits

val_frac = 0.1

_FEATURE_DESC_TRAIN = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}
_FEATURE_DESC_TEST = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


@tf.function
def _decode_resize_uint8(img_bytes):
    img = tf.image.decode_jpeg(img_bytes, channels=3)  # uint8 [0..255]
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method="bilinear")
    img = tf.cast(img, tf.float32)  # float32 in [0..255]
    return img


@tf.function
def _parse_train_aug(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURE_DESC_TRAIN)
    x = _decode_resize_uint8(ex["image"])
    y = tf.cast(ex["target"], tf.int32)
    x = tf.image.random_flip_left_right(x, seed=SEED)
    x = tf.image.random_flip_up_down(x, seed=SEED)
    return x, y


@tf.function
def _parse_train_noaug(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURE_DESC_TRAIN)
    x = _decode_resize_uint8(ex["image"])
    y = tf.cast(ex["target"], tf.int32)
    return x, y


@tf.function
def _parse_test_only_x(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURE_DESC_TEST)
    x = _decode_resize_uint8(ex["image"])
    return x


_DATA_OPTS = tf.data.Options()
_DATA_OPTS.experimental_deterministic = True
_DATA_OPTS.autotune.enabled = True
_DATA_OPTS.experimental_slack = True
_DATA_OPTS.experimental_optimization.apply_default_optimizations = True
_DATA_OPTS.experimental_optimization.map_fusion = True
_DATA_OPTS.experimental_optimization.map_parallelization = True
_DATA_OPTS.experimental_optimization.parallel_batch = True


def _list_tfrecords(directory):
    files = tf.io.gfile.glob(os.path.join(directory, "*.tfrec"))
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No TFRecord files found in {directory}")
    return files


TRAIN_TFRECS_ALL = _list_tfrecords(TRAIN_TFREC_DIR)
_n = len(TRAIN_TFRECS_ALL)
_n_val = max(1, int(round(_n * val_frac)))
VAL_TFRECS = TRAIN_TFRECS_ALL[:_n_val]
TRN_TFRECS = TRAIN_TFRECS_ALL[_n_val:]
print(
    f"TFRec files: total={_n} train_files={len(TRN_TFRECS)} val_files={len(VAL_TFRECS)}"
)

_HAS_GPU = bool(tf.config.list_physical_devices("GPU"))


def _final_prefetch(ds):
    if _HAS_GPU:
        try:
            return ds.apply(
                tf.data.experimental.prefetch_to_device(
                    "/GPU:0", buffer_size=tf.data.AUTOTUNE
                )
            )
        except Exception:
            return ds.prefetch(tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


_TRAIN_CACHE_PATH = "/kaggle/working/train_decoded_cache"
_VAL_CACHE_PATH = "/kaggle/working/val_decoded_cache"
_TEST_CACHE_PATH = "/kaggle/working/test_decoded_cache"


def make_train_ds():
    ds = tf.data.TFRecordDataset(
        TRN_TFRECS,
        num_parallel_reads=tf.data.AUTOTUNE,
        compression_type=None,
    ).with_options(_DATA_OPTS)

    ds = ds.apply(tf.data.experimental.ignore_errors())

    ds = ds.map(
        _parse_train_aug,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )

    ds = ds.cache(_TRAIN_CACHE_PATH)
    ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.batch(BATCH_SIZE, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)
    ds = _final_prefetch(ds)
    return ds


def make_val_ds():
    ds = tf.data.TFRecordDataset(
        VAL_TFRECS,
        num_parallel_reads=tf.data.AUTOTUNE,
        compression_type=None,
    ).with_options(_DATA_OPTS)

    ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = ds.map(
        _parse_train_noaug,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )

    ds = ds.cache(_VAL_CACHE_PATH)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False, num_parallel_calls=tf.data.AUTOTUNE)
    ds = _final_prefetch(ds)
    return ds


train_ds = make_train_ds()
val_ds = make_val_ds()

base = keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)
base.trainable = False  # fast, stable

inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = inputs
x = layers.Lambda(lambda t: keras.applications.efficientnet.preprocess_input(t))(x)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2, seed=SEED)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    run_eagerly=False,
)

steps_per_epoch = int(np.ceil((len(train_df) * (1.0 - val_frac)) / BATCH_SIZE))
validation_steps = int(np.ceil((len(train_df) * val_frac) / BATCH_SIZE))

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
)



## === cell 4
base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    run_eagerly=False,
)

history_ft = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=1,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
)



## === cell 5
test_image_ids = sample_df["image_id"].astype(str).tolist()

test_tfrec_files = sorted(tf.io.gfile.glob(os.path.join(TEST_TFREC_DIR, "*.tfrec")))
if not test_tfrec_files:
    raise FileNotFoundError(f"No TFRecord files found in {TEST_TFREC_DIR}")

test_ds = tf.data.TFRecordDataset(
    test_tfrec_files,
    num_parallel_reads=tf.data.AUTOTUNE,
    compression_type=None,
).with_options(_DATA_OPTS)

test_ds = test_ds.apply(tf.data.experimental.ignore_errors())
test_ds = test_ds.map(
    _parse_test_only_x, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
)

test_ds = test_ds.cache(_TEST_CACHE_PATH)
test_ds = test_ds.batch(
    BATCH_SIZE, drop_remainder=False, num_parallel_calls=tf.data.AUTOTUNE
)
test_ds = _final_prefetch(test_ds)

probs = model.predict(test_ds, verbose=1)
y_preds = np.argmax(probs, axis=1).astype(int)

print("Preds:", y_preds.shape, "Expected:", len(test_image_ids))

if len(y_preds) != len(test_image_ids):
    raise RuntimeError(
        f"Prediction count {len(y_preds)} != sample_submission {len(test_image_ids)}"
    )



## === cell 6
df_sub = pd.DataFrame({"image_id": test_image_ids, "label": y_preds})
df_sub.head()



## === cell 7
assert len(df_sub) == len(sample_df), "Submission length must match sample_submission"
assert list(df_sub.columns) == ["image_id", "label"]
assert df_sub["label"].between(0, num_classes - 1).all()



## === cell 8
df_sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df_sub.shape)
print(df_sub.head())
