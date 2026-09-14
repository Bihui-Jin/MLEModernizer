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
import math
import numpy as np
import pandas as pd

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
try:
    import google.protobuf  # noqa: F401
except Exception:
    pass

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

TRAIN_IMG_LOC = "../input/cassava-leaf-disease-classification/train_images"
TEST_IMG_LOC = "../input/cassava-leaf-disease-classification/test_images"
TRAIN_CSV = "../input/cassava-leaf-disease-classification/train.csv"
SAMPLE_CSV = "../input/cassava-leaf-disease-classification/sample_submission.csv"

TRAIN_TFREC_GLOB = (
    "../input/cassava-leaf-disease-classification/train_tfrecords/*.tfrec"
)
TEST_TFREC_GLOB = "../input/cassava-leaf-disease-classification/test_tfrecords/*.tfrec"

print("TensorFlow:", tf.__version__)
print("ALL Modules are successfully loaded")



## === cell 1
train_df = pd.read_csv(TRAIN_CSV)
ss = pd.read_csv(SAMPLE_CSV)

assert {"image_id", "label"}.issubset(train_df.columns)
assert {"image_id", "label"}.issubset(ss.columns)

train_df["filepath"] = (TRAIN_IMG_LOC.rstrip("/") + "/" + train_df["image_id"]).astype(
    str
)

missing = [p for p in train_df["filepath"].head(10).tolist() if not os.path.exists(p)]
if missing:
    raise FileNotFoundError(f"Some train image files not found (sample): {missing[:3]}")

NUM_CLASSES = int(train_df["label"].nunique())
print("Train rows:", len(train_df), " Num classes:", NUM_CLASSES)
print("Test rows:", len(ss))



## === cell 2
IMG_SIZE = (300, 300)
BATCH_SIZE = 32
VAL_SPLIT = 0.1
AUTOTUNE = tf.data.AUTOTUNE
SHUFFLE_BUFFER = 8192

DATA_OPTIONS = tf.data.Options()
DATA_OPTIONS.experimental_deterministic = True
try:
    DATA_OPTIONS.autotune.enabled = True
except Exception:
    pass


def _decode_jpeg_bytes(img_bytes):
    img = tf.image.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    return img


def decode_image(path):
    img = tf.io.read_file(path)
    return _decode_jpeg_bytes(img)


_TFREC_FEATURES_TRAIN_OPTIONAL_LABEL = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}
_TFREC_FEATURES_TEST = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


def _parse_tfrecord_train(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TFREC_FEATURES_TRAIN_OPTIONAL_LABEL)
    img = _decode_jpeg_bytes(ex["image"])
    label = tf.cast(ex["label"], tf.int32)
    image_name = ex["image_name"]
    return img, label, image_name


def _parse_tfrecord_test(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TFREC_FEATURES_TEST)
    img = _decode_jpeg_bytes(ex["image"])
    image_name = ex["image_name"]
    return img, image_name


def _list_tfrecord_files(glob_pattern):
    files = tf.io.gfile.glob(glob_pattern)
    if not files:
        raise FileNotFoundError(f"No TFRecord files found for pattern: {glob_pattern}")
    return sorted(files)  # deterministic shard order


def make_ds_from_paths(paths, labels=None, training=False):
    paths = tf.convert_to_tensor(paths, dtype=tf.string)
    if labels is None:
        ds = tf.data.Dataset.from_tensor_slices(paths)
        ds = ds.map(decode_image, num_parallel_calls=AUTOTUNE)
    else:
        labels = tf.convert_to_tensor(labels, dtype=tf.int32)
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.map(lambda p, y: (decode_image(p), y), num_parallel_calls=AUTOTUNE)

    ds = ds.with_options(DATA_OPTIONS)

    if not training:
        ds = ds.cache()

    if training:
        n = (
            int(paths.shape[0])
            if paths.shape.rank == 1 and paths.shape[0] is not None
            else 0
        )
        buf = min(n, SHUFFLE_BUFFER) if n > 0 else SHUFFLE_BUFFER
        ds = ds.shuffle(buffer_size=buf, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


def make_train_val_from_tfrecords_hashsplit(val_split=VAL_SPLIT):
    files = _list_tfrecord_files(TRAIN_TFREC_GLOB)
    ds_all = tf.data.TFRecordDataset(files, num_parallel_reads=AUTOTUNE).with_options(
        DATA_OPTIONS
    )
    ds_all = ds_all.map(_parse_tfrecord_train, num_parallel_calls=AUTOTUNE)

    ds_all = ds_all.filter(lambda img, lab, name: tf.greater_equal(lab, 0))

    def _basename(x):
        x = tf.strings.regex_replace(x, r"\\", "/")
        parts = tf.strings.split(x, sep="/")
        return parts[-1]

    def _is_val(img, lab, name):
        key = _basename(name)
        bucket = tf.strings.to_hash_bucket_fast(key, 1000003)
        threshold = tf.cast(tf.floor(val_split * 1000003.0), tf.int64)
        return tf.less(bucket, threshold)

    train_ds = ds_all.filter(
        lambda img, lab, name: tf.logical_not(_is_val(img, lab, name))
    )
    val_ds = ds_all.filter(_is_val)

    train_ds = train_ds.map(
        lambda img, lab, name: (img, lab), num_parallel_calls=AUTOTUNE
    )
    val_ds = val_ds.map(lambda img, lab, name: (img, lab), num_parallel_calls=AUTOTUNE)

    val_ds = val_ds.cache()
    train_ds = train_ds.shuffle(
        SHUFFLE_BUFFER, seed=SEED, reshuffle_each_iteration=True
    )

    train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    return train_ds, val_ds


idx = np.arange(len(train_df))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
val_size = int(round(len(idx) * VAL_SPLIT))
val_idx = idx[:val_size]
trn_idx = idx[val_size:]

expected_train_batches = int(math.ceil(len(trn_idx) / BATCH_SIZE))
expected_val_batches = int(math.ceil(len(val_idx) / BATCH_SIZE))

use_tfrecord_train = True
try:
    train_ds, val_ds = make_train_val_from_tfrecords_hashsplit(VAL_SPLIT)

    train_card = int(tf.data.experimental.cardinality(train_ds).numpy())
    val_card = int(tf.data.experimental.cardinality(val_ds).numpy())

    if train_card <= 0 or val_card <= 0:
        raise RuntimeError(
            f"TFRecord hash split has invalid cardinality: train={train_card}, val={val_card}"
        )

    train_ds = train_ds.apply(tf.data.experimental.assert_cardinality(train_card))
    val_ds = val_ds.apply(tf.data.experimental.assert_cardinality(val_card))

except Exception as e:
    print(
        "TFRecord train/val build failed; falling back to image files. Error:", repr(e)
    )
    use_tfrecord_train = False

    trn_df = train_df.iloc[trn_idx].reset_index(drop=True)
    val_df = train_df.iloc[val_idx].reset_index(drop=True)

    train_ds = make_ds_from_paths(
        trn_df["filepath"].values, labels=trn_df["label"].values, training=True
    )
    val_ds = make_ds_from_paths(
        val_df["filepath"].values, labels=val_df["label"].values, training=False
    )

    train_ds = train_ds.apply(
        tf.data.experimental.assert_cardinality(expected_train_batches)
    )
    val_ds = val_ds.apply(tf.data.experimental.assert_cardinality(expected_val_batches))

print(
    "Built datasets.",
    "use_tfrecord_train:",
    use_tfrecord_train,
    "train batches:",
    int(tf.data.experimental.cardinality(train_ds).numpy()),
    "val batches:",
    int(tf.data.experimental.cardinality(val_ds).numpy()),
)



## === cell 3
base = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=IMG_SIZE + (3,),
)
base.trainable = False  # initial training head only (fast, stable)

inputs = keras.Input(shape=IMG_SIZE + (3,), name="image")
x = tf.keras.applications.efficientnet.preprocess_input(inputs)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="pred")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

model.summary()



## === cell 4
EPOCHS_HEAD = 2
history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_HEAD,
    verbose=1,
)

base.trainable = True
fine_tune_at = max(0, len(base.layers) - 20)
for layer in base.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

EPOCHS_FT = 1
history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_HEAD + EPOCHS_FT,
    initial_epoch=(history1.epoch[-1] + 1) if history1.epoch else 0,
    verbose=1,
)

print("Training complete.")




## === cell 5
def make_test_ds_from_tfrecords():
    files = _list_tfrecord_files(TEST_TFREC_GLOB)
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTOTUNE).with_options(
        DATA_OPTIONS
    )
    ds = ds.map(_parse_tfrecord_test, num_parallel_calls=AUTOTUNE)
    ds = ds.cache()
    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


test_paths = (TEST_IMG_LOC.rstrip("/") + "/" + ss["image_id"]).astype(str).tolist()
missing_test = [p for p in test_paths[:20] if not os.path.exists(p)]
if missing_test:
    raise FileNotFoundError(
        f"Some test image files not found (sample): {missing_test[:3]}"
    )

use_tfrecord_test = True
try:
    test_ds_named = make_test_ds_from_tfrecords()
except Exception as e:
    print("TFRecord test build failed; falling back to image files. Error:", repr(e))
    use_tfrecord_test = False
    test_ds = make_ds_from_paths(np.array(test_paths), labels=None, training=False)

if use_tfrecord_test:

    def _basename(x):
        x = tf.strings.regex_replace(x, r"\\", "/")
        parts = tf.strings.split(x, sep="/")
        return parts[-1]

    test_names = []
    test_imgs_only = test_ds_named.map(
        lambda img, name: img, num_parallel_calls=AUTOTUNE
    )

    for _, batch_names in test_ds_named:
        batch_names = tf.map_fn(_basename, batch_names, fn_output_signature=tf.string)
        test_names.extend([n.decode("utf-8") for n in batch_names.numpy().tolist()])

    probs = model.predict(test_imgs_only, verbose=1)
    batch_preds = np.argmax(probs, axis=1).astype(int)

    pred_map = dict(zip(test_names, batch_preds.tolist()))
    preds = np.array(
        [pred_map[i] for i in ss["image_id"].astype(str).tolist()], dtype=int
    )
else:
    probs = model.predict(test_ds, verbose=1)
    preds = np.argmax(probs, axis=1).astype(int)

assert len(preds) == len(ss), (len(preds), len(ss))
assert preds.min() >= 0 and preds.max() < NUM_CLASSES

my_submission = pd.DataFrame({"image_id": ss["image_id"].values, "label": preds})
my_submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", my_submission.shape)



## === cell 6
my_submission.head()
