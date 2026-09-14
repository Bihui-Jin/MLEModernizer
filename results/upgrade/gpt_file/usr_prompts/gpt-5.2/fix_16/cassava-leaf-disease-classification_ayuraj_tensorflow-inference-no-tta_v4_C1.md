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
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_PROTOBUF_USE_C_PROTOBUF"] = "0"

import math
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.models import *

print("TF version:", tf.__version__)



## === cell 1
DATA_ROOT_CANDIDATES = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "../input/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
    "../data/cassava-leaf-disease-classification",
]
DATA_ROOT = None
for p in DATA_ROOT_CANDIDATES:
    if os.path.exists(p):
        DATA_ROOT = p
        break

if DATA_ROOT is None:
    raise FileNotFoundError(
        f"Could not find cassava dataset folder in: {DATA_ROOT_CANDIDATES}"
    )

TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_DIR = os.path.join(DATA_ROOT, "test_images")
TRAIN_TFREC_DIR = os.path.join(DATA_ROOT, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(DATA_ROOT, "test_tfrecords")

print("DATA_ROOT:", DATA_ROOT)
print("Train CSV exists:", os.path.exists(TRAIN_CSV))
print("Sample submission exists:", os.path.exists(SAMPLE_SUB))
print("Train images dir exists:", os.path.exists(TRAIN_DIR))
print("Test images dir exists:", os.path.exists(TEST_DIR))
print("Train TFRecords dir exists:", os.path.exists(TRAIN_TFREC_DIR))
print("Test TFRecords dir exists:", os.path.exists(TEST_TFREC_DIR))



## === cell 2
IMAGE_SIZE = 380
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE
CLASS_NUMS = 5
SEED = 42

tf.keras.utils.set_random_seed(SEED)

tf.config.optimizer.set_jit(True)

train_df = pd.read_csv(TRAIN_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB)

sample_sub["image_id"] = sample_sub["image_id"].astype(str)
test_images = sample_sub["image_id"].tolist()
image_paths_test = [os.path.join(TEST_DIR, img) for img in test_images]

assert len(test_images) == len(sample_sub), "Sample submission rows mismatch."
missing = [p for p in image_paths_test[:50] if not os.path.exists(p)]
print("Example missing test paths (should be empty):", missing)

train_df["image_id"] = train_df["image_id"].astype(str)
train_df["path"] = (TRAIN_DIR + "/" + train_df["image_id"]).astype(str)

_check_n = min(200, len(train_df))
_check_missing = sum(
    0 if os.path.exists(p) else 1 for p in train_df["path"].iloc[:_check_n].tolist()
)
if _check_missing > 0:
    raise FileNotFoundError(
        f"Found {_check_missing} missing train images in first {_check_n} paths."
    )

print("Train rows:", len(train_df), "Test rows:", len(sample_sub))




## === cell 3
def _parse_tfrecord_labeled(example_proto):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.int64),
    }
    ex = tf.io.parse_single_example(example_proto, feature_description)
    img = tf.image.decode_jpeg(ex["image"], channels=3)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    return img, tf.cast(ex["target"], tf.int32)


def _parse_tfrecord_test(example_proto):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.VarLenFeature(tf.string),
        "image_id": tf.io.VarLenFeature(tf.string),
    }
    ex = tf.io.parse_single_example(example_proto, feature_description)

    img = tf.image.decode_jpeg(ex["image"], channels=3)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

    name_dense = tf.sparse.to_dense(ex["image_name"], default_value=b"")
    id_dense = tf.sparse.to_dense(ex["image_id"], default_value=b"")

    has_name = tf.greater(tf.size(name_dense), 0) & tf.not_equal(name_dense[0], b"")
    chosen = tf.cond(has_name, lambda: name_dense[0], lambda: id_dense[0])

    return img, chosen


def augment(img, label):
    img = tf.image.random_flip_left_right(img)
    return img, label


train_tfrecord_files = tf.io.gfile.glob(os.path.join(TRAIN_TFREC_DIR, "*.tfrec"))
test_tfrecord_files = tf.io.gfile.glob(os.path.join(TEST_TFREC_DIR, "*.tfrec"))
if not train_tfrecord_files:
    raise FileNotFoundError(f"No train tfrecords found in {TRAIN_TFREC_DIR}")
if not test_tfrecord_files:
    raise FileNotFoundError(f"No test tfrecords found in {TEST_TFREC_DIR}")
train_tfrecord_files = sorted(train_tfrecord_files)
test_tfrecord_files = sorted(test_tfrecord_files)

data_opts = tf.data.Options()
data_opts.experimental_deterministic = True
data_opts.threading.private_threadpool_size = 0
data_opts.threading.max_intra_op_parallelism = 0


def _tfrecord_dataset_with_gzip_fallback(filename):
    try:
        return tf.data.TFRecordDataset(
            filename, compression_type="GZIP", num_parallel_reads=AUTOTUNE
        )
    except (tf.errors.InvalidArgumentError, tf.errors.DataLossError, ValueError):
        return tf.data.TFRecordDataset(filename, num_parallel_reads=AUTOTUNE)


def _make_raw_ds(tfrecord_files):
    files_ds = tf.data.Dataset.from_tensor_slices(tfrecord_files)
    ds = files_ds.interleave(
        lambda x: _tfrecord_dataset_with_gzip_fallback(x),
        cycle_length=min(16, len(tfrecord_files)),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
        block_length=16,
    )
    ds = ds.apply(tf.data.experimental.ignore_errors())
    return ds


SHARD_SIZE = 1338
valid_size = int(0.1 * len(train_df))  # identical definition
valid_shards = int(np.ceil(valid_size / SHARD_SIZE))
valid_shards = max(1, min(valid_shards, len(train_tfrecord_files) - 1))

valid_tfrecord_files = train_tfrecord_files[:valid_shards]
train_tfrecord_files_split = train_tfrecord_files[valid_shards:]

expected_valid = valid_shards * SHARD_SIZE
expected_train = (len(train_tfrecord_files) - valid_shards) * SHARD_SIZE
print(
    "Using deterministic TFRecord shard-based split. Shards(valid/train):",
    valid_shards,
    len(train_tfrecord_files) - valid_shards,
    "Approx counts:",
    expected_train,
    expected_valid,
)

train_raw = _make_raw_ds(train_tfrecord_files_split).with_options(data_opts)
valid_raw = _make_raw_ds(valid_tfrecord_files).with_options(data_opts)
test_raw = _make_raw_ds(test_tfrecord_files).with_options(data_opts)

train_ds = train_raw.map(
    _parse_tfrecord_labeled, num_parallel_calls=AUTOTUNE, deterministic=True
)
train_ds = train_ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
train_ds = train_ds.map(augment, num_parallel_calls=AUTOTUNE, deterministic=True)
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

valid_ds = valid_raw.map(
    _parse_tfrecord_labeled, num_parallel_calls=AUTOTUNE, deterministic=True
)
valid_ds = valid_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

test_named_ds = test_raw.map(
    _parse_tfrecord_test, num_parallel_calls=AUTOTUNE, deterministic=True
)
test_ds = test_named_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

steps_per_epoch = max(1, int(math.ceil(expected_train / BATCH_SIZE)))
validation_steps = max(1, int(math.ceil(expected_valid / BATCH_SIZE)))
print("steps_per_epoch:", steps_per_epoch, "validation_steps:", validation_steps)



## === cell 4
base = tf.keras.applications.EfficientNetB4(
    include_top=False,
    weights="imagenet",
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
)
x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
x = tf.keras.layers.Dropout(0.3)(x)
out = tf.keras.layers.Dense(CLASS_NUMS, activation="softmax")(x)
model = tf.keras.Model(inputs=base.input, outputs=out)

base.trainable = False
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history1 = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=2,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
)

base.trainable = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
history2 = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=1,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
)

model.summary()




## === cell 5
def _decode_img_from_path(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    return img


all_names = []
all_preds = []

for batch_imgs, batch_names in test_ds:
    probs = model(batch_imgs, training=False).numpy()
    preds = np.argmax(probs, axis=1).astype(np.int32)
    all_preds.append(preds)

    bn = batch_names.numpy().tolist()
    bn = [
        x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x) for x in bn
    ]
    all_names.extend(bn)

if len(all_preds) == 0 or len(all_names) == 0:
    print(
        "Warning: TFRecord-based test dataset produced no batches. Falling back to test_images/ decoding."
    )
    test_paths_ds = tf.data.Dataset.from_tensor_slices(image_paths_test)
    test_img_ds = test_paths_ds.map(_decode_img_from_path, num_parallel_calls=AUTOTUNE)
    test_img_ds = test_img_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

    all_preds = []
    for batch_imgs in test_img_ds:
        probs = model(batch_imgs, training=False).numpy()
        preds = np.argmax(probs, axis=1).astype(np.int32)
        all_preds.append(preds)

    pred_labels = np.concatenate(all_preds, axis=0)[: len(test_images)]
    sub = pd.DataFrame({"image_id": test_images, "label": pred_labels.astype(int)})
    sub.to_csv("submission.csv", index=False)
    print(sub.head())
    print("Wrote submission.csv with rows:", len(sub))
    print("submission.csv exists:", os.path.exists("submission.csv"))
else:
    pred_labels = np.concatenate(all_preds, axis=0)
    pred_names = np.array(all_names, dtype=object)

    m = min(len(pred_names), len(pred_labels))
    pred_names = pred_names[:m]
    pred_labels = pred_labels[:m]

    print("Raw predicted pairs:", len(pred_names), len(pred_labels))
    print("Example names:", pred_names[:5].tolist())
    print("Example labels:", pred_labels[:10].tolist())

    name_to_pred = {
        n: int(p) for n, p in zip(pred_names.tolist(), pred_labels.tolist())
    }

    missing_keys = [img for img in test_images if img not in name_to_pred]
    if missing_keys:
        print(
            f"Warning: Missing {len(missing_keys)} test image_ids in TFRecord predictions. "
            f"Example missing: {missing_keys[:5]}. Falling back to test_images/ decoding."
        )
        test_paths_ds = tf.data.Dataset.from_tensor_slices(image_paths_test)
        test_img_ds = test_paths_ds.map(
            _decode_img_from_path, num_parallel_calls=AUTOTUNE
        )
        test_img_ds = test_img_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(
            AUTOTUNE
        )

        all_preds = []
        for batch_imgs in test_img_ds:
            probs = model(batch_imgs, training=False).numpy()
            preds = np.argmax(probs, axis=1).astype(np.int32)
            all_preds.append(preds)

        pred_labels = np.concatenate(all_preds, axis=0)[: len(test_images)]
        sub = pd.DataFrame({"image_id": test_images, "label": pred_labels.astype(int)})
        sub.to_csv("submission.csv", index=False)
        print(sub.head())
        print("Wrote submission.csv with rows:", len(sub))
        print("submission.csv exists:", os.path.exists("submission.csv"))
    else:
        ordered_pred_labels = np.array(
            [name_to_pred[img] for img in test_images], dtype=int
        )

        if len(ordered_pred_labels) != len(sample_sub):
            raise ValueError(
                f"Prediction length {len(ordered_pred_labels)} != sample_submission length {len(sample_sub)}"
            )

        sub = pd.DataFrame({"image_id": test_images, "label": ordered_pred_labels})
        sub.to_csv("submission.csv", index=False)

        print(sub.head())
        print("Wrote submission.csv with rows:", len(sub))
        print("submission.csv exists:", os.path.exists("submission.csv"))
