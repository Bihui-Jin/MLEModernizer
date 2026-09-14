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
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import random
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import ReduceLROnPlateau

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

print("TF version:", tf.__version__)




## === cell 1
BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train_images")
TEST_IMG_DIR = os.path.join(BASE_DIR, "test_images")
SAMPLE_SUB = os.path.join(BASE_DIR, "sample_submission.csv")
TRAIN_TFREC_DIR = os.path.join(BASE_DIR, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(BASE_DIR, "test_tfrecords")

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing: {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing: {TEST_IMG_DIR}"
assert os.path.exists(SAMPLE_SUB), f"Missing: {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_TFREC_DIR), f"Missing: {TRAIN_TFREC_DIR}"
assert os.path.isdir(TEST_TFREC_DIR), f"Missing: {TEST_TFREC_DIR}"

train_df = pd.read_csv(TRAIN_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB)

train_df["label"] = train_df["label"].astype(int)
train_df["image_id"] = train_df["image_id"].astype(str)
sample_sub["image_id"] = sample_sub["image_id"].astype(str)

print(train_df.head())
print("Train rows:", len(train_df), "Test rows:", len(sample_sub))




## === cell 2
from sklearn.model_selection import train_test_split

train_split, val_split = train_test_split(
    train_df,
    test_size=0.1,
    random_state=SEED,
    stratify=train_df["label"],
)

IMG_SIZE = 380
BATCH_SIZE = 16  # keep identical
NUM_CLASSES = 5

AUTO = tf.data.AUTOTUNE

augment_layer = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal", seed=SEED),
        layers.RandomRotation(0.10, seed=SEED),
        layers.RandomTranslation(0.05, 0.05, seed=SEED),
        layers.RandomZoom(0.10, seed=SEED),
    ],
    name="augmentation",
)


def _decode_resize_from_bytes(img_bytes):
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32)
    return img


def _preprocess(img):
    return preprocess_input(img)


_FEATURE_DESC = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}


def _parse_train_example_with_name(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURE_DESC)
    img = _decode_resize_from_bytes(ex["image"])
    y = tf.cast(ex["target"], tf.int32)
    name = tf.strings.strip(
        tf.strings.unicode_transcode(ex["image_name"], "UTF-8", "UTF-8")
    )
    return img, y, name


def _parse_test_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURE_DESC)
    img = _decode_resize_from_bytes(ex["image"])
    image_name = tf.strings.strip(
        tf.strings.unicode_transcode(ex["image_name"], "UTF-8", "UTF-8")
    )
    return img, image_name


train_tfrecs = sorted(tf.io.gfile.glob(os.path.join(TRAIN_TFREC_DIR, "*.tfrec")))
test_tfrecs = sorted(tf.io.gfile.glob(os.path.join(TEST_TFREC_DIR, "*.tfrec")))
assert len(train_tfrecs) > 0, "No train TFRecords found"
assert len(test_tfrecs) > 0, "No test TFRecords found"

_OPTIONS = tf.data.Options()
_OPTIONS.experimental_deterministic = True
_OPTIONS.threading.private_threadpool_size = 0  # let TF choose
_OPTIONS.threading.max_intra_op_parallelism = 0

print("Train/Val sizes:", len(train_split), len(val_split))

_train_ids = train_split["image_id"].astype(str).tolist()
_val_ids = val_split["image_id"].astype(str).tolist()

_train_keys = tf.constant(_train_ids, dtype=tf.string)
_val_keys = tf.constant(_val_ids, dtype=tf.string)
_dummy_vals_train = tf.ones([len(_train_ids)], dtype=tf.int32)
_dummy_vals_val = tf.ones([len(_val_ids)], dtype=tf.int32)

_train_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(_train_keys, _dummy_vals_train),
    default_value=0,
)
_val_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(_val_keys, _dummy_vals_val),
    default_value=0,
)


def _make_ds_from_tfrecs_masked(tfrecs, training: bool):
    ds = tf.data.TFRecordDataset(tfrecs, num_parallel_reads=AUTO).with_options(_OPTIONS)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = ds.map(
        _parse_train_example_with_name, num_parallel_calls=AUTO, deterministic=True
    )

    if training:
        ds = ds.filter(lambda img, y, name: _train_table.lookup(name) > 0)

        def _map_train(img, y, name):
            img = augment_layer(img, training=True)
            img = _preprocess(img)
            return img, y

        ds = ds.map(_map_train, num_parallel_calls=AUTO, deterministic=True)
        ds = ds.shuffle(
            buffer_size=min(len(train_split), 4096),
            seed=SEED,
            reshuffle_each_iteration=True,
        )
    else:
        ds = ds.filter(lambda img, y, name: _val_table.lookup(name) > 0)

        def _map_val(img, y, name):
            img = _preprocess(img)
            return img, y

        ds = ds.map(_map_val, num_parallel_calls=AUTO, deterministic=True)
        ds = ds.cache()

    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTO)
    return ds


train_ds = _make_ds_from_tfrecs_masked(train_tfrecs, training=True)
val_ds = _make_ds_from_tfrecs_masked(train_tfrecs, training=False)




## === cell 3
base = ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)
base.trainable = False  # keep fast and stable

model = models.Sequential(
    [
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()




## === cell 4
callbacks = [
    ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=1, verbose=1, min_lr=1e-6
    ),
]

EPOCHS = 2  # keep identical

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1,
)




## === cell 5
test_df = sample_sub[["image_id"]].copy()


def make_test_ds_with_names():
    ds = tf.data.TFRecordDataset(test_tfrecs, num_parallel_reads=AUTO).with_options(
        _OPTIONS
    )
    ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = ds.map(_parse_test_example, num_parallel_calls=AUTO, deterministic=True)

    def _map(img, name):
        img = _preprocess(img)
        return img, name

    ds = ds.map(_map, num_parallel_calls=AUTO, deterministic=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTO)
    return ds


test_ds = make_test_ds_with_names()

probs_list = []
names_list = []
for xb, nb in test_ds:
    probs_list.append(model(xb, training=False).numpy())
    names_list.append(nb.numpy())

probs = np.concatenate(probs_list, axis=0)
names = np.concatenate(names_list, axis=0)

if names.dtype.kind in ("S", "O"):
    names_str = np.array(
        [
            (
                n.decode("utf-8", errors="ignore").strip()
                if isinstance(n, (bytes, bytearray))
                else str(n).strip()
            )
            for n in names
        ],
        dtype=object,
    )
else:
    names_str = names.astype(str)

preds_all = probs.argmax(axis=1).astype(int)

name_to_pred = dict(zip(names_str.tolist(), preds_all.tolist()))

sample_ids = sample_sub["image_id"].astype(str).tolist()
missing = [sid for sid in sample_ids if sid not in name_to_pred]
if len(missing) > 0:
    raise ValueError(
        f"Missing {len(missing)} sample_submission image_ids in TFRecords. Example: {missing[:5]}"
    )

preds = np.array([name_to_pred[sid] for sid in sample_ids], dtype=int)

assert len(preds) == len(sample_sub), (len(preds), len(sample_sub))

submission = pd.DataFrame({"image_id": sample_sub["image_id"].values, "label": preds})
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with rows:", len(submission))




## === cell 6
predictions = preds.tolist()
predictions[:20]
