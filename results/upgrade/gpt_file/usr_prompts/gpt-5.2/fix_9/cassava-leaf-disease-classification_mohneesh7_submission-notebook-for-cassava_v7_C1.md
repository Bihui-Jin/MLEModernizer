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
import json
import glob
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception as e:
    print("Could not enable TF determinism:", e)

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TRAIN_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_DIR = os.path.join(DATA_ROOT, "test_images")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

TRAIN_TFREC_DIR = os.path.join(DATA_ROOT, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(DATA_ROOT, "test_tfrecords")

print("TensorFlow:", tf.__version__)
print("Train CSV exists:", os.path.exists(TRAIN_CSV))
print("Train dir exists:", os.path.exists(TRAIN_DIR))
print("Test dir exists:", os.path.exists(TEST_DIR))
print("Sample submission exists:", os.path.exists(SAMPLE_SUB))
print("Train TFRecords dir exists:", os.path.exists(TRAIN_TFREC_DIR))
print("Test TFRecords dir exists:", os.path.exists(TEST_TFREC_DIR))

try:
    tf.config.optimizer.set_jit(True)
except Exception as e:
    print("Could not enable XLA JIT:", e)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF pick
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception as e:
    print("Could not set threading:", e)

try:
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "1")
except Exception:
    pass




## === cell 1
IMG_SIZE = 512
BATCH_SIZE = 8  # keep as-is to preserve behavior/accuracy characteristics
NUM_CLASSES = 5

train_df = pd.read_csv(TRAIN_CSV)
assert {"image_id", "label"}.issubset(train_df.columns)

filepaths = (TRAIN_DIR.rstrip("/") + "/" + train_df["image_id"].astype(str)).values
train_df = train_df.assign(filepath=filepaths)

trn_df, val_df = train_test_split(
    train_df,
    test_size=0.1,
    random_state=SEED,
    stratify=train_df["label"],
)

print("Train size:", len(trn_df), "Valid size:", len(val_df))
print("Label distribution (train):")
print(trn_df["label"].value_counts().sort_index())




## === cell 2
_TRAIN_TFRECS = sorted(glob.glob(os.path.join(TRAIN_TFREC_DIR, "*.tfrec")))
_TEST_TFRECS = sorted(glob.glob(os.path.join(TEST_TFREC_DIR, "*.tfrec")))
if len(_TRAIN_TFRECS) == 0 or len(_TEST_TFRECS) == 0:
    raise FileNotFoundError(
        "Expected TFRecord files were not found in train_tfrecords/test_tfrecords."
    )

_FEATURE_DESCRIPTION = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


@tf.function
def _decode_from_tfrecord(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURE_DESCRIPTION)
    img = tf.image.decode_jpeg(ex["image"], channels=3)  # uint8
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method="bilinear", antialias=True)
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    label = tf.cast(ex["target"], tf.int32)
    image_name = ex["image_name"]
    return img, label, image_name


def _parse_tfrecord_shard_index(path: str) -> int:
    base = os.path.basename(path)
    part = base.split("ld_train", 1)[1]
    idx = int(part.split("-", 1)[0])
    return idx


_train_shard_by_idx = {}
for p in _TRAIN_TFRECS:
    _train_shard_by_idx[_parse_tfrecord_shard_index(p)] = p

_shard_size = int(os.path.basename(_TRAIN_TFRECS[0]).split("-", 1)[1].split(".")[0])

_trn_row_idx = trn_df.index.to_numpy(dtype=np.int64)
_val_row_idx = val_df.index.to_numpy(dtype=np.int64)

_trn_shard_ids = np.unique((_trn_row_idx // _shard_size).astype(np.int32))
_val_shard_ids = np.unique((_val_row_idx // _shard_size).astype(np.int32))

_TRN_TFRECS = [
    _train_shard_by_idx[int(i)]
    for i in sorted(_trn_shard_ids.tolist())
    if int(i) in _train_shard_by_idx
]
_VAL_TFRECS = [
    _train_shard_by_idx[int(i)]
    for i in sorted(_val_shard_ids.tolist())
    if int(i) in _train_shard_by_idx
]

if len(_TRN_TFRECS) == 0 or len(_VAL_TFRECS) == 0:
    raise RuntimeError(
        "Failed to resolve train/val TFRecord shard lists; check TFRecord naming/shard size assumptions."
    )


def _tfrecord_dataset(filenames, training):
    options = tf.data.Options()
    options.experimental_deterministic = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True

    ds = tf.data.TFRecordDataset(
        filenames,
        num_parallel_reads=tf.data.AUTOTUNE,
        compression_type=None,
    ).with_options(options)

    ds = ds.map(_decode_from_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = ds.cache()

    if training:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)

    return ds


def make_ds_from_tfrecord(split, training):
    if split == "train":
        ds = _tfrecord_dataset(_TRN_TFRECS, training=training)
    elif split == "val":
        ds = _tfrecord_dataset(_VAL_TFRECS, training=training)
    else:
        raise ValueError("split must be 'train' or 'val'")

    ds = ds.map(
        lambda img, label, name: (img, label), num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.batch(
        BATCH_SIZE, drop_remainder=training, num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


train_ds = make_ds_from_tfrecord("train", training=True)
val_ds = make_ds_from_tfrecord("val", training=False)




## === cell 3
base = ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)
base.trainable = False  # keep frozen backbone as in original

inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    jit_compile=True,
)

model.summary()




## === cell 4
callbacks = [
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_accuracy", factor=0.5, patience=1, verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        "best_model.keras", monitor="val_accuracy", save_best_only=True, verbose=1
    ),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3,
    callbacks=callbacks,
    verbose=1,
)

if os.path.exists("best_model.keras"):
    model = keras.models.load_model("best_model.keras")




## === cell 5
sub = pd.read_csv(SAMPLE_SUB)
assert {"image_id", "label"}.issubset(sub.columns)

options = tf.data.Options()
options.experimental_deterministic = True
options.experimental_optimization.map_parallelization = True
options.experimental_optimization.parallel_batch = True

test_raw = tf.data.TFRecordDataset(
    _TEST_TFRECS,
    num_parallel_reads=tf.data.AUTOTUNE,
    compression_type=None,
).with_options(options)

test_parsed = (
    test_raw.map(_decode_from_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    .apply(tf.data.experimental.ignore_errors())
    .cache()
)

test_batched = (
    test_parsed.map(
        lambda img, label, name: (img, name), num_parallel_calls=tf.data.AUTOTUNE
    )
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

all_names = []
all_preds = []
for batch_imgs, batch_names in test_batched:
    probs = model.predict_on_batch(batch_imgs)
    all_preds.append(tf.argmax(probs, axis=1, output_type=tf.int32).numpy())
    all_names.append(batch_names.numpy())

test_names = np.concatenate(all_names, axis=0)
test_names = [x.decode("utf-8") for x in test_names.tolist()]
preds = np.concatenate(all_preds, axis=0).astype(int)

if len(test_names) != len(preds):
    raise RuntimeError(
        f"Test names count {len(test_names)} != predictions count {len(preds)}"
    )

pred_map = dict(zip(test_names, preds))
sub["label"] = sub["image_id"].map(pred_map).astype(int)

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())




## === cell 6
predictions = sub["label"].tolist()
predictions[:10]
