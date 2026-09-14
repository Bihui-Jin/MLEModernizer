# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


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

# 5. Target score

0.8124811121184647

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

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




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
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
    options.experimental_optimization.autotune_buffers = True
    options.experimental_optimization.apply_default_optimizations = True

    ds = tf.data.TFRecordDataset(
        filenames,
        num_parallel_reads=tf.data.AUTOTUNE,
        compression_type=None,
    ).with_options(options)

    ds = ds.apply(tf.data.experimental.ignore_errors())

    if training:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(_decode_from_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
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

    if not training:
        ds = ds.cache()

    ds = ds.batch(BATCH_SIZE, drop_remainder=training)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


train_ds = make_ds_from_tfrecord("train", training=True)
val_ds = make_ds_from_tfrecord("val", training=False)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3831737504.py in <cell line: 0>()
    108 
    109 
--> 110 train_ds = make_ds_from_tfrecord("train", training=True)
    111 val_ds = make_ds_from_tfrecord("val", training=False)
    112 

/tmp/ipykernel_11/3831737504.py in make_ds_from_tfrecord(split, training)
     90 def make_ds_from_tfrecord(split, training):
     91     if split == "train":
---> 92         ds = _tfrecord_dataset(_TRN_TFRECS, training=training)
     93     elif split == "val":
     94         ds = _tfrecord_dataset(_VAL_TFRECS, training=training)

/tmp/ipykernel_11/3831737504.py in _tfrecord_dataset(filenames, training)
     68     options.experimental_optimization.map_parallelization = True
     69     options.experimental_optimization.parallel_batch = True
---> 70     options.experimental_optimization.autotune_buffers = True
     71     options.experimental_optimization.apply_default_optimizations = True
     72 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/util/options.py in __setattr__(self, name, value)
     59       object.__setattr__(self, name, value)
     60     else:
---> 61       raise AttributeError("Cannot set the property {} on {}.".format(
     62           name,
     63           type(self).__name__))

AttributeError: Cannot set the property autotune_buffers on OptimizationOptions.

## === cell 2
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




## === cell 3
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




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/473591572.py in <cell line: 0>()
      9 
     10 history = model.fit(
---> 11     train_ds,
     12     validation_data=val_ds,
     13     epochs=3,

NameError: name 'train_ds' is not defined

## === cell 4
sub = pd.read_csv(SAMPLE_SUB)
assert {"image_id", "label"}.issubset(sub.columns)

options = tf.data.Options()
options.experimental_deterministic = True
options.experimental_optimization.map_parallelization = True
options.experimental_optimization.parallel_batch = True
options.experimental_optimization.autotune_buffers = True
options.experimental_optimization.apply_default_optimizations = True

test_raw = tf.data.TFRecordDataset(
    _TEST_TFRECS,
    num_parallel_reads=tf.data.AUTOTUNE,
    compression_type=None,
).with_options(options)

test_raw = test_raw.apply(tf.data.experimental.ignore_errors())
test_parsed = test_raw.map(_decode_from_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)

test_img_ds = (
    test_parsed.map(lambda img, label, name: img, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(tf.data.AUTOTUNE)
)
test_name_ds = (
    test_parsed.map(lambda img, label, name: name, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(tf.data.AUTOTUNE)
)

probs = model.predict(test_img_ds, verbose=0)
preds = np.argmax(probs, axis=1).astype(int)

all_names = []
for nb in test_name_ds:
    all_names.extend([x.decode("utf-8") for x in nb.numpy().tolist()])

if len(all_names) != len(preds):
    raise RuntimeError(
        f"Test names count {len(all_names)} != predictions count {len(preds)}"
    )

pred_map = dict(zip(all_names, preds))
sub["label"] = sub["image_id"].map(pred_map).astype(int)

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2271368752.py in <cell line: 0>()
      6 options.experimental_optimization.map_parallelization = True
      7 options.experimental_optimization.parallel_batch = True
----> 8 options.experimental_optimization.autotune_buffers = True
      9 options.experimental_optimization.apply_default_optimizations = True
     10 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/util/options.py in __setattr__(self, name, value)
     59       object.__setattr__(self, name, value)
     60     else:
---> 61       raise AttributeError("Cannot set the property {} on {}.".format(
     62           name,
     63           type(self).__name__))

AttributeError: Cannot set the property autotune_buffers on OptimizationOptions.

## === cell 5
predictions = sub["label"].tolist()
predictions[:10]
