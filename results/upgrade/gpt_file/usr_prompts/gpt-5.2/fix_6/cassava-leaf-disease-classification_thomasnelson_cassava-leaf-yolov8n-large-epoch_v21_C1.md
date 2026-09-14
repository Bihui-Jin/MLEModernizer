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

3.13

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

0.870806890299184

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import sys
import random
from csv import writer

import numpy as np

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)




## === cell 1
CANDIDATE_ROOTS = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    "/kaggle/working/cassava-leaf-disease-classification",
    "/kaggle/working/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
]

DATA_ROOT = None
for r in CANDIDATE_ROOTS:
    if os.path.isfile(os.path.join(r, "train.csv")) and os.path.isdir(
        os.path.join(r, "train_images")
    ):
        DATA_ROOT = r
        break

if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not locate cassava dataset root with train.csv and train_images/. "
        f"Tried: {CANDIDATE_ROOTS}"
    )

TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_DIR = os.path.join(DATA_ROOT, "test_images")
TRAIN_TFREC_DIR = os.path.join(DATA_ROOT, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(DATA_ROOT, "test_tfrecords")

SUB_PATH = "/kaggle/working/submission.csv"

for p in [TRAIN_CSV, SAMPLE_SUB]:
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Required file not found: {p}")
for d in [TRAIN_DIR, TEST_DIR]:
    if not os.path.isdir(d):
        raise FileNotFoundError(f"Required directory not found: {d}")

if os.path.exists(SUB_PATH):
    os.remove(SUB_PATH)

print("DATA_ROOT:", DATA_ROOT)
print("TRAIN_CSV:", TRAIN_CSV)
print("TRAIN_DIR:", TRAIN_DIR)
print("TEST_DIR :", TEST_DIR)
print("TRAIN_TFREC_DIR:", TRAIN_TFREC_DIR, "exists:", os.path.isdir(TRAIN_TFREC_DIR))
print("TEST_TFREC_DIR :", TEST_TFREC_DIR, "exists:", os.path.isdir(TEST_TFREC_DIR))




## === cell 2
import pandas as pd
import tensorflow as tf

tf.random.set_seed(SEED)

_DATA_OPTIONS = tf.data.Options()
_DATA_OPTIONS.experimental_deterministic = True  # preserve stable behavior across runs
_DATA_OPTIONS.threading.private_threadpool_size = 0  # let TF decide
try:
    _DATA_OPTIONS.autotune.enabled = True
except Exception:
    pass

train_df = pd.read_csv(TRAIN_CSV)
if not {"image_id", "label"}.issubset(train_df.columns):
    raise ValueError(
        f"train.csv missing required columns. Found: {train_df.columns.tolist()}"
    )

train_df = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

num_classes = int(train_df["label"].nunique())
if num_classes != 5:
    print(f"Warning: expected 5 classes but found {num_classes}")

val_frac = 0.1
val_size = int(len(train_df) * val_frac)
val_df = train_df.iloc[:val_size].reset_index(drop=True)
tr_df = train_df.iloc[val_size:].reset_index(drop=True)

IMG_SIZE = 160
BATCH_SIZE = 32

_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


def _parse_train_example(serialized):
    ex = tf.io.parse_single_example(serialized, _FEATURES)
    img = tf.image.decode_jpeg(ex["image"], channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    label = tf.cast(ex["label"], tf.int32)
    image_id = tf.cast(ex["image_name"], tf.string)
    return image_id, img, label


def _augment(img, label):
    img = tf.image.random_flip_left_right(img, seed=SEED)
    img = tf.image.random_flip_up_down(img, seed=SEED)
    img = tf.image.random_brightness(img, max_delta=0.10, seed=SEED)
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img, label


tr_ids = tf.constant(tr_df["image_id"].astype(str).values)
val_ids = tf.constant(val_df["image_id"].astype(str).values)

tr_key_init = tf.lookup.KeyValueTensorInitializer(
    keys=tr_ids, values=tf.ones([tf.shape(tr_ids)[0]], dtype=tf.int8)
)
val_key_init = tf.lookup.KeyValueTensorInitializer(
    keys=val_ids, values=tf.ones([tf.shape(val_ids)[0]], dtype=tf.int8)
)

TR_ID_TABLE = tf.lookup.StaticHashTable(
    tr_key_init, default_value=tf.constant(0, tf.int8)
)
VAL_ID_TABLE = tf.lookup.StaticHashTable(
    val_key_init, default_value=tf.constant(0, tf.int8)
)


def _is_in_train(image_id, img, label):
    return tf.equal(TR_ID_TABLE.lookup(image_id), 1)


def _is_in_val(image_id, img, label):
    return tf.equal(VAL_ID_TABLE.lookup(image_id), 1)


def _drop_id(image_id, img, label):
    return img, label


def _list_tfrecords(tfrecord_dir):
    if not os.path.isdir(tfrecord_dir):
        return []
    files = tf.io.gfile.glob(os.path.join(tfrecord_dir, "*.tfrec"))
    files.sort()
    return files


train_tfrec_files = _list_tfrecords(TRAIN_TFREC_DIR)
if not train_tfrec_files:
    print("Warning: train_tfrecords not found; falling back to JPEG pipeline (slower).")

    def _load_image_and_label(image_id, label):
        image_path = tf.strings.join([TRAIN_DIR, "/", image_id])
        img_bytes = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img_bytes, channels=3)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method="bilinear")
        img = tf.cast(img, tf.float32) / 255.0
        return img, tf.cast(label, tf.int32)

    def make_ds(df, training):
        ids = df["image_id"].astype(str).values
        labels = df["label"].astype(np.int32).values
        ds = tf.data.Dataset.from_tensor_slices((ids, labels))
        if training:
            ds = ds.shuffle(
                buffer_size=min(len(df), 8192), seed=SEED, reshuffle_each_iteration=True
            )
        ds = ds.map(_load_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)
        if training:
            ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = make_ds(tr_df, training=True)
    val_ds = make_ds(val_df, training=False)
else:
    raw_ds = tf.data.TFRecordDataset(
        train_tfrec_files, num_parallel_reads=tf.data.AUTOTUNE
    )
    raw_ds = raw_ds.with_options(_DATA_OPTIONS)
    parsed = raw_ds.map(_parse_train_example, num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = (
        parsed.filter(_is_in_train)
        .map(_drop_id, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(buffer_size=8192, seed=SEED, reshuffle_each_iteration=True)
        .map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        parsed.filter(_is_in_val)
        .map(_drop_id, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.25)(x)
outputs = tf.keras.layers.Dense(5, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

EPOCHS = 5
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=2)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
sample_sub = pd.read_csv(SAMPLE_SUB)
if not {"image_id", "label"}.issubset(sample_sub.columns):
    raise ValueError(
        f"sample_submission.csv missing required columns. Found: {sample_sub.columns.tolist()}"
    )

test_image_ids = sample_sub["image_id"].astype(str).values

test_tfrec_files = (
    tf.io.gfile.glob(os.path.join(TEST_TFREC_DIR, "*.tfrec"))
    if os.path.isdir(TEST_TFREC_DIR)
    else []
)
test_tfrec_files.sort()

_TEST_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


def _parse_test_example(serialized):
    ex = tf.io.parse_single_example(serialized, _TEST_FEATURES)
    img = tf.image.decode_jpeg(ex["image"], channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    image_id = tf.cast(ex["image_name"], tf.string)
    return image_id, img


if test_tfrec_files:
    raw_test = tf.data.TFRecordDataset(
        test_tfrec_files, num_parallel_reads=tf.data.AUTOTUNE
    )
    raw_test = raw_test.with_options(_DATA_OPTIONS)
    parsed_test = raw_test.map(_parse_test_example, num_parallel_calls=tf.data.AUTOTUNE)

    batched = parsed_test.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    probs_by_id = {}
    for batch_ids, batch_imgs in batched:
        batch_probs = model(batch_imgs, training=False).numpy()
        for k, p in zip(batch_ids.numpy().tolist(), batch_probs):
            probs_by_id[k.decode("utf-8")] = p

    probs = np.stack([probs_by_id[i] for i in test_image_ids], axis=0)
else:

    def _load_test_image(image_id):
        image_path = tf.strings.join([TEST_DIR, "/", image_id])
        img_bytes = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img_bytes, channels=3)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method="bilinear")
        img = tf.cast(img, tf.float32) / 255.0
        return img

    test_ds = tf.data.Dataset.from_tensor_slices(test_image_ids)
    test_ds = test_ds.map(_load_test_image, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    probs = model.predict(test_ds, verbose=0)

pred_labels = np.argmax(probs, axis=1).astype(int)

sub = pd.DataFrame({"image_id": test_image_ids, "label": pred_labels})
sub.to_csv(SUB_PATH, index=False)

print("Saved:", SUB_PATH)
print(sub.head())
print("shape:", sub.shape)
print("label dtype:", sub["label"].dtype)

assert sub.shape[0] == sample_sub.shape[0]
assert list(sub.columns) == ["image_id", "label"]
assert sub["label"].between(0, 4).all()
assert os.path.isfile(SUB_PATH)

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3728555413.py in <cell line: 0>()
     40     probs_by_id = {}
     41     for batch_ids, batch_imgs in batched:
---> 42         batch_probs = model(batch_imgs, training=False).numpy()
     43         for k, p in zip(batch_ids.numpy().tolist(), batch_probs):
     44             probs_by_id[k.decode("utf-8")] = p

NameError: name 'model' is not defined
