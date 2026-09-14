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

import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("TF version:", tf.__version__)



## === cell 1
DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV_PATH = os.path.join(DATA_DIR, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
SUB_PATH = "/kaggle/working/submission.csv"

TRAIN_TFREC_DIR = os.path.join(DATA_DIR, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(DATA_DIR, "test_tfrecords")

train_df = pd.read_csv(TRAIN_CSV_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

assert {"image_id", "label"}.issubset(
    train_df.columns
), "train.csv must have image_id and label"
assert {"image_id", "label"}.issubset(
    sample_sub.columns
), "sample_submission.csv must have image_id and label"

assert os.path.isdir(TRAIN_IMG_DIR), f"Missing train_images dir: {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing test_images dir: {TEST_IMG_DIR}"

USE_TFRECORDS = os.path.isdir(TRAIN_TFREC_DIR) and os.path.isdir(TEST_TFREC_DIR)
if USE_TFRECORDS:
    train_tfrecs = sorted(
        [
            os.path.join(TRAIN_TFREC_DIR, f)
            for f in os.listdir(TRAIN_TFREC_DIR)
            if f.endswith(".tfrec")
        ]
    )
    test_tfrecs = sorted(
        [
            os.path.join(TEST_TFREC_DIR, f)
            for f in os.listdir(TEST_TFREC_DIR)
            if f.endswith(".tfrec")
        ]
    )
    assert (
        len(train_tfrecs) > 0 and len(test_tfrecs) > 0
    ), "TFRecords dirs exist but no .tfrec files found"

num_classes = train_df["label"].nunique()
assert num_classes == 5, f"Expected 5 classes, got {num_classes}"

print("Train rows:", len(train_df), "Test rows:", len(sample_sub))
print("Using TFRecords:", USE_TFRECORDS)



## === cell 2
IMG_SIZE = 224
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE


def decode_and_resize_from_jpeg_bytes(img_bytes):
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    return img


def decode_and_resize(path, label=None):
    img_bytes = tf.io.read_file(path)
    img = decode_and_resize_from_jpeg_bytes(img_bytes)
    if label is None:
        return img
    return img, tf.cast(label, tf.int32)


TFREC_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
}


def parse_train_tfrecord(example_proto):
    ex = tf.io.parse_single_example(example_proto, TFREC_FEATURES)
    img = decode_and_resize_from_jpeg_bytes(ex["image"])
    label = tf.cast(ex["target"], tf.int32)
    return img, label


def parse_train_tfrecord_with_name(example_proto):
    ex = tf.io.parse_single_example(example_proto, TFREC_FEATURES)
    img = decode_and_resize_from_jpeg_bytes(ex["image"])
    label = tf.cast(ex["target"], tf.int32)
    name = ex["image_name"]
    return img, label, name


def parse_test_tfrecord(example_proto):
    ex = tf.io.parse_single_example(example_proto, TFREC_FEATURES)
    img = decode_and_resize_from_jpeg_bytes(ex["image"])
    return img


train_df_shuffled = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
val_frac = 0.1
val_size = int(len(train_df_shuffled) * val_frac)
val_df = train_df_shuffled.iloc[:val_size].copy()
trn_df = train_df_shuffled.iloc[val_size:].copy()

opts = tf.data.Options()
opts.deterministic = True  # preserve stable iteration order given the same shuffle seed
try:
    opts.experimental_optimization.map_parallelization = True
    opts.experimental_optimization.map_and_batch_fusion = True
    opts.experimental_optimization.parallel_batch = True
except Exception:
    pass

if USE_TFRECORDS:
    trn_names = tf.constant(trn_df["image_id"].astype(str).tolist())
    val_names = tf.constant(val_df["image_id"].astype(str).tolist())
    trn_name_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            trn_names, tf.ones_like(trn_names, dtype=tf.int64)
        ),
        default_value=tf.constant(0, dtype=tf.int64),
    )
    val_name_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            val_names, tf.ones_like(val_names, dtype=tf.int64)
        ),
        default_value=tf.constant(0, dtype=tf.int64),
    )

    raw_all = tf.data.TFRecordDataset(
        train_tfrecs, num_parallel_reads=AUTOTUNE
    ).with_options(opts)
    raw_all = raw_all.map(parse_train_tfrecord_with_name, num_parallel_calls=AUTOTUNE)

    def _is_in_train(img, label, name):
        return trn_name_table.lookup(name) > 0

    def _is_in_val(img, label, name):
        return val_name_table.lookup(name) > 0

    train_ds = raw_all.filter(_is_in_train)
    train_ds = train_ds.map(
        lambda img, label, name: (img, label), num_parallel_calls=AUTOTUNE
    )
    train_ds = train_ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=True).prefetch(AUTOTUNE)

    val_ds = raw_all.filter(_is_in_val)
    val_ds = val_ds.map(
        lambda img, label, name: (img, label), num_parallel_calls=AUTOTUNE
    )
    val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

    raw_test = tf.data.TFRecordDataset(
        test_tfrecs, num_parallel_reads=AUTOTUNE
    ).with_options(opts)
    raw_test = raw_test.map(parse_test_tfrecord, num_parallel_calls=AUTOTUNE)
    test_ds = raw_test.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

else:
    trn_paths = tf.constant((TRAIN_IMG_DIR + "/" + trn_df["image_id"]).tolist())
    trn_labels = tf.constant(trn_df["label"].to_numpy(dtype=np.int32))
    val_paths = tf.constant((TRAIN_IMG_DIR + "/" + val_df["image_id"]).tolist())
    val_labels = tf.constant(val_df["label"].to_numpy(dtype=np.int32))

    assert os.path.exists(
        TRAIN_IMG_DIR + "/" + trn_df["image_id"].iloc[0]
    ), "Example train image not found: " + (
        TRAIN_IMG_DIR + "/" + trn_df["image_id"].iloc[0]
    )

    train_ds = tf.data.Dataset.from_tensor_slices((trn_paths, trn_labels)).with_options(
        opts
    )
    train_ds = train_ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    train_ds = train_ds.map(decode_and_resize, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=True).prefetch(AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels)).with_options(
        opts
    )
    val_ds = val_ds.map(decode_and_resize, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

    test_paths = tf.constant((TEST_IMG_DIR + "/" + sample_sub["image_id"]).tolist())
    test_ds = tf.data.Dataset.from_tensor_slices(test_paths).with_options(opts)
    test_ds = test_ds.map(
        lambda p: decode_and_resize(p, None), num_parallel_calls=AUTOTUNE
    )
    test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

train_steps = len(trn_df) // BATCH_SIZE  # drop_remainder=True => exact integer steps
val_steps = int(np.ceil(len(val_df) / BATCH_SIZE))
test_steps = int(np.ceil(len(sample_sub) / BATCH_SIZE))

print(
    "Datasets ready:",
    "train steps =",
    train_steps,
    "val steps =",
    val_steps,
    "test steps =",
    test_steps,
)



## === cell 3
inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(5, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    jit_compile=True,
)

EPOCHS = 5

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
    verbose=2,
)



## === cell 4
probs = model.predict(test_ds, steps=test_steps, verbose=0)
preds = np.argmax(probs, axis=1).astype(int).tolist()

preds = preds[: len(sample_sub)]

my_submission = pd.DataFrame(
    {"image_id": sample_sub["image_id"].values, "label": preds}
)
my_submission.to_csv(SUB_PATH, index=False)

assert os.path.exists(SUB_PATH), "submission.csv was not created"
assert len(my_submission) == len(sample_sub), "Submission row count mismatch"
assert set(my_submission.columns) == {"image_id", "label"}, "Wrong submission columns"
assert (
    pd.api.types.is_integer_dtype(my_submission["label"])
    or my_submission["label"].map(type).eq(int).all()
)
assert my_submission["label"].between(0, 4).all(), "Labels must be integers in [0,4]"

print("Wrote:", SUB_PATH)
print(my_submission.head())
