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

ROOT_DIR = "../input/cassava-leaf-disease-classification/"
print("ROOT_DIR exists:", os.path.exists(ROOT_DIR))
print("ROOT_DIR listing (head):", os.listdir(ROOT_DIR)[:10])



## === cell 1
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("TensorFlow:", tf.__version__)

try:
    tf.keras.utils.set_random_seed(42)
except Exception:
    pass



## === cell 2
TRAIN_CSV = os.path.join(ROOT_DIR, "train.csv")
TRAIN_DIR = os.path.join(ROOT_DIR, "train_images")
TEST_DIR = os.path.join(ROOT_DIR, "test_images")
SAMPLE_SUB = os.path.join(ROOT_DIR, "sample_submission.csv")

TRAIN_TFRECORDS_DIR = os.path.join(ROOT_DIR, "train_tfrecords")
TEST_TFRECORDS_DIR = os.path.join(ROOT_DIR, "test_tfrecords")

train_df = pd.read_csv(TRAIN_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB)

print("train_df:", train_df.shape, train_df.columns.tolist())
print("sample_sub:", sample_sub.shape, sample_sub.columns.tolist())
print("Train images dir exists:", os.path.isdir(TRAIN_DIR))
print("Test images dir exists:", os.path.isdir(TEST_DIR))
print("Train TFRecords dir exists:", os.path.isdir(TRAIN_TFRECORDS_DIR))
print("Test TFRecords dir exists:", os.path.isdir(TEST_TFRECORDS_DIR))



## === cell 3
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

IMG_SIZE = 300
BATCH_SIZE = 32
EPOCHS = 5  # preserve core training setup
NUM_CLASSES = 5



## === cell 4
train_df["filepath"] = (TRAIN_DIR.rstrip("/") + "/") + train_df["image_id"].astype(str)

from sklearn.model_selection import train_test_split

train_idx, val_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.1,
    random_state=SEED,
    stratify=train_df["label"].values,
)

tr_df = train_df.iloc[train_idx].reset_index(drop=True)
va_df = train_df.iloc[val_idx].reset_index(drop=True)

print("Train split:", tr_df.shape, "Val split:", va_df.shape)
print("Label distribution (train):")
print(tr_df["label"].value_counts(normalize=True).sort_index())
print("Label distribution (val):")
print(va_df["label"].value_counts(normalize=True).sort_index())



## === cell 5
AUTOTUNE = tf.data.AUTOTUNE

DATASET_OPTS = tf.data.Options()
DATASET_OPTS.experimental_deterministic = False
try:
    DATASET_OPTS.autotune.enabled = True
except Exception:
    pass
try:
    DATASET_OPTS.experimental_optimization.apply_default_optimizations = True
    DATASET_OPTS.experimental_optimization.autotune_buffers = True
    DATASET_OPTS.experimental_optimization.map_parallelization = True
except Exception:
    pass  # compatibility across TF builds

TRAIN_TFRECORD_FILES = sorted(
    [
        os.path.join(TRAIN_TFRECORDS_DIR, f)
        for f in os.listdir(TRAIN_TFRECORDS_DIR)
        if f.endswith(".tfrec")
    ]
)
TEST_TFRECORD_FILES = sorted(
    [
        os.path.join(TEST_TFRECORDS_DIR, f)
        for f in os.listdir(TEST_TFRECORDS_DIR)
        if f.endswith(".tfrec")
    ]
)

print("Num train tfrecs:", len(TRAIN_TFRECORD_FILES))
print("Num test tfrecs:", len(TEST_TFRECORD_FILES))


@tf.function
def _decode_resize_from_path(path, label):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR, antialias=True
    )
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img, tf.cast(label, tf.int32)


_FEATURE_DESC = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}


@tf.function
def _parse_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURE_DESC)
    img = tf.image.decode_jpeg(ex["image"], channels=3)
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR, antialias=True
    )
    img = tf.image.convert_image_dtype(img, tf.float32)
    label = tf.cast(ex["target"], tf.int32)
    image_id = ex["image_name"]
    return img, label, image_id


TR_EX_PER_FILE = 1338  # Cassava TFRecord shard size in this dataset
num_train_total = len(train_df)

n = num_train_total
is_train = np.zeros(n, dtype=np.uint8)
is_train[train_idx] = 1

num_shards = len(TRAIN_TFRECORD_FILES)
train_files, val_files, mixed_files = [], [], []

for shard_i, fp in enumerate(TRAIN_TFRECORD_FILES):
    start = shard_i * TR_EX_PER_FILE
    end = min((shard_i + 1) * TR_EX_PER_FILE, n)
    if start >= end:
        continue
    shard_len = end - start
    tr_count = int(is_train[start:end].sum())
    if tr_count == shard_len:
        train_files.append(fp)
    elif tr_count == 0:
        val_files.append(fp)
    else:
        mixed_files.append((fp, start, end))

print(
    "Train-only tfrecs:",
    len(train_files),
    "Val-only tfrecs:",
    len(val_files),
    "Mixed tfrecs:",
    len(mixed_files),
)

tr_ids = tr_df["image_id"].astype(str).values
va_ids = va_df["image_id"].astype(str).values

tr_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        tf.constant(tr_ids),
        tf.ones([len(tr_ids)], dtype=tf.int32),
    ),
    default_value=0,
)
va_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        tf.constant(va_ids),
        tf.ones([len(va_ids)], dtype=tf.int32),
    ),
    default_value=0,
)


@tf.function
def _is_in_train(img, label, image_id):
    return tf.equal(tr_table.lookup(image_id), 1)


@tf.function
def _is_in_val(img, label, image_id):
    return tf.equal(va_table.lookup(image_id), 1)


@tf.function
def _drop_id(img, label, image_id):
    return img, label


def _ds_from_files(files):
    return tf.data.TFRecordDataset(files, num_parallel_reads=AUTOTUNE).with_options(
        DATASET_OPTS
    )


_EMPTY_IMG = tf.zeros([IMG_SIZE, IMG_SIZE, 3], dtype=tf.float32)
_EMPTY_LBL = tf.zeros([], dtype=tf.int32)
_EMPTY_DS = tf.data.Dataset.from_tensors((_EMPTY_IMG, _EMPTY_LBL)).take(0)


def _make_ds_from_tfrecords(split="train"):
    if split == "train":
        base_files = list(train_files)
    else:
        base_files = list(val_files)

    if base_files:
        raw = _ds_from_files(base_files)
        ds = raw.map(_parse_example, num_parallel_calls=AUTOTUNE).map(
            _drop_id, num_parallel_calls=AUTOTUNE
        )
    else:
        ds = _EMPTY_DS

    if mixed_files:
        mixed_only_files = [fp for fp, _, _ in mixed_files]
        mixed_raw = _ds_from_files(mixed_only_files)
        mixed_ds = mixed_raw.map(_parse_example, num_parallel_calls=AUTOTUNE)
        if split == "train":
            mixed_ds = mixed_ds.filter(_is_in_train).map(
                _drop_id, num_parallel_calls=AUTOTUNE
            )
        else:
            mixed_ds = mixed_ds.filter(_is_in_val).map(
                _drop_id, num_parallel_calls=AUTOTUNE
            )
        ds = ds.concatenate(mixed_ds)

    if split == "train":
        ds = ds.shuffle(4096, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def _make_ds_from_filepaths(df, training):
    paths = df["filepath"].astype(str).values
    labels = df["label"].values.astype(np.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels)).with_options(DATASET_OPTS)
    if training:
        ds = ds.shuffle(4096, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(_decode_resize_from_path, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = _make_ds_from_tfrecords(split="train")
val_ds = _make_ds_from_tfrecords(split="val")

STEPS_PER_EPOCH = int(np.ceil(len(tr_df) / BATCH_SIZE))
VAL_STEPS = int(np.ceil(len(va_df) / BATCH_SIZE))
print("steps_per_epoch:", STEPS_PER_EPOCH, "validation_steps:", VAL_STEPS)



## === cell 6
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal", seed=SEED),
        layers.RandomRotation(0.05, seed=SEED),
        layers.RandomZoom(0.1, seed=SEED),
    ],
    name="aug",
)

inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = data_augmentation(inputs)
x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2, seed=SEED)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

new_model = keras.Model(inputs, outputs)

compile_kwargs = dict(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    steps_per_execution=32,
)
new_model.compile(**compile_kwargs)

new_model.summary()



## === cell 7
history = new_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=VAL_STEPS,
    verbose=2,
)



## === cell 8
test_order = sample_sub["image_id"].astype(str).values
order_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        tf.constant(test_order),
        tf.range(len(test_order), dtype=tf.int32),
    ),
    default_value=-1,
)

test_raw = tf.data.TFRecordDataset(
    TEST_TFRECORD_FILES, num_parallel_reads=AUTOTUNE
).with_options(DATASET_OPTS)

test_parsed = test_raw.map(_parse_example, num_parallel_calls=AUTOTUNE)


@tf.function
def _keep_img_and_order_index(img, label, image_id):
    return img, order_table.lookup(image_id)


test_ds = test_parsed.map(_keep_img_and_order_index, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

test_img_ds = test_ds.map(lambda x, idx: x, num_parallel_calls=AUTOTUNE)
pred_probs = new_model.predict(test_img_ds, verbose=0)

idx = np.concatenate([b.numpy() for _, b in test_ds], axis=0).astype(np.int32)
preds = pred_probs.argmax(axis=1).astype(int)

valid_mask = idx >= 0
valid_idx = idx[valid_mask]
valid_preds = preds[valid_mask]

ordered_preds = np.full(len(test_order), 0, dtype=int)
if len(valid_idx) > 0:
    ordered_preds[valid_idx] = valid_preds

print("Preds length:", len(ordered_preds), "Test ids length:", len(test_order))
print("Unique predicted labels:", np.unique(ordered_preds))

sub = pd.DataFrame({"image_id": test_order, "label": ordered_preds})
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("submission.csv exists:", os.path.exists("submission.csv"))
