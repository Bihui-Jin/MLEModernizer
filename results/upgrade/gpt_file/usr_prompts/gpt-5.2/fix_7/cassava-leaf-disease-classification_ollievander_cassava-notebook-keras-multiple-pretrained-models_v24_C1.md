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

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

INPUT_DIR = "../input/cassava-leaf-disease-classification/"
OUTPUT_DIR = "./"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train_images")
TEST_IMG_DIR = os.path.join(INPUT_DIR, "test_images")
TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
SAMPLE_SUB = os.path.join(INPUT_DIR, "sample_submission.csv")
LABEL_MAP_JSON = os.path.join(INPUT_DIR, "label_num_to_disease_map.json")

TRAIN_TFREC_DIR = os.path.join(INPUT_DIR, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(INPUT_DIR, "test_tfrecords")

print("TRAIN_IMG_DIR exists:", os.path.exists(TRAIN_IMG_DIR))
print("TEST_IMG_DIR exists:", os.path.exists(TEST_IMG_DIR))
print("TRAIN_CSV exists:", os.path.exists(TRAIN_CSV))
print("SAMPLE_SUB exists:", os.path.exists(SAMPLE_SUB))
print("TRAIN_TFREC_DIR exists:", os.path.exists(TRAIN_TFREC_DIR))
print("TEST_TFREC_DIR exists:", os.path.exists(TEST_TFREC_DIR))



## === cell 1
import json
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split

try:
    import albumentations as A
except Exception as e:
    A = None
    print(
        "Albumentations not available/compatible; continuing without it. Error:",
        repr(e),
    )

SEED = 100
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception as e:
    print("Could not enable op determinism (continuing). Error:", repr(e))

print("TF version:", tf.__version__)



## === cell 2
train_df = pd.read_csv(TRAIN_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB)

with open(LABEL_MAP_JSON, "r") as f:
    classes = json.load(f)

print(train_df.head())
print(sample_sub.head())
print("Num train:", len(train_df), "Num test:", len(sample_sub))
print("Classes map:", classes)



## === cell 3
train_df["label"] = train_df["label"].astype(np.int32)

train_split, val_split = train_test_split(
    train_df, test_size=0.05, random_state=SEED, stratify=train_df["label"].values
)

print("Train split:", train_split.shape, "Val split:", val_split.shape)
print(
    "Label dist train:\n",
    train_split["label"].value_counts(normalize=True).sort_index(),
)
print("Label dist val:\n", val_split["label"].value_counts(normalize=True).sort_index())



## === cell 4
AUTOTUNE = tf.data.AUTOTUNE
NUM_CLASSES = 5

IMG_SIZE = 224
BATCH_SIZE = 16

TRAIN_TFRECS = sorted(tf.io.gfile.glob(os.path.join(TRAIN_TFREC_DIR, "*.tfrec")))
TEST_TFRECS = sorted(tf.io.gfile.glob(os.path.join(TEST_TFREC_DIR, "*.tfrec")))
print("Num train tfrecs:", len(TRAIN_TFRECS), "Num test tfrecs:", len(TEST_TFRECS))

all_ids_np = train_df["image_id"].astype(str).values
all_labels_np = train_df["label"].values.astype(np.int32)

train_ids_np = train_split["image_id"].astype(str).values
val_ids_np = val_split["image_id"].astype(str).values

id_to_label_lut = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        tf.constant(all_ids_np), tf.constant(all_labels_np)
    ),
    default_value=tf.constant(-1, dtype=tf.int32),
)

train_id_set = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        tf.constant(train_ids_np), tf.ones([len(train_ids_np)], dtype=tf.int32)
    ),
    default_value=tf.constant(0, dtype=tf.int32),
)

val_id_set = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        tf.constant(val_ids_np), tf.ones([len(val_ids_np)], dtype=tf.int32)
    ),
    default_value=tf.constant(0, dtype=tf.int32),
)


def _parse_tfrec(example):
    feat = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string),
    }
    x = tf.io.parse_single_example(example, feat)
    image_id = x["image_name"]
    img_bytes = x["image"]
    img = tf.image.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    return image_id, img


def augment(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, max_delta=0.10)
    return img, label


def _dataset_options():
    options = tf.data.Options()
    options.deterministic = True
    options.experimental_optimization.apply_default_optimizations = True
    options.threading.private_threadpool_size = 0
    options.threading.max_intra_op_parallelism = 0
    return options


def make_ds_from_tfrecs_filtered(tfrecs, training=True):
    ds = tf.data.TFRecordDataset(tfrecs, num_parallel_reads=AUTOTUNE).with_options(
        _dataset_options()
    )
    ds = ds.map(_parse_tfrec, num_parallel_calls=AUTOTUNE, deterministic=True)

    if training:
        ds = ds.filter(lambda image_id, img: tf.equal(train_id_set.lookup(image_id), 1))
    else:
        ds = ds.filter(lambda image_id, img: tf.equal(val_id_set.lookup(image_id), 1))

    ds = ds.map(
        lambda image_id, img: (img, id_to_label_lut.lookup(image_id)),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )

    ds = ds.filter(lambda img, label: tf.greater_equal(label, 0))

    if training:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
        ds = ds.map(augment, num_parallel_calls=AUTOTUNE, deterministic=True)

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_ds_from_tfrecs_filtered(TRAIN_TFRECS, training=True)
val_ds = make_ds_from_tfrecs_filtered(TRAIN_TFRECS, training=False)



## === cell 5
inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()



## === cell 6
EPOCHS = 5
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=2)




## === cell 7
def _parse_test_tfrec(example):
    feat = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string),
    }
    x = tf.io.parse_single_example(example, feat)
    image_id = x["image_name"]
    img_bytes = x["image"]
    img = tf.image.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    return image_id, img


options = tf.data.Options()
options.deterministic = True
options.experimental_optimization.apply_default_optimizations = True
options.threading.private_threadpool_size = 0
options.threading.max_intra_op_parallelism = 0

test_ds = tf.data.TFRecordDataset(
    TEST_TFRECS, num_parallel_reads=AUTOTUNE
).with_options(options)
test_ds = test_ds.map(
    _parse_test_tfrec, num_parallel_calls=AUTOTUNE, deterministic=True
)

test_id_ds = test_ds.map(
    lambda image_id, img: image_id, num_parallel_calls=AUTOTUNE, deterministic=True
)
test_img_ds = test_ds.map(
    lambda image_id, img: img, num_parallel_calls=AUTOTUNE, deterministic=True
)

test_img_ds = test_img_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

probs = model.predict(test_img_ds, verbose=0)
pred_labels = np.argmax(probs, axis=1).astype(np.int32)

test_ids_from_tfrecs = []
for batch_ids in test_id_ds.batch(1024):
    test_ids_from_tfrecs.extend([x.decode("utf-8") for x in batch_ids.numpy().tolist()])

print("Pred shape:", pred_labels.shape, "Num ids:", len(test_ids_from_tfrecs))
print("Unique preds:", np.unique(pred_labels))

pred_df = pd.DataFrame({"image_id": test_ids_from_tfrecs, "label": pred_labels})

submission = sample_sub[["image_id"]].merge(pred_df, on="image_id", how="left")
if submission["label"].isna().any():
    fallback = int(train_df["label"].value_counts().idxmax())
    submission["label"] = submission["label"].fillna(fallback).astype(np.int32)

assert list(submission.columns) == ["image_id", "label"]
assert len(submission) == len(sample_sub)

out_path = os.path.join(OUTPUT_DIR, "submission.csv")
submission.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(submission.head())
