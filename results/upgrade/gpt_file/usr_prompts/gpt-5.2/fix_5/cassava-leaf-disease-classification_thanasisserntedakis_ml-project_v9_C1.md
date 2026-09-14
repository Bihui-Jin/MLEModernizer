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

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Activation,
    Dropout,
    Flatten,
    Dense,
)

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("TensorFlow:", tf.__version__)




## === cell 1
BASE = "../input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(BASE, "train.csv")
TRAIN_IMG_DIR = os.path.join(BASE, "train_images")
TEST_IMG_DIR = os.path.join(BASE, "test_images")
SAMPLE_SUB = os.path.join(BASE, "sample_submission.csv")

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing: {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing: {TEST_IMG_DIR}"
assert os.path.exists(SAMPLE_SUB), f"Missing: {SAMPLE_SUB}"

train_df = pd.read_csv(TRAIN_CSV)
sample_sub_df = pd.read_csv(SAMPLE_SUB)

train_files = {e.name for e in os.scandir(TRAIN_IMG_DIR) if e.is_file()}
test_files = {e.name for e in os.scandir(TEST_IMG_DIR) if e.is_file()}

train_df = train_df[train_df["image_id"].isin(train_files)].reset_index(drop=True)
sample_sub_df = sample_sub_df[sample_sub_df["image_id"].isin(test_files)].reset_index(
    drop=True
)

train_df["path"] = (TRAIN_IMG_DIR + "/" + train_df["image_id"].astype(str)).astype(str)
sample_sub_df["path"] = (
    TEST_IMG_DIR + "/" + sample_sub_df["image_id"].astype(str)
).astype(str)

train_df["label_str"] = train_df["label"].astype(str)

NUM_CLASSES = train_df["label"].nunique()
print("Train rows:", len(train_df), "Num classes:", NUM_CLASSES)
print("Test rows:", len(sample_sub_df))




## === cell 2
IMG_SIZE = 224
SIZE = (IMG_SIZE, IMG_SIZE)
BATCH_SIZE = 32
EPOCHS = 5

AUTOTUNE = tf.data.AUTOTUNE

val_frac = 0.1
n_total = len(train_df)
n_val = int(np.floor(n_total * val_frac))
n_train = n_total - n_val

train_df_split = train_df.iloc[:n_train].reset_index(drop=True)
valid_df_split = train_df.iloc[n_train:].reset_index(drop=True)

class_names = sorted(train_df_split["label_str"].unique().tolist())
class_indices = {name: idx for idx, name in enumerate(class_names)}
print("Class indices:", class_indices)

keys = tf.constant(class_names, dtype=tf.string)
vals = tf.constant([class_indices[k] for k in class_names], dtype=tf.int32)
table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(keys, vals),
    default_value=-1,
)


@tf.function
def _decode_resize_rescale(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)  # [0,1], float32
    return img


@tf.function
def _parse_train(path, label_str):
    img = _decode_resize_rescale(path)
    label_idx = table.lookup(label_str)
    label_oh = tf.one_hot(label_idx, depth=NUM_CLASSES, dtype=tf.float32)
    return img, label_oh


@tf.function
def _parse_test(path):
    img = _decode_resize_rescale(path)
    return img


train_paths = train_df_split["path"].to_numpy()
train_labels = train_df_split["label_str"].to_numpy()

valid_paths = valid_df_split["path"].to_numpy()
valid_labels = valid_df_split["label_str"].to_numpy()

test_paths = sample_sub_df["path"].to_numpy()

options = tf.data.Options()
options.experimental_deterministic = True

shuffle_buf = min(len(train_df_split), 4096)

train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels)).with_options(
    options
)
train_ds = train_ds.shuffle(
    buffer_size=shuffle_buf, seed=SEED, reshuffle_each_iteration=True
)
train_ds = train_ds.map(_parse_train, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.cache()
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

valid_ds = tf.data.Dataset.from_tensor_slices((valid_paths, valid_labels)).with_options(
    options
)
valid_ds = valid_ds.map(_parse_train, num_parallel_calls=AUTOTUNE)
valid_ds = valid_ds.cache()
valid_ds = valid_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices(test_paths).with_options(options)
test_ds = test_ds.map(_parse_test, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.cache()
test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

inv_map = {v: int(k) for k, v in class_indices.items()}




## === cell 3
model = Sequential(
    [
        Conv2D(32, (3, 3), input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        Activation("relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(32, (3, 3)),
        Activation("relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3)),
        Activation("relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(64),
        Activation("relu"),
        Dropout(0.5),
        Dense(NUM_CLASSES),
        Activation("softmax"),
    ]
)

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()




## === cell 4
history = model.fit(train_ds, validation_data=valid_ds, epochs=EPOCHS, verbose=1)

val_loss, val_acc = model.evaluate(valid_ds, verbose=0)
print("Validation accuracy:", val_acc)




## === cell 5
pred_test = model.predict(test_ds, verbose=1)
pred_class_indices = np.argmax(pred_test, axis=-1).astype(int)

lut = np.fromiter(
    (inv_map[i] for i in range(NUM_CLASSES)), count=NUM_CLASSES, dtype=np.int32
)
pred_labels = lut[pred_class_indices].astype(int)

submission = sample_sub_df[["image_id"]].copy()
submission["label"] = pred_labels

base_sub = sample_sub_df[["image_id"]].copy()
submission = base_sub.merge(submission, on="image_id", how="left")
submission["label"] = submission["label"].fillna(0).astype(int)

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with rows:", len(submission))
assert os.path.exists("submission.csv")
assert list(submission.columns) == ["image_id", "label"]
