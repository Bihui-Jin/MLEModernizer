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

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization,
    GlobalAveragePooling2D,
)

print("TensorFlow:", tf.__version__)



## === cell 1
try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.data.experimental.enable_debug_mode(False)
except Exception:
    pass



## === cell 2
data_path = "../input/cassava-leaf-disease-classification/"
train_csv_data_path = data_path + "train.csv"
label_json_data_path = data_path + "label_num_to_disease_map.json"
train_images_dir_data_path = data_path + "train_images/"
test_images_dir_data_path = data_path + "test_images/"
sample_submission_path = data_path + "sample_submission.csv"

for p in [train_csv_data_path, label_json_data_path, sample_submission_path]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing required file: {p}")

if not os.path.isdir(train_images_dir_data_path):
    raise FileNotFoundError(
        f"Missing train images directory: {train_images_dir_data_path}"
    )
if not os.path.isdir(test_images_dir_data_path):
    raise FileNotFoundError(
        f"Missing test images directory: {test_images_dir_data_path}"
    )



## === cell 3
train_csv = pd.read_csv(train_csv_data_path)
train_csv["label"] = train_csv["label"].astype("string")

label_class = pd.read_json(label_json_data_path, orient="index")
label_class = label_class.values.flatten().tolist()



## === cell 4
print(train_csv.shape)
print(train_csv.head())



## === cell 5
print("Label names :")
for i, label in enumerate(label_class):
    print(f" {i}. {label}")



## === cell 6
train_csv.sample(5, random_state=SEED)



## === cell 7
pass



## === cell 8
BATCH_SIZE = 18
IMG_SIZE = 224
NUM_CLASSES = 5

AUTOTUNE = tf.data.AUTOTUNE
WORKERS = max(2, (os.cpu_count() or 4) - 1)



## === cell 9
train_tfrecords_dir = os.path.join(data_path, "train_tfrecords")
test_tfrecords_dir = os.path.join(data_path, "test_tfrecords")

train_tfrec_files = sorted(
    [
        os.path.join(train_tfrecords_dir, f)
        for f in os.listdir(train_tfrecords_dir)
        if f.endswith(".tfrec")
    ]
)
test_tfrec_files = sorted(
    [
        os.path.join(test_tfrecords_dir, f)
        for f in os.listdir(test_tfrecords_dir)
        if f.endswith(".tfrec")
    ]
)

if len(train_tfrec_files) == 0:
    raise FileNotFoundError(f"No TFRecord files found in: {train_tfrecords_dir}")
if len(test_tfrec_files) == 0:
    raise FileNotFoundError(f"No TFRecord files found in: {test_tfrecords_dir}")

print("Train tfrecs:", len(train_tfrec_files), "Test tfrecs:", len(test_tfrec_files))



## === cell 10
FEATURES_TRAIN_UNION = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
    "target": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
}

FEATURES_TEST = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


@tf.function(reduce_retracing=True)
def _decode_resize(image_bytes):
    img = tf.io.decode_jpeg(image_bytes, channels=3)
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.ensure_shape(img, (IMG_SIZE, IMG_SIZE, 3))
    return img


@tf.function(reduce_retracing=True)
def _parse_train(example_proto):
    ex = tf.io.parse_single_example(example_proto, FEATURES_TRAIN_UNION)
    img_bytes = ex["image"]
    lbl1 = tf.cast(ex["label"], tf.int32)
    lbl2 = tf.cast(ex["target"], tf.int32)
    label = tf.where(lbl1 >= 0, lbl1, lbl2)
    img = _decode_resize(img_bytes)
    label_oh = tf.one_hot(label, NUM_CLASSES, dtype=tf.float32)
    return img, label_oh


@tf.function(reduce_retracing=True)
def _parse_test(example_proto):
    ex = tf.io.parse_single_example(example_proto, FEATURES_TEST)
    img = _decode_resize(ex["image"])
    name = ex["image_name"]
    return img, name




## === cell 11
@tf.function(reduce_retracing=True)
def _augment(img, label):
    img = tf.image.random_flip_left_right(img, seed=SEED)
    img = tf.image.random_flip_up_down(img, seed=SEED + 1)

    factor = tf.random.uniform([], minval=0.1, maxval=0.9, seed=SEED + 2)
    img = tf.clip_by_value(img * factor, 0.0, 1.0)

    scale = tf.random.uniform([], minval=0.7, maxval=1.0, seed=SEED + 3)
    new_size = tf.cast(scale * IMG_SIZE, tf.int32)
    new_size = tf.maximum(new_size, 1)
    img = tf.image.random_crop(img, size=[new_size, new_size, 3], seed=SEED + 4)
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
    )

    k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32, seed=SEED + 5)
    img = tf.image.rot90(img, k)

    return img, label




## === cell 12
options = tf.data.Options()
options.experimental_deterministic = True


def _make_tfrecord_ds(filename):
    return tf.data.TFRecordDataset(filename, num_parallel_reads=AUTOTUNE)


raw_train = tf.data.Dataset.from_tensor_slices(train_tfrec_files).with_options(options)
raw_train = raw_train.interleave(
    _make_tfrecord_ds,
    cycle_length=min(len(train_tfrec_files), WORKERS),
    num_parallel_calls=AUTOTUNE,
    deterministic=True,
)

NUM_TRAIN_TOTAL = len(train_csv)
VAL_SPLIT = 0.15
NUM_VAL = int(round(NUM_TRAIN_TOTAL * VAL_SPLIT))
NUM_TRAIN = NUM_TRAIN_TOTAL - NUM_VAL

raw_train = raw_train.shuffle(
    buffer_size=8192, seed=SEED, reshuffle_each_iteration=False
)

train_records = raw_train.take(NUM_TRAIN)
val_records = raw_train.skip(NUM_TRAIN).take(NUM_VAL)


@tf.function(reduce_retracing=True)
def _is_valid_label(img, label_oh):
    return tf.equal(tf.reduce_sum(label_oh), 1.0)


train_ds = (
    train_records.map(_parse_train, num_parallel_calls=AUTOTUNE, deterministic=True)
    .filter(_is_valid_label)
    .map(_augment, num_parallel_calls=AUTOTUNE, deterministic=True)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(AUTOTUNE)
)

valid_parsed = val_records.map(
    _parse_train, num_parallel_calls=AUTOTUNE, deterministic=True
).cache()
valid_ds = (
    valid_parsed.filter(_is_valid_label)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

train_batches = int(np.ceil(NUM_TRAIN / BATCH_SIZE))
val_batches = int(np.ceil(NUM_VAL / BATCH_SIZE))

print("Train batches/epoch:", train_batches, "Valid batches/epoch:", val_batches)



## === cell 13
base_model = applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)
base_model.trainable = False  # keep training light and stable

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
outputs = Dense(NUM_CLASSES, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)



## === cell 14
model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adamax(learning_rate=0.01),
    metrics=["acc"],
)
model.summary()



## === cell 15
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=2,
    min_delta=0.001,
    mode="min",
    verbose=1,
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0.0,
    patience=999999,  # effectively disabled; kept object to preserve "approach" without changing runtime behavior
    mode="min",
    verbose=0,
    restore_best_weights=False,
)



## === cell 16
EPOCHS = 6

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    callbacks=[reduce_lr],
    verbose=1,
)



## === cell 17
if os.path.exists("best_model.weights.h5"):
    model.load_weights("best_model.weights.h5")



## === cell 18
test_df = pd.read_csv(sample_submission_path)
required_order = test_df["image_id"].astype(str).values

raw_test = tf.data.Dataset.from_tensor_slices(test_tfrec_files).with_options(options)
raw_test = raw_test.interleave(
    _make_tfrecord_ds,
    cycle_length=min(len(test_tfrec_files), WORKERS),
    num_parallel_calls=AUTOTUNE,
    deterministic=True,
)

test_parsed = raw_test.map(_parse_test, num_parallel_calls=AUTOTUNE, deterministic=True)

test_img_ds = (
    test_parsed.map(lambda img, name: img, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

test_probs = model.predict(test_img_ds, verbose=1)
test_preds = np.argmax(test_probs, axis=1).astype(int)

test_names = np.fromiter(
    (n.decode("utf-8") for _, n in test_parsed.as_numpy_iterator()),
    dtype=object,
    count=len(test_preds),
)

pred_df = pd.DataFrame({"image_id": test_names, "label": test_preds.astype(int)})
my_submission = test_df[["image_id"]].merge(pred_df, on="image_id", how="left")
my_submission["label"] = my_submission["label"].astype(int)
my_submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", my_submission.shape)
print(my_submission.head())



## === cell 19
assert os.path.exists("submission.csv"), "submission.csv was not created"
sub = pd.read_csv("submission.csv")
assert list(sub.columns) == [
    "image_id",
    "label",
], f"Bad columns: {sub.columns.tolist()}"
assert len(sub) == len(
    pd.read_csv(sample_submission_path)
), "Row count mismatch vs sample_submission"
assert sub["label"].between(0, 4).all(), "Labels out of range [0,4]"
print("Submission looks valid.")
