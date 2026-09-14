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

if "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION" in os.environ:
    del os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import json, warnings
import math
import random

warnings.simplefilter("ignore")

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    GlobalAveragePooling2D,
    Dense,
    Dropout,
)
from tensorflow.keras.optimizers import Adam

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    _CPU_COUNT = os.cpu_count() or 2
    tf.config.threading.set_inter_op_parallelism_threads(min(4, _CPU_COUNT))
    tf.config.threading.set_intra_op_parallelism_threads(_CPU_COUNT)
except Exception:
    pass

general_path = "../input/cassava-leaf-disease-classification/"
if not os.path.exists(general_path):
    general_path = "/kaggle/input/cassava-leaf-disease-classification/"

print("general_path:", general_path)
print("train.csv exists:", os.path.exists(os.path.join(general_path, "train.csv")))
print(
    "sample_submission exists:",
    os.path.exists(os.path.join(general_path, "sample_submission.csv")),
)

_CPU = os.cpu_count() or 2
AUTOTUNE = tf.data.AUTOTUNE
TFDATA_NUM_CALLS = AUTOTUNE  # let runtime pick best parallelism

print("CPU:", _CPU)



## === cell 1
with open(os.path.join(general_path, "label_num_to_disease_map.json")) as file:
    map_classes = json.loads(file.read())
    map_classes = {int(k): v for k, v in map_classes.items()}
print("Class map:", map_classes)



## === cell 2
df_train = pd.read_csv(os.path.join(general_path, "train.csv"))
df_train["class_name"] = df_train["label"].map(map_classes)
df_train.head()



## === cell 3
img_width, img_height = 260, 260

train_df = df_train.copy()
train_df["label_str"] = train_df["label"].astype(str)

perm = np.random.RandomState(SEED).permutation(len(train_df))
val_size = int(round(0.2 * len(train_df)))
val_idx = perm[:val_size]
trn_idx = perm[val_size:]

trn_df = train_df.iloc[trn_idx].reset_index(drop=True)
val_df = train_df.iloc[val_idx].reset_index(drop=True)

num_classes = int(train_df["label"].nunique())
print("num_classes:", num_classes)

sorted_label_str = sorted(train_df["label_str"].unique(), key=lambda x: x)
class_indices = {k: i for i, k in enumerate(sorted_label_str)}
idx_to_label = {v: int(k) for k, v in class_indices.items()}
print("class_indices:", class_indices)
print("idx_to_label:", idx_to_label)

train_steps = int(math.ceil(len(trn_df) / 64))
valid_steps = int(math.ceil(len(val_df) / 64))
print("train_steps:", train_steps, "valid_steps:", valid_steps)

train_img_dir = os.path.join(general_path, "train_images")


def _read_decode_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)  # cassava dataset is jpg
    img = tf.image.resize(
        img, [img_width, img_height], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) / 255.0
    return img


def _augment(img):
    raise NotImplementedError


def _make_ds(df, training):
    paths = tf.constant(
        [os.path.join(train_img_dir, fn) for fn in df["image_id"].values]
    )
    labels = tf.constant(
        [class_indices[s] for s in df["label_str"].values], dtype=tf.int32
    )

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(buffer_size=len(df), seed=SEED, reshuffle_each_iteration=True)

    ds = ds.enumerate()

    def _map_fn(i, x):
        path, label = x
        img = _read_decode_resize(path)

        if training:
            seed = tf.stack([tf.cast(SEED, tf.int64), tf.cast(i, tf.int64)], axis=0)

            rnd = tf.random.stateless_uniform([2], seed=seed)
            img = tf.cond(
                rnd[0] < 0.5, lambda: tf.image.flip_left_right(img), lambda: img
            )
            img = tf.cond(rnd[1] < 0.5, lambda: tf.image.flip_up_down(img), lambda: img)

            z = tf.random.stateless_uniform(
                [], seed=seed + tf.constant([0, 1], tf.int64), minval=0.8, maxval=1.2
            )
            new_h = tf.cast(tf.round(z * img_height), tf.int32)
            new_w = tf.cast(tf.round(z * img_width), tf.int32)
            img_zoom = tf.image.resize(
                img, [new_h, new_w], method=tf.image.ResizeMethod.BILINEAR
            )
            img_zoom = tf.image.resize_with_crop_or_pad(img_zoom, img_height, img_width)
            img = img_zoom

            shear = tf.random.stateless_uniform(
                [], seed=seed + tf.constant([0, 2], tf.int64), minval=-0.2, maxval=0.2
            )
            a0 = tf.constant(1.0, tf.float32)
            a1 = -tf.cast(shear, tf.float32)
            a2 = tf.constant(0.0, tf.float32)
            b0 = tf.constant(0.0, tf.float32)
            b1 = tf.constant(1.0, tf.float32)
            b2 = tf.constant(0.0, tf.float32)
            c0 = tf.constant(0.0, tf.float32)
            c1 = tf.constant(0.0, tf.float32)
            transform = tf.stack([a0, a1, a2, b0, b1, b2, c0, c1])
            img = tf.raw_ops.ImageProjectiveTransformV3(
                images=tf.expand_dims(img, 0),
                transforms=tf.expand_dims(transform, 0),
                output_shape=tf.constant([img_height, img_width], dtype=tf.int32),
                interpolation="BILINEAR",
                fill_mode="REFLECT",
                fill_value=0.0,
            )[0]

        y = tf.one_hot(label, depth=num_classes, dtype=tf.float32)
        return img, y

    ds = ds.map(_map_fn, num_parallel_calls=TFDATA_NUM_CALLS)

    if not training:
        ds = ds.cache()

    ds = ds.batch(64, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


train_ds = _make_ds(trn_df, training=True)
valid_ds = _make_ds(val_df, training=False)



## === cell 4
pass



## === cell 5
model = Sequential(
    [
        Conv2D(32, (3, 3), activation="relu", input_shape=(img_width, img_height, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        GlobalAveragePooling2D(),
        Dropout(0.3),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax"),
    ]
)

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()



## === cell 6
EPOCHS = 5  # keep as given (no early stopping / approximations)

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    steps_per_epoch=train_steps,
    validation_steps=valid_steps,
    verbose=1,
)



## === cell 7
val_loss, val_acc = model.evaluate(
    valid_ds,
    steps=valid_steps,
    verbose=0,
)
print("Validation accuracy:", val_acc)



## === cell 8
ss = pd.read_csv(os.path.join(general_path, "sample_submission.csv"))

test_img_dir = os.path.join(general_path, "test_images")
test_paths = tf.constant(
    [os.path.join(test_img_dir, fn) for fn in ss["image_id"].values]
)

test_ds = tf.data.Dataset.from_tensor_slices(test_paths)


def _test_map(path):
    img = _read_decode_resize(path)
    return img


test_ds = (
    test_ds.map(_test_map, num_parallel_calls=TFDATA_NUM_CALLS)
    .batch(64)
    .prefetch(AUTOTUNE)
)

test_steps = int(math.ceil(len(ss) / 64))

probs = model.predict(
    test_ds,
    steps=test_steps,
    verbose=1,
)

pred_class_idx = np.argmax(probs, axis=1).astype(int)

lut = np.fromiter(
    (idx_to_label[i] for i in range(num_classes)), count=num_classes, dtype=np.int64
)
preds = lut[pred_class_idx].astype(int)

if len(preds) != len(ss):
    raise RuntimeError(
        f"Prediction length {len(preds)} != sample_submission length {len(ss)}"
    )

my_submission = pd.DataFrame({"image_id": ss["image_id"].values, "label": preds})
my_submission.to_csv("submission.csv", index=False)

print("Saved submission.csv with shape:", my_submission.shape)
my_submission.head()
