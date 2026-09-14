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
import random

import numpy as np
import pandas as pd
import cv2  # kept (present in original), even though unused

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
)  # kept (present in original), even though unused
from tensorflow.keras.applications.vgg16 import (
    VGG16,
    preprocess_input,
)  # preprocess_input kept for API parity
from tensorflow.keras import layers

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass




## === cell 1
BASE_DIR = "../input/cassava-leaf-disease-classification"
source_dir = os.path.join(BASE_DIR, "train_images")
train_csv_path = os.path.join(BASE_DIR, "train.csv")
label_name_path = os.path.join(BASE_DIR, "label_num_to_disease_map.json")

data_label = pd.read_csv(train_csv_path)

with open(label_name_path, "r") as f:
    label_name = json.load(f)

csv_ids = data_label["image_id"].astype(str).tolist()

all_images = list(csv_ids)

random.shuffle(all_images)

print("Train labels in CSV:", len(data_label))
print("Images from CSV (assumed present):", len(all_images))




## === cell 2
IMG_H, IMG_W = 100, 100

train_size = 0.9
split_size = int(len(all_images) * train_size)
train_images = all_images[:split_size]
test_images = all_images[split_size:]

id_to_label = dict(
    zip(data_label["image_id"].astype(str).values, data_label["label"].values)
)

train_paths = [os.path.join(source_dir, img) for img in train_images]
test_paths = [os.path.join(source_dir, img) for img in test_images]

train_labels = np.array([id_to_label[img] for img in train_images], dtype=np.int32)
test_labels = np.array([id_to_label[img] for img in test_images], dtype=np.int32)

if len(train_paths) == 0 or len(test_paths) == 0:
    raise RuntimeError(
        f"Empty train/test split: train={len(train_paths)}, test={len(test_paths)}"
    )

y_train = to_categorical(train_labels, 5)
y_test = to_categorical(test_labels, 5)

BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE


def _load_and_resize_tf(path, label):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, [IMG_H, IMG_W], method=tf.image.ResizeMethod.AREA)
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.vgg16.preprocess_input(img)
    return img, label


_SHUFFLE_BUFFER = min(len(train_paths), 4096)

options = tf.data.Options()
options.autotune.enabled = True

cache_dir = "/kaggle/working/tf_cache"
os.makedirs(cache_dir, exist_ok=True)
train_cache_path = os.path.join(
    cache_dir, f"train_{IMG_H}x{IMG_W}_bs{BATCH_SIZE}.cache"
)
test_cache_path = os.path.join(cache_dir, f"val_{IMG_H}x{IMG_W}_bs{BATCH_SIZE}.cache")

train_ds = tf.data.Dataset.from_tensor_slices((train_paths, y_train)).with_options(
    options
)
train_ds = train_ds.map(
    _load_and_resize_tf, num_parallel_calls=AUTOTUNE, deterministic=True
)
train_ds = train_ds.apply(tf.data.experimental.ignore_errors())
train_ds = train_ds.cache(train_cache_path)
train_ds = train_ds.shuffle(
    buffer_size=_SHUFFLE_BUFFER, seed=SEED, reshuffle_each_iteration=True
)
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((test_paths, y_test)).with_options(options)
test_ds = test_ds.map(
    _load_and_resize_tf, num_parallel_calls=AUTOTUNE, deterministic=True
)
test_ds = test_ds.apply(tf.data.experimental.ignore_errors())
test_ds = test_ds.cache(test_cache_path)
test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

print("Train samples:", len(train_paths), "Test samples:", len(test_paths))
print("IMG:", (IMG_H, IMG_W), "Batch:", BATCH_SIZE, "Shuffle buffer:", _SHUFFLE_BUFFER)
print("Disk cache:", train_cache_path)




## === cell 3
def define_directory():
    base = "/kaggle/working/Training"
    subdirs = ["CBB", "CBSD", "CGM", "CMD", "Healthy"]
    if not os.path.exists(base):
        os.mkdir(base)
    for sd in subdirs:
        p = os.path.join(base, sd)
        if not os.path.exists(p):
            os.mkdir(p)




## === cell 4
pre_trained_model = VGG16(
    include_top=False, weights="imagenet", input_shape=(IMG_H, IMG_W, 3)
)

for layer in pre_trained_model.layers:
    layer.trainable = False

last_layer = pre_trained_model.get_layer("block5_pool")
last_output = last_layer.output

x = layers.Reshape((-1,))(last_output)
x = layers.Dense(5, activation="softmax")(x)

model = keras.Model(pre_trained_model.input, x)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    steps_per_execution=32,
)

model.summary()




## === cell 5
history = model.fit(train_ds, validation_data=test_ds, epochs=10, verbose=2)




## === cell 6
test_dir = os.path.join(BASE_DIR, "test_images")

sample_sub_path = os.path.join(BASE_DIR, "sample_submission.csv")
sample_sub = pd.read_csv(sample_sub_path)

ordered_test_ids = sample_sub["image_id"].astype(str).tolist()

test_existing_paths = tf.io.gfile.glob(os.path.join(test_dir, "*.jpg"))
test_existing = set(os.path.basename(p) for p in test_existing_paths)

if any(img not in test_existing for img in ordered_test_ids):
    ordered_test_ids = sorted(list(test_existing))

test_paths = [os.path.join(test_dir, img) for img in ordered_test_ids]


def _load_test_tf(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, [IMG_H, IMG_W], method=tf.image.ResizeMethod.AREA)
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.vgg16.preprocess_input(img)
    return img


test_pred_ds = tf.data.Dataset.from_tensor_slices(test_paths).with_options(options)
test_pred_ds = test_pred_ds.map(
    _load_test_tf, num_parallel_calls=AUTOTUNE, deterministic=True
)
test_pred_ds = test_pred_ds.apply(tf.data.experimental.ignore_errors())
test_pred_ds = test_pred_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

probs = model.predict(test_pred_ds, verbose=0)
pred_labels = probs.argmax(axis=1).astype(int).tolist()

submission_df = pd.DataFrame({"image_id": ordered_test_ids, "label": pred_labels})
submission_path = "./submission.csv"
submission_df.to_csv(submission_path, index=False)

print("Wrote:", submission_path)
print(submission_df.head())
print("Rows:", len(submission_df), "Cols:", submission_df.columns.tolist())
assert submission_path.endswith(".csv")
assert submission_df.shape[0] == sample_sub.shape[0]
assert submission_df.columns.tolist() == ["image_id", "label"]
