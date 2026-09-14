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
import re
import math
import random

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from functools import partial
from sklearn.model_selection import train_test_split

print("Tensorflow version " + tf.__version__)

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.optimizer.set_jit(True)
except Exception as e:
    print("Could not enable XLA JIT:", repr(e))

tf.data.experimental.enable_debug_mode = False



## === cell 1
BASE_PATH = "../input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")
TRAIN_TFRECORDS_GLOB = os.path.join(BASE_PATH, "train_tfrecords", "ld_train*.tfrec")
TEST_TFRECORDS_GLOB = os.path.join(BASE_PATH, "test_tfrecords", "ld_test*.tfrec")

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"



## === cell 2
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 16
IMAGE_SIZE = [512, 512]
CLASSES = ["0", "1", "2", "3", "4"]
CLASS_NAMES = [
    "Cassava Bacterial Blight",
    "Cassava Brown Streak Disease",
    "Cassava Green Mottle",
    "Cassava Mosaic Disease",
    "Healthy",
]

EPOCHS = 7



## === cell 3
TRAIN_FILENAMES = tf.io.gfile.glob(TRAIN_TFRECORDS_GLOB)
TEST_FILENAMES = tf.io.gfile.glob(TEST_TFRECORDS_GLOB)

if len(TRAIN_FILENAMES) == 0:
    raise FileNotFoundError(f"No train tfrecords found at: {TRAIN_TFRECORDS_GLOB}")
if len(TEST_FILENAMES) == 0:
    raise FileNotFoundError(f"No test tfrecords found at: {TEST_TFRECORDS_GLOB}")

print(f"Train TFRecords: {len(TRAIN_FILENAMES)}")
print(f"Test TFRecords: {len(TEST_FILENAMES)}")




## === cell 4
def decode_image(image):
    image = tf.image.decode_jpeg(image, channels=3, dct_method="INTEGER_FAST")
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    return image




## === cell 5
def read_tfrecord(example, labeled):
    tfrecord_format = (
        {
            "image": tf.io.FixedLenFeature([], tf.string),
            "target": tf.io.FixedLenFeature([], tf.int64),
        }
        if labeled
        else {
            "image": tf.io.FixedLenFeature([], tf.string),
            "image_name": tf.io.FixedLenFeature([], tf.string),
        }
    )

    example = tf.io.parse_single_example(example, tfrecord_format)
    image = decode_image(example["image"])

    if labeled:
        label = tf.cast(example["target"], tf.int32)
        return image, label

    idnum = example["image_name"]
    return image, idnum




## === cell 6
def load_dataset(filenames, labeled=True, ordered=False):
    opts = tf.data.Options()
    if not ordered:
        opts.experimental_deterministic = False

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
    dataset = dataset.with_options(opts)
    dataset = dataset.map(
        partial(read_tfrecord, labeled=labeled), num_parallel_calls=AUTOTUNE
    )
    return dataset




## === cell 7
_COUNT_RE = re.compile(r"-([0-9]*)\.")


def _file_count(fn: str) -> int:
    m = _COUNT_RE.search(fn)
    if m is None:
        raise ValueError(f"Could not parse count from filename: {fn}")
    return int(m.group(1))


TRAIN_FILE_COUNTS = {fn: _file_count(fn) for fn in TRAIN_FILENAMES}
TEST_FILE_COUNTS = {fn: _file_count(fn) for fn in TEST_FILENAMES}


def count_data_items(filenames):
    return int(
        sum(
            TRAIN_FILE_COUNTS.get(fn, TEST_FILE_COUNTS.get(fn, _file_count(fn)))
            for fn in filenames
        )
    )


NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)
NUM_TRAIN_ITEMS = count_data_items(TRAIN_FILENAMES)

print("NUM_TRAIN_ITEMS:", NUM_TRAIN_ITEMS)
print("NUM_TEST_IMAGES:", NUM_TEST_IMAGES)




## === cell 8
def data_augment(image, label):
    image = tf.image.random_flip_left_right(image)
    return image, label


def get_test_dataset(ordered=False):
    dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    return dataset


def get_train_dataset(ordered=False):
    dataset = load_dataset(TRAIN_FILENAMES, labeled=True, ordered=ordered)
    dataset = dataset.map(data_augment, num_parallel_calls=AUTOTUNE)
    dataset = dataset.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    return dataset




## === cell 9
print("Test data shapes:")
for image, idnum in get_test_dataset(ordered=True).take(1):
    print(image.shape, idnum.shape)
    print("Example IDs:", idnum.numpy()[:5].astype("U"))



## === cell 10
base = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(*IMAGE_SIZE, 3),
)
base.trainable = False  # stabilize & fit within time budget

inputs = keras.Input(shape=(*IMAGE_SIZE, 3))
x = inputs
x = tf.keras.applications.efficientnet.preprocess_input(x * 255.0)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(5, activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
)

model.summary()



## === cell 11
train_files, val_files = train_test_split(
    TRAIN_FILENAMES, test_size=0.15, random_state=SEED, shuffle=True
)

train_ds = load_dataset(train_files, labeled=True, ordered=False)
train_ds = train_ds.map(data_augment, num_parallel_calls=AUTOTUNE)
train_ds = (
    train_ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

val_ds = load_dataset(val_files, labeled=True, ordered=True)
val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

steps_per_epoch = max(1, count_data_items(train_files) // BATCH_SIZE)
val_steps = max(1, count_data_items(val_files) // BATCH_SIZE)

print("steps_per_epoch:", steps_per_epoch, "val_steps:", val_steps)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    verbose=2,
)



## === cell 12
base.trainable = True
for layer in base.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
)

fine_tune_epochs = 2
history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS + fine_tune_epochs,
    initial_epoch=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    verbose=2,
)



## === cell 13
test_ds = get_test_dataset(ordered=True)

id_inp = keras.Input(shape=(), dtype=tf.string, name="image_id")
img_inp = keras.Input(shape=(*IMAGE_SIZE, 3), dtype=tf.float32, name="image")
probs_out = model(img_inp, training=False)
pred_model = keras.Model(inputs=[img_inp, id_inp], outputs=[probs_out, id_inp])

print("Computing predictions...")
probabilities, test_ids = pred_model.predict(test_ds, verbose=1)
predictions = np.argmax(probabilities, axis=-1).astype(np.int64)
test_ids = test_ids.astype("U")

print("Predictions shape:", predictions.shape)

print("Generating submission.csv file...")

if len(test_ids) != len(predictions):
    raise ValueError(
        f"ID/pred length mismatch: {len(test_ids)} ids vs {len(predictions)} preds"
    )

sub = pd.DataFrame({"image_id": test_ids, "label": predictions})
sample = pd.read_csv(SAMPLE_SUB)
sub = sample[["image_id"]].merge(sub, on="image_id", how="left")
if sub["label"].isna().any():
    raise ValueError("Some test image_ids missing predictions after merge.")
sub["label"] = sub["label"].astype(int)

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
