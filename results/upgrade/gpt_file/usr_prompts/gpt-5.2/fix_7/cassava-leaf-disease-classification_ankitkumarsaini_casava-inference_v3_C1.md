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

0.8786642490178301

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

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

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception as e:
    print("Could not set threading config:", repr(e))

try:
    tf.config.experimental.enable_op_determinism(False)
except Exception:
    pass



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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

IMAGE_SIZE = [224, 224]

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
@tf.function
def decode_image(image):
    image = tf.image.decode_jpeg(image, channels=3, dct_method="INTEGER_FAST")
    image = tf.image.resize(
        image, IMAGE_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.ensure_shape(image, [*IMAGE_SIZE, 3])
    return image




## === cell 5
_LABELED_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}
_UNLABELED_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


def _read_tfrecord_batch(examples, labeled):
    feats = tf.io.parse_example(
        examples, _LABELED_FEATURES if labeled else _UNLABELED_FEATURES
    )
    images = tf.map_fn(
        decode_image,
        feats["image"],
        fn_output_signature=tf.float32,
        parallel_iterations=AUTOTUNE,
        back_prop=False,
    )
    if labeled:
        labels = tf.cast(feats["target"], tf.int32)
        return images, labels
    ids = feats["image_name"]
    return images, ids


def load_dataset(filenames, labeled=True, ordered=False):
    opts = tf.data.Options()
    if not ordered:
        opts.experimental_deterministic = False
    opts.experimental_optimization.apply_default_optimizations = True
    opts.experimental_optimization.map_parallelization = True
    opts.experimental_optimization.autotune_buffers = True

    ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE).with_options(
        opts
    )
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.map(
        partial(_read_tfrecord_batch, labeled=labeled), num_parallel_calls=AUTOTUNE
    )
    return ds




## === cell 6
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




## === cell 7
def data_augment(image, label):
    image = tf.image.random_flip_left_right(image)
    return image, label


_HAS_GPU = bool(tf.config.list_logical_devices("GPU"))


def _maybe_prefetch_to_device(ds):
    if _HAS_GPU:
        try:
            return ds.apply(
                tf.data.experimental.prefetch_to_device("/GPU:0", buffer_size=AUTOTUNE)
            )
        except Exception:
            return ds.prefetch(AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


def get_test_dataset(ordered=False):
    dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
    dataset = dataset.cache()
    dataset = _maybe_prefetch_to_device(dataset)
    return dataset


def get_train_dataset(ordered=False):
    dataset = load_dataset(TRAIN_FILENAMES, labeled=True, ordered=ordered)
    dataset = dataset.map(data_augment, num_parallel_calls=AUTOTUNE)
    dataset = (
        dataset.unbatch()
        .shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
        .batch(BATCH_SIZE, drop_remainder=False)
    )
    dataset = _maybe_prefetch_to_device(dataset)
    return dataset




## === cell 8
print(
    "Test dataset configured. (Skipping eager inspection to avoid extra I/O/decoding pass.)"
)



## === cell 9
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



## === cell 10
train_files, val_files = train_test_split(
    TRAIN_FILENAMES, test_size=0.15, random_state=SEED, shuffle=True
)

train_ds = load_dataset(train_files, labeled=True, ordered=False)
train_ds = train_ds.unbatch().map(data_augment, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True).batch(
    BATCH_SIZE, drop_remainder=False
)
train_ds = _maybe_prefetch_to_device(train_ds)

val_ds = load_dataset(val_files, labeled=True, ordered=True).cache()
val_ds = _maybe_prefetch_to_device(val_ds)

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



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3530272584.py in <cell line: 0>()
      5 # Speed: training dataset uses batched parsing already; shuffle must operate on individual examples,
      6 # so we temporarily unbatch->shuffle->batch to preserve exact shuffle semantics.
----> 7 train_ds = load_dataset(train_files, labeled=True, ordered=False)
      8 train_ds = train_ds.unbatch().map(data_augment, num_parallel_calls=AUTOTUNE)
      9 train_ds = train_ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True).batch(

/tmp/ipykernel_11/3315104761.py in load_dataset(filenames, labeled, ordered)
     37     opts.experimental_optimization.apply_default_optimizations = True
     38     opts.experimental_optimization.map_parallelization = True
---> 39     opts.experimental_optimization.autotune_buffers = True
     40 
     41     ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE).with_options(

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/util/options.py in __setattr__(self, name, value)
     59       object.__setattr__(self, name, value)
     60     else:
---> 61       raise AttributeError("Cannot set the property {} on {}.".format(
     62           name,
     63           type(self).__name__))

AttributeError: Cannot set the property autotune_buffers on OptimizationOptions.

## === cell 11
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



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3305480801.py in <cell line: 0>()
     12 fine_tune_epochs = 2
     13 history2 = model.fit(
---> 14     train_ds,
     15     validation_data=val_ds,
     16     epochs=EPOCHS + fine_tune_epochs,

NameError: name 'train_ds' is not defined

## === cell 12
test_ds = get_test_dataset(ordered=True)

test_img_ds = test_ds.map(lambda img, img_id: img, num_parallel_calls=AUTOTUNE)
test_img_ds = test_img_ds.prefetch(AUTOTUNE)

print("Computing predictions...")
probabilities = model.predict(test_img_ds, verbose=1)
predictions = np.argmax(probabilities, axis=-1).astype(np.int64)

test_ids_batches = []
for _, batch_ids in test_ds:
    test_ids_batches.append(batch_ids.numpy())
test_ids = np.concatenate(test_ids_batches, axis=0).astype("U")

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

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/775921259.py in <cell line: 0>()
      1 # Speed: single-pass inference that collects ids and predicts without iterating test_ds twice.
      2 # Correctness: identical model.predict inputs; ids come from the same dataset order (ordered=True).
----> 3 test_ds = get_test_dataset(ordered=True)
      4 
      5 test_img_ds = test_ds.map(lambda img, img_id: img, num_parallel_calls=AUTOTUNE)

/tmp/ipykernel_11/4031515067.py in get_test_dataset(ordered)
     21 
     22 def get_test_dataset(ordered=False):
---> 23     dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
     24     # Speed: cache decoded/resized test set in memory for a single pass predict+id collection.
     25     dataset = dataset.cache()

/tmp/ipykernel_11/3315104761.py in load_dataset(filenames, labeled, ordered)
     37     opts.experimental_optimization.apply_default_optimizations = True
     38     opts.experimental_optimization.map_parallelization = True
---> 39     opts.experimental_optimization.autotune_buffers = True
     40 
     41     ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE).with_options(

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/util/options.py in __setattr__(self, name, value)
     59       object.__setattr__(self, name, value)
     60     else:
---> 61       raise AttributeError("Cannot set the property {} on {}.".format(
     62           name,
     63           type(self).__name__))

AttributeError: Cannot set the property autotune_buffers on OptimizationOptions.
