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

0.8164097914777878

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.models import *

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

print("TF version:", tf.__version__)

tf.config.experimental.enable_op_determinism()

try:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except Exception as e:
    print("GPU mem growth set failed (non-fatal):", e)

try:
    tf.config.optimizer.set_jit(True)
except Exception as e:
    print("XLA JIT enable failed (non-fatal):", e)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_DIR = "../input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(TRAIN_IMG_DIR), f"Missing {TRAIN_IMG_DIR}"
assert os.path.exists(TEST_IMG_DIR), f"Missing {TEST_IMG_DIR}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"

IMAGE_SIZE = 380
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE
CLASS_NUMS = 5

train_df = pd.read_csv(TRAIN_CSV)
print("Train rows:", len(train_df), "columns:", train_df.columns.tolist())
print("Label distribution:\n", train_df["label"].value_counts().sort_index())



## === cell 2
tf.keras.utils.set_random_seed(42)

train_paths = (TRAIN_IMG_DIR + "/" + train_df["image_id"].astype(str)).to_numpy()
train_labels = train_df["label"].astype("int32").to_numpy()

idx = np.arange(len(train_paths))
rng = np.random.default_rng(42)
rng.shuffle(idx)
split = int(0.9 * len(idx))
tr_idx, va_idx = idx[:split], idx[split:]

tr_paths, va_paths = train_paths[tr_idx], train_paths[va_idx]
tr_labels, va_labels = train_labels[tr_idx], train_labels[va_idx]

print("Train/Val sizes:", len(tr_paths), len(va_paths))




## === cell 3
@tf.function
def decode_and_resize(img_path, label=None):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    if label is None:
        return img
    return img, tf.cast(label, tf.int32)


@tf.function
def augment_if_training(img, label=None, training=False):
    if training:
        img = tf.image.random_flip_left_right(img)
    if label is None:
        return img
    return img, label


def make_ds(paths, labels=None, training=False, cache=False, cache_path=None):
    options = tf.data.Options()
    options.experimental_deterministic = True
    options.threading.private_threadpool_size = 0
    options.threading.max_intra_op_parallelism = 0
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    options.experimental_optimization.autotune_buffers = True

    if labels is None:
        ds = tf.data.Dataset.from_tensor_slices(paths)
        ds = ds.map(decode_and_resize, num_parallel_calls=AUTOTUNE)
        ds = ds.apply(tf.data.experimental.ignore_errors())
        if cache:
            ds = ds.cache()
        ds = ds.batch(BATCH_SIZE, drop_remainder=False)
        ds = ds.prefetch(AUTOTUNE)
        try:
            ds = ds.apply(tf.data.experimental.prefetch_to_device("/GPU:0"))
        except Exception:
            pass
    else:
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if training:
            ds = ds.shuffle(2048, seed=42, reshuffle_each_iteration=True)
        ds = ds.map(decode_and_resize, num_parallel_calls=AUTOTUNE)
        ds = ds.apply(tf.data.experimental.ignore_errors())
        if cache:
            ds = ds.cache()
        if training:
            ds = ds.repeat()
            ds = ds.map(
                lambda img, y: augment_if_training(img, y, training=True),
                num_parallel_calls=AUTOTUNE,
            )
        ds = ds.batch(BATCH_SIZE, drop_remainder=bool(training))
        ds = ds.prefetch(AUTOTUNE)
        try:
            ds = ds.apply(tf.data.experimental.prefetch_to_device("/GPU:0"))
        except Exception:
            pass

    ds = ds.with_options(options)
    return ds


TRAIN_CACHE = os.path.join("/kaggle/working", "train_cache.tfdata")
VAL_CACHE = os.path.join("/kaggle/working", "val_cache.tfdata")

train_ds = make_ds(
    tr_paths, tr_labels, training=True, cache=True, cache_path=TRAIN_CACHE
)
val_ds = make_ds(va_paths, va_labels, training=False, cache=True, cache_path=VAL_CACHE)

steps_per_epoch = int(np.ceil(len(tr_paths) / BATCH_SIZE))
validation_steps = int(np.ceil(len(va_paths) / BATCH_SIZE))



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1847649889.py in <cell line: 0>()
     75 VAL_CACHE = os.path.join("/kaggle/working", "val_cache.tfdata")
     76 
---> 77 train_ds = make_ds(
     78     tr_paths, tr_labels, training=True, cache=True, cache_path=TRAIN_CACHE
     79 )

/tmp/ipykernel_11/1847649889.py in make_ds(paths, labels, training, cache, cache_path)
     31     options.experimental_optimization.map_parallelization = True
     32     options.experimental_optimization.parallel_batch = True
---> 33     options.experimental_optimization.autotune_buffers = True
     34 
     35     if labels is None:

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/util/options.py in __setattr__(self, name, value)
     59       object.__setattr__(self, name, value)
     60     else:
---> 61       raise AttributeError("Cannot set the property {} on {}.".format(
     62           name,
     63           type(self).__name__))

AttributeError: Cannot set the property autotune_buffers on OptimizationOptions.

## === cell 4
base = tf.keras.applications.EfficientNetB4(
    include_top=False,
    weights="imagenet",
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    pooling="avg",
)
base.trainable = False  # minimal training time and stable behavior

inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
x = inputs
x = tf.keras.applications.efficientnet.preprocess_input(x)
x = base(x, training=False)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(CLASS_NUMS, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()



## === cell 5
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=2,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
)

base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
history_ft = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=1,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2131079875.py in <cell line: 0>()
      1 history = model.fit(
----> 2     train_ds,
      3     validation_data=val_ds,
      4     epochs=2,
      5     steps_per_epoch=steps_per_epoch,

NameError: name 'train_ds' is not defined

## === cell 6
sample_sub = pd.read_csv(SAMPLE_SUB)
test_images = sample_sub[
    "image_id"
].tolist()  # use sample submission ordering (canonical)

test_paths = (TEST_IMG_DIR + "/" + sample_sub["image_id"].astype(str)).to_numpy()

missing = [p for p in test_paths if not os.path.exists(p)]
print("Missing test images:", len(missing))
assert len(missing) == 0, f"Some test images are missing, e.g. {missing[:3]}"

TEST_CACHE = os.path.join("/kaggle/working", "test_cache.tfdata")
test_ds = make_ds(
    test_paths, labels=None, training=False, cache=True, cache_path=TEST_CACHE
)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1316435507.py in <cell line: 0>()
     11 
     12 TEST_CACHE = os.path.join("/kaggle/working", "test_cache.tfdata")
---> 13 test_ds = make_ds(
     14     test_paths, labels=None, training=False, cache=True, cache_path=TEST_CACHE
     15 )

/tmp/ipykernel_11/1847649889.py in make_ds(paths, labels, training, cache, cache_path)
     31     options.experimental_optimization.map_parallelization = True
     32     options.experimental_optimization.parallel_batch = True
---> 33     options.experimental_optimization.autotune_buffers = True
     34 
     35     if labels is None:

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/util/options.py in __setattr__(self, name, value)
     59       object.__setattr__(self, name, value)
     60     else:
---> 61       raise AttributeError("Cannot set the property {} on {}.".format(
     62           name,
     63           type(self).__name__))

AttributeError: Cannot set the property autotune_buffers on OptimizationOptions.

## === cell 7
pass



## === cell 8
probs = model.predict(test_ds, verbose=1)
predictions = np.argmax(probs, axis=1).astype(int).tolist()

print("Num test images:", len(test_images))
print("Num predictions:", len(predictions))
assert len(test_images) == len(
    predictions
), "Prediction count must match test image count"



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2751492849.py in <cell line: 0>()
----> 1 probs = model.predict(test_ds, verbose=1)
      2 predictions = np.argmax(probs, axis=1).astype(int).tolist()
      3 
      4 print("Num test images:", len(test_images))
      5 print("Num predictions:", len(predictions))

NameError: name 'test_ds' is not defined

## === cell 9
predictions[:20], test_images[:5]



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2149292075.py in <cell line: 0>()
----> 1 predictions[:20], test_images[:5]
      2 

NameError: name 'predictions' is not defined

## === cell 10
sub = pd.DataFrame({"image_id": test_images, "label": predictions})
print(sub.head())
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print("Saved to:", os.path.abspath("submission.csv"))

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1257540920.py in <cell line: 0>()
----> 1 sub = pd.DataFrame({"image_id": test_images, "label": predictions})
      2 print(sub.head())
      3 sub.to_csv("submission.csv", index=False)
      4 print("Wrote submission.csv with shape:", sub.shape)
      5 print("Saved to:", os.path.abspath("submission.csv"))

NameError: name 'predictions' is not defined
