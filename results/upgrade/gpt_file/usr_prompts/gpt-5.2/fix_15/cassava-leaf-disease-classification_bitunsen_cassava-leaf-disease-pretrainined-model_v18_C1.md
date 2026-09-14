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

0.8584164400120883

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import json
import random
import numpy as np
import pandas as pd



## === cell 1
BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification/"
TRAIN_DIR = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
TEST_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_images/"

assert os.path.isdir(BASE_DIR), f"BASE_DIR not found: {BASE_DIR}"
assert os.path.isdir(TRAIN_DIR), f"TRAIN_DIR not found: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"TEST_DIR not found: {TEST_DIR}"



## === cell 2
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault(
    "TF_XLA_FLAGS", "--tf_xla_auto_jit=0 --tf_xla_cpu_global_jit=false"
)

from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

keras.backend.clear_session()
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

print("TF version:", tf.__version__)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
with open(os.path.join(BASE_DIR, "label_num_to_disease_map.json")) as file:
    map_classes = json.loads(file.read())
print(json.dumps(map_classes, indent=2))

label_list = sorted([int(k) for k in map_classes.keys()])
NUM_CLASSES = len(label_list)
assert NUM_CLASSES == 5, f"Expected 5 classes, got {NUM_CLASSES}"



## === cell 4
pass



## === cell 5
IMG_HEIGHT = 300
IMG_WIDTH = 300
batch_size = 16

PRE_TRAINED_MODEL = "../input/unionmodelv05/Cassava_Best_UnitedModel_V05.hdf5"
print("Pretrained model exists?:", os.path.exists(PRE_TRAINED_MODEL))

RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS



## === cell 6
train_df = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
train_df["label"] = train_df["label"].astype(int)
train_df["filepath"] = (TRAIN_DIR + train_df["image_id"].astype(str)).astype(str)

fps = train_df["filepath"].to_numpy()
exists_mask = np.fromiter(
    (os.path.exists(p) for p in fps), dtype=bool, count=fps.shape[0]
)
train_df = train_df.loc[exists_mask].reset_index(drop=True)

print(
    "Train rows:",
    len(train_df),
    "Unique labels:",
    sorted(train_df["label"].unique().tolist()),
)



## === cell 7
sample_sub = pd.read_csv(os.path.join(BASE_DIR, "sample_submission.csv"))
test_df = sample_sub[["image_id"]].copy()
test_df["filepath"] = (TEST_DIR + test_df["image_id"].astype(str)).astype(str)

fps = test_df["filepath"].to_numpy()
exists_mask = np.fromiter(
    (os.path.exists(p) for p in fps), dtype=bool, count=fps.shape[0]
)
test_df = test_df.loc[exists_mask].reset_index(drop=True)

print("Test rows:", len(test_df))




## === cell 8
@tf.function
def decode_and_resize(path, label=None, training=False):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH], method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0

    if training:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, max_delta=0.12)
        img = tf.image.random_contrast(img, lower=0.85, upper=1.15)

    if label is None:
        return img
    return img, tf.cast(label, tf.int32)


def stratified_split_df(df, label_col="label", test_size=0.2, seed=42):
    rng = np.random.RandomState(seed)
    train_idx = []
    valid_idx = []
    for lab, g in df.groupby(label_col):
        idx = g.index.values.copy()
        rng.shuffle(idx)
        n_valid = int(np.floor(len(idx) * test_size))
        valid_idx.extend(idx[:n_valid].tolist())
        train_idx.extend(idx[n_valid:].tolist())
    return df.loc[train_idx].reset_index(drop=True), df.loc[valid_idx].reset_index(
        drop=True
    )


train_part, valid_part = stratified_split_df(
    train_df, label_col="label", test_size=0.2, seed=42
)

train_idx_set = set(train_part.index.tolist())
valid_idx_set = set(valid_part.index.tolist())

AUTOTUNE = tf.data.AUTOTUNE

options = tf.data.Options()
options.experimental_deterministic = True
try:
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.OFF
    )
except Exception:
    pass

TFREC_TRAIN_DIR = os.path.join(BASE_DIR, "train_tfrecords")
TFREC_TEST_DIR = os.path.join(BASE_DIR, "test_tfrecords")
assert os.path.isdir(TFREC_TRAIN_DIR), f"train_tfrecords not found: {TFREC_TRAIN_DIR}"
assert os.path.isdir(TFREC_TEST_DIR), f"test_tfrecords not found: {TFREC_TEST_DIR}"

train_tfrecs = sorted(
    [
        os.path.join(TFREC_TRAIN_DIR, f)
        for f in os.listdir(TFREC_TRAIN_DIR)
        if f.endswith(".tfrec")
    ]
)
test_tfrecs = sorted(
    [
        os.path.join(TFREC_TEST_DIR, f)
        for f in os.listdir(TFREC_TEST_DIR)
        if f.endswith(".tfrec")
    ]
)
assert len(train_tfrecs) > 0, "No train tfrecords found"
assert len(test_tfrecs) > 0, "No test tfrecords found"

FEATURES_TRAIN = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
    "class": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
}
FEATURES_TEST = {
    "image": tf.io.FixedLenFeature([], tf.string),
}


@tf.function
def _parse_train_example(ex):
    x = tf.io.parse_single_example(ex, FEATURES_TRAIN)
    img = tf.image.decode_jpeg(x["image"], channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH], method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0

    y_class = tf.cast(x["class"], tf.int32)
    y_label = tf.cast(x["label"], tf.int32)
    y = tf.where(y_class >= 0, y_class, y_label)  # prefer "class" if present
    return img, y


@tf.function
def _parse_test_example(ex):
    x = tf.io.parse_single_example(ex, FEATURES_TEST)
    img = tf.image.decode_jpeg(x["image"], channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH], method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    return img


@tf.function
def _augment_img(img, y):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, max_delta=0.12)
    img = tf.image.random_contrast(img, lower=0.85, upper=1.15)
    return img, y


ds_all = tf.data.TFRecordDataset(
    train_tfrecs, num_parallel_reads=AUTOTUNE
).with_options(options)
ds_all = ds_all.map(_parse_train_example, num_parallel_calls=AUTOTUNE)

train_mask = tf.constant(
    np.array([i in train_idx_set for i in range(len(train_df))], dtype=np.bool_),
    dtype=tf.bool,
)
valid_mask = tf.logical_not(train_mask)

ds_all_enum = ds_all.enumerate()  # (idx, (img, y))

ds_train = ds_all_enum.filter(
    lambda idx, xy: tf.gather(train_mask, tf.cast(idx, tf.int32))
)
ds_valid = ds_all_enum.filter(
    lambda idx, xy: tf.gather(valid_mask, tf.cast(idx, tf.int32))
)

ds_train = ds_train.map(lambda idx, xy: xy, num_parallel_calls=AUTOTUNE)
ds_valid = ds_valid.map(lambda idx, xy: xy, num_parallel_calls=AUTOTUNE)

ds_train = ds_train.cache()
ds_train = ds_train.shuffle(2048, seed=42, reshuffle_each_iteration=True)
ds_train = ds_train.map(_augment_img, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)

ds_valid = ds_valid.cache()
ds_valid = ds_valid.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)

ds_test_tfr = tf.data.TFRecordDataset(
    test_tfrecs, num_parallel_reads=AUTOTUNE
).with_options(options)
ds_test_tfr = ds_test_tfr.map(_parse_test_example, num_parallel_calls=AUTOTUNE)
ds_test_tfr = ds_test_tfr.cache()

try:
    n_train_tfr = int(tf.data.experimental.cardinality(ds_all).numpy())
    print("TFRecord train count:", n_train_tfr, "CSV train count:", len(train_df))
except Exception as e:
    print("Could not compute TFRecord train count:", repr(e))

try:
    n_test_tfr = int(tf.data.experimental.cardinality(ds_test_tfr).numpy())
    print(
        "TFRecord test count:", n_test_tfr, "Sample submission count:", len(sample_sub)
    )
except Exception as e:
    print("Could not compute TFRecord test count:", repr(e))



## === cell 9
inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.25)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()



## === cell 10
EPOCHS = 8
history = model.fit(ds_train, validation_data=ds_valid, epochs=EPOCHS, verbose=2)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/1044257117.py in <cell line: 0>()
      1 EPOCHS = 8
----> 2 history = model.fit(ds_train, validation_data=ds_valid, epochs=EPOCHS, verbose=2)
      3 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/execute.py in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
     57       e.message += " name: " + name
     58     raise core._status_to_exception(e) from None
---> 59   except TypeError as e:
     60     keras_symbolic_tensors = [x for x in inputs if _is_keras_symbolic_tensor(x)]
     61     if keras_symbolic_tensors:

InvalidArgumentError: Graph execution error:

Detected at node compile_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits defined at (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main

  File "<frozen runpy>", line 88, in _run_code

  File "/usr/local/lib/python3.11/dist-packages/colab_kernel_launcher.py", line 37, in <module>

  File "/usr/local/lib/python3.11/dist-packages/traitlets/config/application.py", line 992, in launch_instance

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelapp.py", line 712, in start

  File "/usr/local/lib/python3.11/dist-packages/tornado/platform/asyncio.py", line 211, in start

  File "/usr/lib/python3.11/asyncio/base_events.py", line 608, in run_forever

  File "/usr/lib/python3.11/asyncio/base_events.py", line 1936, in _run_once

  File "/usr/lib/python3.11/asyncio/events.py", line 84, in _run

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 510, in dispatch_queue

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 499, in process_one

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 406, in dispatch_shell

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 730, in execute_request

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/ipkernel.py", line 383, in do_execute

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/zmqshell.py", line 528, in run_cell

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 2975, in run_cell

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3030, in _run_cell

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/async_helpers.py", line 78, in _pseudo_sync_runner

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3257, in run_cell_async

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3473, in run_ast_nodes

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3553, in run_code

  File "/tmp/ipykernel_11/1044257117.py", line 2, in <cell line: 0>

  File "/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py", line 117, in error_handler

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 371, in fit

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 219, in function

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 132, in multi_step_on_iterator

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 113, in one_step_on_data

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 60, in train_step

  File "/usr/local/lib/python3.11/dist-packages/keras/src/trainers/trainer.py", line 383, in _compute_loss

  File "/usr/local/lib/python3.11/dist-packages/keras/src/trainers/trainer.py", line 351, in compute_loss

  File "/usr/local/lib/python3.11/dist-packages/keras/src/trainers/compile_utils.py", line 691, in __call__

  File "/usr/local/lib/python3.11/dist-packages/keras/src/trainers/compile_utils.py", line 700, in call

  File "/usr/local/lib/python3.11/dist-packages/keras/src/losses/loss.py", line 67, in __call__

  File "/usr/local/lib/python3.11/dist-packages/keras/src/losses/losses.py", line 33, in call

  File "/usr/local/lib/python3.11/dist-packages/keras/src/losses/losses.py", line 2246, in sparse_categorical_crossentropy

  File "/usr/local/lib/python3.11/dist-packages/keras/src/ops/nn.py", line 1963, in sparse_categorical_crossentropy

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/nn.py", line 744, in sparse_categorical_crossentropy

Received a label value of -1 which is outside the valid range of [0, 5).  Label values: -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
	 [[{{node compile_loss/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits}}]] [Op:__inference_multi_step_on_iterator_1972]

## === cell 11
GLOBAL_SEED = 42
N_AUG = 5
N_VIEWS = 1 + N_AUG


@tf.function
def _seed2(a, b):
    return tf.stack([tf.cast(a, tf.int32), tf.cast(b, tf.int32)], axis=0)


@tf.function
def _tta_views_batch(imgs, idxs):
    imgs = tf.convert_to_tensor(imgs, tf.float32)
    idxs = tf.cast(idxs, tf.int32)
    b = tf.shape(imgs)[0]

    views0 = imgs[:, None, :, :, :]  # [B,1,H,W,3]
    out = tf.broadcast_to(imgs[:, None, :, :, :], [b, N_AUG, IMG_HEIGHT, IMG_WIDTH, 3])

    js = tf.range(N_AUG, dtype=tf.int32)[None, :]  # [1, N_AUG]
    idxs2 = idxs[:, None]  # [B,1]
    seed_second = idxs2 * 1000 + js  # [B,N_AUG] unique per (image, aug)

    s1 = _seed2(GLOBAL_SEED + 1, seed_second + 11)
    r = tf.random.stateless_uniform([b, N_AUG], seed=s1)
    do = r < 0.5
    out = tf.where(do[:, :, None, None, None], tf.reverse(out, axis=[3]), out)

    s2 = _seed2(GLOBAL_SEED + 2, seed_second + 22)
    r = tf.random.stateless_uniform([b, N_AUG], seed=s2)
    do = r < 0.2
    out = tf.where(do[:, :, None, None, None], tf.reverse(out, axis=[2]), out)

    s3 = _seed2(GLOBAL_SEED + 3, seed_second + 33)
    r = tf.random.stateless_uniform([b, N_AUG], seed=s3)
    do = r < 0.5
    s4 = _seed2(GLOBAL_SEED + 4, seed_second + 44)
    delta = tf.random.stateless_uniform([b, N_AUG], seed=s4, minval=-0.12, maxval=0.12)
    out = tf.where(
        do[:, :, None, None, None],
        tf.clip_by_value(out + delta[:, :, None, None, None], 0.0, 1.0),
        out,
    )

    s5 = _seed2(GLOBAL_SEED + 5, seed_second + 55)
    r = tf.random.stateless_uniform([b, N_AUG], seed=s5)
    do = r < 0.5
    s6 = _seed2(GLOBAL_SEED + 6, seed_second + 66)
    c = tf.random.stateless_uniform([b, N_AUG], seed=s6, minval=0.85, maxval=1.15)
    mean = tf.reduce_mean(out, axis=[2, 3], keepdims=True)  # [B,N_AUG,1,1,3]
    out_contrast = tf.clip_by_value(
        (out - mean) * c[:, :, None, None, None] + mean, 0.0, 1.0
    )
    out = tf.where(do[:, :, None, None, None], out_contrast, out)

    views = tf.concat([views0, out], axis=1)  # [B,N_VIEWS,H,W,3]
    return tf.reshape(views, [b * N_VIEWS, IMG_HEIGHT, IMG_WIDTH, 3])


def _collect_test_images_from_tfrecord(ds, n_expected):
    imgs = []
    for img in ds:
        imgs.append(img.numpy())
    assert len(imgs) == n_expected, (len(imgs), n_expected)
    return np.stack(imgs, axis=0).astype(np.float32, copy=False)


test_imgs_np = _collect_test_images_from_tfrecord(ds_test_tfr, len(test_df))
pred_ids = test_df["image_id"].astype(str).tolist()

ds_test_imgs = tf.data.Dataset.from_tensor_slices(test_imgs_np).with_options(options)
ds_test_imgs = ds_test_imgs.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
ds_test_imgs = ds_test_imgs.enumerate()  # (batch_idx, img_batch)


@tf.function
def _batch_img_to_views(batch_idx, imgs):
    b = tf.shape(imgs)[0]
    start = tf.cast(batch_idx, tf.int32) * tf.cast(batch_size, tf.int32)
    idxs = start + tf.range(b, dtype=tf.int32)  # global per-image indices
    views_flat = _tta_views_batch(imgs, idxs)  # [B*N_VIEWS,H,W,3]
    return views_flat


ds_views_flat = ds_test_imgs.map(
    lambda bi, imgs: _batch_img_to_views(bi, imgs), num_parallel_calls=AUTOTUNE
)
ds_views_flat = ds_views_flat.prefetch(AUTOTUNE)

preds_flat = model.predict(ds_views_flat, verbose=0)

n_images = len(pred_ids)
assert preds_flat.shape[0] == n_images * N_VIEWS, (preds_flat.shape, n_images, N_VIEWS)

preds = preds_flat.reshape(n_images, N_VIEWS, NUM_CLASSES)
preds_mean = preds.mean(axis=1)
pred_labels = preds_mean.argmax(axis=1).astype(np.int32, copy=False).tolist()

submission = pd.DataFrame({"image_id": pred_ids, "label": pred_labels})
submission = sample_sub[["image_id"]].merge(submission, on="image_id", how="left")
assert submission["label"].notnull().all(), "Some test image_ids were not predicted."
submission["label"] = submission["label"].astype(int)

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print("Wrote:", submission_path)
print(submission.head())
print(submission.shape)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1718736927.py in <cell line: 0>()
     85 
     86 
---> 87 ds_views_flat = ds_test_imgs.map(
     88     lambda bi, imgs: _batch_img_to_views(bi, imgs), num_parallel_calls=AUTOTUNE
     89 )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in map(self, map_func, num_parallel_calls, deterministic, synchronous, use_unbounded_threadpool, name)
   2339     from tensorflow.python.data.ops import map_op
   2340 
-> 2341     return map_op._map_v2(
   2342         self,
   2343         map_func,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/map_op.py in _map_v2(input_dataset, map_func, num_parallel_calls, deterministic, synchronous, use_unbounded_threadpool, name)
     55           num_parallel_calls,
     56       )
---> 57     return _ParallelMapDataset(
     58         input_dataset,
     59         map_func,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/map_op.py in __init__(self, input_dataset, map_func, num_parallel_calls, deterministic, use_inter_op_parallelism, preserve_cardinality, use_legacy_function, use_unbounded_threadpool, name)
    200     self._input_dataset = input_dataset
    201     self._use_inter_op_parallelism = use_inter_op_parallelism
--> 202     self._map_func = structured_function.StructuredFunctionWrapper(
    203         map_func,
    204         self._transformation_name(),

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in __init__(self, func, transformation_name, dataset, input_classes, input_shapes, input_types, input_structure, add_to_graph, use_legacy_function, defun_kwargs)
    263         fn_factory = trace_tf_function(defun_kwargs)
    264 
--> 265     self._function = fn_factory()
    266     # There is no graph to add in eager mode.
    267     add_to_graph &= not context.executing_eagerly()

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in get_concrete_function(self, *args, **kwargs)
   1249   def get_concrete_function(self, *args, **kwargs):
   1250     # Implements PolymorphicFunction.get_concrete_function.
-> 1251     concrete = self._get_concrete_function_garbage_collected(*args, **kwargs)
   1252     concrete._garbage_collector.release()  # pylint: disable=protected-access
   1253     return concrete

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in _get_concrete_function_garbage_collected(self, *args, **kwargs)
   1219       if self._variable_creation_config is None:
   1220         initializers = []
-> 1221         self._initialize(args, kwargs, add_initializers_to=initializers)
   1222         self._initialize_uninitialized_variables(initializers)
   1223 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in _initialize(self, args, kwds, add_initializers_to)
    694     )
    695     # Force the definition of the function for these arguments
--> 696     self._concrete_variable_creation_fn = tracing_compilation.trace_function(
    697         args, kwds, self._variable_creation_config
    698     )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in trace_function(args, kwargs, tracing_options)
    176       kwargs = {}
    177 
--> 178     concrete_function = _maybe_define_function(
    179         args, kwargs, tracing_options
    180     )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in _maybe_define_function(args, kwargs, tracing_options)
    281         else:
    282           target_func_type = lookup_func_type
--> 283         concrete_function = _create_concrete_function(
    284             target_func_type, lookup_func_context, func_graph, tracing_options
    285         )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in _create_concrete_function(function_type, type_context, func_graph, tracing_options)
    308       attributes_lib.DISABLE_ACD, False
    309   )
--> 310   traced_func_graph = func_graph_module.func_graph_from_py_func(
    311       tracing_options.name,
    312       tracing_options.python_function,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/func_graph.py in func_graph_from_py_func(name, python_func, args, kwargs, signature, func_graph, add_control_dependencies, arg_names, op_return_value, collections, capture_by_value, create_placeholders)
   1057 
   1058     _, original_func = tf_decorator.unwrap(python_func)
-> 1059     func_outputs = python_func(*func_args, **func_kwargs)
   1060 
   1061     # invariant: `func_outputs` contains only Tensors, CompositeTensors,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in wrapped_fn(*args, **kwds)
    597         # the function a weak reference to itself to avoid a reference cycle.
    598         with OptionalXlaContext(compile_with_xla):
--> 599           out = weak_wrapped_fn().__wrapped__(*args, **kwds)
    600         return out
    601 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in wrapped_fn(*args)
    229       # Note: wrapper_helper will apply autograph based on context.
    230       def wrapped_fn(*args):  # pylint: disable=missing-docstring
--> 231         ret = wrapper_helper(*args)
    232         ret = structure.to_tensor_list(self._output_structure, ret)
    233         return [ops.convert_to_tensor(t) for t in ret]

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in wrapper_helper(*args)
    159       if not _should_unpack(nested_args):
    160         nested_args = (nested_args,)
--> 161       ret = autograph.tf_convert(self._func, ag_ctx)(*nested_args)
    162       ret = variable_utils.convert_variables_to_tensors(ret)
    163       if _should_pack(ret):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in wrapper(*args, **kwargs)
    691       except Exception as e:  # pylint:disable=broad-except
    692         if hasattr(e, 'ag_error_metadata'):
--> 693           raise e.ag_error_metadata.to_exception(e)
    694         else:
    695           raise

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in wrapper(*args, **kwargs)
    688       try:
    689         with conversion_ctx:
--> 690           return converted_call(f, args, kwargs, options=options)
    691       except Exception as e:  # pylint:disable=broad-except
    692         if hasattr(e, 'ag_error_metadata'):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    437     try:
    438       if kwargs is not None:
--> 439         result = converted_f(*effective_args, **kwargs)
    440       else:
    441         result = converted_f(*effective_args)

/tmp/__autograph_generated_filen3vybgiv.py in <lambda>(bi, imgs)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda bi, imgs: ag__.with_function_scope(lambda lscope: ag__.converted_call(_batch_img_to_views, (bi, imgs), None, lscope), 'lscope', ag__.STD)
      6         return tf__lam
      7     return inner_factory

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/function_wrappers.py in with_function_scope(thunk, scope_name, options)
    111   """Inline version of the FunctionScope context manager."""
    112   with FunctionScope('lambda_', scope_name, options) as scope:
--> 113     return thunk(scope)

/tmp/__autograph_generated_filen3vybgiv.py in <lambda>(lscope)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda bi, imgs: ag__.with_function_scope(lambda lscope: ag__.converted_call(_batch_img_to_views, (bi, imgs), None, lscope), 'lscope', ag__.STD)
      6         return tf__lam
      7     return inner_factory

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    375 
    376   if not options.user_requested and conversion.is_allowlisted(f):
--> 377     return _call_unconverted(f, args, kwargs, options)
    378 
    379   # internal_convert_user_code is for example turned off when issuing a dynamic

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in _call_unconverted(f, args, kwargs, options, update_cache)
    458   if kwargs is not None:
    459     return f(*args, **kwargs)
--> 460   return f(*args)
    461 
    462 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/tmp/__autograph_generated_filejqyhddax.py in tf___batch_img_to_views(batch_idx, imgs)
     11                 start = ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(batch_idx), ag__.ld(tf).int32), None, fscope) * ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(batch_size), ag__.ld(tf).int32), None, fscope)
     12                 idxs = ag__.ld(start) + ag__.converted_call(ag__.ld(tf).range, (ag__.ld(b),), dict(dtype=ag__.ld(tf).int32), fscope)
---> 13                 views_flat = ag__.converted_call(ag__.ld(_tta_views_batch), (ag__.ld(imgs), ag__.ld(idxs)), None, fscope)
     14                 try:
     15                     do_return = True

/tmp/__autograph_generated_filenufoch4p.py in tf___tta_views_batch(imgs, idxs)
     16                 idxs2 = ag__.ld(idxs)[:, None]
     17                 seed_second = ag__.ld(idxs2) * 1000 + ag__.ld(js)
---> 18                 s1 = ag__.converted_call(ag__.ld(_seed2), (ag__.ld(GLOBAL_SEED) + 1, ag__.ld(seed_second) + 11), None, fscope)
     19                 r = ag__.converted_call(ag__.ld(tf).random.stateless_uniform, ([ag__.ld(b), ag__.ld(N_AUG)],), dict(seed=ag__.ld(s1)), fscope)
     20                 do = ag__.ld(r) < 0.5

/tmp/__autograph_generated_file_b9bbfj1.py in tf___seed2(a, b)
     13                 except:
     14                     do_return = False
---> 15                     raise
     16                 return fscope.ret(retval_, do_return)
     17         return tf___seed2

ValueError: in user code:

    File "/tmp/ipykernel_11/1718736927.py", line 88, in None  *
        lambda bi, imgs: _batch_img_to_views(bi, imgs)
    File "/tmp/ipykernel_11/1718736927.py", line 83, in _batch_img_to_views  *
        views_flat = _tta_views_batch(imgs, idxs)  # [B*N_VIEWS,H,W,3]
    File "/tmp/ipykernel_11/1718736927.py", line 26, in _tta_views_batch  *
        s1 = _seed2(GLOBAL_SEED + 1, seed_second + 11)
    File "/tmp/ipykernel_11/1718736927.py", line 9, in _seed2  *
        return tf.stack([tf.cast(a, tf.int32), tf.cast(b, tf.int32)], axis=0)

    ValueError: Shapes must be equal rank, but are 0 and 2
    	From merging shape 0 with other shapes. for '{{node stack_1}} = Pack[N=2, T=DT_INT32, axis=0](Cast/x, b)' with input shapes: [], [?,5].


## === cell 12
chk = pd.read_csv("submission.csv")
assert list(chk.columns) == ["image_id", "label"], chk.columns
assert len(chk) == len(sample_sub), (len(chk), len(sample_sub))
assert (
    chk["image_id"].astype(str).tolist() == sample_sub["image_id"].astype(str).tolist()
)
assert chk["label"].between(0, NUM_CLASSES - 1).all()
print(chk.head(3))
print("Submission OK:", chk.shape)

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/722539377.py in <cell line: 0>()
----> 1 chk = pd.read_csv("submission.csv")
      2 assert list(chk.columns) == ["image_id", "label"], chk.columns
      3 assert len(chk) == len(sample_sub), (len(chk), len(sample_sub))
      4 assert (
      5     chk["image_id"].astype(str).tolist() == sample_sub["image_id"].astype(str).tolist()

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: 'submission.csv'
