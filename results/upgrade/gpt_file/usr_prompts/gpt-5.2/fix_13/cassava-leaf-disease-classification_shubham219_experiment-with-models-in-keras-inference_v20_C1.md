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

0.710486551828347

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

for k in [
    "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION",
    "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION",
]:
    if k in os.environ:
        del os.environ[k]

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", str(os.cpu_count() or 4))
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")

import glob
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras import layers

SEED = 42
DEBUG = False

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

BASE_PATH = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(BASE_PATH, "train_images")
TEST_IMG_DIR = os.path.join(BASE_PATH, "test_images")
TRAIN_TFREC_DIR = os.path.join(BASE_PATH, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(BASE_PATH, "test_tfrecords")

print("TF version:", tf.__version__)
print("Base path exists:", os.path.exists(BASE_PATH))
print("Train CSV exists:", os.path.exists(TRAIN_CSV))
print("Sample submission exists:", os.path.exists(SAMPLE_SUB))
print("Train img dir exists:", os.path.exists(TRAIN_IMG_DIR))
print("Test img dir exists:", os.path.exists(TEST_IMG_DIR))
print("Train tfrec dir exists:", os.path.exists(TRAIN_TFREC_DIR))
print("Test tfrec dir exists:", os.path.exists(TEST_TFREC_DIR))

assert os.path.isdir(TRAIN_IMG_DIR), "TRAIN_IMG_DIR not found"
assert os.path.isdir(TEST_IMG_DIR), "TEST_IMG_DIR not found"



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
IMG_SIZE = (300, 300)
NUM_CLASSES = 5
BATCH_SIZE = 32
EPOCHS = 2  # keep as-is per original core logic

df_train = pd.read_csv(TRAIN_CSV)
df_train["label"] = df_train["label"].astype(str)
df_train["path"] = TRAIN_IMG_DIR.rstrip("/") + "/" + df_train["image_id"].astype(str)
df_train = df_train.reset_index(drop=True)

idx = np.arange(len(df_train))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
split = int(0.9 * len(idx))
train_idx, val_idx = idx[:split], idx[split:]
train_df = df_train.iloc[train_idx].reset_index(drop=True)
val_df = df_train.iloc[val_idx].reset_index(drop=True)

print("Train/Val sizes:", len(train_df), len(val_df))
print("Train label sample:", train_df["label"].value_counts().sort_index().to_dict())

AUTOTUNE = tf.data.AUTOTUNE


def _decode_and_resize_bytes(img_bytes):
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    return img


def _decode_and_resize(path):
    img = tf.io.read_file(path)
    return _decode_and_resize_bytes(img)


_ROT_MAX = 10.0 * np.pi / 180.0
_WSHIFT = 0.05
_HSHIFT = 0.05
_ZOOM = 0.1


def _augment_fast(img, seed):
    seed = tf.cast(seed, tf.int32)  # shape [2]

    img = tf.image.stateless_random_flip_left_right(img, seed=seed)

    s0, s1, s2, s3 = tf.unstack(
        tf.random.experimental.stateless_split(seed, num=4), axis=0
    )

    h = tf.cast(tf.shape(img)[0], tf.float32)
    w = tf.cast(tf.shape(img)[1], tf.float32)

    dx = tf.random.stateless_uniform((), s0, -_WSHIFT, _WSHIFT, dtype=tf.float32) * w
    dy = tf.random.stateless_uniform((), s1, -_HSHIFT, _HSHIFT, dtype=tf.float32) * h
    z = tf.random.stateless_uniform((), s2, 1.0 - _ZOOM, 1.0 + _ZOOM, dtype=tf.float32)
    ang = tf.random.stateless_uniform((), s3, -_ROT_MAX, _ROT_MAX, dtype=tf.float32)

    cx = (w - 1.0) / 2.0
    cy = (h - 1.0) / 2.0

    cos_a = tf.cos(ang)
    sin_a = tf.sin(ang)
    inv_z = 1.0 / z
    a0 = cos_a * inv_z
    a1 = -sin_a * inv_z
    a3 = sin_a * inv_z
    a4 = cos_a * inv_z

    tx = cx - (a0 * cx + a1 * cy) - dx
    ty = cy - (a3 * cx + a4 * cy) - dy

    transform = tf.stack([a0, a1, tx, a3, a4, ty, 0.0, 0.0])
    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[None, ...],
        transforms=transform[None, ...],
        output_shape=tf.constant([IMG_SIZE[0], IMG_SIZE[1]], dtype=tf.int32),
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )[0]
    return img


_data_opts = tf.data.Options()
_data_opts.experimental_deterministic = True
try:
    _data_opts.experimental_slack = True
except Exception:
    pass
try:
    _data_opts.threading.private_threadpool_size = max(8, (os.cpu_count() or 8))
except Exception:
    pass

_TFREC_IMG_KEY = "image"
_TFREC_LABEL_KEY = "target"


def _parse_train_tfrecord(example_proto):
    feat = {
        _TFREC_IMG_KEY: tf.io.FixedLenFeature([], tf.string),
        _TFREC_LABEL_KEY: tf.io.FixedLenFeature([], tf.int64),
    }
    x = tf.io.parse_single_example(example_proto, feat)
    img = _decode_and_resize_bytes(x[_TFREC_IMG_KEY])
    label = tf.cast(x[_TFREC_LABEL_KEY], tf.int32)
    return img, label


def _parse_test_tfrecord(example_proto):
    feat = {_TFREC_IMG_KEY: tf.io.FixedLenFeature([], tf.string)}
    x = tf.io.parse_single_example(example_proto, feat)
    img = _decode_and_resize_bytes(x[_TFREC_IMG_KEY])
    return img


def _tfrecord_files(folder, pattern="*.tfrec"):
    files = sorted(glob.glob(os.path.join(folder, pattern)))
    if not files:
        files = sorted(glob.glob(os.path.join(folder, "*.tfrecord")))
    return files


def make_train_val_datasets(train_df, val_df, batch_size):
    train_files = _tfrecord_files(TRAIN_TFREC_DIR)
    assert train_files, "No TFRecord files found in train_tfrecords."
    n_files = len(train_files)
    split_files = int(0.9 * n_files)
    tr_files = train_files[:split_files]
    va_files = train_files[split_files:] if split_files < n_files else train_files[-1:]

    n_train = len(train_df)
    n_val = len(val_df)

    ds_train = tf.data.TFRecordDataset(
        tr_files, num_parallel_reads=AUTOTUNE, compression_type="GZIP"
    )
    ds_train = ds_train.map(
        _parse_train_tfrecord, num_parallel_calls=AUTOTUNE, deterministic=True
    )
    ds_train = ds_train.shuffle(8192, seed=SEED, reshuffle_each_iteration=True)

    ds_train = ds_train.enumerate()
    ds_train = ds_train.map(
        lambda i, xy: (
            _augment_fast(
                xy[0], tf.stack([tf.cast(i, tf.int32), tf.cast(SEED, tf.int32)])
            ),
            xy[1],
        ),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )

    ds_train = ds_train.batch(batch_size, drop_remainder=True).prefetch(AUTOTUNE)
    ds_train = ds_train.with_options(_data_opts)

    ds_val = tf.data.TFRecordDataset(
        va_files, num_parallel_reads=AUTOTUNE, compression_type="GZIP"
    )
    ds_val = ds_val.map(
        _parse_train_tfrecord, num_parallel_calls=AUTOTUNE, deterministic=True
    )
    ds_val = ds_val.cache()
    ds_val = ds_val.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    ds_val = ds_val.with_options(_data_opts)

    steps_per_epoch = n_train // batch_size
    validation_steps = int(np.ceil(n_val / batch_size))

    return ds_train, ds_val, steps_per_epoch, validation_steps


train_ds, val_ds, STEPS_PER_EPOCH, VAL_STEPS = make_train_val_datasets(
    train_df, val_df, BATCH_SIZE
)

unique_labels = set(train_df["label"].unique().tolist())
assert unique_labels == set(
    map(str, range(NUM_CLASSES))
), "Unexpected number of classes parsed."

base = EfficientNetB3(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    pooling="avg",
)
base.trainable = False

inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base(inputs, training=False)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
my_model = Model(inputs, outputs)

my_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

my_model.summary()

history = my_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=VAL_STEPS,
    verbose=1,
)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
DataLossError                             Traceback (most recent call last)
/tmp/ipykernel_11/2739520978.py in <cell line: 0>()
    215 my_model.summary()
    216 
--> 217 history = my_model.fit(
    218     train_ds,
    219     validation_data=val_ds,

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

DataLossError: Graph execution error:

Detected at node IteratorGetNext defined at (most recent call last):
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

  File "/tmp/ipykernel_11/2739520978.py", line 217, in <cell line: 0>

  File "/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py", line 117, in error_handler

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 371, in fit

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 219, in function

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 132, in multi_step_on_iterator

inflate() failed with error -3: incorrect header check
	 [[{{node IteratorGetNext}}]] [Op:__inference_multi_step_on_iterator_25058]

## === cell 2
df_sub = pd.read_csv(SAMPLE_SUB)
df_test = df_sub.copy()
df_test["path"] = TEST_IMG_DIR.rstrip("/") + "/" + df_test["image_id"].astype(str)

print("Test rows:", len(df_test))


def make_test_gen(batch_size=64):
    test_files = _tfrecord_files(TEST_TFREC_DIR)
    assert test_files, "No TFRecord files found in test_tfrecords."
    ds_test = tf.data.TFRecordDataset(
        test_files, num_parallel_reads=AUTOTUNE, compression_type="GZIP"
    )
    ds_test = ds_test.map(
        _parse_test_tfrecord, num_parallel_calls=AUTOTUNE, deterministic=True
    )
    ds_test = ds_test.cache()
    ds_test = ds_test.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    ds_test = ds_test.with_options(_data_opts)

    n_test = len(df_test)
    test_steps = int(np.ceil(n_test / batch_size))
    return ds_test, test_steps




## === cell 3
test_gen, TEST_STEPS = make_test_gen(batch_size=128)

pred_test = my_model.predict(
    test_gen,
    steps=TEST_STEPS,
    verbose=1,
)

pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

final_csv = df_test[["image_id"]].copy()
final_csv["label"] = pred_test_labels

assert len(final_csv) == len(df_sub), "Submission row count mismatch."
assert list(final_csv.columns) == ["image_id", "label"], "Submission columns mismatch."
assert final_csv["label"].between(0, 4).all(), "Predicted labels out of range [0,4]."

final_csv.to_csv("submission.csv", index=False)
print(final_csv.head())
print("Wrote submission.csv with", len(final_csv), "rows")



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
DataLossError                             Traceback (most recent call last)
/tmp/ipykernel_11/2814656694.py in <cell line: 0>()
      1 test_gen, TEST_STEPS = make_test_gen(batch_size=128)
      2 
----> 3 pred_test = my_model.predict(
      4     test_gen,
      5     steps=TEST_STEPS,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

DataLossError: {{function_node __wrapped__IteratorGetNext_output_types_1_device_/job:localhost/replica:0/task:0/device:CPU:0}} inflate() failed with error -3: incorrect header check [Op:IteratorGetNext] name: 

## === cell 4
print(pd.read_csv("submission.csv").head())
print(
    "submission.csv exists:",
    os.path.exists("submission.csv"),
    "size:",
    os.path.getsize("submission.csv") if os.path.exists("submission.csv") else None,
)

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/5545704.py in <cell line: 0>()
----> 1 print(pd.read_csv("submission.csv").head())
      2 print(
      3     "submission.csv exists:",
      4     os.path.exists("submission.csv"),
      5     "size:",

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
