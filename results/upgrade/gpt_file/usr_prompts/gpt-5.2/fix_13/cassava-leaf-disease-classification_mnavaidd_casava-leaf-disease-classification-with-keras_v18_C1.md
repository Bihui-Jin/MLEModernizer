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

0.7719854941069809

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.05531) has done: 'The timeout is dominated by slow Python-side image loading/augmentation in `ImageDataGenerator.flow_from_dataframe` plus expensive augmentation settings, causing the GPU/CPU to idle while waiting on batches. I keep the same model and training loop, but switch the input pipeline to `tf.data` reading from the provided TFRecords (same images/labels), using parallel map, prefetch, and deterministic options to eliminate the generator bottleneck while preserving the same augmentation semantics as closely as TensorFlow provides. I also remove redundant `steps_per_epoch/validation_steps` (Keras can infer them) and ensure we cache/prefetch appropriately to reduce per-epoch overhead. All paths remain unchanged, and seeds/determinism are kept.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization,
    GlobalAveragePooling2D,
)

print("TensorFlow:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/1786389515.py in <cell line: 0>()
     12 import matplotlib.pyplot as plt
     13 
---> 14 import tensorflow as tf
     15 from tensorflow.keras import applications
     16 from tensorflow.keras.preprocessing.image import ImageDataGenerator

/usr/local/lib/python3.11/dist-packages/tensorflow/__init__.py in <module>
     47 _tf2.enable()
     48 
---> 49 from tensorflow._api.v2 import __internal__
     50 from tensorflow._api.v2 import __operators__
     51 from tensorflow._api.v2 import audio

/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/__init__.py in <module>
      6 import sys as _sys
      7 
----> 8 from tensorflow._api.v2.__internal__ import autograph
      9 from tensorflow._api.v2.__internal__ import decorator
     10 from tensorflow._api.v2.__internal__ import dispatch

/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/autograph/__init__.py in <module>
      6 import sys as _sys
      7 
----> 8 from tensorflow.python.autograph.core.ag_ctx import control_status_ctx # line: 34
      9 from tensorflow.python.autograph.impl.api import tf_convert # line: 493

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/ag_ctx.py in <module>
     19 import threading
     20 
---> 21 from tensorflow.python.autograph.utils import ag_logging
     22 from tensorflow.python.util.tf_export import tf_export
     23 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/__init__.py in <module>
     15 """Utility module that contains APIs usable in the generated code."""
     16 
---> 17 from tensorflow.python.autograph.utils.context_managers import control_dependency_on_returns
     18 from tensorflow.python.autograph.utils.misc import alias_tensors
     19 from tensorflow.python.autograph.utils.tensor_list import dynamic_list_append

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/context_managers.py in <module>
     17 import contextlib
     18 
---> 19 from tensorflow.python.framework import ops
     20 from tensorflow.python.ops import tensor_array_ops
     21 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in <module>
     31 
     32 from google.protobuf import message
---> 33 from tensorflow.core.framework import attr_value_pb2
     34 from tensorflow.core.framework import full_type_pb2
     35 from tensorflow.core.framework import function_pb2

/usr/local/lib/python3.11/dist-packages/tensorflow/core/framework/attr_value_pb2.py in <module>
      3 # source: tensorflow/core/framework/attr_value.proto
      4 """Generated protocol buffer code."""
----> 5 from google.protobuf.internal import builder as _builder
      6 from google.protobuf import descriptor as _descriptor
      7 from google.protobuf import descriptor_pool as _descriptor_pool

/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/builder.py in <module>
     16 
     17 from google.protobuf.internal import enum_type_wrapper
---> 18 from google.protobuf.internal import python_message
     19 from google.protobuf import message as _message
     20 from google.protobuf import reflection as _reflection

/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/python_message.py in <module>
     36 import weakref
     37 
---> 38 from google.protobuf import descriptor as descriptor_mod
     39 from google.protobuf import message as message_mod
     40 from google.protobuf import text_format

/usr/local/lib/python3.11/dist-packages/google/protobuf/descriptor.py in <module>
     27   # TODO: Remove this import after fix api_implementation
     28   if _message is None:
---> 29     from google.protobuf.pyext import _message
     30   _USE_C_DESCRIPTORS = True
     31 

ImportError: cannot import name '_message' from 'google.protobuf.pyext' (/usr/local/lib/python3.11/dist-packages/google/protobuf/pyext/__init__.py)

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



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2957434096.py in <cell line: 0>()
      7 random.seed(SEED)
      8 np.random.seed(SEED)
----> 9 tf.random.set_seed(SEED)
     10 
     11 try:

NameError: name 'tf' is not defined

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
train_gen = ImageDataGenerator(
    rotation_range=360,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.1, 0.9],
    shear_range=25,
    zoom_range=0.3,
    channel_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    rescale=1 / 255,
    validation_split=0.15,
)

valid_gen = ImageDataGenerator(rescale=1 / 255, validation_split=0.15)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1487264624.py in <cell line: 0>()
----> 1 train_gen = ImageDataGenerator(
      2     rotation_range=360,
      3     width_shift_range=0.1,
      4     height_shift_range=0.1,
      5     brightness_range=[0.1, 0.9],

NameError: name 'ImageDataGenerator' is not defined

## === cell 8
BATCH_SIZE = 18
IMG_SIZE = 224
NUM_CLASSES = 5

AUTOTUNE = tf.data.AUTOTUNE
WORKERS = max(2, (os.cpu_count() or 4) - 1)
USE_MP = True
MAX_QUEUE = 32



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/304149889.py in <cell line: 0>()
      3 NUM_CLASSES = 5
      4 
----> 5 AUTOTUNE = tf.data.AUTOTUNE
      6 WORKERS = max(2, (os.cpu_count() or 4) - 1)
      7 USE_MP = True

NameError: name 'tf' is not defined

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
FEATURES_TRAIN_LABEL = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
}
FEATURES_TRAIN_TARGET = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
}

FEATURES_TEST = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


def _decode_resize(image_bytes):
    img = tf.io.decode_jpeg(image_bytes, channels=3)
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) / 255.0
    return img


def _parse_train(example_proto):
    ex1 = tf.io.parse_single_example(example_proto, FEATURES_TRAIN_LABEL)
    ex2 = tf.io.parse_single_example(example_proto, FEATURES_TRAIN_TARGET)

    img_bytes = ex1["image"]
    lbl1 = tf.cast(ex1["label"], tf.int32)
    lbl2 = tf.cast(ex2["target"], tf.int32)

    label = tf.where(lbl1 >= 0, lbl1, lbl2)

    img = _decode_resize(img_bytes)
    label_oh = tf.one_hot(label, NUM_CLASSES, dtype=tf.float32)
    return img, label_oh


def _parse_test(example_proto):
    ex = tf.io.parse_single_example(example_proto, FEATURES_TEST)
    img = _decode_resize(ex["image"])
    name = ex["image_name"]
    return img, name




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1901053380.py in <cell line: 0>()
      3 # We parse both possible keys and pick whichever is present.
      4 FEATURES_TRAIN_LABEL = {
----> 5     "image": tf.io.FixedLenFeature([], tf.string),
      6     "label": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
      7 }

NameError: name 'tf' is not defined

## === cell 11
@tf.function
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




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3786531832.py in <cell line: 0>()
----> 1 @tf.function
      2 def _augment(img, label):
      3     img = tf.image.random_flip_left_right(img, seed=SEED)
      4     img = tf.image.random_flip_up_down(img, seed=SEED + 1)
      5 

NameError: name 'tf' is not defined

## === cell 12
options = tf.data.Options()
options.experimental_deterministic = True

raw_train = tf.data.Dataset.from_tensor_slices(train_tfrec_files).with_options(options)
raw_train = raw_train.interleave(
    lambda f: tf.data.TFRecordDataset(f, num_parallel_reads=1),
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


def _is_valid_label(img, label_oh):
    return tf.equal(tf.reduce_sum(label_oh), 1.0)


train_ds = (
    train_records.map(_parse_train, num_parallel_calls=AUTOTUNE, deterministic=True)
    .filter(_is_valid_label)
    .map(_augment, num_parallel_calls=AUTOTUNE, deterministic=True)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

valid_ds = (
    val_records.map(_parse_train, num_parallel_calls=AUTOTUNE, deterministic=True)
    .filter(_is_valid_label)
    .cache()
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

train_batches = int(np.ceil(NUM_TRAIN / BATCH_SIZE))
val_batches = int(np.ceil(NUM_VAL / BATCH_SIZE))

print("Train batches/epoch:", train_batches, "Valid batches/epoch:", val_batches)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1987855203.py in <cell line: 0>()
----> 1 options = tf.data.Options()
      2 options.experimental_deterministic = True
      3 
      4 raw_train = tf.data.Dataset.from_tensor_slices(train_tfrec_files).with_options(options)
      5 raw_train = raw_train.interleave(

NameError: name 'tf' is not defined

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



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3033780549.py in <cell line: 0>()
----> 1 base_model = applications.EfficientNetB0(
      2     include_top=False,
      3     weights="imagenet",
      4     input_shape=(IMG_SIZE, IMG_SIZE, 3),
      5 )

NameError: name 'applications' is not defined

## === cell 14
model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adamax(learning_rate=0.01),
    metrics=["acc"],
)
model.summary()



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2954976561.py in <cell line: 0>()
----> 1 model.compile(
      2     loss=tf.keras.losses.CategoricalCrossentropy(),
      3     optimizer=tf.keras.optimizers.Adamax(learning_rate=0.01),
      4     metrics=["acc"],
      5 )

NameError: name 'model' is not defined

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



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3471048574.py in <cell line: 0>()
----> 1 reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
      2     monitor="val_loss",
      3     factor=0.3,
      4     patience=2,
      5     min_delta=0.001,

NameError: name 'tf' is not defined

## === cell 16
EPOCHS = 6

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    callbacks=[reduce_lr],
    verbose=1,
)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3423181560.py in <cell line: 0>()
      1 EPOCHS = 6
      2 
----> 3 history = model.fit(
      4     train_ds,
      5     validation_data=valid_ds,

NameError: name 'model' is not defined

## === cell 17
if os.path.exists("best_model.weights.h5"):
    model.load_weights("best_model.weights.h5")



## === cell 18
test_df = pd.read_csv(sample_submission_path)
required_order = test_df["image_id"].astype(str).values

raw_test = tf.data.Dataset.from_tensor_slices(test_tfrec_files).with_options(options)
raw_test = raw_test.interleave(
    lambda f: tf.data.TFRecordDataset(f, num_parallel_reads=1),
    cycle_length=min(len(test_tfrec_files), WORKERS),
    num_parallel_calls=AUTOTUNE,
    deterministic=True,
)

test_parsed = raw_test.map(_parse_test, num_parallel_calls=AUTOTUNE, deterministic=True)
test_parsed = test_parsed.cache()

test_img_ds = (
    test_parsed.map(lambda img, name: img, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

test_probs = model.predict(test_img_ds, verbose=1)
test_preds = np.argmax(test_probs, axis=1).astype(int)

test_names = np.array(
    [n.decode("utf-8") for _, n in test_parsed.as_numpy_iterator()], dtype=object
)

name_to_pred = dict(zip(test_names, test_preds))
ordered_preds = np.fromiter(
    (name_to_pred[n] for n in required_order), dtype=np.int64, count=len(required_order)
)

my_submission = pd.DataFrame(
    {"image_id": required_order, "label": ordered_preds.astype(int)}
)
my_submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", my_submission.shape)
print(my_submission.head())



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/970133361.py in <cell line: 0>()
      2 required_order = test_df["image_id"].astype(str).values
      3 
----> 4 raw_test = tf.data.Dataset.from_tensor_slices(test_tfrec_files).with_options(options)
      5 raw_test = raw_test.interleave(
      6     lambda f: tf.data.TFRecordDataset(f, num_parallel_reads=1),

NameError: name 'tf' is not defined

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

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_11/2462009828.py in <cell line: 0>()
----> 1 assert os.path.exists("submission.csv"), "submission.csv was not created"
      2 sub = pd.read_csv("submission.csv")
      3 assert list(sub.columns) == [
      4     "image_id",
      5     "label",

AssertionError: submission.csv was not created
