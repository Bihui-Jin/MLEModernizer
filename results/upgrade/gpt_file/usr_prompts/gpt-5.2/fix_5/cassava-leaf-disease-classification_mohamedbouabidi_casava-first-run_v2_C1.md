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

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
tf_keras==2.18.0

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

0.7802

# 6. Current score

0.11584

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'I keep the same EfficientNetB0 model, loss/optimizer, epochs, and ImageDataGenerator augmentation, but remove Python-side overhead that causes timeouts. The main wins are (1) enabling TF graph compilation for training and inference, (2) ensuring the Keras generators use multiprocessing workers with a larger prefetch queue, and (3) replacing the per-image PIL prediction loop with a single batched `flow_from_dataframe` test generator (same resizing/rescaling), which is mathematically equivalent but far faster. I also avoid unnecessary model checkpoint disk writes each epoch by keeping the callback (same semantics) but reducing I/O overhead via less verbose output (does not affect results). Paths and core training logic remain unchanged.'
- What this solution (achieved 0.11584) has done: 'The timeout is dominated by Python-side `ImageDataGenerator` JPEG decoding/augmentation, which can’t keep the GPU/CPU efficiently fed and adds heavy per-step overhead; the model itself is fine. I keep the same split, augmentations, model, loss, optimizer, epochs, and steps, but replace the generators with an equivalent `tf.data` pipeline that performs the same geometric transforms and rescaling inside the TensorFlow graph, with caching/prefetching/parallel map to remove the input bottleneck. I also enable deterministic `tf.data` execution and use `steps_per_execution` to reduce Python→TF call overhead per epoch without changing training semantics. Prediction similarly use a fast `tf.data` input pipeline while preserving identical output formatting and file paths.'

# 9. Code solution

## === cell 0
import os, json
import numpy as np
import pandas as pd

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "cpp")

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

AUTOTUNE = tf.data.AUTOTUNE




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/4128211606.py in <cell line: 0>()
      7 os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "cpp")
      8 
----> 9 import tensorflow as tf
     10 from tensorflow.keras import layers, models
     11 from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

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
base_dir = "../input/cassava-leaf-disease-classification"
os.listdir(base_dir)




## === cell 2
train_labels = pd.read_csv(os.path.join(base_dir, "train.csv"))
train_labels.head()




## === cell 3
BATCH_SIZE = 20
EPOCHS = 10
TARGET_SIZE = 224

STEPS_PER_EPOCH = int(np.ceil(len(train_labels) * 0.8 / BATCH_SIZE))
VALIDATION_STEPS = int(np.ceil(len(train_labels) * 0.2 / BATCH_SIZE))

GEN_WORKERS = max(2, (os.cpu_count() or 2) - 1)
GEN_MAX_QUEUE_SIZE = 32




## === cell 4
train_labels = train_labels.copy()
train_labels["label"] = train_labels["label"].astype(np.int32)

train_img_dir = os.path.join(base_dir, "train_images")

shuffled = train_labels.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
split_idx = int(len(shuffled) * 0.8)
train_df = shuffled.iloc[:split_idx].reset_index(drop=True)
val_df = shuffled.iloc[split_idx:].reset_index(drop=True)

train_paths = (train_img_dir + "/" + train_df["image_id"].values).astype(str)
train_y = train_df["label"].values.astype(np.int32)

val_paths = (train_img_dir + "/" + val_df["image_id"].values).astype(str)
val_y = val_df["label"].values.astype(np.int32)

rotation_range = 40.0 * np.pi / 180.0
zoom_range = 0.2
shear_range = (
    0.2  # in radians for tfa shear; here used as a small-angle shear approximation
)

height_shift_range = 0.2
width_shift_range = 0.2


def _decode_resize(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(
        img,
        [TARGET_SIZE, TARGET_SIZE],
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False,
    )
    img = tf.cast(img, tf.float32) / 255.0  # rescale=1./255
    return img


def _augment(img, seed):
    img = tf.image.stateless_random_flip_left_right(img, seed=tf.stack([seed, 1]))
    img = tf.image.stateless_random_flip_up_down(img, seed=tf.stack([seed, 2]))

    angle = tf.random.stateless_uniform(
        [], seed=tf.stack([seed, 3]), minval=-rotation_range, maxval=rotation_range
    )
    img = tfa.image.rotate(
        img, angles=angle, interpolation="BILINEAR", fill_mode="nearest"
    )

    z = tf.random.stateless_uniform(
        [], seed=tf.stack([seed, 4]), minval=1.0 - zoom_range, maxval=1.0 + zoom_range
    )
    new_size = tf.cast(tf.round(tf.cast(TARGET_SIZE, tf.float32) * z), tf.int32)
    new_size = tf.maximum(new_size, 1)
    img2 = tf.image.resize(
        img,
        [new_size, new_size],
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False,
    )
    img2 = tf.image.resize_with_crop_or_pad(img2, TARGET_SIZE, TARGET_SIZE)
    img = img2

    sx = tf.random.stateless_uniform(
        [], seed=tf.stack([seed, 5]), minval=-shear_range, maxval=shear_range
    )
    sy = tf.random.stateless_uniform(
        [], seed=tf.stack([seed, 6]), minval=-shear_range, maxval=shear_range
    )
    tx = (
        tf.random.stateless_uniform(
            [],
            seed=tf.stack([seed, 7]),
            minval=-width_shift_range,
            maxval=width_shift_range,
        )
        * TARGET_SIZE
    )
    ty = (
        tf.random.stateless_uniform(
            [],
            seed=tf.stack([seed, 8]),
            minval=-height_shift_range,
            maxval=height_shift_range,
        )
        * TARGET_SIZE
    )

    a0 = 1.0
    a1 = tf.tan(sx)
    a2 = tx
    b0 = tf.tan(sy)
    b1 = 1.0
    b2 = ty
    c0 = 0.0
    c1 = 0.0
    transform = tf.stack([a0, a1, a2, b0, b1, b2, c0, c1])
    img = tfa.image.transform(
        img, transform, interpolation="BILINEAR", fill_mode="nearest"
    )

    return img


import tensorflow_addons as tfa


def make_dataset(paths, labels=None, training=False):
    if labels is None:
        ds = tf.data.Dataset.from_tensor_slices(paths)
    else:
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    options = tf.data.Options()
    options.deterministic = True
    ds = ds.with_options(options)

    if training:
        ds = ds.shuffle(
            buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True
        )

    if labels is None:

        def _map_fn(path, idx):
            img = _decode_resize(path)
            if training:
                img = _augment(img, tf.cast(idx, tf.int32))
            return img

        ds = ds.enumerate()
        ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    else:

        def _map_fn(path, y, idx):
            img = _decode_resize(path)
            if training:
                img = _augment(img, tf.cast(idx, tf.int32))
            return img, y

        ds = ds.enumerate().map(
            lambda idx, xy: (xy[0], xy[1], idx), num_parallel_calls=AUTOTUNE
        )
        ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_dataset(train_paths, train_y, training=True)
val_ds = make_dataset(val_paths, val_y, training=False)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2933257599.py in <cell line: 0>()
     13 # flow_from_dataframe uses a deterministic shuffle with the given seed and then takes subsets.
     14 # We reproduce that by shuffling the dataframe once with the same seed and slicing.
---> 15 shuffled = train_labels.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
     16 split_idx = int(len(shuffled) * 0.8)
     17 train_df = shuffled.iloc[:split_idx].reset_index(drop=True)

NameError: name 'SEED' is not defined

## === cell 5
from tensorflow.keras.applications import EfficientNetB0

eff_base = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(TARGET_SIZE, TARGET_SIZE, 3),
)
eff_base.summary()




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/2862384465.py in <cell line: 0>()
----> 1 from tensorflow.keras.applications import EfficientNetB0
      2 
      3 eff_base = EfficientNetB0(
      4     include_top=False,
      5     weights="imagenet",

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

## === cell 6
model = models.Sequential()
model.add(eff_base)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(5, activation="softmax", name="Output"))
model.summary()




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2479460309.py in <cell line: 0>()
----> 1 model = models.Sequential()
      2 model.add(eff_base)
      3 model.add(layers.GlobalAveragePooling2D())
      4 model.add(layers.Dense(5, activation="softmax", name="Output"))
      5 model.summary()

NameError: name 'models' is not defined

## === cell 7
model.compile(
    optimizer="Adam",
    loss="sparse_categorical_crossentropy",
    metrics=["acc"],
    jit_compile=True,
    steps_per_execution=32,
)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/520975639.py in <cell line: 0>()
      1 # Speed fix: Increase steps_per_execution to reduce Python overhead per epoch while keeping identical semantics.
      2 # This does not change the model, loss, optimizer, data, or number of optimizer steps; it only batches graph execution.
----> 3 model.compile(
      4     optimizer="Adam",
      5     loss="sparse_categorical_crossentropy",

NameError: name 'model' is not defined

## === cell 8
model_save = ModelCheckpoint(
    "./EffNetB0_512_8_best_weights.weights.h5",
    save_best_only=True,
    save_weights_only=True,
    monitor="val_loss",
    mode="min",
    verbose=0,
)
early_stop = EarlyStopping(
    monitor="val_loss",
    min_delta=0.001,
    patience=5,
    mode="min",
    verbose=1,
    restore_best_weights=True,
)
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=2,
    min_delta=0.001,
    mode="min",
    verbose=1,
)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2493472439.py in <cell line: 0>()
----> 1 model_save = ModelCheckpoint(
      2     "./EffNetB0_512_8_best_weights.weights.h5",
      3     save_best_only=True,
      4     save_weights_only=True,
      5     monitor="val_loss",

NameError: name 'ModelCheckpoint' is not defined

## === cell 9
history = model.fit(
    train_ds,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    validation_data=val_ds,
    validation_steps=VALIDATION_STEPS,
    callbacks=[model_save, early_stop, reduce_lr],
    verbose=1,
)




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/204962341.py in <cell line: 0>()
----> 1 history = model.fit(
      2     train_ds,
      3     steps_per_epoch=STEPS_PER_EPOCH,
      4     epochs=EPOCHS,
      5     validation_data=val_ds,

NameError: name 'model' is not defined

## === cell 10
acc = history.history.get("acc", history.history.get("accuracy", []))
val_acc = history.history.get("val_acc", history.history.get("val_accuracy", []))
loss = history.history.get("loss", [])
val_loss = history.history.get("val_loss", [])

epochs_range = range(1, len(loss) + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
sns.set_style("white")
plt.suptitle("Train history", size=15)

if len(acc) and len(val_acc):
    ax1.plot(epochs_range, acc, "bo", label="Training acc")
    ax1.plot(epochs_range, val_acc, "b", label="Validation acc")
    ax1.set_title("Training and validation acc")
    ax1.legend()
else:
    ax1.set_title("Accuracy history unavailable (key mismatch)")

ax2.plot(epochs_range, loss, "bo", label="Training loss", color="red")
ax2.plot(epochs_range, val_loss, "b", label="Validation loss", color="red")
ax2.set_title("Training and validation loss")
ax2.legend()

plt.show()




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/696798842.py in <cell line: 0>()
----> 1 acc = history.history.get("acc", history.history.get("accuracy", []))
      2 val_acc = history.history.get("val_acc", history.history.get("val_accuracy", []))
      3 loss = history.history.get("loss", [])
      4 val_loss = history.history.get("val_loss", [])
      5 

NameError: name 'history' is not defined

## === cell 11
sub = pd.read_csv(os.path.join(base_dir, "sample_submission.csv"))
sub.head()




## === cell 12
test_dir = os.path.join(base_dir, "test_images")
test_df = sub[["image_id"]].copy()
test_paths = (test_dir + "/" + test_df["image_id"].values).astype(str)

test_ds = make_dataset(test_paths, labels=None, training=False)
test_steps = int(np.ceil(len(test_df) / BATCH_SIZE))

probs = model.predict(
    test_ds,
    steps=test_steps,
    verbose=0,
)

preds = np.argmax(probs, axis=1).astype(int)[: len(test_df)]
sub["label"] = preds
sub.head()




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2646027670.py in <cell line: 0>()
      4 test_paths = (test_dir + "/" + test_df["image_id"].values).astype(str)
      5 
----> 6 test_ds = make_dataset(test_paths, labels=None, training=False)
      7 test_steps = int(np.ceil(len(test_df) / BATCH_SIZE))
      8 

NameError: name 'make_dataset' is not defined

## === cell 13
sub = sub[["image_id", "label"]]
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
