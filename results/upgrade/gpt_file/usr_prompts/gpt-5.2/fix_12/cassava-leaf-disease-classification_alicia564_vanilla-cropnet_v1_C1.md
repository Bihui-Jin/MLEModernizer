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

3.13

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

0.8779087337564219

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.21413) has done: 'The timeout is most likely dominated by input pipeline overhead (JPEG decode/resize + heavy projective transforms) starving the GPU/CPU during training, plus extra per-element Python/TF graph work for label mapping and redundant caching choices. I keep the exact same model, loss, batch size, and augmentation math, but optimize the `tf.data` pipeline: fuse operations, vectorize label encoding (avoid lookup table + one_hot inside the map), enable `map_and_batch`, add `cache()` for decoded+resized images (before augmentation) so decoding is done once per file, and ensure aggressive parallelism/prefetch while keeping determinism. I also remove unnecessary `.cache()` on test data (it’s only iterated once and can waste time/memory) and keep all seeds/deterministic behavior intact.'
- What this solution (achieved 0.21413) has done: 'The main timeout driver is the expensive Python-based protobuf implementation and a tf.data pipeline that adds heavy per-example work (stateless RNG + projective transform) and caches full decoded images in RAM, increasing overhead and stalls. I switch protobuf back to the default C++ implementation (huge speedup for TFRecord/JPEG parsing paths), keep determinism, and restructure the dataset pipeline to do map→batch→map (vectorized per-batch augmentation) while preserving identical augmentation math and preprocessing. I also remove redundant seed-mangling that doesn’t depend on image content but costs ops, keep caching only where it’s safe/beneficial, and enable standard tf.data optimizations/prefetch with AUTOTUNE to keep the GPU busy. Model, loss, epochs, callbacks, and evaluation semantics remain unchanged.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import re
from datetime import datetime

import numpy as np
import pandas as pd

import tensorflow as tf

print("TF version:", tf.__version__)
print("Eager:", tf.executing_eagerly())

tf.keras.utils.set_random_seed(42)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    tf.data.experimental.enable_debug_mode(False)
except Exception:
    pass



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/314469508.py in <cell line: 0>()
     13 import pandas as pd
     14 
---> 15 import tensorflow as tf
     16 
     17 print("TF version:", tf.__version__)

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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

label_to_disease = pd.read_json(
    "/kaggle/input/cassava-leaf-disease-classification/label_num_to_disease_map.json",
    typ="series",
)

train_csv = pd.read_csv("/kaggle/input/cassava-leaf-disease-classification/train.csv")

train_csv["disease"] = train_csv["label"].map(label_to_disease)

train_csv["path"] = (
    "/kaggle/input/cassava-leaf-disease-classification/train_images/"
    + train_csv["image_id"].astype(str)
)

le = LabelEncoder()
train_csv["label_encoded"] = le.fit_transform(train_csv["disease"])

train_csv["disease"] = train_csv["disease"].astype(str)
train_csv["label"] = train_csv["label"].astype(str)

train, valid = train_test_split(
    train_csv, test_size=0.2, stratify=train_csv["label"], random_state=42
)

class_names = sorted(train["disease"].unique().tolist())
class_indices = {name: i for i, name in enumerate(class_names)}

NUM_CLASSES = len(class_names)
print("Num classes:", NUM_CLASSES)
print("Class indices (disease->idx):", class_indices)
print("Train samples:", len(train))
print("Valid samples:", len(valid))



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/299018795.py in <cell line: 0>()
      2 from sklearn.preprocessing import LabelEncoder
      3 
----> 4 from tensorflow.keras.applications.efficientnet import preprocess_input
      5 from tensorflow.keras.applications import EfficientNetB0
      6 from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

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

## === cell 2
AUTOTUNE = tf.data.AUTOTUNE
IMG_SIZE = 224
BATCH_SIZE = 32  # keep identical
SEED = 42

_IMG_SIZE_F = tf.constant(float(IMG_SIZE), dtype=tf.float32)
_HALF_C = _IMG_SIZE_F / 2.0
_MAX_ANGLE = tf.constant(45.0 * np.pi / 180.0, dtype=tf.float32)

_disease_to_idx = {name: i for i, name in enumerate(class_names)}


@tf.function(jit_compile=False)
def _decode_and_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(
        img, (IMG_SIZE, IMG_SIZE), method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.image.convert_image_dtype(img, tf.float32) * 255.0  # match previous 0..255
    return img


@tf.function(jit_compile=False)
def _augment_like_datagen_batch(imgs, seed2s):
    b = tf.shape(imgs)[0]
    seed2s = tf.cast(seed2s, tf.int32)

    def _flip_lr(args):
        im, s2 = args
        seed = tf.stack([tf.constant(SEED, tf.int32), s2])
        return tf.image.stateless_random_flip_left_right(im, seed=seed)

    def _flip_ud(args):
        im, s2 = args
        seed = tf.stack([tf.constant(SEED + 1, tf.int32), s2])
        return tf.image.stateless_random_flip_up_down(im, seed=seed)

    imgs = tf.map_fn(
        _flip_lr,
        (imgs, seed2s),
        fn_output_signature=tf.float32,
        parallel_iterations=32,
    )
    imgs = tf.map_fn(
        _flip_ud,
        (imgs, seed2s),
        fn_output_signature=tf.float32,
        parallel_iterations=32,
    )

    rnd = tf.random.stateless_uniform(
        [b, 6],
        seed=tf.stack([tf.constant(SEED + 2, tf.int32), tf.constant(12345, tf.int32)]),
        minval=0.0,
        maxval=1.0,
    )
    rnd = tf.random.stateless_uniform(
        [b, 6],
        seed=tf.stack([tf.constant(SEED + 2, tf.int32), tf.reduce_sum(seed2s)]),
        minval=0.0,
        maxval=1.0,
    )

    angle = (rnd[:, 0] * 2.0 - 1.0) * _MAX_ANGLE
    tx = (rnd[:, 1] * 2.0 - 1.0) * 0.2 * _IMG_SIZE_F
    ty = (rnd[:, 2] * 2.0 - 1.0) * 0.2 * _IMG_SIZE_F
    zoom = 1.0 + (rnd[:, 3] * 2.0 - 1.0) * 0.2
    shear = (rnd[:, 4] * 2.0 - 1.0) * 0.2

    cos_a = tf.cos(angle) / zoom
    sin_a = tf.sin(angle) / zoom

    a0 = cos_a + shear * sin_a
    a1 = -sin_a
    a2 = (1.0 - a0) * _HALF_C - a1 * _HALF_C - tx

    b0 = sin_a + shear * cos_a
    b1 = cos_a
    b2 = (1.0 - b1) * _HALF_C - b0 * _HALF_C - ty

    transform = tf.stack(
        [a0, a1, a2, b0, b1, b2, tf.zeros_like(a0), tf.zeros_like(a0)], axis=1
    )

    imgs = tf.raw_ops.ImageProjectiveTransformV3(
        images=imgs,
        transforms=transform,
        output_shape=tf.constant([IMG_SIZE, IMG_SIZE], dtype=tf.int32),
        interpolation="BILINEAR",
        fill_mode="NEAREST",
        fill_value=0.0,
    )
    return imgs


@tf.function(jit_compile=False)
def _train_after_decode_batch(imgs, label_idx):
    label_i32 = tf.cast(label_idx, tf.int32)
    seed2s = tf.bitcast(label_i32 * tf.constant(2654435761, tf.int32), tf.int32)

    imgs = _augment_like_datagen_batch(imgs, seed2s)
    imgs = preprocess_input(imgs)
    labels = tf.one_hot(
        tf.cast(label_idx, tf.int32), depth=NUM_CLASSES, dtype=tf.float32
    )
    return imgs, labels


@tf.function(jit_compile=False)
def _valid_after_decode(img, label_idx):
    img = preprocess_input(img)
    label = tf.one_hot(
        tf.cast(label_idx, tf.int32), depth=NUM_CLASSES, dtype=tf.float32
    )
    return img, label


train_paths = train["path"].to_numpy(dtype=object)
valid_paths = valid["path"].to_numpy(dtype=object)

train_label_idx = train["disease"].map(_disease_to_idx).to_numpy(dtype=np.int64)
valid_label_idx = valid["disease"].map(_disease_to_idx).to_numpy(dtype=np.int64)

opts = tf.data.Options()
opts.deterministic = True
opts.experimental_optimization.map_parallelization = True
opts.experimental_optimization.parallel_batch = True
opts.experimental_optimization.apply_default_optimizations = True
opts.threading.private_threadpool_size = 0  # let TF pick
opts.threading.max_intra_op_parallelism = 0

shuffle_buffer = min(len(train_paths), 4096)

train_ds = tf.data.Dataset.from_tensor_slices(
    (train_paths, train_label_idx)
).with_options(opts)

train_ds = train_ds.map(
    lambda p, y: (_decode_and_resize(p), y),
    num_parallel_calls=AUTOTUNE,
    deterministic=True,
).apply(tf.data.experimental.ignore_errors())

train_ds = train_ds.shuffle(shuffle_buffer, seed=SEED, reshuffle_each_iteration=True)
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False)
train_ds = train_ds.map(
    _train_after_decode_batch, num_parallel_calls=AUTOTUNE, deterministic=True
)

try:
    train_ds = train_ds.apply(
        tf.data.experimental.prefetch_to_device("/GPU:0", buffer_size=AUTOTUNE)
    )
except Exception:
    train_ds = train_ds.prefetch(AUTOTUNE)

valid_ds = tf.data.Dataset.from_tensor_slices(
    (valid_paths, valid_label_idx)
).with_options(opts)
valid_ds = valid_ds.map(
    lambda p, y: (_decode_and_resize(p), y),
    num_parallel_calls=AUTOTUNE,
    deterministic=True,
).apply(tf.data.experimental.ignore_errors())

valid_ds = valid_ds.cache()
valid_ds = valid_ds.map(
    _valid_after_decode, num_parallel_calls=AUTOTUNE, deterministic=True
)
valid_ds = valid_ds.batch(BATCH_SIZE, drop_remainder=False)

try:
    valid_ds = valid_ds.apply(
        tf.data.experimental.prefetch_to_device("/GPU:0", buffer_size=AUTOTUNE)
    )
except Exception:
    valid_ds = valid_ds.prefetch(AUTOTUNE)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3930720351.py in <cell line: 0>()
----> 1 AUTOTUNE = tf.data.AUTOTUNE
      2 IMG_SIZE = 224
      3 BATCH_SIZE = 32  # keep identical
      4 SEED = 42
      5 

NameError: name 'tf' is not defined

## === cell 3
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback


class EarlyStoppingCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if self.model.stop_training:
            print(f"Early stopping triggered at epoch {epoch + 1}.")


early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1,
)

learning_rate_reduction = ReduceLROnPlateau(
    monitor="val_loss",
    patience=2,
    factor=0.5,
    min_lr=1e-7,
    verbose=1,
)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/3757893064.py in <cell line: 0>()
----> 1 from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
      2 
      3 
      4 class EarlyStoppingCallback(Callback):
      5     def on_epoch_end(self, epoch, logs=None):

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

## === cell 4
base_model = EfficientNetB0(
    include_top=False, weights="imagenet", input_shape=(224, 224, 3)
)
x = base_model.output
x = GlobalAveragePooling2D()(x)
outputs = Dense(NUM_CLASSES, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

try:
    tf.data.experimental.assert_cardinality(
        train_ds, int(np.ceil(len(train_paths) / BATCH_SIZE))
    )
except Exception:
    pass

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=10,
    callbacks=[early_stopping, learning_rate_reduction, EarlyStoppingCallback()],
    verbose=1,
)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/571443318.py in <cell line: 0>()
----> 1 base_model = EfficientNetB0(
      2     include_top=False, weights="imagenet", input_shape=(224, 224, 3)
      3 )
      4 x = base_model.output
      5 x = GlobalAveragePooling2D()(x)

NameError: name 'EfficientNetB0' is not defined

## === cell 5
sample_sub = pd.read_csv(
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
test_image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"

sample_sub["path"] = (
    test_image_dir.rstrip("/") + "/" + sample_sub["image_id"].astype(str)
)
test_paths = sample_sub["path"].to_numpy(dtype=object)


@tf.function(jit_compile=False)
def make_test_example(path):
    img = _decode_and_resize(path)
    img = preprocess_input(img)
    return img


test_ds = tf.data.Dataset.from_tensor_slices(test_paths).with_options(opts)
test_ds = test_ds.map(
    make_test_example, num_parallel_calls=AUTOTUNE, deterministic=True
)
test_ds = test_ds.apply(tf.data.experimental.ignore_errors())
test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False)
try:
    test_ds = test_ds.apply(
        tf.data.experimental.prefetch_to_device("/GPU:0", buffer_size=AUTOTUNE)
    )
except Exception:
    test_ds = test_ds.prefetch(AUTOTUNE)

probs = model.predict(test_ds, verbose=1)
pred_idx = np.argmax(probs, axis=1)

idx_to_disease = {v: k for k, v in class_indices.items()}

label_to_disease_dict = label_to_disease.to_dict()
disease_to_label = {str(v): int(k) for k, v in label_to_disease_dict.items()}

pred_disease = [idx_to_disease[int(i)] for i in pred_idx]
pred_label = [disease_to_label[str(d)] for d in pred_disease]

submission_df = pd.DataFrame(
    {"image_id": sample_sub["image_id"].values, "label": pred_label}
)
submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)

print("Submission file created:", submission_path)
print(submission_df.head())
print("Rows:", len(submission_df), "Cols:", submission_df.shape[1])
print("Unique labels:", sorted(submission_df["label"].unique().tolist()))

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3329936978.py in <cell line: 0>()
     10 
     11 
---> 12 @tf.function(jit_compile=False)
     13 def make_test_example(path):
     14     img = _decode_and_resize(path)

NameError: name 'tf' is not defined
