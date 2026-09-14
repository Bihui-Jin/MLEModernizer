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

0.6668177697189483

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.06839) has done: 'Main bottlenecks are (1) the expensive Python-based protobuf implementation forced via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`, (2) rebuilding and re-reading the entire `tf.data` pipeline from disk for each TTA pass, and (3) non-vectorized existence checks over ~2.6k files. I switch protobuf back to the default C++ implementation (keeping behavior identical), build the base decoded dataset once and reuse it across TTA passes (so JPEG decode/resize happens only once), and replace the Python loop file check with a vectorized `tf.io.gfile.glob`-based set membership check. TTA randomness/semantics and the model call pattern remain the same (still one non-TTA pass + `TTA_PASSES` augmented passes), but runtime drops significantly because disk decode/resize is no longer repeated.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "cpp")

import glob
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import load_model

SEED = 42
DEBUG = False

tf.random.set_seed(SEED)
np.random.seed(SEED)

print("TF version:", tf.__version__)

try:
    tf.config.threading.set_inter_op_parallelism_threads(0)
    tf.config.threading.set_intra_op_parallelism_threads(0)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/2394424102.py in <cell line: 0>()
     10 import pandas as pd
     11 
---> 12 import tensorflow as tf
     13 from tensorflow.keras.models import load_model
     14 

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
preferred_weight_path = "../input/model-v14/model_v0.25.h5"

candidate_paths = []
if os.path.exists(preferred_weight_path):
    candidate_paths = [preferred_weight_path]
else:
    likely_paths = [
        "../input",
        "/kaggle/input",
    ]
    for base in likely_paths:
        if os.path.isdir(base):
            candidate_paths += sorted(glob.glob(os.path.join(base, "*", "*.h5")))
            candidate_paths += sorted(glob.glob(os.path.join(base, "*", "*", "*.h5")))

if candidate_paths:
    weight_path = candidate_paths[0]
    print("Loading model from:", weight_path)
    my_model = load_model(weight_path, compile=False)

    in_shape = my_model.input_shape
    if isinstance(in_shape, list):
        in_shape = in_shape[0]
    if (
        in_shape is None
        or len(in_shape) != 4
        or in_shape[1] is None
        or in_shape[2] is None
    ):
        MODEL_INPUT_SIZE = (300, 300)
    else:
        MODEL_INPUT_SIZE = (int(in_shape[1]), int(in_shape[2]))
    NEED_PREPROCESS = False  # keep original behavior for trained .h5
    print("Model input size:", MODEL_INPUT_SIZE)
else:
    print(
        "No .h5 model found under /kaggle/input or ../input. Using EfficientNetB3(ImageNet) fallback."
    )
    from tensorflow.keras import layers, Model
    from tensorflow.keras.applications import EfficientNetB3
    from tensorflow.keras.applications.efficientnet import (
        preprocess_input as effnet_preprocess,
    )

    MODEL_INPUT_SIZE = (300, 300)
    base = EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=(MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1], 3),
        pooling="avg",
    )
    x = layers.Dense(5, activation="softmax")(base.output)
    my_model = Model(inputs=base.input, outputs=x)
    NEED_PREPROCESS = True

    _ = my_model(
        tf.zeros((1, MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1], 3), dtype=tf.float32)
    )
    print("Fallback model built:", my_model.name)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/1679159182.py in <cell line: 0>()
     41         "No .h5 model found under /kaggle/input or ../input. Using EfficientNetB3(ImageNet) fallback."
     42     )
---> 43     from tensorflow.keras import layers, Model
     44     from tensorflow.keras.applications import EfficientNetB3
     45     from tensorflow.keras.applications.efficientnet import (

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
sample_sub_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
if not os.path.exists(sample_sub_path):
    sample_sub_path = (
        "../input/cassava-leaf-disease-classification/sample_submission.csv"
    )

sample_sub = pd.read_csv(sample_sub_path)

test_img_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
if not os.path.isdir(test_img_dir):
    test_img_dir = "../input/cassava-leaf-disease-classification/test_images"

df_test = pd.DataFrame({"image_id": sample_sub["image_id"].astype(str).values})
df_test["path"] = test_img_dir.rstrip("/") + "/" + df_test["image_id"].values

paths_np = df_test["path"].to_numpy()

existing = set(tf.io.gfile.glob(test_img_dir.rstrip("/") + "/*.jpg"))
missing = sum(1 for p in paths_np if p not in existing)
if missing:
    raise FileNotFoundError(
        f"{missing} test images listed in sample_submission.csv were not found in {test_img_dir}"
    )

if NEED_PREPROCESS:
    from tensorflow.keras.applications.efficientnet import (
        preprocess_input as effnet_preprocess,
    )

    def _preprocess_if_needed(x):
        return effnet_preprocess(x)

else:

    def _preprocess_if_needed(x):
        return x


def _decode_and_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, MODEL_INPUT_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    img = _preprocess_if_needed(img)
    return img


def _apply_tta(img):
    img = tf.image.random_flip_left_right(img, seed=SEED)
    h = MODEL_INPUT_SIZE[0]
    w = MODEL_INPUT_SIZE[1]
    pad_h = tf.cast(tf.round(0.05 * tf.cast(h, tf.float32)), tf.int32)
    pad_w = tf.cast(tf.round(0.05 * tf.cast(w, tf.float32)), tf.int32)
    img = tf.image.resize_with_crop_or_pad(img, h + 2 * pad_h, w + 2 * pad_w)
    img = tf.image.random_crop(img, size=[h, w, 3], seed=SEED)

    zoom = tf.random.uniform([], minval=0.90, maxval=1.0, seed=SEED)
    crop_h = tf.cast(tf.round(zoom * tf.cast(h, tf.float32)), tf.int32)
    crop_w = tf.cast(tf.round(zoom * tf.cast(w, tf.float32)), tf.int32)
    img = tf.image.random_crop(img, size=[crop_h, crop_w, 3], seed=SEED)
    img = tf.image.resize(img, MODEL_INPUT_SIZE, method=tf.image.ResizeMethod.BILINEAR)

    angle = tf.random.uniform([], minval=-10.0, maxval=10.0, seed=SEED) * (
        np.pi / 180.0
    )
    cos_a = tf.math.cos(angle)
    sin_a = tf.math.sin(angle)
    tx = 0.0
    ty = 0.0
    transform = tf.stack([cos_a, -sin_a, tx, sin_a, cos_a, ty, 0.0, 0.0])[tf.newaxis, :]
    try:
        img = tf.raw_ops.ImageProjectiveTransformV3(
            images=img[tf.newaxis, ...],
            transforms=transform,
            output_shape=tf.constant([h, w], dtype=tf.int32),
            interpolation="BILINEAR",
            fill_mode="REFLECT",
            fill_value=0.0,
        )[0]
    except Exception:
        pass

    return img


def make_base_test_ds():
    ds = tf.data.Dataset.from_tensor_slices(paths_np)
    ds = ds.map(_decode_and_resize, num_parallel_calls=AUTOTUNE)
    ds = ds.cache()
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_test_ds_from_base(base_ds, batch_size=64, tta=False):
    ds = base_ds
    if tta:
        ds = ds.map(_apply_tta, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2203593718.py in <cell line: 0>()
     18 paths_np = df_test["path"].to_numpy()
     19 
---> 20 existing = set(tf.io.gfile.glob(test_img_dir.rstrip("/") + "/*.jpg"))
     21 missing = sum(1 for p in paths_np if p not in existing)
     22 if missing:

NameError: name 'tf' is not defined

## === cell 3
BATCH_SIZE = 128
TTA_PASSES = 3

base_test_ds = make_base_test_ds()

test_ds = make_test_ds_from_base(base_test_ds, batch_size=BATCH_SIZE, tta=False)
pred_sum = my_model.predict(test_ds, verbose=1)
pred_sum = np.asarray(pred_sum, dtype=np.float64)

for i in range(TTA_PASSES):
    tta_ds = make_test_ds_from_base(base_test_ds, batch_size=BATCH_SIZE, tta=True)
    pred_i = my_model.predict(tta_ds, verbose=0)
    pred_sum += np.asarray(pred_i, dtype=np.float64)

pred_mean = pred_sum / (TTA_PASSES + 1)
pred_test_labels = np.argmax(pred_mean, axis=-1).astype(int)

final_csv = pd.DataFrame(
    {"image_id": df_test["image_id"].values, "label": pred_test_labels}
)

if len(final_csv) != len(sample_sub):
    raise ValueError(
        f"Submission length mismatch: got {len(final_csv)} rows, expected {len(sample_sub)}"
    )
if list(final_csv.columns) != ["image_id", "label"]:
    raise ValueError(f"Submission columns incorrect: {final_csv.columns.tolist()}")

final_csv.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", final_csv.shape)
print(final_csv.head())



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1209849643.py in <cell line: 0>()
      2 TTA_PASSES = 3
      3 
----> 4 base_test_ds = make_base_test_ds()
      5 
      6 test_ds = make_test_ds_from_base(base_test_ds, batch_size=BATCH_SIZE, tta=False)

NameError: name 'make_base_test_ds' is not defined

## === cell 4
sub = pd.read_csv("submission.csv")
assert sub.shape[0] == sample_sub.shape[0]
assert sub.columns.tolist() == ["image_id", "label"]
assert sub["image_id"].iloc[0] == sample_sub["image_id"].iloc[0]
print("submission.csv OK:", sub.shape)
print(sub["label"].value_counts().sort_index())

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2932655558.py in <cell line: 0>()
----> 1 sub = pd.read_csv("submission.csv")
      2 assert sub.shape[0] == sample_sub.shape[0]
      3 assert sub.columns.tolist() == ["image_id", "label"]
      4 assert sub["image_id"].iloc[0] == sample_sub["image_id"].iloc[0]
      5 print("submission.csv OK:", sub.shape)

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
