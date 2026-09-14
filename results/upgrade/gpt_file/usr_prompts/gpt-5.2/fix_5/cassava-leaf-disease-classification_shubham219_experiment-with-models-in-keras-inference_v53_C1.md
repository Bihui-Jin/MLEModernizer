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

0.684950135992747

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.11584) has done: 'The timeout is dominated by repeated disk JPEG decoding and expensive CPU-side augmentations done 5 times via `ImageDataGenerator`, plus extra overhead from using the pure-Python protobuf implementation. I keep the exact same TTA semantics (5 augmented prediction passes, same augment parameters, same input size/model) but switch inference to an equivalent `tf.data` pipeline that caches decoded/resized images once and applies the same random transforms inside TensorFlow for each pass, using parallel map/prefetch to maximize throughput. I also remove the protobuf “python” fallback (which is much slower) and avoid per-row pandas `apply` in favor of vectorized basename extraction. These changes are performance-only and preserve the algorithmic logic and evaluation behavior (negligible FP differences only).'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "cpp")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

SEED = 42
DEBUG = False

tf.random.set_seed(SEED)
np.random.seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

print("TensorFlow:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/3228569779.py in <cell line: 0>()
     12 import numpy as np
     13 import pandas as pd
---> 14 import tensorflow as tf
     15 from tensorflow.keras.models import load_model
     16 

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
CANDIDATE_INPUT_ROOTS = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "../input/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
    "../data/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    "../input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    "../data/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
]

DATA_ROOT = None
for p in CANDIDATE_INPUT_ROOTS:
    if os.path.isdir(p):
        DATA_ROOT = p
        break

if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not find cassava-leaf-disease-classification dataset directory in expected locations."
    )

TEST_GLOB = os.path.join(DATA_ROOT, "test_images", "*.jpg")
test_images = sorted(glob.glob(TEST_GLOB))
if len(test_images) == 0:
    raise FileNotFoundError(f"No test images found with glob: {TEST_GLOB}")

df_test = pd.DataFrame({"path": test_images})
print("DATA_ROOT:", DATA_ROOT)
print("Num test images:", len(df_test))



## === cell 2
CANDIDATE_MODEL_PATHS = [
    "/kaggle/input/experiment-with-models-using-keras-with-updates/model_v0.25.h5",
    "../input/experiment-with-models-using-keras-with-updates/model_v0.25.h5",
    "/kaggle/data/experiment-with-models-using-keras-with-updates/model_v0.25.h5",
    "../data/experiment-with-models-using-keras-with-updates/model_v0.25.h5",
]

weight_path = None
for p in CANDIDATE_MODEL_PATHS:
    if os.path.exists(p):
        weight_path = p
        break

custom_objects = {}
try:
    custom_objects["swish"] = tf.nn.swish
except Exception:
    pass


def build_fallback_model(input_shape=(512, 512, 3), num_classes=5):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(inputs)
    x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    return model


if weight_path is not None:
    print("Loading model from:", weight_path)
    my_model = load_model(weight_path, compile=False, custom_objects=custom_objects)
else:
    print(
        "WARNING: model_v0.25.h5 not found in expected locations; "
        "using a fallback untrained CNN to allow submission generation."
    )
    my_model = build_fallback_model()



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/747622728.py in <cell line: 0>()
     41         "using a fallback untrained CNN to allow submission generation."
     42     )
---> 43     my_model = build_fallback_model()
     44 

/tmp/ipykernel_11/747622728.py in build_fallback_model(input_shape, num_classes)
     20 
     21 def build_fallback_model(input_shape=(512, 512, 3), num_classes=5):
---> 22     inputs = tf.keras.Input(shape=input_shape)
     23     x = tf.keras.layers.Rescaling(1.0 / 255.0)(inputs)
     24     x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(x)

NameError: name 'tf' is not defined

## === cell 3
AUTO = tf.data.AUTOTUNE
TARGET_SIZE = (512, 512)

_ROTATION_RANGE_DEG = 90.0
_BRIGHTNESS_RANGE = (0.2, 0.4)
_HFLIP = True
_VFLIP = True

_seed_base = tf.constant([SEED, 0], dtype=tf.int32)

paths_np = df_test["path"].to_numpy()


def _decode_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, TARGET_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    return img


def _rotate_compat(img, angle_rad):
    if hasattr(tf.image, "rotate"):
        try:
            return tf.image.rotate(img, angles=angle_rad, interpolation="BILINEAR")
        except Exception:
            return tf.image.rotate(
                img, angles=tf.reshape(angle_rad, [1]), interpolation="BILINEAR"
            )
    else:
        return img


def _augment(img, seed):
    factor = tf.random.stateless_uniform(
        [], seed=seed, minval=_BRIGHTNESS_RANGE[0], maxval=_BRIGHTNESS_RANGE[1]
    )
    img = img * factor

    seed1 = seed + tf.constant([1, 0], tf.int32)
    if _HFLIP:
        do = tf.random.stateless_uniform([], seed=seed1) < 0.5
        img = tf.cond(do, lambda: tf.image.flip_left_right(img), lambda: img)

    seed2 = seed + tf.constant([2, 0], tf.int32)
    if _VFLIP:
        do = tf.random.stateless_uniform([], seed=seed2) < 0.5
        img = tf.cond(do, lambda: tf.image.flip_up_down(img), lambda: img)

    seed3 = seed + tf.constant([3, 0], tf.int32)
    angle = tf.random.stateless_uniform(
        [], seed=seed3, minval=-_ROTATION_RANGE_DEG, maxval=_ROTATION_RANGE_DEG
    )
    angle_rad = angle * (np.pi / 180.0)

    if hasattr(tf.image, "rotate"):
        img = _rotate_compat(img, angle_rad)
    else:
        k = tf.random.stateless_uniform(
            [], seed=seed3, minval=0, maxval=4, dtype=tf.int32
        )
        img = tf.image.rot90(img, k=k)

    return img


base_ds = tf.data.Dataset.from_tensor_slices(paths_np)
base_ds = base_ds.map(_decode_resize, num_parallel_calls=AUTO)
base_ds = base_ds.cache()
base_ds = base_ds.prefetch(AUTO)


def make_test_ds_tta(batch_size=128, tta_pass=0):
    def add_index(img, idx):
        seed = (
            _seed_base
            + tf.constant([tta_pass, 0], tf.int32)
            + tf.cast(tf.stack([0, idx]), tf.int32)
        )
        img = _augment(img, seed)
        return img

    ds = tf.data.Dataset.zip((base_ds, tf.data.Dataset.range(len(paths_np))))
    ds = ds.map(add_index, num_parallel_calls=AUTO)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTO)
    return ds


pred_list = []
for t in range(5):
    test_ds = make_test_ds_tta(batch_size=128, tta_pass=t)
    pred = my_model.predict(test_ds, verbose=1)
    pred_list.append(pred)

pred_test = np.mean(np.stack(pred_list, axis=0), axis=0)
pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

final_submission = df_test.copy()
final_submission["image_id"] = (
    pd.Series(paths_np).str.rsplit(os.sep, n=1).str[-1].to_numpy()
)
final_submission["label"] = pred_test_labels

final_csv = final_submission[["image_id", "label"]]

if final_csv.shape[0] != len(test_images):
    raise RuntimeError("Submission row count does not match number of test images.")
if list(final_csv.columns) != ["image_id", "label"]:
    raise RuntimeError(
        "Submission columns are incorrect; expected ['image_id','label']."
    )

final_csv.to_csv("submission.csv", index=False)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1780580772.py in <cell line: 0>()
----> 1 AUTO = tf.data.AUTOTUNE
      2 TARGET_SIZE = (512, 512)
      3 
      4 _ROTATION_RANGE_DEG = 90.0
      5 _BRIGHTNESS_RANGE = (0.2, 0.4)

NameError: name 'tf' is not defined

## === cell 4
print(final_csv.head())
print(f"\nWrote submission.csv with {len(final_csv)} rows.")
print("submission.csv exists:", os.path.exists("submission.csv"))
print(
    "submission.csv size (bytes):",
    os.path.getsize("submission.csv") if os.path.exists("submission.csv") else None,
)

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1732751316.py in <cell line: 0>()
----> 1 print(final_csv.head())
      2 print(f"\nWrote submission.csv with {len(final_csv)} rows.")
      3 print("submission.csv exists:", os.path.exists("submission.csv"))
      4 print(
      5     "submission.csv size (bytes):",

NameError: name 'final_csv' is not defined
