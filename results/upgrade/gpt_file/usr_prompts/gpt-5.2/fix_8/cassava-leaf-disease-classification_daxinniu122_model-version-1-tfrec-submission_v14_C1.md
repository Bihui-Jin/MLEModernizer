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

0.0959504381988516

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'The timeout is dominated by per-image Python overhead: you load/resize images one-by-one and call `model.predict()` three times per image (≈ 8000 predict calls for 2676 images). I keep the exact same models and weighted-ensemble logic, but switch inference to a batched `tf.data` pipeline that decodes/resizes images in parallel and runs each model on full batches (only a few dozen predict calls total). I also force deterministic execution and use TensorFlow’s faster native protobuf (removing the pure-Python protobuf slowdown), while preserving identical preprocessing semantics (RGB decode + /255 + resize to 224). These changes are provably equivalent to the original logic but cut constant overhead drastically to fit under 600 seconds.'
- What this solution (achieved 0.61099) has done: 'You’re failing before any modeling runs due to an environment-level protobuf/TensorFlow incompatibility triggered by forcing the C++ protobuf implementation. I remove that problematic env var manipulation (keep TF log level and determinism seeds) so TensorFlow imports cleanly in Kaggle. I also keep your exact model/ensemble/inference pipeline intact, only adding a safe fallback to disable determinism if the build doesn’t support it, and ensure the submission CSV is always written with the required columns and row count.'
- What this solution (achieved 0.19395) has done: 'The crash happens at TensorFlow import time due to an incompatible protobuf runtime being picked up in this environment, which triggers `MessageFactory.GetPrototype` failures before any of your logic runs. The minimal fix is to force the pure-Python protobuf implementation *before* importing TensorFlow (this avoids the broken C++/upb path in this setup), while keeping your determinism/seed setup and the exact model/ensemble/inference semantics unchanged. I also keep the submission-writing checks as-is to guarantee a valid `submission.csv` with the required columns and row count.'
- What this solution (achieved 0.11024) has done: 'The crash happens before your code runs because TensorFlow is importing an incompatible protobuf runtime, producing `MessageFactory.GetPrototype` errors. The minimal fix is to force the pure-Python protobuf implementation *and* disable the C++/upb fast path via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=2` before importing TensorFlow; this is a common Kaggle workaround and is score-neutral (it only affects import/runtime backend, not model math). I also renumber the notebook-style cells to start at 1 (your input had `cell 0`) but keep all core model/ensemble/inference logic identical. Finally, I keep the submission writing checks to guarantee a valid `submission.csv` with the required columns and row count.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "cpp")
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

tf.keras.backend.clear_session()

np.random.seed(42)
tf.random.set_seed(42)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
if not os.path.exists(DATA_DIR):
    DATA_DIR = "/kaggle/data/cassava-leaf-disease-classification"

TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

assert os.path.exists(
    SAMPLE_SUB_PATH
), f"sample_submission.csv not found at {SAMPLE_SUB_PATH}"
assert os.path.exists(
    TEST_IMG_DIR
), f"test_images directory not found at {TEST_IMG_DIR}"

print("TensorFlow:", tf.__version__)
print("DATA_DIR:", DATA_DIR)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/2499087128.py in <cell line: 0>()
      9 import numpy as np
     10 import pandas as pd
---> 11 import tensorflow as tf
     12 from tensorflow import keras
     13 

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
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)


def build_model(seed: int):
    tf.random.set_seed(seed)
    inputs = keras.Input(shape=(224, 224, 3))
    x = keras.layers.Rescaling(1.0)(inputs)
    x = keras.layers.Conv2D(16, 3, strides=2, padding="same", activation="relu")(x)
    x = keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(x)
    x = keras.layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    outputs = keras.layers.Dense(5, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    return model


model1 = build_model(101)
model2 = build_model(202)
model3 = build_model(303)

norm_constant = 0.87 + 0.86 + 0.84
alpha_1 = 0.86 / norm_constant
alpha_2 = 0.87 / norm_constant
alpha_3 = 0.84 / norm_constant

print("Loaded sample_sub rows:", len(sample_sub))
print("Ensemble weights:", alpha_1, alpha_2, alpha_3)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3438611710.py in <cell line: 0>()
----> 1 sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
      2 
      3 
      4 def build_model(seed: int):
      5     tf.random.set_seed(seed)

NameError: name 'SAMPLE_SUB_PATH' is not defined

## === cell 2
AUTOTUNE = tf.data.AUTOTUNE

image_ids = sample_sub["image_id"].astype(str).tolist()
img_paths = [os.path.join(TEST_IMG_DIR, img_id) for img_id in image_ids]

BATCH_SIZE = 64


@tf.function
def _load_preprocess(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)  # RGB
    img = tf.image.convert_image_dtype(img, tf.float32)  # scales to [0,1] like /255.0
    img = tf.image.resize(img, (224, 224))  # same as original
    return img


ds = tf.data.Dataset.from_tensor_slices(img_paths)
options = tf.data.Options()
options.experimental_deterministic = True
ds = ds.with_options(options)
ds = ds.map(_load_preprocess, num_parallel_calls=AUTOTUNE)
ds = ds.batch(BATCH_SIZE, drop_remainder=False)
ds = ds.prefetch(AUTOTUNE)

preds = []
for batch in ds:
    p1 = model1(batch, training=False) * alpha_1
    p2 = model2(batch, training=False) * alpha_2
    p3 = model3(batch, training=False) * alpha_3
    p = p1 + p2 + p3
    preds.extend(tf.argmax(p, axis=1, output_type=tf.int32).numpy().tolist())

my_submission = pd.DataFrame({"image_id": sample_sub["image_id"], "label": preds})

out_path = "/kaggle/working/submission.csv"
my_submission.to_csv(out_path, index=False)

print(my_submission.head())
print(f"Wrote {out_path} with {len(my_submission)} rows")
print("Submission columns:", list(my_submission.columns))
assert my_submission.shape[0] == sample_sub.shape[0]
assert list(my_submission.columns) == ["image_id", "label"]

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3165863702.py in <cell line: 0>()
----> 1 AUTOTUNE = tf.data.AUTOTUNE
      2 
      3 image_ids = sample_sub["image_id"].astype(str).tolist()
      4 img_paths = [os.path.join(TEST_IMG_DIR, img_id) for img_id in image_ids]
      5 

NameError: name 'tf' is not defined
