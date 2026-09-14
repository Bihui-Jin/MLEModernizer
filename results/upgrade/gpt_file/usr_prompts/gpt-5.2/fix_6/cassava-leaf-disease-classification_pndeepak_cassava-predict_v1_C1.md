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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
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

0.8768510123904503

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.13229) has done: 'I fix the runtime crash caused by an incompatible `protobuf` version (the `MessageFactory.GetPrototype` error) by forcing the pure-Python protobuf implementation before importing TensorFlow. Then I keep your exact model architecture/inference logic, but make the image preprocessing deterministic and consistent with EfficientNet (convert to RGB, resize to 300×300, and apply `efficientnet.preprocess_input`) to improve accuracy toward your target without changing the core approach. Finally, I batch predictions for speed and ensure a correctly formatted `submission.csv` is written.'
- What this solution (achieved 0.10052) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *and* disabling the C++ implementation before any TensorFlow import, which avoids the `MessageFactory.GetPrototype` issue in this environment. Next, I fix the missing weights error by removing the external weights dependency and instead use EfficientNetB0’s built-in `imagenet` weights (same architecture), which should move accuracy substantially toward your target. Finally, I keep your inference logic the same but make sure preprocessing matches EfficientNet expectations and that a correctly formatted `submission.csv` is always written.'
- What this solution (achieved 0.19806) has done: 'I fix the protobuf/TensorFlow import crash by removing the incompatible pure-Python protobuf forcing and instead forcing TensorFlow to use the Python protobuf implementation via `TF_PROTOBUF_IMPLEMENTATION=python` before importing TensorFlow (this avoids the `MessageFactory.GetPrototype` failure under protobuf 6.x). I keep your exact model and inference logic unchanged, only adding a robust fallback that searches the known dataset locations so the notebook runs regardless of whether Kaggle mounts data under `/kaggle/input` or `../input`. Finally, I ensure the submission is always written as `submission.csv` with the required columns and row order matching `sample_submission.csv`.'
- What this solution (achieved 0.05755) has done: 'I fix the TensorFlow/protobuf crash by forcing protobuf to use the pure-Python backend *and* disabling the C++ implementation before TensorFlow is imported, which avoids the `MessageFactory.GetPrototype` error under protobuf 6.x. Then I keep your model architecture and inference logic the same, only ensuring the environment variables are set early enough and adding a small safety check that all test image paths exist (so it fails fast with a clear message instead of crashing mid-loop). This should restore end-to-end execution and produce a valid `submission.csv`. Any score change come only from the code actually running correctly with the same preprocessing/model logic.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "cpp")
os.environ.setdefault("TF_PROTOBUF_IMPLEMENTATION", "cpp")

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
except AttributeError as e:
    if "GetPrototype" in str(e) or "MessageFactory" in str(e):
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
        os.environ["TF_PROTOBUF_IMPLEMENTATION"] = "python"
        import importlib

        tf = importlib.import_module("tensorflow")
    else:
        raise

from tensorflow import keras

tf.random.set_seed(42)
np.random.seed(42)

print("TF version:", tf.__version__)
print("Eager:", tf.executing_eagerly())



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_55/3807577949.py in <cell line: 0>()
     14 # Import TensorFlow with a robust fallback (without changing core model/inference logic)
     15 try:
---> 16     import tensorflow as tf
     17 except AttributeError as e:
     18     # If the environment still hits protobuf factory issues, fall back to python implementation.

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
model = keras.models.Sequential()
model.add(
    keras.applications.EfficientNetB0(
        input_shape=(300, 300, 3), weights="imagenet", include_top=False
    )
)
model.add(keras.layers.GlobalAveragePooling2D())
model.add(keras.layers.Dense(5, activation="softmax"))

model.trainable = False




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1340769430.py in <cell line: 0>()
----> 1 model = keras.models.Sequential()
      2 model.add(
      3     keras.applications.EfficientNetB0(
      4         input_shape=(300, 300, 3), weights="imagenet", include_top=False
      5     )

NameError: name 'keras' is not defined

## === cell 2
def find_existing_path(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "None of the candidate paths exist:\n" + "\n".join(candidates)
    )


sample_csv = find_existing_path(
    [
        "../input/cassava-leaf-disease-classification/sample_submission.csv",
        "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv",
        "/kaggle/data/cassava-leaf-disease-classification/sample_submission.csv",
        "/kaggle/input/sample_submission.csv",
        "/kaggle/data/sample_submission.csv",
    ]
)

test_dir = find_existing_path(
    [
        "../input/cassava-leaf-disease-classification/test_images/",
        "/kaggle/input/cassava-leaf-disease-classification/test_images/",
        "/kaggle/data/cassava-leaf-disease-classification/test_images/",
        "/kaggle/input/test_images/",
        "/kaggle/data/test_images/",
    ]
)

ss = pd.read_csv(sample_csv)
image_paths = (test_dir.rstrip("/") + "/" + ss["image_id"]).tolist()

missing = [p for p in image_paths[:50] if not os.path.exists(p)]
if missing:
    raise FileNotFoundError(
        "Some test image paths do not exist (showing up to 5):\n"
        + "\n".join(missing[:5])
    )

preprocess = keras.applications.efficientnet.preprocess_input


def load_and_preprocess(path):
    img = keras.preprocessing.image.load_img(
        path, color_mode="rgb", target_size=(300, 300)
    )
    arr = keras.preprocessing.image.img_to_array(img)
    arr = preprocess(arr)
    return arr




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2075748579.py in <cell line: 0>()
     38     )
     39 
---> 40 preprocess = keras.applications.efficientnet.preprocess_input
     41 
     42 

NameError: name 'keras' is not defined

## === cell 3
batch_size = 32
n = len(image_paths)
preds = []

for start in range(0, n, batch_size):
    batch_paths = image_paths[start : start + batch_size]
    batch = np.stack([load_and_preprocess(p) for p in batch_paths], axis=0)
    prob = model.predict(batch, verbose=0)
    preds.extend(np.argmax(prob, axis=1).astype(int).tolist())

my_submission = pd.DataFrame({"image_id": ss["image_id"], "label": preds})
my_submission.to_csv("submission.csv", index=False)

assert list(my_submission.columns) == ["image_id", "label"]
assert len(my_submission) == len(ss)
print("Wrote submission.csv with", len(my_submission), "rows")
print(my_submission.head())

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2800166004.py in <cell line: 0>()
      5 for start in range(0, n, batch_size):
      6     batch_paths = image_paths[start : start + batch_size]
----> 7     batch = np.stack([load_and_preprocess(p) for p in batch_paths], axis=0)
      8     prob = model.predict(batch, verbose=0)
      9     preds.extend(np.argmax(prob, axis=1).astype(int).tolist())

/tmp/ipykernel_55/2800166004.py in <listcomp>(.0)
      5 for start in range(0, n, batch_size):
      6     batch_paths = image_paths[start : start + batch_size]
----> 7     batch = np.stack([load_and_preprocess(p) for p in batch_paths], axis=0)
      8     prob = model.predict(batch, verbose=0)
      9     preds.extend(np.argmax(prob, axis=1).astype(int).tolist())

NameError: name 'load_and_preprocess' is not defined
