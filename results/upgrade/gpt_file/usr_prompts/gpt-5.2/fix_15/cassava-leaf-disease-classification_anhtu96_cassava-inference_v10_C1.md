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

0.6500453309156845

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.8083) has done: 'I fix the environment crash caused by a protobuf incompatibility by pinning protobuf to a TensorFlow-compatible version at runtime before importing TensorFlow. Then I remove the dependency on a missing external `.h5` model file by training the same kind of image classifier directly from the provided TFRecords (keeping the same general TFRecord parsing + Keras training/inference flow). Finally, I make submission generation robust by collecting predictions for every test image, ordering them to match `sample_submission.csv`, and writing a valid `submission.csv` with the exact required columns.'
- What this solution (achieved 0.8012) has done: 'I fix the crash before TensorFlow imports by removing the protobuf runtime pinning (it’s now causing incompatibility with TF 2.18) and instead rely on the Kaggle environment’s preinstalled protobuf. Then I correct a data loading bug where a Python list of TFRecord filenames was passed into `tf.io.gfile.glob`, which can silently break dataset creation; this should restore proper training/inference behavior without changing the model architecture or training loop. Finally, I keep submission generation the same but make ordering robust by defaulting missing IDs (shouldn’t happen) to a safe class so a valid `submission.csv` is always produced.'
- What this solution (achieved 0.78326) has done: 'I remove the protobuf runtime pinning/restart logic because TensorFlow 2.18 in your environment already works with protobuf 6.x, and the forced downgrade/re-exec is the most likely reason you get “Not yielded” (kernel restart/timeout before writing the CSV). Then I make one minimal training change that doesn’t alter your architecture or pipeline: unfreeze the last ResNet50 stage (conv5) so the model can adapt slightly to cassava images within the same 2-epoch budget, which should improve accuracy toward your target. Finally, I keep your submission logic but add a small safeguard to ensure every `image_id` is unique and ordered exactly like `sample_submission.csv`, so the generated `submission.csv` is always valid.'
- What this solution (achieved 0.78326) has done: 'I fix the TensorFlow import crash caused by an incompatible protobuf C++ implementation by forcing the safe-Python protobuf runtime before importing TensorFlow (this is the direct cause of the `MessageFactory.GetPrototype` error). I keep your model architecture, TFRecord parsing, training loop, and submission logic the same, only adding this compatibility shim plus a small safety fallback if TensorFlow still fails to import. Since your current score (0.78326) is already above the target (0.6500) and within the ±10% tolerance band, I not make any score-improving changes—just ensure the notebook runs end-to-end and reliably writes `submission.csv`.'
- What this solution (achieved 0.78326) has done: 'I fix the TensorFlow/protobuf crash by removing the forced `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` override, which is what triggers the `MessageFactory.GetPrototype` error under TF 2.18 + protobuf 6.x in this environment. I keep your model, TFRecord parsing, training loop, and prediction logic unchanged to avoid unnecessary score drift (your current score is already within ±10% of the target). I also make the TFRecord file discovery and submission generation a bit more robust (stable ordering + duplicate guard) without changing evaluation semantics. The result run end-to-end and always write a valid `submission.csv`.'
- What this solution (achieved 0.78326) has done: 'I fix the TensorFlow import crash by removing the forced protobuf C++ implementation override (it’s causing the `_message` import error in this environment) and explicitly setting protobuf to the safe “python” implementation before importing TensorFlow. Then I keep the rest of your pipeline (TFRecord parsing, ResNet50 model, training loop, and submission ordering) the same, only adding small robustness guards so the dataset discovery always finds files and the submission always matches `sample_submission.csv` exactly. These changes are required to make the notebook run end-to-end and yield a valid `submission.csv`; they should be score-neutral aside from restoring the intended training/inference execution.'
- What this solution (achieved 0.78513) has done: 'I fix the TensorFlow/protobuf crash by removing the forced `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` override, which triggers the `MessageFactory.GetPrototype` error in TF 2.18 + protobuf 6.x. I keep the model, TFRecord parsing, training loop, and prediction logic the same to avoid unnecessary score drift (your current score is already above the target band, so no score-improving changes). I also add a small guard to ensure TFRecord discovery works even if `DATA_DIR` points to a nested directory, and keep submission ordering exactly matching `sample_submission.csv` so a valid `submission.csv` is always produced.'
- What this solution (achieved 0.78326) has done: 'I fix the TensorFlow import crash by forcing protobuf to use the pure-Python implementation before importing TensorFlow, which avoids the `MessageFactory.GetPrototype` AttributeError seen with TF 2.18 + protobuf 6.x in Kaggle. I keep your model, TFRecord parsing, training loop, and prediction logic unchanged to avoid unnecessary score drift (your current score is already above the target band). I also make TFRecord discovery slightly more robust (fallback to nested directory) while preserving the same file patterns and submission ordering, ensuring a valid `submission.csv` is always produced.'
- What this solution (achieved 0.78326) has done: 'I fix the crash in the first cell by removing the protobuf “python implementation” override, which is what triggers the `MessageFactory.GetPrototype` AttributeError under TensorFlow 2.18 + protobuf 6.x in this Kaggle environment. I keep your TFRecord parsing, ResNet50 architecture, unfreezing of conv5, training loop, and inference logic unchanged to avoid unnecessary score drift (your current score is already above the target band). I also make the DATA_DIR discovery not depend on TensorFlow (so it works even if TF import fails early) and keep the submission ordering exactly aligned to `sample_submission.csv` so a valid `submission.csv` is always produced.'
- What this solution (achieved 0.78326) has done: 'I fix the TensorFlow/protobuf import crash in the first cell by setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` *before* importing TensorFlow (and avoiding the current “pop” behavior that triggers the `MessageFactory.GetPrototype` error in this environment). I keep the rest of your pipeline (TFRecord parsing, ResNet50 + conv5 unfreeze, 2-epoch training, and submission ordering) unchanged to avoid unnecessary score drift, since your current score is already above the target band. I also add a tiny safety guard so the submission always has exactly the sample’s row order and no duplicated `image_id`s, without changing prediction semantics.'
- What this solution (achieved 0.78326) has done: 'I fix the immediate TensorFlow/protobuf crash by removing the forced `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` override, which is incompatible with TF 2.18 + protobuf 6.x here and triggers the `MessageFactory.GetPrototype` error. I keep your TFRecord parsing, ResNet50 architecture (including conv5 unfreezing), 2-epoch training loop, and inference logic unchanged to avoid unnecessary score drift (your current score is already above the target band). I also make TFRecord discovery robust to both directory layouts without changing which files are used, and keep submission ordering exactly aligned to `sample_submission.csv` so `submission.csv` is always valid.'

# 9. Code solution

## === cell 0
import os
import sys

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "cpp")
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
except AttributeError as e:
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    import tensorflow as tf

print("Python:", sys.version.split()[0])
print("TensorFlow:", tf.__version__)
try:
    import google.protobuf

    print("protobuf:", google.protobuf.__version__)
    print(
        "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:",
        os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "<unset>"),
    )
except Exception as e:
    print("protobuf: <unavailable>", e)

SEED = 42
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_55/2889582.py in <cell line: 0>()
     12 
     13 try:
---> 14     import tensorflow as tf
     15 except AttributeError as e:
     16     # Fallback: if protobuf C++ runtime is unavailable/mismatched, retry with python.

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
ROOT_CANDIDATES = [
    "/kaggle/input/cassava-leaf-disease-classification/",
    "/kaggle/data/cassava-leaf-disease-classification/",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/",
    "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification/",
]

DATA_DIR = None
for cand in ROOT_CANDIDATES:
    if os.path.exists(cand):
        DATA_DIR = cand
        break

if DATA_DIR is None:
    DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification/"

AUTOTUNE = tf.data.AUTOTUNE

HEIGHT, WIDTH = 240, 240
BATCH_SIZE = 32
NUM_CLASSES = 5

TRAIN_TFRECS = os.path.join(DATA_DIR, "train_tfrecords/ld_train*.tfrec")
TEST_TFRECS = os.path.join(DATA_DIR, "test_tfrecords/ld_test*.tfrec")

train_csv_path = os.path.join(DATA_DIR, "train.csv")
sample_sub_path = os.path.join(DATA_DIR, "sample_submission.csv")

if (not os.path.exists(train_csv_path)) or (not os.path.exists(sample_sub_path)):
    alt_dir = os.path.join(DATA_DIR, "cassava-leaf-disease-classification")
    alt_train_csv = os.path.join(alt_dir, "train.csv")
    alt_sample_sub = os.path.join(alt_dir, "sample_submission.csv")
    alt_train_tfrec = os.path.join(alt_dir, "train_tfrecords/ld_train*.tfrec")
    alt_test_tfrec = os.path.join(alt_dir, "test_tfrecords/ld_test*.tfrec")
    if os.path.exists(alt_train_csv) and os.path.exists(alt_sample_sub):
        DATA_DIR = alt_dir + ("" if alt_dir.endswith("/") else "/")
        train_csv_path = alt_train_csv
        sample_sub_path = alt_sample_sub
        TRAIN_TFRECS = alt_train_tfrec
        TEST_TFRECS = alt_test_tfrec

train_df = pd.read_csv(train_csv_path)
sample_sub = pd.read_csv(sample_sub_path)

print("DATA_DIR:", DATA_DIR)
print("Train rows:", len(train_df), " Sample submission rows:", len(sample_sub))




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3890006159.py in <cell line: 0>()
     15     DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification/"
     16 
---> 17 AUTOTUNE = tf.data.AUTOTUNE
     18 
     19 HEIGHT, WIDTH = 240, 240

NameError: name 'tf' is not defined

## === cell 2
def _parse_function(example, feature_description):
    parsed_example = tf.io.parse_single_example(example, feature_description)
    image = tf.io.decode_jpeg(parsed_example["image"], channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (HEIGHT, WIDTH))
    image = tf.keras.applications.resnet50.preprocess_input(image)

    if "target" in feature_description:
        target = tf.cast(parsed_example["target"], tf.int32)
        return image, target

    return image, parsed_example["image_name"]


def load_data(path_or_files, batch_size=32, train=True, shuffle=False):
    if isinstance(path_or_files, (list, tuple)):
        filenames = list(path_or_files)
    else:
        filenames = tf.io.gfile.glob(path_or_files)

    filenames = sorted(filenames)

    if not filenames:
        raise RuntimeError(f"No TFRecord files found for: {path_or_files}")

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)

    if train:
        feature_description = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "image_name": tf.io.FixedLenFeature([], tf.string),
            "target": tf.io.FixedLenFeature([], tf.int64),
        }
    else:
        feature_description = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "image_name": tf.io.FixedLenFeature([], tf.string),
        }

    dataset = dataset.map(
        lambda x: _parse_function(x, feature_description), num_parallel_calls=AUTOTUNE
    )
    if shuffle:
        dataset = dataset.shuffle(2048, reshuffle_each_iteration=True, seed=SEED)
    dataset = dataset.batch(batch_size).prefetch(AUTOTUNE)
    return dataset




## === cell 3
def build_model():
    base = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(HEIGHT, WIDTH, 3),
        pooling="avg",
    )

    for layer in base.layers:
        layer.trainable = False
    for layer in base.layers:
        if layer.name.startswith("conv5_"):
            layer.trainable = True

    x = tf.keras.layers.Dropout(0.3)(base.output)
    out = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = tf.keras.Model(inputs=base.input, outputs=out)
    return model


train_files = sorted(tf.io.gfile.glob(TRAIN_TFRECS))
if len(train_files) < 2:
    fallback_pattern = os.path.join(DATA_DIR, "train_tfrecords", "ld_train*.tfrec")
    train_files = sorted(tf.io.gfile.glob(fallback_pattern))

if len(train_files) < 2:
    raise RuntimeError(
        "Expected multiple train TFRecord shards but found: %d" % len(train_files)
    )

val_files = train_files[-2:]  # small val on last shards
tr_files = train_files[:-2]

train_ds = load_data(tr_files, batch_size=BATCH_SIZE, train=True, shuffle=True)
val_ds = load_data(val_files, batch_size=BATCH_SIZE, train=True, shuffle=False)

model = build_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2663551073.py in <cell line: 0>()
     19 
     20 
---> 21 train_files = sorted(tf.io.gfile.glob(TRAIN_TFRECS))
     22 if len(train_files) < 2:
     23     fallback_pattern = os.path.join(DATA_DIR, "train_tfrecords", "ld_train*.tfrec")

NameError: name 'tf' is not defined

## === cell 4
EPOCHS = 2

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1,
)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2067493987.py in <cell line: 0>()
      1 EPOCHS = 2
      2 
----> 3 history = model.fit(
      4     train_ds,
      5     validation_data=val_ds,

NameError: name 'model' is not defined

## === cell 5
test_set = load_data(TEST_TFRECS, batch_size=32, train=False, shuffle=False)

test_IDs = []
pred_labels = []

for images, names in test_set:
    probs = model.predict(images, verbose=0)
    preds = np.argmax(probs, axis=1).astype(int)

    names = [n.decode("utf-8") for n in names.numpy().tolist()]
    test_IDs.extend(names)
    pred_labels.extend(preds.tolist())

pred_map = {}
for k, v in zip(test_IDs, pred_labels):
    if k not in pred_map:
        pred_map[k] = int(v)

ordered_ids = sample_sub["image_id"].astype(str).tolist()
if len(ordered_ids) != len(set(ordered_ids)):
    ordered_ids = pd.Index(ordered_ids).drop_duplicates(keep="first").tolist()

ordered_preds = [int(pred_map.get(iid, 0)) for iid in ordered_ids]

submission_df = pd.DataFrame({"image_id": ordered_ids, "label": ordered_preds})
submission_df["label"] = submission_df["label"].astype(int)
submission_df.to_csv("submission.csv", index=False)

print(submission_df.head())
print("Wrote submission.csv with rows:", len(submission_df))
print(
    "Missing IDs filled with class 0:", sum(iid not in pred_map for iid in ordered_ids)
)
print("Unique predicted labels:", submission_df["label"].value_counts().to_dict())

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/196343047.py in <cell line: 0>()
----> 1 test_set = load_data(TEST_TFRECS, batch_size=32, train=False, shuffle=False)
      2 
      3 test_IDs = []
      4 pred_labels = []
      5 

NameError: name 'TEST_TFRECS' is not defined
