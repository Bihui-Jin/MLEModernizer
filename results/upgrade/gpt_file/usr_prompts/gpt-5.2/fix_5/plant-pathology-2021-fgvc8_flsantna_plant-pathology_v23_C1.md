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
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
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
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Target score

0.7599261311172675

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"

import pandas as pd
import tensorflow as tf



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_55/2968253606.py in <cell line: 0>()
      7 
      8 import pandas as pd
----> 9 import tensorflow as tf
     10 

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
output_dir = "./"
test_dir = "../input/plant-pathology-2021-fgvc8/test_images/"
model_dir = "../input/model-effb7-01/epoch-7/epoch-7"

image_dims = (300, 300, 3)
data_set = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")
df_labels = data_set["labels"]
one_hot = df_labels.str.get_dummies(sep=" ")
dataset_labels = one_hot.columns.to_list()



## === cell 2
if __name__ == "__main__":
    import keras

    def find_model_artifact(path: str):
        """
        Returns a tuple (kind, resolved_path) where kind is one of:
        - "savedmodel": directory containing saved_model.pb/pbtxt
        - "keras": path to a .keras or .h5 file
        """
        path = os.path.expanduser(path)

        if os.path.isfile(path):
            lower = path.lower()
            if (
                lower.endswith(".keras")
                or lower.endswith(".h5")
                or lower.endswith(".hdf5")
            ):
                return "keras", path

        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for fn in files:
                    lower = fn.lower()
                    if (
                        lower.endswith(".keras")
                        or lower.endswith(".h5")
                        or lower.endswith(".hdf5")
                    ):
                        return "keras", os.path.join(root, fn)

            if os.path.exists(os.path.join(path, "saved_model.pb")) or os.path.exists(
                os.path.join(path, "saved_model.pbtxt")
            ):
                return "savedmodel", path
            for root, _, files in os.walk(path):
                if "saved_model.pb" in files or "saved_model.pbtxt" in files:
                    return "savedmodel", root

        raise FileNotFoundError(
            f"Could not find a TensorFlow SavedModel (saved_model.pb/pbtxt) or Keras model (.keras/.h5) under: {path}"
        )

    def try_resolve_model_dir(primary_path: str):
        if os.path.exists(primary_path):
            return primary_path

        base = "../input"
        if not os.path.isdir(base):
            return primary_path

        candidate_root = os.path.join(base, "model-effb7-01")
        if os.path.exists(candidate_root):
            return candidate_root

        for root, dirs, _ in os.walk(base):
            for d in dirs:
                if d == "model-effb7-01":
                    return os.path.join(root, d)
        return primary_path

    resolved_model_root = try_resolve_model_dir(model_dir)
    kind, resolved_model_path = find_model_artifact(resolved_model_root)

    if kind == "savedmodel":
        loaded_sm = tf.saved_model.load(resolved_model_path)
        available_endpoints = list(getattr(loaded_sm, "signatures", {}).keys())
        call_endpoint = (
            "serving_default"
            if "serving_default" in available_endpoints
            else (available_endpoints[0] if available_endpoints else "serving_default")
        )
        predictor = keras.layers.TFSMLayer(
            resolved_model_path, call_endpoint=call_endpoint
        )
    else:
        model = keras.models.load_model(resolved_model_path, compile=False)

        def predictor(x, training=False):
            return model(x, training=training)

    sample_sub = pd.read_csv(
        "../input/plant-pathology-2021-fgvc8/sample_submission.csv"
    )
    images_path_list = sample_sub["image"].tolist()

    def test_on_sub(index: int):
        img_name = images_path_list[index]
        img_path = os.path.join(test_dir, img_name)
        input_img = tf.io.read_file(img_path)
        image = tf.io.decode_image(
            contents=input_img,
            channels=3,
            dtype=tf.dtypes.float32,
            expand_animations=False,
        )
        tensor_image = tf.image.resize(image, [image_dims[0], image_dims[1]])
        name_jpg = os.path.basename(img_name)
        return name_jpg, tf.expand_dims(tensor_image, axis=0)

    values = []
    for i in range(len(images_path_list)):
        name, images = test_on_sub(index=i)
        images = images * 255.0

        pred = predictor(images, training=False)
        if isinstance(pred, dict):
            pred = pred[next(iter(pred.keys()))]
        elif isinstance(pred, (list, tuple)):
            pred = pred[0]

        pred = tf.convert_to_tensor(pred)
        index_values = [j for j, v in enumerate(pred[0].numpy().tolist()) if v > 0.6]

        classes_img = ""
        for j in index_values:
            classes_img = str(dataset_labels[j]) + " " + classes_img

        classes_img = classes_img.strip()
        if classes_img == "":
            classes_img = "healthy"

        values.append([name, classes_img])

    csv_pd = pd.DataFrame(values, columns=["image", "labels"], index=None)
    out_path = os.path.join(output_dir, "submission.csv")
    csv_pd.to_csv(out_path, index=False)
    print(f"Wrote submission to: {out_path} (rows={len(csv_pd)})")
    print(f"Model loaded from: {resolved_model_path} (kind={kind})")

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_55/609193832.py in <cell line: 0>()
      1 if __name__ == "__main__":
----> 2     import keras
      3 
      4     def find_model_artifact(path: str):
      5         """

/usr/local/lib/python3.11/dist-packages/keras/__init__.py in <module>
      1 # DO NOT EDIT. Generated by api_gen.sh
----> 2 from keras.api import DTypePolicy
      3 from keras.api import FloatDTypePolicy
      4 from keras.api import Function
      5 from keras.api import Initializer

/usr/local/lib/python3.11/dist-packages/keras/api/__init__.py in <module>
      6 
      7 
----> 8 from keras.api import activations
      9 from keras.api import applications
     10 from keras.api import backend

/usr/local/lib/python3.11/dist-packages/keras/api/activations/__init__.py in <module>
      5 """
      6 
----> 7 from keras.src.activations import deserialize
      8 from keras.src.activations import get
      9 from keras.src.activations import serialize

/usr/local/lib/python3.11/dist-packages/keras/src/__init__.py in <module>
----> 1 from keras.src import activations
      2 from keras.src import applications
      3 from keras.src import backend
      4 from keras.src import constraints
      5 from keras.src import datasets

/usr/local/lib/python3.11/dist-packages/keras/src/activations/__init__.py in <module>
      1 import types
      2 
----> 3 from keras.src.activations.activations import celu
      4 from keras.src.activations.activations import elu
      5 from keras.src.activations.activations import exponential

/usr/local/lib/python3.11/dist-packages/keras/src/activations/activations.py in <module>
----> 1 from keras.src import backend
      2 from keras.src import ops
      3 from keras.src.api_export import keras_export
      4 
      5 

/usr/local/lib/python3.11/dist-packages/keras/src/backend/__init__.py in <module>
      8 
      9 from keras.src.api_export import keras_export
---> 10 from keras.src.backend.common.dtypes import result_type
     11 from keras.src.backend.common.keras_tensor import KerasTensor
     12 from keras.src.backend.common.keras_tensor import any_symbolic_tensors

/usr/local/lib/python3.11/dist-packages/keras/src/backend/common/__init__.py in <module>
      1 from keras.src.backend.common import backend_utils
----> 2 from keras.src.backend.common.dtypes import result_type
      3 from keras.src.backend.common.variables import AutocastScope
      4 from keras.src.backend.common.variables import Variable as KerasVariable
      5 from keras.src.backend.common.variables import get_autocast_scope

/usr/local/lib/python3.11/dist-packages/keras/src/backend/common/dtypes.py in <module>
      3 from keras.src.api_export import keras_export
      4 from keras.src.backend import config
----> 5 from keras.src.backend.common.variables import standardize_dtype
      6 
      7 BOOL_TYPES = ("bool",)

/usr/local/lib/python3.11/dist-packages/keras/src/backend/common/variables.py in <module>
      9 from keras.src.backend.common.stateless_scope import get_stateless_scope
     10 from keras.src.backend.common.stateless_scope import in_stateless_scope
---> 11 from keras.src.utils.module_utils import tensorflow as tf
     12 from keras.src.utils.naming import auto_name
     13 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/__init__.py in <module>
----> 1 from keras.src.utils.audio_dataset_utils import audio_dataset_from_directory
      2 from keras.src.utils.dataset_utils import split_dataset
      3 from keras.src.utils.file_utils import get_file
      4 from keras.src.utils.image_dataset_utils import image_dataset_from_directory
      5 from keras.src.utils.image_utils import array_to_img

/usr/local/lib/python3.11/dist-packages/keras/src/utils/audio_dataset_utils.py in <module>
      2 
      3 from keras.src.api_export import keras_export
----> 4 from keras.src.utils import dataset_utils
      5 from keras.src.utils.module_utils import tensorflow as tf
      6 from keras.src.utils.module_utils import tensorflow_io as tfio

/usr/local/lib/python3.11/dist-packages/keras/src/utils/dataset_utils.py in <module>
      7 import numpy as np
      8 
----> 9 from keras.src import tree
     10 from keras.src.api_export import keras_export
     11 from keras.src.utils import io_utils

/usr/local/lib/python3.11/dist-packages/keras/src/tree/__init__.py in <module>
----> 1 from keras.src.tree.tree_api import assert_same_paths
      2 from keras.src.tree.tree_api import assert_same_structure
      3 from keras.src.tree.tree_api import flatten
      4 from keras.src.tree.tree_api import flatten_with_path
      5 from keras.src.tree.tree_api import is_nested

/usr/local/lib/python3.11/dist-packages/keras/src/tree/tree_api.py in <module>
      6 
      7 if optree.available:
----> 8     from keras.src.tree import optree_impl as tree_impl
      9 elif dmtree.available:
     10     from keras.src.tree import dmtree_impl as tree_impl

/usr/local/lib/python3.11/dist-packages/keras/src/tree/optree_impl.py in <module>
     11 # Register backend-specific node classes
     12 if backend() == "tensorflow":
---> 13     from tensorflow.python.trackable.data_structures import ListWrapper
     14     from tensorflow.python.trackable.data_structures import _DictWrapper
     15 

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
