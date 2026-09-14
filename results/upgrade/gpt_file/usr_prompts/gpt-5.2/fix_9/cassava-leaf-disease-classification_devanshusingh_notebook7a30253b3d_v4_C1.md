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

0.053

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.11958) has done: 'The timeout is driven mainly by per-image Python I/O and `model.predict()` calls inside a loop (2676 separate predicts) plus relatively slow CPU image augmentation/feeding during training. To fix this without changing the model or training semantics, I batch the test inference using a `tf.data` pipeline (vectorized decode/resize/rescale + batched `model.predict`) and I speed up training input by enabling generator shuffling (no semantic change), using `workers`/`use_multiprocessing` for background preprocessing, and avoiding unnecessary plotting/prediction work. I also switch protobuf to the faster default implementation (the current forced “python” implementation is slower) while keeping determinism controls so results remain stable aside from negligible float noise.'
- What this solution (achieved 0.11958) has done: 'The timeout is most likely coming from slow Python-side image loading/augmentation in `ImageDataGenerator.flow_from_dataframe` and from not using worker multiprocessing; the model itself is small and prediction is already using a fast `tf.data` pipeline. I keep the exact same model and training loop semantics, but speed up input throughput by enabling multi-worker prefetching for `.fit()` and by ensuring we don’t do any unnecessary generator work. I also make `steps_per_epoch`/`validation_steps` cover the full dataset (same semantics as “one pass”), and add a deterministic tf.data-based training pipeline only if the pretrained model is missing is **not allowed** (would change core approach), so we stay with the generator but optimize its execution. Finally, I keep determinism and paths unchanged.'
- What this solution (achieved 0.11958) has done: 'Main bottlenecks are (1) forcing pure-Python protobuf (very slow TFRecord/graph ops), (2) Keras `ImageDataGenerator` I/O/augmentation overhead with small batches and multiprocessing, and (3) extra overhead from OpenCV/Matplotlib imports that aren’t used in the timed path. The changes below keep the same model, loss, optimizer, epochs, and data splits, but speed up by letting TensorFlow use the default optimized protobuf backend, reducing Python-side generator overhead (disable multiprocessing for generators to avoid IPC/pickling costs on Kaggle), and ensuring deterministic, efficient tf.data pipelines for test inference. All paths and core training/inference semantics remain the same; only performance-relevant plumbing is adjusted.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "cpp")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

SEED = 42
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
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
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

print("TF version:", tf.__version__)
print("Keras version:", tf.keras.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/929424835.py in <cell line: 0>()
      9 import pandas as pd
     10 
---> 11 import tensorflow as tf
     12 from tensorflow.keras.preprocessing.image import ImageDataGenerator
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
train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"
label_json_path = (
    "../input/cassava-leaf-disease-classification/label_num_to_disease_map.json"
)
images_dir_path = "../input/cassava-leaf-disease-classification/train_images"



## === cell 2
train_csv = pd.read_csv(train_csv_path)
train_csv["label"] = train_csv["label"].astype("string")

label_class = pd.read_json(label_json_path, orient="index")
label_class = label_class.values.flatten().tolist()



## === cell 3
print(train_csv.head())
print(train_csv.shape)



## === cell 4
train_csv = train_csv.reset_index(drop=True)



## === cell 5
print(train_csv.dtypes)



## === cell 6
print("Label names :")
for i, label in enumerate(label_class):
    print(f" {i}. {label}")



## === cell 7
BATCH_SIZE = 50
IMG_SIZE = 200



## === cell 8
train_gen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    rescale=1 / 255.0,
    validation_split=0.2,
)

valid_gen = ImageDataGenerator(
    rescale=1 / 255.0,
    validation_split=0.2,
)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/300521450.py in <cell line: 0>()
----> 1 train_gen = ImageDataGenerator(
      2     rotation_range=40,
      3     width_shift_range=0.2,
      4     height_shift_range=0.2,
      5     shear_range=0.2,

NameError: name 'ImageDataGenerator' is not defined

## === cell 9
train_generator = train_gen.flow_from_dataframe(
    dataframe=train_csv,
    directory=images_dir_path,
    x_col="image_id",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
    subset="training",
)

valid_generator = valid_gen.flow_from_dataframe(
    dataframe=train_csv,
    directory=images_dir_path,
    x_col="image_id",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=False,
    subset="validation",
)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2833211644.py in <cell line: 0>()
----> 1 train_generator = train_gen.flow_from_dataframe(
      2     dataframe=train_csv,
      3     directory=images_dir_path,
      4     x_col="image_id",
      5     y_col="label",

NameError: name 'train_gen' is not defined

## === cell 10
SHOW_BATCH = False

if SHOW_BATCH:
    import matplotlib.pyplot as plt

    batch = next(train_generator)
    images = batch[0]
    labels = batch[1]

    plt.figure(figsize=(12, 9))
    for i, (img, label) in enumerate(zip(images, labels)):
        plt.subplot(2, 3, i % 6 + 1)
        plt.axis("off")
        plt.imshow(img)
        plt.title(label_class[int(np.argmax(label))])
        if i == 15:
            break



## === cell 11
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(5))
model.add(Activation("softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/110580054.py in <cell line: 0>()
----> 1 from keras.models import Sequential
      2 from keras.layers import Conv2D, MaxPooling2D
      3 from keras.layers import Activation, Dropout, Flatten, Dense
      4 
      5 model = Sequential()

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

## === cell 12
def scheduler(epoch, lr):
    if epoch > 6 and epoch % 2 == 0:
        lr = lr / 1.5
        return lr
    else:
        return lr


callback0 = tf.keras.callbacks.ModelCheckpoint(
    "./CasavaLeafDiseaseModel.h5", monitor="val_loss", save_best_only=True
)
callback1 = tf.keras.callbacks.LearningRateScheduler(scheduler)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/82240149.py in <cell line: 0>()
      7 
      8 
----> 9 callback0 = tf.keras.callbacks.ModelCheckpoint(
     10     "./CasavaLeafDiseaseModel.h5", monitor="val_loss", save_best_only=True
     11 )

NameError: name 'tf' is not defined

## === cell 13
PRETRAINED_PATH = (
    "../input/casavaleafdiseasemodel-tf/CasavaLeafDiseaseModel_epoch_12_acc_85.h5"
)
loaded_pretrained = False
try:
    model = tf.keras.models.load_model(PRETRAINED_PATH)
    loaded_pretrained = True
    print("Loaded saved model:", PRETRAINED_PATH)
except Exception as e:
    print("No saved model. So Train the model !")
    print("Load error:", repr(e))



## === cell 14
his = None
if not loaded_pretrained:
    import math

    steps_per_epoch = max(1, int(math.ceil(train_generator.n / BATCH_SIZE)))
    validation_steps = max(1, int(math.ceil(valid_generator.n / BATCH_SIZE)))

    his = model.fit(
        x=train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=10,
        validation_data=valid_generator,
        validation_steps=validation_steps,
        callbacks=[callback0, callback1],
        verbose=2,
        workers=max(2, (os.cpu_count() or 4) // 2),
        use_multiprocessing=True,
        max_queue_size=20,
    )



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3525461276.py in <cell line: 0>()
      3     import math
      4 
----> 5     steps_per_epoch = max(1, int(math.ceil(train_generator.n / BATCH_SIZE)))
      6     validation_steps = max(1, int(math.ceil(valid_generator.n / BATCH_SIZE)))
      7 

NameError: name 'train_generator' is not defined

## === cell 15
RUN_VALID_PRED_DEBUG = False
if RUN_VALID_PRED_DEBUG:
    print(model.predict(next(valid_generator)[0]))



## === cell 16
if his is not None:
    stats = pd.DataFrame(his.history)
    print(stats.tail())
else:
    print("Training skipped (pretrained model loaded).")



## === cell 17
SHOW_TEST_EXAMPLE = False

test_dir = "../input/cassava-leaf-disease-classification/test_images"
if SHOW_TEST_EXAMPLE:
    import cv2
    import matplotlib.pyplot as plt

    test_images = sorted(
        [f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")]
    )
    test_img_path = os.path.join(test_dir, test_images[0]) if test_images else None
    print("Example test image:", test_img_path)

    if test_img_path is not None:
        img = cv2.imread(test_img_path)
        if img is not None:
            resized_img = (
                cv2.resize(img, (IMG_SIZE, IMG_SIZE)).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
                / 255.0
            )
            plt.figure(figsize=(8, 4))
            plt.title("TEST IMAGE")
            plt.imshow(resized_img[0][:, :, ::-1])  # BGR->RGB for display
            plt.axis("off")
        else:
            print("cv2.imread failed for:", test_img_path)



## === cell 18
ss = pd.read_csv("../input/cassava-leaf-disease-classification/sample_submission.csv")

test_paths = (
    ("../input/cassava-leaf-disease-classification/test_images/") + ss["image_id"]
).to_numpy()

AUTOTUNE = tf.data.AUTOTUNE


@tf.function
def _load_and_preprocess(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize_with_pad(
        img, IMG_SIZE, IMG_SIZE, method="bilinear", antialias=False
    )
    img = tf.cast(img, tf.float32) / 255.0
    return img


ds = tf.data.Dataset.from_tensor_slices(test_paths)
opts = tf.data.Options()
opts.experimental_deterministic = True
ds = ds.with_options(opts)

ds = ds.map(_load_and_preprocess, num_parallel_calls=AUTOTUNE).cache()
ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

probs = model.predict(ds, verbose=0)
preds = np.argmax(probs, axis=1).astype(int).tolist()

my_submission = pd.DataFrame({"image_id": ss["image_id"].values, "label": preds})
my_submission.to_csv("submission.csv", index=False)



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1098474567.py in <cell line: 0>()
      7 ).to_numpy()
      8 
----> 9 AUTOTUNE = tf.data.AUTOTUNE
     10 
     11 

NameError: name 'tf' is not defined

## === cell 19
print("Submission File: \n---------------\n")
print(my_submission.head())
print("\nWrote:", os.path.abspath("submission.csv"), "rows:", len(my_submission))
assert os.path.basename("submission.csv").endswith(".csv")
assert list(my_submission.columns) == ["image_id", "label"]
assert len(my_submission) == len(
    pd.read_csv("../input/cassava-leaf-disease-classification/sample_submission.csv")
)

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1694048329.py in <cell line: 0>()
      1 print("Submission File: \n---------------\n")
----> 2 print(my_submission.head())
      3 print("\nWrote:", os.path.abspath("submission.csv"), "rows:", len(my_submission))
      4 assert os.path.basename("submission.csv").endswith(".csv")
      5 assert list(my_submission.columns) == ["image_id", "label"]

NameError: name 'my_submission' is not defined
