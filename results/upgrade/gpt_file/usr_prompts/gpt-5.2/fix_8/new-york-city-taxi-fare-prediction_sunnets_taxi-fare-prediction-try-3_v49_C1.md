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
Predict the fare amount for a taxi ride given the pickup and dropoff locations.

## Metric
Root mean-squared error.

## Submission Format
For each `key` in the test set, you must predict a value for the `fare_amount` variable. The file should contain a header and have the following format:

```
key,fare_amount
2015-01-27 13:08:24.0000002,11.00
2015-02-27 13:08:24.0000002,12.05
2015-03-27 13:08:24.0000002,11.23
2015-04-27 13:08:24.0000002,14.17
2015-05-27 13:08:24.0000002,15.12
etc
```

## Dataset
- **train.csv** - Input features and target `fare_amount` values for the training set (about 55M rows).
- **test.csv** - Input features for the test set (about 10K rows). Your goal is to predict `fare_amount` for each row.
- **sample_submission.csv** - a sample submission file in the correct format (columns `key` and `fare_amount`). This file 'predicts' `fare_amount` to be $`11.35` for all rows, which is the mean `fare_amount` from the training set.

### Data fields
**ID**

- **key** - Unique `string` identifying each row in both the training and test sets. Comprised of **pickup_datetime** plus a unique integer, but this doesn't matter, it should just be used as a unique ID field.Required in your submission CSV. Not necessarily needed in the training set, but could be useful to simulate a 'submission file' while doing cross-validation within the training set.

**Features**

- **pickup_datetime** - `timestamp` value indicating when the taxi ride started.
- **pickup_longitude** - `float` for longitude coordinate of where the taxi ride started.
- **pickup_latitude** - `float` for latitude coordinate of where the taxi ride started.
- **dropoff_longitude** - `float` for longitude coordinate of where the taxi ride ended.
- **dropoff_latitude** - `float` for latitude coordinate of where the taxi ride ended.
- **passenger_count** - `integer` indicating the number of passengers in the taxi ride.

**Target**

- **fare_amount** - `float` dollar amount of the cost of the taxi ride. This value is only in the training set; this is what you are predicting in the test set and it is required in your submission CSV.

# 2. Python version

3.7

# 3. Installed packages

geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
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
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            GCP-Coupons-Instructions.rtf (486 Bytes)
            description.md (100 lines)
            labels.csv (55413943 lines)
            labels.csv.zip (1.6 GB)
            sample_submission.csv (9915 lines)
            sample_submission.csv.zip (76.2 kB)
            test.csv (9915 lines)
            test.csv.zip (273.0 kB)
            train.csv (55423857 lines)
            train.csv.zip (1.6 GB)
            new-york-city-taxi-fare-prediction/
                GCP-Coupons-Instructions.rtf (486 Bytes)
                description.md (100 lines)
                ... and 8 other files
                new-york-city-taxi-fare-prediction/
        input/
            GCP-Coupons-Instructions.rtf (486 Bytes)
            description.md (100 lines)
            labels.csv (55413943 lines)
            labels.csv.zip (1.6 GB)
            sample_submission.csv (9915 lines)
            sample_submission.csv.zip (76.2 kB)
            test.csv (9915 lines)
            test.csv.zip (273.0 kB)
            train.csv (55423857 lines)
            train.csv.zip (1.6 GB)
            new-york-city-taxi-fare-prediction/
                GCP-Coupons-Instructions.rtf (486 Bytes)
                description.md (100 lines)
                ... and 8 other files
                new-york-city-taxi-fare-prediction/
        working/
            new-york-city-taxi-fare-prediction/
                GCP-Coupons-Instructions.rtf (486 Bytes)
                description.md (100 lines)
                ... and 8 other files
                new-york-city-taxi-fare-prediction/
```

-> data/labels.csv has 55413942 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/new-york-city-taxi-fare-prediction/labels.csv has 55413942 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/new-york-city-taxi-fare-prediction/sample_submission.csv has 9914 rows and 2 columns.
The columns are: key, fare_amount

-> data/new-york-city-taxi-fare-prediction/test.csv has 9914 rows and 7 columns.
The columns are: key, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/new-york-city-taxi-fare-prediction/train.csv has 55423856 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/sample_submission.csv has 9914 rows and 2 columns.
The columns are: key, fare_amount

-> data/test.csv has 9914 rows and 7 columns.
The columns are: key, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/train.csv has 55423856 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> (stopped after 10 files for performance)

# 5. Target score

18.47546

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 15.21252) has done: 'I make the notebook run end-to-end in the Kaggle offline environment by removing the URL-based water-mask dependency (which currently crashes) and fixing Keras 3 API incompatibilities (optimizer constructor, missing vis_utils, and compile/fit failures). I keep the model architecture and training loop the same, but fix a scaling bug where you accidentally refit the scaler on validation data (data leakage), which should modestly worsen the score (increase RMSE) and move you closer to the target. I also correct minor logic errors in the time-feature helper functions (they currently always return 1), and ensure the CSV submission is written with the required columns and a `.csv` suffix. Finally, I keep the original paths but add a safe fallback to `/kaggle/input/` because your provided environment uses that layout.'
- What this solution (achieved 93.22692) has done: 'I fix the Keras backend/runtime errors by switching the custom `rmse` metric to use `tf_keras.backend` (which provides `sqrt/mean/square` in this environment) while keeping the model, loss, and training loop unchanged. I also prevent the early import crash (`MessageFactory.GetPrototype`) by avoiding the standalone `keras` package and consistently using `tf_keras` for models/layers/optimizers/regularizers. These changes are score-neutral (they only restore compatibility) and allow training, evaluation, prediction, and writing a valid `.csv` submission to complete end-to-end. I keep all paths, hyperparameters, and feature logic the same.'
- What this solution (achieved 50.46953) has done: 'I fix the immediate crash (`MessageFactory.GetPrototype`) by forcing protobuf to use the pure-Python implementation before any TensorFlow/Keras import, which is a common Kaggle/offline compatibility issue. I also correct the training CSV read (`usecols` currently drops the `fare_amount` target), which can silently break training/labels and is the most likely cause of the very poor RMSE. Finally, I make the path fallback robust for your provided `/kaggle/input/...` layout (including the nested competition folder) and keep the model/training loop/feature engineering unchanged, ensuring a valid `submissiontry_water.csv` is always written.'
- What this solution (achieved 94.07505) has done: 'We fix the immediate protobuf/TensorFlow import crash by setting the additional environment flag that disables the C++ protobuf implementation (this is the root cause of the `MessageFactory.GetPrototype` error in many Kaggle offline images). Then we fix a major logic issue that hurts RMSE: the training CSV currently drops the `key` column, and downstream you also drop `passenger_count`, which is a useful feature; adding `key` to `usecols` and keeping `passenger_count` restores the intended submission alignment and improves predictive signal without changing the model/training loop. Finally, we make the submission writer robust by enforcing the correct column order and ensuring predictions are 1D floats so the CSV matches Kaggle’s expected format exactly.'
- What this solution (achieved 213.41556) has done: 'I fix the protobuf/TensorFlow import crash that prevents the notebook from running by pinning protobuf to the pure-Python implementation and ensuring that `keras` is never imported (only `tf_keras`), which is the root cause of the `MessageFactory.GetPrototype` error. Then, to move RMSE substantially toward your target (lower is better), I fix a key logic issue: you currently train the model to output raw dollars while scaling only the inputs; with this network/regularization, that often collapses to poor predictions—so I minimally add target scaling (fit on train labels only, inverse-transform for prediction) while keeping the same model, loss, and training loop. Finally, I add a small guard to ensure `pickup_datetime` parsing never produces invalid integer dtypes (avoiding occasional NaT-related crashes) and keep the submission format exactly `key,fare_amount` with a `.csv` suffix.'
- What this solution (achieved 289.38361) has done: 'We fix the crash happening before training by preventing the standalone `keras` stack from being imported indirectly (this is what typically triggers the protobuf `MessageFactory.GetPrototype` error), and instead consistently route Keras usage through `tf_keras` (TensorFlow’s bundled Keras). We also add a robust fallback for the data paths to match the provided `/kaggle/input/...` layout without changing any I/O semantics. These changes are execution/stability fixes and should be score-neutral relative to your current logic (same features, same model, same training loop, same scaling). Finally, we keep the submission writer as-is but add a small safety check to ensure the output CSV is valid and non-empty.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["KERAS_BACKEND"] = "tensorflow"

try:
    import google.protobuf  # noqa: F401
    from google.protobuf import message_factory as _mf  # noqa: F401

    _ = _mf.MessageFactory().GetPrototype  # noqa: F841
except Exception:
    import sys
    import subprocess

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "protobuf==3.20.3"]
    )

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import tf_keras
from tf_keras.models import Sequential
from tf_keras.layers import Dense, BatchNormalization
from tf_keras import optimizers
from tf_keras import regularizers
from tf_keras import backend

TRAIN_PATH = "../input/train.csv"
TEST_PATH = "../input/test.csv"

if not os.path.exists(TRAIN_PATH):
    if os.path.exists("/kaggle/input/train.csv"):
        TRAIN_PATH = "/kaggle/input/train.csv"
    elif os.path.exists("/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"):
        TRAIN_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/train.csv"
    elif os.path.exists(
        "/kaggle/input/new-york-city-taxi-fare-prediction/new-york-city-taxi-fare-prediction/train.csv"
    ):
        TRAIN_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/new-york-city-taxi-fare-prediction/train.csv"
    elif os.path.exists("/kaggle/input/labels.csv"):
        TRAIN_PATH = "/kaggle/input/labels.csv"
    elif os.path.exists("/kaggle/input/new-york-city-taxi-fare-prediction/labels.csv"):
        TRAIN_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/labels.csv"

if not os.path.exists(TEST_PATH):
    if os.path.exists("/kaggle/input/test.csv"):
        TEST_PATH = "/kaggle/input/test.csv"
    elif os.path.exists("/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"):
        TEST_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/test.csv"
    elif os.path.exists(
        "/kaggle/input/new-york-city-taxi-fare-prediction/new-york-city-taxi-fare-prediction/test.csv"
    ):
        TEST_PATH = "/kaggle/input/new-york-city-taxi-fare-prediction/new-york-city-taxi-fare-prediction/test.csv"

SUBMISSION_NAME = "submissiontry_water.csv"  # must end with .csv

BATCH_SIZE = 256
EPOCHS = 30
LEARNING_RATE = 0.001
DATASET_SIZE = 80000

np.random.seed(1)

print("Resolved TRAIN_PATH:", TRAIN_PATH)
print("Resolved TEST_PATH:", TEST_PATH)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/950256650.py in <cell line: 0>()
     31 from sklearn.model_selection import train_test_split
     32 
---> 33 import tf_keras
     34 from tf_keras.models import Sequential
     35 from tf_keras.layers import Dense, BatchNormalization

/usr/local/lib/python3.11/dist-packages/tf_keras/__init__.py in <module>
      1 """AUTOGENERATED. DO NOT EDIT."""
      2 
----> 3 from tf_keras import __internal__
      4 from tf_keras import activations
      5 from tf_keras import applications

/usr/local/lib/python3.11/dist-packages/tf_keras/__internal__/__init__.py in <module>
      1 """AUTOGENERATED. DO NOT EDIT."""
      2 
----> 3 from tf_keras.__internal__ import backend
      4 from tf_keras.__internal__ import layers
      5 from tf_keras.__internal__ import losses

/usr/local/lib/python3.11/dist-packages/tf_keras/__internal__/backend/__init__.py in <module>
      1 """AUTOGENERATED. DO NOT EDIT."""
      2 
----> 3 from tf_keras.src.backend import _initialize_variables as initialize_variables
      4 from tf_keras.src.backend import track_variable

/usr/local/lib/python3.11/dist-packages/tf_keras/src/__init__.py in <module>
     19 """
     20 
---> 21 from tf_keras.src import applications
     22 from tf_keras.src import distribute
     23 from tf_keras.src import layers

/usr/local/lib/python3.11/dist-packages/tf_keras/src/applications/__init__.py in <module>
     16 
     17 
---> 18 from tf_keras.src.applications.convnext import ConvNeXtBase
     19 from tf_keras.src.applications.convnext import ConvNeXtLarge
     20 from tf_keras.src.applications.convnext import ConvNeXtSmall

/usr/local/lib/python3.11/dist-packages/tf_keras/src/applications/convnext.py in <module>
     24 
     25 import numpy as np
---> 26 import tensorflow.compat.v2 as tf
     27 
     28 from tf_keras.src import backend

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
     12 
     13 
---> 14 from tensorflow.core.framework import tensor_pb2 as tensorflow_dot_core_dot_framework_dot_tensor__pb2
     15 from tensorflow.core.framework import tensor_shape_pb2 as tensorflow_dot_core_dot_framework_dot_tensor__shape__pb2
     16 from tensorflow.core.framework import types_pb2 as tensorflow_dot_core_dot_framework_dot_types__pb2

/usr/local/lib/python3.11/dist-packages/tensorflow/core/framework/tensor_pb2.py in <module>
     12 
     13 
---> 14 from tensorflow.core.framework import resource_handle_pb2 as tensorflow_dot_core_dot_framework_dot_resource__handle__pb2
     15 from tensorflow.core.framework import tensor_shape_pb2 as tensorflow_dot_core_dot_framework_dot_tensor__shape__pb2
     16 from tensorflow.core.framework import types_pb2 as tensorflow_dot_core_dot_framework_dot_types__pb2

/usr/local/lib/python3.11/dist-packages/tensorflow/core/framework/resource_handle_pb2.py in <module>
     12 
     13 
---> 14 from tensorflow.core.framework import tensor_shape_pb2 as tensorflow_dot_core_dot_framework_dot_tensor__shape__pb2
     15 from tensorflow.core.framework import types_pb2 as tensorflow_dot_core_dot_framework_dot_types__pb2
     16 

/usr/local/lib/python3.11/dist-packages/tensorflow/core/framework/tensor_shape_pb2.py in <module>
     14 
     15 
---> 16 DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n,tensorflow/core/framework/tensor_shape.proto\x12\ntensorflow\"z\n\x10TensorShapeProto\x12-\n\x03\x64im\x18\x02 \x03(\x0b\x32 .tensorflow.TensorShapeProto.Dim\x12\x14\n\x0cunknown_rank\x18\x03 \x01(\x08\x1a!\n\x03\x44im\x12\x0c\n\x04size\x18\x01 \x01(\x03\x12\x0c\n\x04name\x18\x02 \x01(\tB\x87\x01\n\x18org.tensorflow.frameworkB\x11TensorShapeProtosP\x01ZSgithub.com/tensorflow/tensorflow/tensorflow/go/core/framework/tensor_shape_go_proto\xf8\x01\x01\x62\x06proto3')
     17 
     18 _builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())

/usr/local/lib/python3.11/dist-packages/google/protobuf/descriptor_pool.py in AddSerializedFile(self, serialized_file_desc_proto)
    186           if isinstance(desc, descriptor.EnumValueDescriptor):
    187             error_msg += ('\nNote: enum values appear as '
--> 188                           'siblings of the enum type instead of '
    189                           'children of it.')
    190 

/usr/local/lib/python3.11/dist-packages/google/protobuf/descriptor_pb2.py in <module>
   1853 else:
   1854   _builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
-> 1855 _builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'google.protobuf.descriptor_pb2', globals())
   1856 if _descriptor._USE_C_DESCRIPTORS == False:
   1857 

/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/builder.py in BuildTopDescriptorsAndMessages(file_des, module_name, module)
    106   # Build messages.
    107   for (name, msg_des) in file_des.message_types_by_name.items():
--> 108     module[name] = BuildMessage(msg_des)
    109 
    110 

/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/builder.py in BuildMessage(msg_des)
     83     create_dict['DESCRIPTOR'] = msg_des
     84     create_dict['__module__'] = module_name
---> 85     message_class = _reflection.GeneratedProtocolMessageType(
     86         msg_des.name, (_message.Message,), create_dict)
     87     _sym_db.RegisterMessage(message_class)

/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/python_message.py in __init__(cls, name, bases, dictionary)
    176     # If this is an _existing_ class looked up via `_concrete_class` in the
    177     # __new__ method above, then we don't need to re-initialize anything.
--> 178     existing_class = getattr(descriptor, '_concrete_class', None)
    179     if existing_class:
    180       assert existing_class is cls, (

/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/python_message.py in _AttachFieldHelpers(cls, field_descriptor)
    271 
    272 
--> 273 def _IsMapField(field):
    274   return (field.type == _FieldDescriptor.TYPE_MESSAGE and
    275           field.message_type.has_options and

/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/python_message.py in _DefaultValueConstructorForField(field)
    425   if _IsMapField(field):
    426     return _GetInitializeDefaultForMap(field)
--> 427 
    428   if field.label == _FieldDescriptor.LABEL_REPEATED:
    429     if field.has_default_value and field.default_value != []:

/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/python_message.py in _IsMapField(field)
    262                              '_oneofs']
    263 
--> 264 
    265 def _IsMessageSetExtension(field):
    266   return (field.is_extension and

/usr/local/lib/python3.11/dist-packages/google/protobuf/descriptor.py in type(self)
    779     index (int): 0-based index giving the order of the oneof field inside
    780       its containing type.
--> 781     containing_type (Descriptor): :class:`Descriptor` of the protocol message
    782       type that contains this field.  Set by the :class:`Descriptor` constructor
    783       if we're passed into one.

/usr/local/lib/python3.11/dist-packages/google/protobuf/descriptor.py in _GetFeatures(self)
    164       return self._options
    165 
--> 166     from google.protobuf import descriptor_pb2
    167     try:
    168       options_class = getattr(descriptor_pb2,

/usr/local/lib/python3.11/dist-packages/google/protobuf/descriptor.py in _LazyLoadOptions(self)
    203       serialized_start: The start index (inclusive) in block in the
    204         file.serialized_pb that describes this descriptor.
--> 205       serialized_end: The end index (exclusive) in block in the
    206         file.serialized_pb that describes this descriptor.
    207       serialized_options: Protocol message serialized options or None.

RuntimeError: Unknown options class name FieldOptions!

## === cell 1
def remove_datapoints_from_water(df):
    """
    Kaggle offline environment: URL-based mask download isn't available.
    To preserve pipeline execution with minimal core-logic change, skip this step.
    """
    return df


def clean(df):
    print(" Old size: %d" % len(df))
    df = df.dropna(how="any", axis="rows")
    print(" New size after dropna: %d" % len(df))

    df = df[
        (df["dropoff_longitude"] != df["pickup_longitude"])
        & (df["dropoff_latitude"] != df["pickup_latitude"])
    ]
    print(" New size after removing same long lat: %d" % len(df))

    df = df[
        (df["dropoff_longitude"] != 0)
        & (df["pickup_longitude"] != 0)
        & (df["dropoff_latitude"] != 0)
        & (df["pickup_latitude"] != 0)
    ]
    print(" New size after removing 0 long lat: %d" % len(df))

    MinMax = (-74.5, -72.8, 40.5, 41.8)
    df = df[
        (MinMax[0] <= df["pickup_longitude"]) & (df["pickup_longitude"] <= MinMax[1])
    ]
    df = df[
        (MinMax[0] <= df["dropoff_longitude"]) & (df["dropoff_longitude"] <= MinMax[1])
    ]
    df = df[(MinMax[2] <= df["pickup_latitude"]) & (df["pickup_latitude"] <= MinMax[3])]
    df = df[
        (MinMax[2] <= df["dropoff_latitude"]) & (df["dropoff_latitude"] <= MinMax[3])
    ]

    print(" New size after only NYC: %d" % len(df))
    df = df[(0 < df["fare_amount"]) & (df["fare_amount"] <= 50)]
    print(" New size after removing outliers: %d" % len(df))

    df = df[(df["passenger_count"] > 0) & (df["passenger_count"] <= 6)]
    print(" New size after removing 6=>passenger_count > 0 : %d" % len(df))

    nyc_coord = (40.7141667, -74.0063889)
    fk_coord = (40.639722, -73.778889)
    ewr_coord = (40.6925, -74.168611)
    lga_coord = (40.77725, -73.872611)
    sol_coord = (40.6892, -74.0445)

    df = df[
        (nyc_coord[1] != df["pickup_longitude"])
        & (df["pickup_latitude"] != nyc_coord[0])
    ]
    df = df[
        (nyc_coord[1] != df["dropoff_longitude"])
        & (df["dropoff_latitude"] != nyc_coord[0])
    ]
    print(" New size after NY airport: %d" % len(df))

    df = df[
        (fk_coord[1] != df["pickup_longitude"]) & (df["pickup_latitude"] != fk_coord[0])
    ]
    df = df[
        (fk_coord[1] != df["dropoff_longitude"])
        & (df["dropoff_latitude"] != fk_coord[0])
    ]
    print(" New size after jfk airport: %d" % len(df))

    df = df[
        (ewr_coord[1] != df["pickup_longitude"])
        & (df["pickup_latitude"] != ewr_coord[0])
    ]
    df = df[
        (ewr_coord[1] != df["dropoff_longitude"])
        & (df["dropoff_latitude"] != ewr_coord[0])
    ]
    print(" New size after ewr airport: %d" % len(df))

    df = df[
        (lga_coord[1] != df["pickup_longitude"])
        & (df["pickup_latitude"] != lga_coord[0])
    ]
    df = df[
        (lga_coord[1] != df["dropoff_longitude"])
        & (df["dropoff_latitude"] != lga_coord[0])
    ]
    print(" New size after lgr airport: %d" % len(df))

    df = df[
        (sol_coord[1] != df["pickup_longitude"])
        & (df["pickup_latitude"] != sol_coord[0])
    ]
    df = df[
        (sol_coord[1] != df["dropoff_longitude"])
        & (df["dropoff_latitude"] != sol_coord[0])
    ]
    print(" New size after sol removed: %d" % len(df))

    print("Old size: %d" % len(df))
    df = remove_datapoints_from_water(df)
    print("New size: %d" % len(df))

    print(" New size: %d" % len(df))
    return df


def late_night(row):
    return 1 if (row["hour"] <= 3) else 0


def night(row):
    return 1 if ((row["hour"] > 20) and (row["weekday"] < 5)) else 0


def rush_hour(row):
    return (
        1
        if ((row["hour"] <= 20) and (row["hour"] >= 16) and (row["weekday"] < 5))
        else 0
    )


def manhattan(pickup_lat, pickup_long, dropoff_lat, dropoff_long):
    return np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)


def add_time_features(df):
    dt = pd.to_datetime(df["pickup_datetime"], errors="coerce", utc=True)

    df["year"] = dt.dt.year.fillna(0).astype("int16")
    df["month"] = dt.dt.month.fillna(0).astype("int8")
    df["day"] = dt.dt.day.fillna(0).astype("int8")
    df["hour"] = dt.dt.hour.fillna(0).astype("int8")
    df["weekday"] = dt.dt.weekday.fillna(0).astype("int8")

    df["pickup_datetime"] = dt.dt.strftime("%Y-%m-%d %H:%M:%S%z").fillna(
        df["pickup_datetime"].astype(str)
    )

    df["night"] = df.apply(night, axis=1).astype("int8")
    df["late_night"] = df.apply(late_night, axis=1).astype("int8")
    df["rush_hour"] = df.apply(rush_hour, axis=1).astype("int8")
    return df


def add_coordinate_features(df):
    lat1 = df["pickup_latitude"]
    lat2 = df["dropoff_latitude"]
    lon1 = df["pickup_longitude"]
    lon2 = df["dropoff_longitude"]
    df["latdiff"] = lat1 - lat2
    df["londiff"] = lon1 - lon2
    return df


def add_distances_features(df):
    lat1 = df["pickup_latitude"]
    lat2 = df["dropoff_latitude"]
    lon1 = df["pickup_longitude"]
    lon2 = df["dropoff_longitude"]
    df["manhattan"] = manhattan(lat1, lon1, lat2, lon2)
    df["distance"] = np.sqrt(np.abs(lon1 - lon2) ** 2 + np.abs(lat1 - lat2) ** 2)
    return df


def output_submission(raw_test, prediction, id_column, prediction_column, file_name):
    pred_1d = np.asarray(prediction).reshape(-1).astype("float32")
    out = pd.DataFrame(
        {id_column: raw_test[id_column].values, prediction_column: pred_1d}
    )
    out.to_csv(file_name, index=False)
    print("Output complete:", file_name, "rows:", len(out))


def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


def plot_loss_accuracy_rmse(history):
    plt.figure(figsize=(20, 6))
    plt.plot(history.history.get("loss", []))
    plt.plot(history.history.get("val_loss", []))
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper right")
    plt.show()

    if "rmse" in history.history:
        plt.figure(figsize=(20, 6))
        plt.plot(history.history.get("rmse", []))
        plt.plot(history.history.get("val_rmse", []))
        plt.title("Model rmse")
        plt.ylabel("rmse")
        plt.xlabel("epoch")
        plt.legend(["train", "val"], loc="upper right")
        plt.show()




## === cell 2
datatypes = {
    "key": "str",
    "fare_amount": "float32",
    "pickup_datetime": "str",
    "pickup_longitude": "float32",
    "pickup_latitude": "float32",
    "dropoff_longitude": "float32",
    "dropoff_latitude": "float32",
    "passenger_count": "uint8",
}

trainKaggle = pd.read_csv(
    TRAIN_PATH,
    nrows=DATASET_SIZE,
    dtype=datatypes,
    usecols=[
        "key",
        "fare_amount",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ],
)

testKaggle = pd.read_csv(
    TEST_PATH, dtype={k: v for k, v in datatypes.items() if k != "fare_amount"}
)
print("Loaded trainKaggle columns:", list(trainKaggle.columns))
print("Loaded testKaggle columns:", list(testKaggle.columns))



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2842994017.py in <cell line: 0>()
     11 
     12 trainKaggle = pd.read_csv(
---> 13     TRAIN_PATH,
     14     nrows=DATASET_SIZE,
     15     dtype=datatypes,

NameError: name 'TRAIN_PATH' is not defined

## === cell 3
train_df, test_df = train_test_split(trainKaggle, test_size=0.50, random_state=1)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2210393952.py in <cell line: 0>()
----> 1 train_df, test_df = train_test_split(trainKaggle, test_size=0.50, random_state=1)
      2 

NameError: name 'trainKaggle' is not defined

## === cell 4
print("testKaggle Size %d" % len(testKaggle))
print("train_df Size %d" % len(train_df))
print("test_df Size %d" % len(test_df))



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/395627971.py in <cell line: 0>()
----> 1 print("testKaggle Size %d" % len(testKaggle))
      2 print("train_df Size %d" % len(train_df))
      3 print("test_df Size %d" % len(test_df))
      4 

NameError: name 'testKaggle' is not defined

## === cell 5
train_df.describe()



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1577644986.py in <cell line: 0>()
----> 1 train_df.describe()
      2 

NameError: name 'train_df' is not defined

## === cell 6
test_df.describe()



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4294142452.py in <cell line: 0>()
----> 1 test_df.describe()
      2 

NameError: name 'test_df' is not defined

## === cell 7
print("train_df clean")
train_df = clean(train_df)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4279415965.py in <cell line: 0>()
      1 print("train_df clean")
----> 2 train_df = clean(train_df)
      3 

NameError: name 'train_df' is not defined

## === cell 8
train_df.describe()



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1577644986.py in <cell line: 0>()
----> 1 train_df.describe()
      2 

NameError: name 'train_df' is not defined

## === cell 9
print("train_df add_time_features")
train_df = add_time_features(train_df)
print("test_df add_time_features")
test_df = add_time_features(test_df)
print("testKaggle add_time_features")
testKaggle = add_time_features(testKaggle)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1145226127.py in <cell line: 0>()
      1 print("train_df add_time_features")
----> 2 train_df = add_time_features(train_df)
      3 print("test_df add_time_features")
      4 test_df = add_time_features(test_df)
      5 print("testKaggle add_time_features")

NameError: name 'train_df' is not defined

## === cell 10
train_df.describe()



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1577644986.py in <cell line: 0>()
----> 1 train_df.describe()
      2 

NameError: name 'train_df' is not defined

## === cell 11
print("train_df add_coordinate_features")
train_df = add_coordinate_features(train_df)
print("test_df add_coordinate_features")
test_df = add_coordinate_features(test_df)
print("testKaggle add_coordinate_features")
testKaggle = add_coordinate_features(testKaggle)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1245147733.py in <cell line: 0>()
      1 print("train_df add_coordinate_features")
----> 2 train_df = add_coordinate_features(train_df)
      3 print("test_df add_coordinate_features")
      4 test_df = add_coordinate_features(test_df)
      5 print("testKaggle add_coordinate_features")

NameError: name 'train_df' is not defined

## === cell 12
train_df.describe()



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1577644986.py in <cell line: 0>()
----> 1 train_df.describe()
      2 

NameError: name 'train_df' is not defined

## === cell 13
print("train_df add_distances_features")
train_df = add_distances_features(train_df)
print("test_df add_distances_features")
test_df = add_distances_features(test_df)
print("testKaggle add_distances_features")
testKaggle = add_distances_features(testKaggle)
print("Done with Adding features")



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1336277678.py in <cell line: 0>()
      1 print("train_df add_distances_features")
----> 2 train_df = add_distances_features(train_df)
      3 print("test_df add_distances_features")
      4 test_df = add_distances_features(test_df)
      5 print("testKaggle add_distances_features")

NameError: name 'train_df' is not defined

## === cell 14
train_df.describe()



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1577644986.py in <cell line: 0>()
----> 1 train_df.describe()
      2 

NameError: name 'train_df' is not defined

## === cell 15
dropped_columns = ["pickup_datetime", "key"]

train_df = train_df.drop(dropped_columns, axis=1)
test_df = test_df.drop(dropped_columns, axis=1)
testKaggle_clean = testKaggle.drop(["pickup_datetime", "key"], axis=1)

print("Done with dropped_columns")



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3021200048.py in <cell line: 0>()
      1 dropped_columns = ["pickup_datetime", "key"]
      2 
----> 3 train_df = train_df.drop(dropped_columns, axis=1)
      4 test_df = test_df.drop(dropped_columns, axis=1)
      5 testKaggle_clean = testKaggle.drop(["pickup_datetime", "key"], axis=1)

NameError: name 'train_df' is not defined

## === cell 16
train_df.shape



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/186828624.py in <cell line: 0>()
----> 1 train_df.shape
      2 

NameError: name 'train_df' is not defined

## === cell 17
train_df.describe()



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1577644986.py in <cell line: 0>()
----> 1 train_df.describe()
      2 

NameError: name 'train_df' is not defined

## === cell 18
test_df.describe()



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4294142452.py in <cell line: 0>()
----> 1 test_df.describe()
      2 

NameError: name 'test_df' is not defined

## === cell 19
train_df, validation_df = train_test_split(train_df, test_size=0.10, random_state=1)



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1154705564.py in <cell line: 0>()
----> 1 train_df, validation_df = train_test_split(train_df, test_size=0.10, random_state=1)
      2 

NameError: name 'train_df' is not defined

## === cell 20
train_labels = train_df["fare_amount"].values.astype("float32")
validation_labels = validation_df["fare_amount"].values.astype("float32")
test_labels = test_df["fare_amount"].values.astype("float32")

train_df = train_df.drop(["fare_amount"], axis=1)
validation_df = validation_df.drop(["fare_amount"], axis=1)
test_df = test_df.drop(["fare_amount"], axis=1)

print("Done with Labels")



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2024178754.py in <cell line: 0>()
----> 1 train_labels = train_df["fare_amount"].values.astype("float32")
      2 validation_labels = validation_df["fare_amount"].values.astype("float32")
      3 test_labels = test_df["fare_amount"].values.astype("float32")
      4 
      5 train_df = train_df.drop(["fare_amount"], axis=1)

NameError: name 'train_df' is not defined

## === cell 21
scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df)
validation_df_scaled = scaler.transform(validation_df)
test_scaled = scaler.transform(test_df)
testKaggle_scaled = scaler.transform(testKaggle_clean)

label_scaler = preprocessing.MinMaxScaler()
train_labels_scaled = (
    label_scaler.fit_transform(train_labels.reshape(-1, 1))
    .reshape(-1)
    .astype("float32")
)
validation_labels_scaled = (
    label_scaler.transform(validation_labels.reshape(-1, 1))
    .reshape(-1)
    .astype("float32")
)
test_labels_scaled = (
    label_scaler.transform(test_labels.reshape(-1, 1)).reshape(-1).astype("float32")
)



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1011097418.py in <cell line: 0>()
      1 scaler = preprocessing.MinMaxScaler()
----> 2 train_df_scaled = scaler.fit_transform(train_df)
      3 validation_df_scaled = scaler.transform(validation_df)
      4 test_scaled = scaler.transform(test_df)
      5 testKaggle_scaled = scaler.transform(testKaggle_clean)

NameError: name 'train_df' is not defined

## === cell 22
model = Sequential()
model.add(
    Dense(
        256,
        activation="relu",
        input_dim=train_df_scaled.shape[1],
        activity_regularizer=regularizers.l1(0.01),
    )
)
model.add(BatchNormalization())
model.add(Dense(128, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(64, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(32, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(8, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(1))

adam = optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(loss="mean_squared_error", optimizer=adam, metrics=["mae", rmse, "mse"])

print("Dataset size: %s" % DATASET_SIZE)
print("Epochs: %s" % EPOCHS)
print("Learning rate: %s" % LEARNING_RATE)
print("Batch size: %s" % BATCH_SIZE)
print("Input dimension: %s" % train_df_scaled.shape[1])
print("Features used: %s" % list(train_df.columns))
model.summary()

history = model.fit(
    x=train_df_scaled,
    y=train_labels_scaled,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=1,
    validation_data=(validation_df_scaled, validation_labels_scaled),
    shuffle=True,
)



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/591154134.py in <cell line: 0>()
----> 1 model = Sequential()
      2 model.add(
      3     Dense(
      4         256,
      5         activation="relu",

NameError: name 'Sequential' is not defined

## === cell 23
plot_loss_accuracy_rmse(history)



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2579130885.py in <cell line: 0>()
----> 1 plot_loss_accuracy_rmse(history)
      2 

NameError: name 'history' is not defined

## === cell 24
score = model.evaluate(test_scaled, test_labels_scaled, verbose=1)
print(score)



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1868949165.py in <cell line: 0>()
----> 1 score = model.evaluate(test_scaled, test_labels_scaled, verbose=1)
      2 print(score)
      3 

NameError: name 'model' is not defined

## === cell 25
prediction = model.predict(test_scaled, batch_size=128, verbose=1)
predictionKaggle = model.predict(testKaggle_scaled, batch_size=128, verbose=1)

prediction = label_scaler.inverse_transform(
    np.asarray(prediction).reshape(-1, 1)
).reshape(-1)
predictionKaggle = label_scaler.inverse_transform(
    np.asarray(predictionKaggle).reshape(-1, 1)
).reshape(-1)



## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1212291988.py in <cell line: 0>()
----> 1 prediction = model.predict(test_scaled, batch_size=128, verbose=1)
      2 predictionKaggle = model.predict(testKaggle_scaled, batch_size=128, verbose=1)
      3 
      4 prediction = label_scaler.inverse_transform(
      5     np.asarray(prediction).reshape(-1, 1)

NameError: name 'model' is not defined

## === cell 26
predictionKaggle = np.maximum(predictionKaggle, 0)
output_submission(testKaggle, predictionKaggle, "key", "fare_amount", SUBMISSION_NAME)

assert SUBMISSION_NAME.endswith(".csv")
assert os.path.exists(SUBMISSION_NAME), "Submission file was not created."
sub_preview = pd.read_csv(SUBMISSION_NAME)
assert list(sub_preview.columns) == ["key", "fare_amount"], "Wrong submission columns."
assert len(sub_preview) == len(testKaggle), "Submission row count mismatch."



## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1364491576.py in <cell line: 0>()
----> 1 predictionKaggle = np.maximum(predictionKaggle, 0)
      2 output_submission(testKaggle, predictionKaggle, "key", "fare_amount", SUBMISSION_NAME)
      3 
      4 assert SUBMISSION_NAME.endswith(".csv")
      5 assert os.path.exists(SUBMISSION_NAME), "Submission file was not created."

NameError: name 'predictionKaggle' is not defined

## === cell 27
print("Example prediction:", float(np.asarray(prediction).reshape(-1)[0]))
print("Example true label:", float(test_labels[0]))
print("Saved submission file exists:", os.path.exists(SUBMISSION_NAME))
print("Submission preview:")
print(pd.read_csv(SUBMISSION_NAME).head())

## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2569006339.py in <cell line: 0>()
----> 1 print("Example prediction:", float(np.asarray(prediction).reshape(-1)[0]))
      2 print("Example true label:", float(test_labels[0]))
      3 print("Saved submission file exists:", os.path.exists(SUBMISSION_NAME))
      4 print("Submission preview:")
      5 print(pd.read_csv(SUBMISSION_NAME).head())

NameError: name 'prediction' is not defined
